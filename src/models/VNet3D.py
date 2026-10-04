from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualStage3d(nn.Module):
    """num_convs x (Conv3d(5x5x5) -> norm -> PReLU), with the stage's input
    added back before the final PReLU (V-Net's core residual-learning idea).
    A 1x1 conv projects the input if in_ch != out_ch, same as a ResNet
    identity-vs-projection shortcut."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_convs: int,
        dropout: float = 0.0,
        norm_layer: type[nn.Module] = nn.InstanceNorm3d,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_ch
        for i in range(num_convs):
            layers.append(nn.Conv3d(ch, out_ch, kernel_size=5, padding=2, bias=False))
            layers.append(norm_layer(out_ch, affine=True) if norm_layer is not nn.Identity else nn.Identity())
            if i < num_convs - 1:  # activation after every conv except the last;
                layers.append(nn.PReLU(out_ch))  # the last one waits for the residual add
            ch = out_ch
        self.convs = nn.Sequential(*layers)

        self.project = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.final_act = nn.PReLU(out_ch)
        if dropout > 0:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.project(x)
        out = self.convs(x)
        return self.dropout(self.final_act(out + residual))


class DownTransition3d(nn.Module):
    """Learned downsampling: strided Conv3d instead of MaxPool3d."""

    def __init__(self, in_ch: int, out_ch: int, pool_kernel: tuple[int, int, int]):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=pool_kernel, stride=pool_kernel, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.PReLU(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class UpTransition3d(nn.Module):
    """Learned upsampling: ConvTranspose3d, corrected to the skip's exact
    size (see module docstring), then concat + a residual stage."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        pool_kernel: tuple[int, int, int],
        num_convs: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=pool_kernel, stride=pool_kernel),
            nn.InstanceNorm3d(skip_ch, affine=True),
            nn.PReLU(skip_ch),
        )
        self.stage = ResidualStage3d(skip_ch * 2, out_ch, num_convs, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.stage(torch.cat((x, skip), dim=1))


class VNet3D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernels: int = 16,
        factor: int = 2,
        dropoutRate: float = 0.01,
        pool_kernels: Sequence[tuple[int, int, int]] = ((2, 2, 2), (2, 2, 2)),
        max_channels: int = 256,
        num_convs: Sequence[int] | None = None,
        norm_layer: type[nn.Module] = nn.InstanceNorm3d,
        **kwargs,
    ):
        """
        kernels, factor, pool_kernels, max_channels: same meaning as in UNet3D.py.
        num_convs: how many conv layers per encoder stage. Defaults to V-Net's
            own 1,2,3,3,3,... progression (one entry per stage, decoder stages
            reuse the same counts in reverse). Pass your own sequence (length
            == len(pool_kernels) + 1) to override.
        norm_layer: nn.InstanceNorm3d by default (stability at small batch
            sizes); pass nn.Identity for the paper-original "no norm" setup.
        """
        super().__init__()
        self.pool_kernels = [tuple(k) for k in pool_kernels]
        depth = len(self.pool_kernels)
        chans = [min(kernels * factor**i, max_channels) for i in range(depth + 1)]

        if num_convs is None:
            num_convs = [min(i + 1, 3) for i in range(depth + 1)]
        assert len(num_convs) == depth + 1, "num_convs must have one entry per stage"
        self.num_convs = list(num_convs)

        # Encoder: one residual stage per level, then a learned downsample
        self.encoders = nn.ModuleList()
        prev = in_dim
        for c, n in zip(chans[:-1], self.num_convs[:-1]):
            self.encoders.append(ResidualStage3d(prev, c, n, dropoutRate, norm_layer))
            prev = c
        self.downs = nn.ModuleList(
            DownTransition3d(c, c, k) for c, k in zip(chans[:-1], self.pool_kernels)
        )

        # Bottleneck
        self.bottleneck = ResidualStage3d(
            chans[-2], chans[-1], self.num_convs[-1], dropoutRate, norm_layer
        )

        # Decoder (deepest first)
        self.decoders = nn.ModuleList(
            UpTransition3d(
                chans[i + 1], chans[i], chans[i], self.pool_kernels[i], self.num_convs[i], dropoutRate
            )
            for i in reversed(range(depth))
        )

        self.final = nn.Conv3d(chans[0], out_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.final(x)

    def init_weights(self, *args, **kwargs):
        def _init(m: nn.Module):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)


if __name__ == "__main__":
    net = VNet3D(1, 5, kernels=8)
    net.init_weights()

    x = torch.randn(2, 1, 36, 36, 32)
    y = net(x)
    assert y.shape == (2, 5, 36, 36, 32), y.shape
    y.mean().backward()

    # Odd sizes: downsampling floor-divides, upsampling corrects back via interpolate
    y = net(torch.randn(1, 1, 33, 35, 31))
    assert y.shape == (1, 5, 33, 35, 31), y.shape

    # paper-original: no norm layer
    net_nonorm = VNet3D(1, 5, kernels=8, norm_layer=nn.Identity)
    net_nonorm.init_weights()
    y = net_nonorm(torch.randn(1, 1, 36, 36, 32))
    assert y.shape == (1, 5, 36, 36, 32), y.shape

    print("OK", sum(p.numel() for p in net.parameters()), "params")