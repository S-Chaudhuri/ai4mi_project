from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock3d(nn.Module):
    """(Conv3d -> InstanceNorm -> LeakyReLU) x 2, then optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock3d(nn.Module):
    """Upsample to the skip's size, concat the skip, then ConvBlock3d."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock3d(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class UNet3D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernels: int = 16,
        factor: int = 2,
        dropoutRate: float = 0.01,
        pool_kernels: Sequence[tuple[int, int, int]] = ((2, 2, 2), (2, 2, 2)),
        max_channels: int = 256,
        **kwargs,
    ):
        """
        kernels:      channels at the first level (was `K` in ENet)
        factor:       channel multiplier between levels (2 = double at each level)
        pool_kernels: one (d, h, w) pooling per downsampling step; the number of
                      entries sets the depth. Default: 2 steps, which suits a
                      132x132x128 box (132 -> 66 -> 33). Use (1, 2, 2) entries to
                      avoid pooling the z axis.
        """
        super().__init__()
        self.pool_kernels = [tuple(k) for k in pool_kernels]
        chans = [
            min(kernels * factor**i, max_channels)
            for i in range(len(self.pool_kernels) + 1)
        ]

        # Encoder
        self.encoders = nn.ModuleList()
        prev = in_dim
        for c in chans[:-1]:
            self.encoders.append(ConvBlock3d(prev, c, dropoutRate))
            prev = c
        self.pools = nn.ModuleList(nn.MaxPool3d(k, stride=k) for k in self.pool_kernels)

        # Bottleneck
        self.bottleneck = ConvBlock3d(chans[-2], chans[-1], dropoutRate)

        # Decoder (deepest first)
        self.decoders = nn.ModuleList(
            UpBlock3d(chans[i + 1], chans[i], chans[i], dropoutRate)
            for i in reversed(range(len(chans) - 1))
        )

        self.final = nn.Conv3d(chans[0], out_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.final(x)

    def init_weights(self, *args, **kwargs):
        def _init(m: nn.Module):
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)
