from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock3d(nn.Module):
    """Conv3d -> norm -> LeakyReLU -> Conv3d -> norm, plus a skip connection,
    then LeakyReLU. The first conv can be strided to downsample; the skip then
    becomes a strided 1x1 conv so both paths have the same shape."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )

        self.skip: nn.Module = nn.Identity()
        if in_ch != out_ch or any(s != 1 for s in stride):
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )

        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.act(self.convs(x) + self.skip(x)))


class EncoderStage3d(nn.Module):
    """num_blocks ResBlock3d's. The first one is strided (except at the top
    level), so downsampling is learned instead of MaxPool3d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        dropout: float = 0.0,
    ):
        super().__init__()
        blocks = [ResBlock3d(in_ch, out_ch, stride, dropout)]
        blocks += [ResBlock3d(out_ch, out_ch, dropout=dropout) for _ in range(num_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DecoderStage3d(nn.Module):
    """ConvTranspose3d up to the skip's resolution (corrected to its exact
    size for odd inputs), concat the skip, then one ResBlock3d."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        pool_kernel: tuple[int, int, int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=pool_kernel, stride=pool_kernel)
        self.block = ResBlock3d(2 * skip_ch, skip_ch, dropout=dropout)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class ResUNet3D(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernels: int = 16,
        factor: int = 2,
        dropoutRate: float = 0.01,
        pool_kernels: Sequence[tuple[int, int, int]] = ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
        max_channels: int = 256,
        num_blocks: Sequence[int] | None = None,
        deep_supervision: bool = False,
        **kwargs,
    ):
        """
        kernels, factor, pool_kernels, max_channels: same meaning as in UNet3D.py.
            pool_kernels here are the strides of the strided convs; use
            (1, 2, 2) entries to avoid downsampling the z axis.
        num_blocks: ResBlock3d's per encoder stage (length == len(pool_kernels) + 1).
            Defaults to 1 at the top level and 2 below it. The decoder always
            uses a single block per stage.
        deep_supervision: when True and in training mode, forward() returns a
            list of logits [full res, 1/2, 1/4, ...] (one per decoder level);
            the loss must then handle the list. In eval mode it always returns
            only the full-resolution logits.
        """
        super().__init__()
        self.pool_kernels = [tuple(k) for k in pool_kernels]
        depth = len(self.pool_kernels)
        chans = [min(kernels * factor**i, max_channels) for i in range(depth + 1)]

        if num_blocks is None:
            num_blocks = [1] + [2] * depth
        assert len(num_blocks) == depth + 1, "num_blocks must have one entry per stage"
        self.num_blocks = list(num_blocks)
        self.deep_supervision = deep_supervision

        # Encoder: the last stage is the bottleneck
        strides = [(1, 1, 1)] + self.pool_kernels
        self.encoders = nn.ModuleList()
        prev = in_dim
        for c, n, s in zip(chans, self.num_blocks, strides):
            self.encoders.append(EncoderStage3d(prev, c, n, s, dropoutRate))
            prev = c

        # Decoder (deepest first)
        self.decoders = nn.ModuleList(
            DecoderStage3d(chans[i + 1], chans[i], self.pool_kernels[i], dropoutRate)
            for i in reversed(range(depth))
        )

        self.final = nn.Conv3d(chans[0], out_dim, kernel_size=1)

        # Deep supervision heads for every decoder level except the last
        # (which is self.final), deepest first like the decoders
        self.ds_heads = nn.ModuleList(
            nn.Conv3d(chans[i], out_dim, kernel_size=1) for i in reversed(range(1, depth))
        )

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        skips: list[Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        x = skips.pop()  # bottleneck output

        use_ds = self.deep_supervision and self.training
        ds_outputs: list[Tensor] = []
        for i, (dec, skip) in enumerate(zip(self.decoders, reversed(skips))):
            x = dec(x, skip)
            if use_ds and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))

        out = self.final(x)
        if use_ds:
            return [out] + ds_outputs[::-1]  # full res first, then coarser
        return out

    def init_weights(self, *args, **kwargs):
        def _init(m: nn.Module):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)
