from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class MedNeXtBlock3d(nn.Module):
    """ConvNeXt-style 3D block (Roy et al., MedNeXt, 2023):
    depthwise kxkxk conv -> GroupNorm -> 1x1 expand (x exp_r) -> GELU
    -> 1x1 compress, plus a residual. Large-kernel depthwise convs play the
    role of attention (spatial mixing), the 1x1 convs the role of the MLP."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        stride: tuple[int, int, int] = (1, 1, 1),
        dropout: float = 0.0,
        transpose: bool = False,
    ):
        super().__init__()
        conv_cls = nn.ConvTranspose3d if transpose else nn.Conv3d
        # output_padding makes a transposed conv exactly multiply the size by the stride
        extra = {"output_padding": tuple(s - 1 for s in stride)} if transpose else {}

        self.dw_conv = conv_cls(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_ch,
            **extra,
        )
        # One group per channel: per-channel normalization, like InstanceNorm
        self.norm = nn.GroupNorm(num_groups=in_ch, num_channels=in_ch)
        self.expand = nn.Conv3d(in_ch, exp_r * in_ch, kernel_size=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(exp_r * in_ch, out_ch, kernel_size=1)

        self.skip: nn.Module = nn.Identity()
        if in_ch != out_ch or any(s != 1 for s in stride):
            self.skip = conv_cls(in_ch, out_ch, kernel_size=1, stride=stride, **extra)

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.compress(self.act(self.expand(self.norm(self.dw_conv(x)))))
        return self.dropout(out + self.skip(x))


class MedNeXt3D(nn.Module):
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
        exp_r: Sequence[int] | None = None,
        kernel_size: int = 3,
        deep_supervision: bool = False,
        grad_checkpoint: bool = False,
        **kwargs,
    ):
        """
        kernels, factor, pool_kernels, max_channels: same meaning as in UNet3D.py.
            pool_kernels are the strides of the down/up blocks; use (1, 2, 2)
            entries to avoid downsampling the z axis.
        num_blocks: MedNeXt blocks per level (length == len(pool_kernels) + 1,
            last entry = bottleneck). Decoder levels reuse the encoder counts.
            Defaults to 2 everywhere (close to MedNeXt-S). MedNeXt-L uses
            3, 4, 8, 8, 8 with 4 downsamplings.
        exp_r: expansion ratio per level (same length as num_blocks).
            Defaults to 2, 3, 4, 4, ... (close to MedNeXt-B).
        kernel_size: depthwise kernel size (3 or 5). Train with 3 first, then
            initialize a kernel-5 model from it with upkern_load_weights().
        deep_supervision: same behaviour as in ResUNet3D.py (list of logits in
            training mode, full-resolution logits in eval mode).
        grad_checkpoint: recompute each level's activations in the backward
            pass instead of storing them (less memory, ~30% slower).
        """
        super().__init__()
        self.pool_kernels = [tuple(k) for k in pool_kernels]
        depth = len(self.pool_kernels)
        chans = [min(kernels * factor**i, max_channels) for i in range(depth + 1)]

        if num_blocks is None:
            num_blocks = [2] * (depth + 1)
        if exp_r is None:
            exp_r = [min(i + 2, 4) for i in range(depth + 1)]
        assert len(num_blocks) == depth + 1, "num_blocks must have one entry per level"
        assert len(exp_r) == depth + 1, "exp_r must have one entry per level"
        self.deep_supervision = deep_supervision
        self.grad_checkpoint = grad_checkpoint

        def level(c: int, n: int, r: int) -> nn.Sequential:
            return nn.Sequential(
                *(MedNeXtBlock3d(c, c, r, kernel_size, dropout=dropoutRate) for _ in range(n))
            )

        self.stem = nn.Conv3d(in_dim, chans[0], kernel_size=1)

        # Encoder: blocks at each level, then a strided MedNeXt block that
        # doubles the channels (learned downsampling)
        self.encoders = nn.ModuleList(level(chans[i], num_blocks[i], exp_r[i]) for i in range(depth))
        self.downs = nn.ModuleList(
            MedNeXtBlock3d(
                chans[i], chans[i + 1], exp_r[i + 1], kernel_size, self.pool_kernels[i], dropoutRate
            )
            for i in range(depth)
        )

        self.bottleneck = level(chans[-1], num_blocks[-1], exp_r[-1])

        # Decoder (deepest first): transposed strided MedNeXt block that halves
        # the channels, add the skip (MedNeXt sums instead of concatenating),
        # then blocks at that level
        self.ups = nn.ModuleList(
            MedNeXtBlock3d(
                chans[i + 1],
                chans[i],
                exp_r[i],
                kernel_size,
                self.pool_kernels[i],
                dropoutRate,
                transpose=True,
            )
            for i in reversed(range(depth))
        )
        self.decoders = nn.ModuleList(
            level(chans[i], num_blocks[i], exp_r[i]) for i in reversed(range(depth))
        )

        self.final = nn.Conv3d(chans[0], out_dim, kernel_size=1)

        # Deep supervision heads for every decoder level except the last
        # (which is self.final), deepest first like the decoders
        self.ds_heads = nn.ModuleList(
            nn.Conv3d(chans[i], out_dim, kernel_size=1) for i in reversed(range(1, depth))
        )

    def _run(self, module: nn.Module, x: Tensor) -> Tensor:
        if self.grad_checkpoint and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        x = self.stem(x)

        skips: list[Tensor] = []
        for enc, down in zip(self.encoders, self.downs):
            x = self._run(enc, x)
            skips.append(x)
            x = self._run(down, x)

        x = self._run(self.bottleneck, x)

        use_ds = self.deep_supervision and self.training
        ds_outputs: list[Tensor] = []
        for i, (up, dec, skip) in enumerate(zip(self.ups, self.decoders, reversed(skips))):
            x = self._run(up, x)
            if x.shape[2:] != skip.shape[2:]:  # odd input sizes
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = self._run(dec, x + skip)
            if use_ds and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))

        out = self.final(x)
        if use_ds:
            return [out] + ds_outputs[::-1]  # full res first, then coarser
        return out

    def init_weights(self, *args, **kwargs):
        def _init(m: nn.Module):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)


def upkern_load_weights(target: MedNeXt3D, source_state: dict[str, Tensor]) -> MedNeXt3D:
    """UpKern (MedNeXt paper): initialize a large-kernel model (e.g. kernel 5)
    from a trained small-kernel one (kernel 3) with the same architecture.
    Weights with matching shapes are copied; depthwise conv weights whose
    spatial size differs are trilinearly upsampled to the new kernel size."""
    target_state = target.state_dict()
    for name, weight in target_state.items():
        src = source_state[name]
        if src.shape == weight.shape:
            target_state[name] = src.clone()
        elif src.ndim == 5 and src.shape[:2] == weight.shape[:2]:
            target_state[name] = F.interpolate(
                src.float(), size=weight.shape[2:], mode="trilinear", align_corners=True
            ).to(weight.dtype)
        else:
            raise ValueError(f"Cannot UpKern {name}: {tuple(src.shape)} -> {tuple(weight.shape)}")
    target.load_state_dict(target_state)
    return target
