"""Benchmark harness: runs the real training loop (runTraining) for 1 epoch with a
small number of batches, to measure end-to-end per-batch wall time.

Run on a GPU node from the project root, e.g.:
    uv run python snellius_bench.py \
        --data_path /scratch-shared/scur0088/data \
        --dataset SEGTHOR --batch_size 8 --num_workers 8 --batches_per_epoch 10

Useful flags:
    --profile           enable torch.profiler (chrome trace saved to <dest>/profiler)
    --hd95_in_train     compute HD95 during training as well (only recognized by the
                        performance branch; ignored by the old code)
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")

import autoroot  # noqa

import torch
import wandb

wandb.init = lambda *a, **k: None
wandb.log = lambda *a, **k: None
wandb.log_artifact = lambda *a, **k: None
wandb.watch = lambda *a, **k: None
wandb.finish = lambda *a, **k: None

from src.train import runTraining  # noqa: E402
from src.utils.config import (  # noqa: E402
    Config,
    DatasetConfig,
    ModelConfig,
    ProfilerConfig,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--dataset", type=str, default="SEGTHOR")
    ap.add_argument("--box", type=int, nargs=3, default=[132, 132, 128])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batches_per_epoch", type=int, default=10)
    ap.add_argument("--dest", type=str, default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--hd95_in_train", action="store_true")
    args = ap.parse_args()

    print(
        f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    profiler = (
        ProfilerConfig(
            enabled=True,
            wait=2,
            warmup=2,
            active=max(1, args.batches_per_epoch - 4),
        )
        if args.profile
        else ProfilerConfig()
    )

    kwargs = dict(
        dataset=DatasetConfig(
            name=args.dataset, num_classes=5, box_size=tuple(args.box)
        ),
        model=ModelConfig(name="UNet3D", kernels=8, factor=2),
        is_3d=True,
        gpu=torch.cuda.is_available(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        batches_per_epoch=args.batches_per_epoch,
        epochs=1,
        mode="full",
        loss="ce",
        profiler=profiler,
    )
    if args.data_path:
        kwargs["data_path"] = Path(args.data_path)
    if args.dest:
        kwargs["dest"] = Path(args.dest)
    config = Config(**kwargs)
    if args.hd95_in_train and hasattr(config, "hd95_in_train"):
        config.hd95_in_train = True

    t0 = time.perf_counter()
    runTraining(config)
    print(
        f">> runTraining wall time (1 epoch, train+val): {time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
