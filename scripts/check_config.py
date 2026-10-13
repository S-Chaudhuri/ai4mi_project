#!/usr/bin/env python3
"""Check a job's train.py options without training: parse them exactly like
train.py does, then check that they fit together. Exit code 2 on any error.

    uv run python jobs/check_config.py <the same options as train.py>
"""

import shutil
import sys
import tempfile
from pathlib import Path

import autoroot  # noqa     Do not remove, puts the project root on sys.path

MODELS_3D = {"UNet3D", "VNet3D", "ResUNet3D", "MedNeXt3D"}
MODELS_2D = {"ENet", "ShallowNet"}


def main() -> int:
    from src.utils.config import get_config

    # get_config creates the results folder: point it at a throwaway one
    scratch = Path(tempfile.mkdtemp(prefix="check_config_"))
    sys.argv += ["--paths.results_dir", str(scratch / "results")]
    try:
        c = get_config()  # unknown or misspelled options stop here (tyro)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    m, d, t = c.model, c.dataset, c.training
    errors, warnings = [], []

    if m.is_3d and m.name not in MODELS_3D:
        errors.append(f"model.is_3d is set but {m.name} is a 2D model")
    if not m.is_3d and m.name in MODELS_3D:
        errors.append(f"{m.name} is a 3D model but model.is_3d is not set")
    if d.name.endswith("_1ch") and m.in_channels != 1:
        errors.append(f"dataset {d.name} has 1 channel but model.in_channels is {m.in_channels}")
    if d.name.endswith("_3ch") and m.in_channels != 3:
        errors.append(f"dataset {d.name} has 3 channels but model.in_channels is {m.in_channels}")
    if d.num_classes != 5:
        errors.append(f"dataset.num_classes is {d.num_classes}; SegTHOR has 5 (background + 4 organs)")
    if t.skeleton_weight > 0 and (not m.is_3d or d.coarse):
        errors.append("training.skeleton_weight > 0 needs a 3D box dataset")
    if m.deep_supervision and not m.is_3d:
        errors.append("model.deep_supervision only works for the 3D models")
    if t.epochs < 1 or t.batch_size < 1:
        errors.append(f"epochs ({t.epochs}) and batch_size ({t.batch_size}) must be at least 1")
    if m.name == "MedNeXt3D" and t.batch_size > 4:
        warnings.append(f"MedNeXt3D with batch_size {t.batch_size} needs ~{5 * t.batch_size} GB: "
                        "does not fit a 40 GB A100 (use --training.batch_size 4)")

    print(f">> config check: model={m.name} ({'3D' if m.is_3d else '2D'}) dataset={d.name} "
          f"in_channels={m.in_channels} classes={d.num_classes}")
    print(f"   loss={t.loss} skeleton_weight={t.skeleton_weight} deep_supervision={m.deep_supervision} "
          f"mode={t.mode}")
    print(f"   skip_augmented={d.skip_augmented} online_augmentation={c.augmentation.enabled} "
          f"epochs={t.epochs} batch_size={t.batch_size} batches_per_epoch={t.batches_per_epoch} "
          f"kernels={m.kernels} lr={t.lr}")
    print(f"   eval={c.eval.enabled} postprocess={list(c.eval.postprocess)}")
    for w in warnings:
        print(f"   WARNING: {w}")
    for e in errors:
        print(f"   ERROR: {e}", file=sys.stderr)
    print(">> config check: " + ("FAILED" if errors else "OK"))
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
