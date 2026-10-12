"""Assemble the course submission folder and tarball for group 12.

Layout, as required in docs/original/readme.md ("Archiving everything"):

    group-12/
        test/pred/Patient_41.nii.gz ...
        val/pred/...
        val/gt/...
        val/3d_dice.npz  val/3d_hd95.npz  val/3d_nsd.npz
        group-12.bundle
        bestmodel-group-12.pkl

Input: the results run directory of the best model, containing
config.yaml, bestweights.pt and eval/ (from src/eval_3D.py on the val
and test splits, with --eval.submission for val).

Usage:
    uv run python scripts/make_submission.py --run results/SEGTHOR_3D_1ch/<run>
    uv run python scripts/make_submission.py        # picks the newest eligible run
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import autoroot  # side effect: puts the project root on sys.path
import nibabel as nib
import numpy as np
import torch
import yaml

from src.train import get_model
from src.utils.config import Config
from src.utils.config_helpers import instantiate_dataclass

GROUP = 12


def find_default_run() -> Path:
    cands = []
    for weights in (autoroot.root / "results").glob("*/*/bestweights.pt"):
        run_dir = weights.parent
        if (run_dir / "config.yaml").is_file() and (
            run_dir / "eval" / "submission" / "val" / "pred"
        ).is_dir():
            cands.append((run_dir.stat().st_mtime, run_dir))
    if not cands:
        sys.exit(
            "No --run given and no results run with bestweights.pt + "
            "eval/submission/val/pred found under results/"
        )
    return max(cands)[1]


def build_model_pkl(run_dir: Path, dest: Path) -> None:
    config = instantiate_dataclass(
        Config, yaml.safe_load((run_dir / "config.yaml").read_text())
    )
    net = get_model(config)
    net.load_state_dict(
        torch.load(run_dir / "bestweights.pt", map_location="cpu", weights_only=True)
    )
    torch.save(net, dest)
    print(
        f">> {dest.name}: {config.model.name}, "
        f"{sum(p.numel() for p in net.parameters()):,} params"
    )


def copy_files(src: Path, dest: Path) -> int:
    if not src.is_dir():
        sys.exit(f"missing {src}")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=None,
        help="results run dir with config.yaml, bestweights.pt and eval/",
    )
    parser.add_argument("--out", type=Path, default=autoroot.root / "submission")
    parser.add_argument("--branch", default="submission",
                        help="branch to pack into the git bundle")
    parser.add_argument("--skip-bundle", action="store_true")
    parser.add_argument("--skip-tar", action="store_true")
    args = parser.parse_args()

    run_dir: Path = args.run or find_default_run()
    group_dir = args.out / f"group-{GROUP}"
    if group_dir.exists():
        sys.exit(f"{group_dir} already exists, remove it first")
    print(f">> run dir: {run_dir}")
    eval_dir = run_dir / "eval"

    group_dir.mkdir(parents=True)

    # 1. best model as a full torch module (the course's torch.save(net, ...) form)
    build_model_pkl(run_dir, group_dir / f"bestmodel-group-{GROUP}.pkl")

    # 2. val predictions, labels and metrics
    val_stems = sorted(
        p.name.removesuffix(".nii.gz")
        for p in (eval_dir / "submission" / "val" / "pred").glob("*.nii.gz")
    )
    if not val_stems:
        sys.exit(f"no val predictions in {eval_dir / 'submission' / 'val' / 'pred'}")
    n_pred = copy_files(eval_dir / "submission" / "val" / "pred", group_dir / "val" / "pred")
    n_gt = copy_files(eval_dir / "submission" / "val" / "gt", group_dir / "val" / "gt")
    if n_gt != n_pred:
        sys.exit(f"val: {n_pred} predictions but {n_gt} GT labels")

    npz = sorted(eval_dir.glob("3d_*.npz"))
    for p in npz:
        shutil.copy2(p, group_dir / "val" / p.name)
    if not npz:
        sys.exit(f"no 3d_*.npz metric files in {eval_dir}")
    for p in npz:
        if sorted(np.load(p).keys()) != val_stems:
            sys.exit(f"{p.name} keys do not match the val patients")

    for f in (group_dir / "val" / "pred").glob("*.nii.gz"):
        shape_p = np.asarray(nib.load(str(f)).dataobj).shape
        shape_g = np.asarray(nib.load(str(group_dir / "val" / "gt" / f.name)).dataobj).shape
        if shape_p != shape_g:
            sys.exit(f"{f.name}: pred {shape_p} != gt {shape_g}")
    print(f">> val: {n_pred} patients, {[p.name for p in npz]}")

    # 3. test predictions
    n_test = copy_files(
        eval_dir / "submission" / "test" / "pred", group_dir / "test" / "pred"
    ) if (eval_dir / "submission" / "test" / "pred").is_dir() else 0
    if n_test:
        print(f">> test: {n_test} predictions")
    else:
        print(">> WARNING: no test predictions, run src/eval_3D.py --eval.split test first")

    # 4. git bundle of the whole project
    if args.skip_bundle:
        print(">> skipped bundle")
    else:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=autoroot.root, check=False,
        ).stdout.strip()
        if dirty:
            print(">> WARNING: uncommitted changes will NOT be in the bundle")
        subprocess.run(
            ["git", "bundle", "create",
             str(group_dir / f"group-{GROUP}.bundle"), f"refs/heads/{args.branch}"],
            cwd=autoroot.root, check=True,
        )
        print(f">> bundle of {args.branch} written")

    # 5. tar
    if args.skip_tar:
        print(">> skipped tar, group dir is at", group_dir)
        return
    tarball = args.out / f"group-{GROUP}.tar.zst"
    subprocess.run(
        f"tar -C {args.out} -cf - group-{GROUP} | zstd -T0 -3 > {tarball}",
        shell=True, check=True,
    )
    print(f">> {tarball}")


if __name__ == "__main__":
    main()
