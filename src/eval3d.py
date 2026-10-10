#!/usr/bin/env python3

"""Whole-volume 3D evaluation of a trained 3D model.

For every patient of a preprocess_3d.py dataset split:
    1. sliding-window inference over the whole volume (Gaussian-weighted,
       overlapping patches), argmax to a label volume
    2. optional postprocessing: keep the largest connected component of the
       heart, aorta and trachea (the esophagus is often legitimately split)
    3. map the prediction back to the original NIfTI grid (undo the body crop
       and the resampling, using meta.json) and write it as NIfTI
    4. score it against the original GT.nii.gz at the original voxel spacing

Metrics per patient and organ: Dice, IoU, precision, recall, relative volume
difference, Hausdorff (HD), HD95, average symmetric surface distance (ASSD,
SegTHOR's "average Hausdorff") and normalized surface distance (NSD).

Output (<dest>):
    pred/Patient_XX.nii.gz     predictions in the original geometry
    gt/Patient_XX.nii.gz       the matching ground truth (submission layout)
    metrics.csv                one row per patient x organ
    summary.csv                mean / std / median per organ and metric
    metrics.npz                one (patients, organs) array per metric

Usage:
    python src/eval3d.py --weights results/ResUNet3D/bestweights.pt \
        --model ResUNet3D --kernels 16 --in_channels 1 \
        --data_dir data/SEGTHOR_3D_1ch --source_dir data/segthor_train_full \
        --dest results/ResUNet3D/eval_val
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import autoroot  # noqa     Do not remove, puts the project root on sys.path
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from src.models.MedNeXt3D import MedNeXt3D
from src.models.ResUNet3D import ResUNet3D
from src.models.UNet3D import UNet3D
from src.models.VNet3D import VNet3D
from src.postprocess import keep_largest_component
from src.utils.dataset import window_starts
from src.utils.utils import (
    average_hausdorff_distance,
    hausdorff_distance,
    normalized_surface_distance,
)

MODELS = {"UNet3D": UNet3D, "VNet3D": VNet3D, "ResUNet3D": ResUNet3D, "MedNeXt3D": MedNeXt3D}
ORGANS = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
METRICS = ["dice", "iou", "precision", "recall", "rel_vol_diff", "hd", "hd95", "assd", "nsd"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def gaussian_importance(patch: tuple[int, int, int], sigma_scale: float = 1 / 8) -> torch.Tensor:
    """Weights that fall off towards the patch border, where predictions have
    the least context. Stitching overlapping patches with them avoids seams."""
    axes = []
    for p in patch:
        x = torch.arange(p, dtype=torch.float32) - (p - 1) / 2
        axes.append(torch.exp(-0.5 * (x / (p * sigma_scale)) ** 2))
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    w = w / w.max()
    return w.clamp(min=w[w > 0].min().item())  # never exactly 0


@torch.no_grad()
def sliding_window(
    net: torch.nn.Module,
    vol: torch.Tensor,
    num_classes: int,
    patch: tuple[int, int, int],
    overlap: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Softmax probabilities (K, D, H, W) for a (C, D, H, W) volume in [0, 1]."""
    shape = tuple(vol.shape[1:])
    # Pad at the end up to one patch; 0 is "below the window", i.e. air
    pad = [max(p - s, 0) for p, s in zip(patch, shape)]
    if any(pad):
        vol = F.pad(vol, (0, pad[2], 0, pad[1], 0, pad[0]), value=0.0)
    padded = tuple(vol.shape[1:])

    starts = [
        window_starts(full, size, max(int(size * (1 - overlap)), 1))
        for full, size in zip(padded, patch)
    ]
    corners = [(d, h, w) for d in starts[0] for h in starts[1] for w in starts[2]]

    weight = gaussian_importance(patch).to(device)
    probs = torch.zeros((num_classes, *padded), device=device)
    norm = torch.zeros(padded, device=device)
    pd, ph, pw = patch

    for i in range(0, len(corners), batch_size):
        chunk = corners[i : i + batch_size]
        x = torch.stack([vol[:, d : d + pd, h : h + ph, w : w + pw] for d, h, w in chunk])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = net(x.to(device))
        p = F.softmax(logits.float(), dim=1)
        for (d, h, w), pi in zip(chunk, p):
            probs[:, d : d + pd, h : h + ph, w : w + pw] += pi * weight
            norm[d : d + pd, h : h + ph, w : w + pw] += weight

    probs /= norm
    return probs[:, : shape[0], : shape[1], : shape[2]].cpu()


# ---------------------------------------------------------------------------
# Back to the original NIfTI grid
# ---------------------------------------------------------------------------


def to_original_space(pred: np.ndarray, meta: dict) -> sitk.Image:
    """Undo preprocess_3d.py: paste the crop back, then nearest-neighbour
    resample from the working spacing onto the original CT grid."""
    full = np.zeros(meta["resampled_size_zyx"], dtype=np.uint8)
    y0, y1, x0, x1 = meta["crop_yx"]
    full[:, y0:y1, x0:x1] = pred

    img = sitk.GetImageFromArray(full)
    img.SetSpacing(tuple(meta["spacing_zyx"][::-1]))  # SimpleITK is (x, y, z)
    img.SetOrigin(tuple(meta["original_origin_xyz"]))
    img.SetDirection(tuple(meta["original_direction"]))

    reference = sitk.Image([int(s) for s in meta["original_size_xyz"]], sitk.sitkUInt8)
    reference.SetSpacing(tuple(meta["original_spacing_xyz"]))
    reference.SetOrigin(tuple(meta["original_origin_xyz"]))
    reference.SetDirection(tuple(meta["original_direction"]))
    return sitk.Resample(img, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def organ_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx: tuple, nsd_tau: float) -> dict:
    """All metrics for one binary organ mask. Distances in mm."""
    tp = int(np.logical_and(gt, pred).sum())
    n_gt, n_pred = int(gt.sum()), int(pred.sum())
    nan = float("nan")
    return {
        "dice": 2 * tp / (n_gt + n_pred) if n_gt + n_pred else 1.0,
        "iou": tp / (n_gt + n_pred - tp) if n_gt + n_pred else 1.0,
        "precision": tp / n_pred if n_pred else nan,
        "recall": tp / n_gt if n_gt else nan,
        "rel_vol_diff": (n_pred - n_gt) / n_gt if n_gt else nan,
        "hd": hausdorff_distance(gt, pred, spacing_zyx, 100.0),
        "hd95": hausdorff_distance(gt, pred, spacing_zyx, 95.0),
        "assd": average_hausdorff_distance(gt, pred, spacing_zyx),
        "nsd": normalized_surface_distance(gt, pred, spacing_zyx, nsd_tau),
    }


def patient_metrics(gt: np.ndarray, pred: np.ndarray, spacing_zyx: tuple, nsd_tau: float) -> dict:
    return {
        name: organ_metrics(gt == k, pred == k, spacing_zyx, nsd_tau)
        for k, name in ORGANS.items()
    }


# ---------------------------------------------------------------------------


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    net = MODELS[args.model](args.in_channels, len(ORGANS) + 1, kernels=args.kernels)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net.to(device).eval()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = load_model(args, device)
    dest = Path(args.dest)
    (dest / "pred").mkdir(parents=True, exist_ok=True)
    (dest / "gt").mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p.parent for p in (Path(args.data_dir) / args.subset).glob("*/img.npy"))
    patient_dirs = [p for p in patient_dirs if not json.loads((p / "meta.json").read_text()).get("augmented")]
    print(f">> {args.model} on {len(patient_dirs)} {args.subset} patients ({device}), "
          f"patch {tuple(args.patch)}, overlap {args.overlap}, lcc {args.lcc}")

    rows: list[dict] = []
    for pdir in patient_dirs:
        t0 = time.time()
        meta = json.loads((pdir / "meta.json").read_text())
        pid = meta["patient"]

        vol = torch.from_numpy(np.load(pdir / "img.npy").astype(np.float32) / 255.0)
        probs = sliding_window(
            net, vol, len(ORGANS) + 1, tuple(args.patch), args.overlap, args.batch_size, device
        )
        pred = probs.argmax(0).numpy().astype(np.uint8)
        if args.lcc:
            pred = keep_largest_component(pred)

        pred_img = to_original_space(pred, meta)
        sitk.WriteImage(pred_img, str(dest / "pred" / f"{pid}.nii.gz"))

        gt_path = Path(args.source_dir) / "train" / pid / "GT.nii.gz"
        shutil.copy2(gt_path, dest / "gt" / f"{pid}.nii.gz")
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.uint8)
        pred_orig = sitk.GetArrayFromImage(pred_img)
        assert gt.shape == pred_orig.shape, (pid, gt.shape, pred_orig.shape)

        spacing_zyx = tuple(meta["original_spacing_xyz"][::-1])
        for organ, m in patient_metrics(gt, pred_orig, spacing_zyx, args.nsd_tau).items():
            rows.append({"model": args.model, "patient": pid, "organ": organ} | m)
        last = rows[-4:]
        print(f"  {pid}  dice " + " ".join(f"{r['organ'][:5]} {r['dice']:.3f}" for r in last)
              + "  hd95 " + " ".join(f"{r['hd95']:.1f}" for r in last)
              + f"  ({time.time() - t0:.0f}s)")

    write_results(rows, dest, args)


def write_results(rows: list[dict], dest: Path, args: argparse.Namespace) -> None:
    with open(dest / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    patients = sorted({r["patient"] for r in rows})
    organs = list(ORGANS.values())
    table = {
        m: np.array([[next(r[m] for r in rows if r["patient"] == p and r["organ"] == o)
                      for o in organs] for p in patients])
        for m in METRICS
    }
    np.savez(dest / "metrics.npz", patients=np.array(patients), organs=np.array(organs), **table)

    summary = []
    for m in METRICS:
        for oi, o in enumerate(organs + ["mean"]):
            vals = table[m][:, oi] if o != "mean" else np.nanmean(table[m], axis=1)
            summary.append({"metric": m, "organ": o, "mean": np.nanmean(vals),
                            "std": np.nanstd(vals), "median": np.nanmedian(vals)})
    with open(dest / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "organ", "mean", "std", "median"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n>> {args.model}: mean +- std over {len(patients)} patients")
    print(f"   {'metric':<13}" + "".join(f"{o:>17}" for o in organs + ["mean"]))
    for m in METRICS:
        line = [s for s in summary if s["metric"] == m]
        print(f"   {m:<13}" + "".join(f"{s['mean']:>10.3f} +-{s['std']:>5.2f}" for s in line))
    print(f">> wrote {dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-volume 3D evaluation")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--kernels", type=int, default=8)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--data_dir", type=str, required=True, help="preprocess_3d.py dataset")
    parser.add_argument("--source_dir", type=str, required=True, help="original NIfTI dataset")
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--subset", default="val")
    parser.add_argument("--patch", type=int, nargs=3, default=[132, 132, 128])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--nsd_tau", type=float, default=2.0, help="NSD tolerance in mm")
    parser.add_argument("--lcc", action="store_true", help="largest component for heart/trachea/aorta")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
