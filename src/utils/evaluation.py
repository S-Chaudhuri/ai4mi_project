import csv
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.transform import resize

from src.utils.utils import hausdorff_distance, normalized_surface_distance

CLASS_NAMES = {1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}  # SegTHOR labels


def to_native_space(pred_dhw: np.ndarray, native_xyz_shape: tuple) -> np.ndarray:
    """(D, 256, 256) class map in (z, x, y) order -> (X, Y, Z) at the original CT grid. Nearest neighbour."""
    X, Y, Z = native_xyz_shape
    assert pred_dhw.shape[0] == Z, (pred_dhw.shape, native_xyz_shape)
    up = np.stack(
        [
            resize(
                s,
                (X, Y),
                order=0,
                mode="constant",
                preserve_range=True,
                anti_aliasing=False,
            )
            for s in pred_dhw
        ],
        axis=2,
    )
    return up.astype(np.uint8)


def load_native_gt(data_root: Path, patient: str, split_dir="train"):
    """GT in native space + spacing (dx, dy, dz) in mm + affine. Adjust the path to your layout."""
    nii = nib.load(str(data_root / "segthor_part1" / split_dir / patient / "GT.nii.gz"))
    return (
        np.asarray(nii.dataobj).astype(np.uint8),
        tuple(float(z) for z in nii.header.get_zooms()[:3]),
        nii.affine,
    )


def dice_score(g: np.ndarray, p: np.ndarray) -> float:
    if not g.any():
        return float("nan")  # class not in this patient: excluded, not 0 or 1
    return float(2 * (g & p).sum() / (g.sum() + p.sum()))


def evaluate_patient(
    pred_xyz, gt_xyz, spacing_xyz, tau_mm=2.0, classes=CLASS_NAMES
) -> dict:
    """Per-organ Dice / HD95 / NSD. Arrays share shape (X, Y, Z); spacing in the same axis order."""
    assert pred_xyz.shape == gt_xyz.shape, (pred_xyz.shape, gt_xyz.shape)
    out = {}
    for k, name in classes.items():
        g, p = gt_xyz == k, pred_xyz == k
        if not g.any():
            out[name] = dict(dice=np.nan, hd95=np.nan, nsd=np.nan)
            continue
        if not p.any():  # missed organ: Dice 0, distances undefined (not "perfect")
            out[name] = dict(dice=0.0, hd95=np.inf, nsd=0.0)
            continue
        out[name] = dict(
            dice=dice_score(g, p),
            hd95=hausdorff_distance(g, p, spacing_xyz, 95.0),
            nsd=normalized_surface_distance(g, p, spacing_xyz, tau_mm),
        )
    return out


def write_csv(results: dict, path: Path):
    """results: {patient: {organ: {metric: value}}} -> long-format CSV plus a MEAN row per organ/metric."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ("dice", "hd95", "nsd")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "organ", *metrics])
        for pat, organs in results.items():
            for organ, m in organs.items():
                w.writerow([pat, organ, *[f"{m[x]:.4f}" for x in metrics]])
        for organ in CLASS_NAMES.values():
            vals = [
                [r[organ][x] for r in results.values() if np.isfinite(r[organ][x])]
                for x in metrics
            ]
            w.writerow(
                ["MEAN", organ, *[f"{np.mean(v):.4f}" if v else "nan" for v in vals]]
            )


def write_submission(pred_xyz: np.ndarray, affine, patient: str, out_dir: Path):
    """One NIfTI per patient, uint8 labels 0-4, native grid and affine of the CT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(pred_xyz.astype(np.uint8), affine),
        str(out_dir / f"{patient}.nii.gz"),
    )

