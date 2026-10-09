"""Selectable SegTHOR postprocessing.

CLI (native-space prediction NIfTIs; no inference needed):
    python post_processing.py --input results/raw --output results/lcc --steps lcc
    python post_processing.py --input results/raw --output results/small --steps small
    python post_processing.py --input results/raw --output results/both --steps lcc small
    python post_processing.py --input results/raw --output results/compare --compare

Optional metrics: add --gt-root data --gt-split train.
Requires Python >= 3.10, numpy, scipy, nibabel.

For inference, call run_eval(Args(...)) from your existing evaluation launcher.
It reuses predict_volume from eval_3D, and your project's evaluation utilities.
Set eval_module to the dotted import path of eval_3D if it is not at repo root.
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

LCC_CLASSES = (2, 3, 4)  # heart, trachea, aorta; esophagus = 1
VALID_STEPS = ("lcc", "small")


def _structure(connectivity):
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2 or 3 (6, 18 or 26 neighbors)")
    return ndimage.generate_binary_structure(3, connectivity)


def _spacing(spacing_xyz):
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing_xyz must contain three finite positive values in mm")
    return spacing


def _labels(pred):
    arr = np.asarray(pred)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D label volume, got shape {arr.shape}")
    if not np.isfinite(arr).all() or not np.isin(arr, (0, 1, 2, 3, 4)).all():
        raise ValueError("Expected discrete SegTHOR labels 0..4, not probabilities or CT intensities")
    return arr.astype(np.uint8, copy=False)


def largest_component(mask, connectivity=3):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    lab, n = ndimage.label(mask, structure=_structure(connectivity))
    if n <= 1:
        return mask.copy()
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return lab == (1 + int(np.argmax(sizes)))


def filter_small_components(mask, spacing_xyz, min_mm3, connectivity=3):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    voxel_mm3 = float(np.prod(_spacing(spacing_xyz)))
    if not np.isfinite(min_mm3) or min_mm3 < 0:
        raise ValueError("min_mm3 must be finite and non-negative")
    lab, n = ndimage.label(mask, structure=_structure(connectivity))
    if n <= 1:
        return mask.copy()
    sizes_mm3 = np.bincount(lab.ravel(), minlength=n + 1)[1:] * voxel_mm3
    keep = sizes_mm3 >= min_mm3
    keep[int(np.argmax(sizes_mm3))] = True  # preserve at least the largest component
    return np.isin(lab, 1 + np.flatnonzero(keep))


def _normalize_steps(steps):
    if isinstance(steps, str):
        steps = (steps,)
    steps = tuple(steps)
    unknown = set(steps) - set(VALID_STEPS)
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}; use {VALID_STEPS}")
    return tuple(dict.fromkeys(steps))


def postprocess(pred, spacing_xyz, esoph_min_mm3=500.0, *,
                steps=VALID_STEPS, lcc_classes=LCC_CLASSES, connectivity=3):
    """Return a copy; use steps=() for an unchanged baseline.

    lcc: keep the largest component of each selected class.
    small: remove small esophagus components, always retaining the largest.
    Removed voxels become background (0); inputs are never modified.
    """
    out = _labels(pred).copy()
    _spacing(spacing_xyz)
    _structure(connectivity)
    steps = _normalize_steps(steps)
    lcc_classes = tuple(lcc_classes)
    if not set(lcc_classes).issubset(LCC_CLASSES):
        raise ValueError("lcc_classes must be a subset of (2, 3, 4)")
    for step in steps:
        if step == "lcc":
            for k in lcc_classes:
                mask = out == k
                out[mask & ~largest_component(mask, connectivity)] = 0
        elif step == "small":
            mask = out == 1
            keep = filter_small_components(mask, spacing_xyz, esoph_min_mm3, connectivity)
            out[mask & ~keep] = 0
    return out


def bbox(mask, margin_vox, shape):
    """Bounding-box slices; empty masks raise a clear error.

    This helper selects a crop; it does not perform refinement inference.
    """
    mask = np.asarray(mask, dtype=bool)
    shape = np.asarray(shape, dtype=int)
    margin = np.asarray(margin_vox)
    if margin.ndim == 0:
        margin = np.repeat(margin, 3)
    if mask.ndim != 3 or shape.shape != (3,) or tuple(shape) != mask.shape:
        raise ValueError("shape must match the 3D mask")
    if margin.shape != (3,) or not np.isfinite(margin).all() or (margin < 0).any() or (margin != np.floor(margin)).any():
        raise ValueError("margin_vox must be a non-negative integer or three integers")
    idx = np.argwhere(mask)
    if not len(idx):
        raise ValueError("Cannot compute a bounding box for an empty mask")
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def paste_back(full_pred, crop_pred, sl, class_ids):
    """Replace selected classes inside the crop, preserving labels elsewhere.

    As in the original helper, new selected labels can overwrite other labels
    at the same voxels inside the crop.
    """
    out = _labels(full_pred).copy()
    crop_pred = _labels(crop_pred)
    class_ids = tuple(class_ids)
    if not set(class_ids).issubset((1, 2, 3, 4)):
        raise ValueError("class_ids must contain foreground labels 1..4")
    if not isinstance(sl, tuple) or len(sl) != 3 or not all(isinstance(s, slice) for s in sl):
        raise ValueError("sl must contain three slices")
    region = out[sl]
    if region.shape != crop_pred.shape:
        raise ValueError(f"Crop shape {crop_pred.shape} does not match region {region.shape}")
    region[np.isin(region, class_ids)] = 0
    selected = np.isin(crop_pred, class_ids)
    region[selected] = crop_pred[selected]
    return out


def load_native_gt(data_root, patient, split_dir="train"):
    path = Path(data_root) / "segthor_part1" / split_dir / patient / "GT.nii.gz"
    nii = nib.load(str(path))
    gt = _labels(np.asarray(nii.dataobj))
    return gt, _spacing(nii.header.get_zooms()[:3]), nii


def write_submission(pred_xyz, ref_nii, patient, out_dir):
    """Save native-grid labels with the reference geometry; never resample here."""
    pred_xyz = _labels(pred_xyz)
    if pred_xyz.shape != ref_nii.shape:
        raise ValueError("Prediction shape must match the reference NIfTI")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = ref_nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(1, 0)
    img = nib.Nifti1Image(pred_xyz, ref_nii.affine, hdr)
    img.set_qform(ref_nii.get_qform(), int(ref_nii.header["qform_code"]))
    img.set_sform(ref_nii.get_sform(), int(ref_nii.header["sform_code"]))
    img.header.set_slope_inter(1, 0)
    img.header["cal_min"], img.header["cal_max"] = 0, 4
    path = out_dir / f"{patient}.nii.gz"
    nib.save(img, str(path))
    return path


def _variants(steps, compare):
    if compare:
        return {"raw": (), "lcc": ("lcc",), "small": ("small",), "both": VALID_STEPS}
    steps = _normalize_steps(steps)
    return {"selected": steps}


@dataclass
class Args:
    # Pass the same ModelConfig and DatasetConfig objects used by eval_3D.
    model: Any
    dataset: Any
    weights: Path
    split: str = "val"
    patient_id: int | None = None
    overlap: float = 0.5
    gpu: bool = True
    batch_size: int = 4
    save_nii: bool = True
    nii_out: Path | None = None
    csv_out: Path = Path("results/postprocessing/metrics.csv")
    submission_dir: Path | None = None
    steps: tuple[str, ...] = VALID_STEPS
    compare: bool = False
    esoph_min_mm3: float = 500.0
    lcc_classes: tuple[int, ...] = LCC_CLASSES
    connectivity: int = 3
    eval_module: str = "eval_3D"


def run_eval(args: Args):
    """Inference -> native grid -> selectable cleanup -> metrics/export.

    Comparison mode performs inference once per patient, then independently
    evaluates raw, lcc, small and both. CSVs go beside csv_out with mode suffixes.
    """
    import autoroot
    import torch
    from src.utils.config import Config
    from src.utils.dataset import BoxDataset
    from src.train import get_model
    from src.utils.evaluation import to_native_space, evaluate_patient, write_csv

    predict_volume = importlib.import_module(args.eval_module).predict_volume
    if not 0 <= args.overlap < 1 or args.batch_size < 1:
        raise ValueError("Require 0 <= overlap < 1 and batch_size >= 1")
    variants = _variants(args.steps, args.compare)
    results = {mode: {} for mode in variants}
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    cfg = Config(model=args.model, dataset=args.dataset)
    K = cfg.dataset.num_classes
    net = get_model(cfg)
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net.to(device).eval()
    ds = BoxDataset(args.split, autoroot.root / "data" / cfg.dataset.name,
                    sub_box_size=None, num_classes=K)
    matched = 0
    for item in ds.items:
        patient = item["stem"]
        if args.patient_id is not None and patient != f"Patient_{args.patient_id:02d}":
            continue
        matched += 1
        img = torch.from_numpy(item["img_vol"]).float() / 255.0
        with torch.no_grad():
            probs = predict_volume(net, img, cfg.dataset.box_size, args.overlap, K,
                                   args.batch_size, device, cfg.dataset.use_coords,
                                   cfg.model.temperature)
            pred = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
        del probs
        gt, spacing, ref = load_native_gt(autoroot.root / "data", patient)
        native = to_native_space(pred, gt.shape)
        for mode, steps in variants.items():
            out = postprocess(native, spacing, args.esoph_min_mm3, steps=steps,
                              lcc_classes=args.lcc_classes, connectivity=args.connectivity)
            results[mode][patient] = evaluate_patient(out, gt, spacing)
            print(patient, mode, results[mode][patient])
            export_dir = args.submission_dir
            if export_dir is None and args.save_nii:
                export_dir = args.nii_out
            if export_dir is not None:
                directory = Path(export_dir) / mode if args.compare else Path(export_dir)
                write_submission(out, ref, patient, directory)
    if not matched:
        raise ValueError("No patients matched the selected split/patient_id")
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    for mode, metrics in results.items():
        path = csv_out.with_name(f"{csv_out.stem}_{mode}{csv_out.suffix}") if args.compare else csv_out
        write_csv(metrics, path)
        print(f">> Wrote metrics to {path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="One prediction NIfTI or a directory of them")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="*", choices=VALID_STEPS, default=list(VALID_STEPS),
                        help="lcc, small, or both; --steps without values saves an unchanged baseline")
    parser.add_argument("--compare", action="store_true", help="Save raw/lcc/small/both, each starting from the original")
    parser.add_argument("--lcc-classes", nargs="+", type=int, choices=LCC_CLASSES, default=list(LCC_CLASSES))
    parser.add_argument("--esoph-min-mm3", type=float, default=500.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--patient-id", type=int)
    parser.add_argument("--gt-root", type=Path, help="Optional data root containing segthor_part1; enables project metrics")
    parser.add_argument("--gt-split", default="train")
    args = parser.parse_args()
    if args.input.is_dir():
        files = sorted(set(args.input.glob("*.nii")) | set(args.input.glob("*.nii.gz")))
    elif args.input.is_file() and str(args.input).endswith((".nii", ".nii.gz")):
        files = [args.input]
    else:
        parser.error("--input must be a NIfTI file or an existing directory")
    if args.patient_id is not None:
        name = f"Patient_{args.patient_id:02d}"
        files = [p for p in files if p.name in (f"{name}.nii", f"{name}.nii.gz")]
    if not files:
        parser.error("No matching prediction NIfTIs found")
    names = [p.name[:-7] if p.name.endswith(".nii.gz") else p.stem for p in files]
    if len(names) != len(set(names)):
        parser.error("Duplicate patient names (.nii and .nii.gz); choose one file per patient")
    variants = _variants(args.steps, args.compare)
    metrics = {mode: {} for mode in variants}
    if args.gt_root is not None:
        from src.utils.evaluation import evaluate_patient, write_csv
    for path, patient in zip(files, names):
        ref = nib.load(str(path))
        native = _labels(np.asarray(ref.dataobj))
        units = ref.header.get_xyzt_units()[0]
        if units not in ("mm", "unknown"):
            raise ValueError(f"{path}: expected millimeter units, found {units}")
        if units == "unknown":
            print(f"Warning: {path.name} has unspecified spatial units; assuming mm")
        spacing = _spacing(ref.header.get_zooms()[:3])
        if args.gt_root is not None:
            gt, gt_spacing, gt_ref = load_native_gt(args.gt_root, patient, args.gt_split)
            if gt.shape != native.shape or not np.allclose(ref.affine, gt_ref.affine, rtol=0, atol=1e-4):
                raise ValueError(f"{patient}: prediction and GT must be on the same native grid")
            spacing = gt_spacing
        for mode, steps in variants.items():
            directory = args.output / mode if args.compare else args.output
            destination = directory / f"{patient}.nii.gz"
            if destination.resolve() == path.resolve():
                raise ValueError("Refusing to overwrite an input prediction; use a different output directory")
            out = postprocess(native, spacing, args.esoph_min_mm3, steps=steps,
                              lcc_classes=args.lcc_classes, connectivity=args.connectivity)
            saved = write_submission(out, ref, patient, directory)
            print(f">> {patient} [{mode}]: removed {np.count_nonzero(native != out)} voxels; saved {saved}")
            if args.gt_root is not None:
                metrics[mode][patient] = evaluate_patient(out, gt, spacing)
                print(patient, mode, metrics[mode][patient])
    if args.gt_root is not None:
        for mode, values in metrics.items():
            directory = args.output / mode if args.compare else args.output
            write_csv(values, directory / "metrics.csv")
            print(f">> Wrote metrics to {directory / 'metrics.csv'}")


if __name__ == "__main__":
    main()