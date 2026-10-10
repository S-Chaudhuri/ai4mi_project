#!/usr/bin/env python3

"""3D preprocessing for the volumetric models: NIfTI -> one .npy volume per patient.

The PNG route (slice_segthor.py, then BoxDataset stacking the slices back)
loses the HU values and the voxel spacing. This script goes straight from the
NIfTI files and, per patient:

    1. resamples CT (linear) and GT (nearest, on the resampled CT grid) to one
       fixed spacing, the same SimpleITK approach as image_reshape.py
    2. crops in-plane to the body (keeps every slice)
    3. applies one or more HU windows from hu_window_to_3channel.py, one
       channel each, stored as uint8 so img_transform_3d's /255 still applies
    4. optionally writes offline-augmented copies of the training patients,
       with the same affine + elastic warp as augment_offline.py (every
       channel gets the same warp, so they stay aligned)

The train/val split is the one slice_segthor.py makes for the same seed,
retains and fold, so the 2D and 3D datasets hold out the same patients.

Output, read by NpyBoxDataset:
    <dest_dir>/train/Patient_02/img.npy     (C, D, H, W) uint8, axes (z, y, x)
    <dest_dir>/train/Patient_02/gt.npy      (D, H, W) uint8 class indices
    <dest_dir>/train/Patient_02/meta.json   spacing, crop, original geometry
    <dest_dir>/train/Patient_02a1/...       augmented variant 1

Usage:
    # one soft-tissue channel
    python src/preprocessing/preprocess_3d.py --source_dir data/segthor_train_full \
        --dest_dir data/SEGTHOR_3D_1ch --windows soft --variants 2

    # all three windows of hu_window_to_3channel.py
    python src/preprocessing/preprocess_3d.py --source_dir data/segthor_train_full \
        --dest_dir data/SEGTHOR_3D_3ch --windows soft lung blood --variants 2
"""

import argparse
import json
import random
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import autoroot  # noqa     Do not remove, puts the project root on sys.path
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.preprocessing.augment_offline import (
    Args as AugArgs,
    affine_matrix,
    elastic_offsets,
    jacobian_min,
    retention,
    sampling_grid,
)
from src.preprocessing.hu_window_to_3channel import DEFAULT_WINDOWS, apply_window
from src.preprocessing.slice_segthor import get_splits

# Names for the (center, width) windows of hu_window_to_3channel.py
WINDOWS: dict[str, tuple[float, float]] = dict(
    zip(["soft", "lung", "blood"], DEFAULT_WINDOWS)
)

NUM_CLASSES = 5

# Anything above this is the patient (or the table), below is air
BODY_HU_THRESHOLD = -500


def resample(
    image: sitk.Image, spacing: tuple[float, float, float], reference=None
) -> sitk.Image:
    """Linear for the CT; nearest neighbour onto the CT's grid for the labels,
    as in image_reshape.py (label-ness is passed explicitly here, the
    filename check there only knows GT_4label_v6.nii.gz)."""
    resampler = sitk.ResampleImageFilter()
    if reference is None:
        size = [
            int(round(n * s / t))
            for n, s, t in zip(image.GetSize(), image.GetSpacing(), spacing)
        ]
        resampler.SetSize(size)
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)  # air outside the scan
    else:
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def body_crop(ct: np.ndarray, gt: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    """In-plane bounding box (y0, y1, x0, x1) of the body, as the largest
    connected component above BODY_HU_THRESHOLD. Always contains every label."""
    labels, n = ndimage.label(ct > BODY_HU_THRESHOLD)
    if n == 0:
        return 0, ct.shape[1], 0, ct.shape[2]
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, n + 1))
    body = labels == (int(np.argmax(sizes)) + 1)

    mask = body | (gt > 0)
    ys = np.where(mask.any(axis=(0, 2)))[0]
    xs = np.where(mask.any(axis=(0, 1)))[0]
    return (
        max(int(ys[0]) - margin, 0),
        min(int(ys[-1]) + 1 + margin, ct.shape[1]),
        max(int(xs[0]) - margin, 0),
        min(int(xs[-1]) + 1 + margin, ct.shape[2]),
    )


def to_channels(ct: np.ndarray, windows: list[str]) -> np.ndarray:
    """(C, D, H, W) uint8, one HU window per channel."""
    chans = [apply_window(ct.astype(np.float32), *WINDOWS[w]) for w in windows]
    return np.round(np.stack(chans) * 255).astype(np.uint8)


def write_patient(dest: Path, img: np.ndarray, gt: np.ndarray, meta: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "img.npy", img)
    np.save(dest / "gt.npy", gt)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))


def augment(
    img: np.ndarray, gt: np.ndarray, patient: str, variant: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Same warp as augment_offline.py, applied to every channel at once."""
    aug_args = AugArgs()
    rng = np.random.default_rng([zlib.crc32(patient.encode()), seed + variant])
    shape = gt.shape
    offsets = elastic_offsets(
        shape, aug_args.elastic_alpha, aug_args.elastic_sigma, rng
    )
    grid = sampling_grid(shape, offsets, affine_matrix(aug_args, rng))

    img_t = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]
    img_out = F.grid_sample(img_t, grid, mode="bilinear", align_corners=True)
    gt_out = F.grid_sample(gt_t, grid, mode="nearest", align_corners=True)

    aug_img = np.round(img_out[0].numpy().clip(0, 1) * 255).astype(np.uint8)
    aug_gt = gt_out[0, 0].numpy().round().astype(np.uint8)
    kept, _ = retention(gt, aug_gt, NUM_CLASSES)
    return aug_img, aug_gt, jacobian_min(offsets), kept


def process_patient(
    patient: str, split: str, args: argparse.Namespace
) -> list[str]:
    src = Path(args.source_dir) / "train" / patient
    ct_img = sitk.ReadImage(str(src / f"{patient}.nii.gz"))
    gt_img = sitk.ReadImage(str(src / "GT.nii.gz"))
    gt_img.CopyInformation(ct_img)

    spacing = tuple(args.spacing)  # (x, y, z), SimpleITK order
    ct_res = resample(ct_img, spacing)
    gt_res = resample(gt_img, spacing, reference=ct_res)

    ct = sitk.GetArrayFromImage(ct_res)  # (z, y, x)
    gt = sitk.GetArrayFromImage(gt_res).astype(np.uint8)
    assert set(np.unique(gt).tolist()) <= set(range(NUM_CLASSES)), patient

    y0, y1, x0, x1 = body_crop(ct, gt, args.margin)
    ct, gt = ct[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1]
    img = to_channels(ct, args.windows)

    meta = {
        "patient": patient,
        "source": patient,
        "augmented": False,
        "axes": "zyx",
        "spacing_zyx": [spacing[2], spacing[1], spacing[0]],
        "windows": {w: list(WINDOWS[w]) for w in args.windows},
        "crop_yx": [y0, y1, x0, x1],
        "resampled_size_zyx": list(ct_res.GetSize()[::-1]),
        "original_size_xyz": list(ct_img.GetSize()),
        "original_spacing_xyz": list(ct_img.GetSpacing()),
        "original_origin_xyz": list(ct_img.GetOrigin()),
        "original_direction": list(ct_img.GetDirection()),
    }
    out = Path(args.dest_dir) / split
    write_patient(out / patient, img, gt, meta)
    lines = [f"  {patient:<14} {split:<5} {tuple(gt.shape)}"]

    if split == "train":
        for i in range(1, args.variants + 1):
            stem = f"{patient}a{i}"
            aug_img, aug_gt, det, kept = augment(img, gt, patient, i, args.seed)
            write_patient(
                out / stem,
                aug_img,
                aug_gt,
                meta | {"patient": stem, "augmented": True, "variant": i},
            )
            flag = "  FOLDS" if det <= 0 else ""
            lines.append(f"  {stem:<14} train jac {det:5.3f} worst class kept {kept:4.0%}{flag}")
    return lines


def main(args: argparse.Namespace) -> None:
    dest = Path(args.dest_dir)
    assert not dest.exists(), f"{dest} already exists"

    random.seed(args.seed)  # the same split as slice_segthor.py
    train_ids, val_ids, _ = get_splits(Path(args.source_dir), args.retains, args.fold)
    jobs = [(p, "train") for p in train_ids] + [(p, "val") for p in val_ids]
    print(f">> {len(train_ids)} train, {len(val_ids)} val patients, windows {args.windows}")
    print(f"   spacing (x, y, z) {tuple(args.spacing)} mm, {args.variants} variant(s) per train patient")

    fn = partial(_process, args=args)
    if args.process == 1:
        results = list(map(fn, jobs))
    else:
        with Pool(args.process) as pool:
            results = pool.map(fn, jobs)
    for lines in results:
        print("\n".join(lines))

    (dest / "dataset.json").write_text(
        json.dumps(
            {
                "windows": {w: list(WINDOWS[w]) for w in args.windows},
                "spacing_zyx": [args.spacing[2], args.spacing[1], args.spacing[0]],
                "train": train_ids,
                "val": val_ids,
                "variants": args.variants,
                "seed": args.seed,
                "retains": args.retains,
                "fold": args.fold,
            },
            indent=2,
        )
    )
    print(f">> wrote {dest}")


def _process(job: tuple[str, str], args: argparse.Namespace) -> list[str]:
    return process_patient(job[0], job[1], args)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIfTI -> .npy for the 3D models")
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["soft"],
        choices=list(WINDOWS),
        help="HU windows from hu_window_to_3channel.py, one channel each",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.5, 1.5, 2.5],
        metavar=("X", "Y", "Z"),
        help="Target voxel spacing in mm",
    )
    parser.add_argument("--margin", type=int, default=4, help="Body crop margin in voxels")
    parser.add_argument("--variants", type=int, default=2, help="Augmented copies per train patient")
    parser.add_argument("--retains", type=int, default=5, help="Validation patients")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--process", "-p", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
