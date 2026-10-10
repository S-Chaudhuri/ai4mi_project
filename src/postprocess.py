#!/usr/bin/env python3

"""Largest-connected-component (LCC) postprocessing for SegTHOR predictions.

The heart, trachea and aorta are each one connected structure, so any extra
component the model predicts for them is a false positive. Those stray blobs
barely move the Dice but dominate the Hausdorff distances (a single voxel far
away sets the HD). For each of those organs this keeps the largest component
and sets the others to background.

The esophagus is left alone by default: on some slices it is legitimately
split (or partly missed), and keeping only its largest piece would delete
true positives.

Works on label maps (0 = background, 1 = esophagus, 2 = heart, 3 = trachea,
4 = aorta), on a single NIfTI file or on every .nii/.nii.gz in a folder. The
geometry (spacing, origin, direction) of each file is kept.

Usage:
    # a folder of predictions
    python src/postprocess.py --src results/run/pred --dest results/run/pred_lcc

    # one file, choosing the organs and 26-connectivity
    python src/postprocess.py --src Patient_01.nii.gz --dest Patient_01_lcc.nii.gz \
        --classes 2 3 4 --connectivity 26
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# Organs that are a single connected structure: heart, trachea, aorta
SINGLE_COMPONENT: tuple[int, ...] = (2, 3, 4)


def remove_small_esophagus_components(
    image: sitk.Image,
    min_volume_mm3: float = 500.0,
) -> sitk.Image:
    """Filter esophagus components; preserve other organs and the largest component."""
    esophagus = sitk.Cast(image == 1, sitk.sitkUInt8)

    # True selects 26-connectivity for a 3D image.
    components = sitk.ConnectedComponent(esophagus, True)

    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(components)
    labels = statistics.GetLabels()

    if not labels:
        return sitk.Image(image)

    largest = max(labels, key=statistics.GetNumberOfPixels)

    keep = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    keep.CopyInformation(image)

    for label in labels:
        if (
            statistics.GetPhysicalSize(label) >= min_volume_mm3
            or label == largest
        ):
            keep = keep | sitk.Cast(components == label, sitk.sitkUInt8)

    # Retain all non-esophagus voxels and accepted esophagus components.
    retain = sitk.Cast(image != 1, sitk.sitkUInt8) | keep
    return sitk.Mask(image, retain)


def keep_largest_component(
    pred: np.ndarray, classes=SINGLE_COMPONENT, connectivity: int = 6
) -> np.ndarray:
    """Keep only the largest connected component of each class in `classes`;
    the other components of that class become background (0).

    connectivity: 6 (voxels sharing a face) or 26 (also edges and corners).
    """
    structure = ndimage.generate_binary_structure(pred.ndim, 1 if connectivity == 6 else pred.ndim)
    out = pred.copy()
    for k in classes:
        labels, n = ndimage.label(pred == k, structure=structure)
        if n <= 1:
            continue
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, n + 1))
        out[(labels > 0) & (labels != int(np.argmax(sizes)) + 1)] = 0
    return out


def postprocess_file(src: Path, dest: Path, classes, connectivity: int) -> dict[int, int]:
    """Apply LCC to one NIfTI label map. Returns voxels removed per class."""
    img = sitk.ReadImage(str(src))
    pred = sitk.GetArrayFromImage(img)
    out = keep_largest_component(pred, classes, connectivity)

    out_img = sitk.GetImageFromArray(out.astype(pred.dtype))
    out_img.CopyInformation(img)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(dest))

    return {k: int(((pred == k) & (out != k)).sum()) for k in classes}


def main(args: argparse.Namespace) -> None:
    src, dest = Path(args.src), Path(args.dest)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.name.endswith((".nii", ".nii.gz")))
        pairs = [(p, dest / p.name) for p in files]
    else:
        pairs = [(src, dest)]
    if not pairs:
        raise SystemExit(f"No .nii/.nii.gz files in {src}")

    print(f">> LCC on classes {tuple(args.classes)} ({args.connectivity}-connectivity), {len(pairs)} file(s)")
    for s, d in pairs:
        removed = postprocess_file(s, d, tuple(args.classes), args.connectivity)
        print(f"  {s.name}: removed " + ", ".join(f"class {k}: {v} vox" for k, v in removed.items()))
    print(f">> wrote {dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Largest-connected-component postprocessing")
    parser.add_argument("--src", required=True, help="Prediction .nii.gz file or folder")
    parser.add_argument("--dest", required=True, help="Output file or folder")
    parser.add_argument(
        "--classes", type=int, nargs="+", default=list(SINGLE_COMPONENT),
        help="Labels to reduce to their largest component (default: heart, trachea, aorta)",
    )
    parser.add_argument("--connectivity", type=int, choices=[6, 26], default=6)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
