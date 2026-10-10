#!/usr/bin/env python3

"""Postprocessing for SegTHOR predictions.

Every method is a function `label map -> label map`, registered by name in
POSTPROCESSING. `postprocess(pred, steps)` applies the named steps in order,
so callers (src/eval_3D.py, this script's CLI) never need to change when a
method is added: write the function, add one line to POSTPROCESSING.

Label maps: 0 = background, 1 = esophagus, 2 = heart, 3 = trachea, 4 = aorta.

Methods:
    lcc     Largest connected component (6-connectivity) of the heart, trachea
            and aorta. Each is one connected structure, so any extra
            component is a false positive. Those stray blobs barely move the
            Dice but dominate the Hausdorff distances (a single voxel far away
            sets the HD). The esophagus is left alone: on some slices it is
            legitimately split (or partly missed), and keeping only its
            largest piece would delete true positives.
    lcc26   Same, with 26-connectivity (also edges and corners).

Usage:
    # a folder of prediction NIfTIs (geometry is kept)
    python src/postprocess.py --src results/run/pred --dest results/run/pred_pp --steps lcc

    # during evaluation
    python src/eval_3D.py ... --eval.postprocess lcc
"""

import argparse
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# Organs that are a single connected structure: heart, trachea, aorta
SINGLE_COMPONENT: tuple[int, ...] = (2, 3, 4)


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


# name -> method. To add a method: write a `pred -> pred` function above and
# register it here; it is then available everywhere as a step name.
POSTPROCESSING: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "lcc": keep_largest_component,
    "lcc26": partial(keep_largest_component, connectivity=26),
}


def postprocess(pred: np.ndarray, steps: Sequence[str]) -> np.ndarray:
    """Apply the named postprocessing steps to a label map, in order."""
    unknown = [s for s in steps if s not in POSTPROCESSING]
    if unknown:
        raise ValueError(f"Unknown postprocessing {unknown}, available: {list(POSTPROCESSING)}")
    for step in steps:
        pred = POSTPROCESSING[step](pred)
    return pred


def postprocess_file(src: Path, dest: Path, steps: Sequence[str]) -> dict[int, int]:
    """Apply the steps to one NIfTI label map, keeping its geometry.
    Returns the number of voxels each class lost."""
    img = sitk.ReadImage(str(src))
    pred = sitk.GetArrayFromImage(img)
    out = postprocess(pred, steps)

    out_img = sitk.GetImageFromArray(out.astype(pred.dtype))
    out_img.CopyInformation(img)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(dest))

    return {int(k): int(((pred == k) & (out != k)).sum()) for k in np.unique(pred) if k != 0}


def main(args: argparse.Namespace) -> None:
    src, dest = Path(args.src), Path(args.dest)
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.name.endswith((".nii", ".nii.gz")))
        pairs = [(p, dest / p.name) for p in files]
    else:
        pairs = [(src, dest)]
    if not pairs:
        raise SystemExit(f"No .nii/.nii.gz files in {src}")

    print(f">> Postprocessing {args.steps} on {len(pairs)} file(s)")
    for s, d in pairs:
        removed = postprocess_file(s, d, args.steps)
        print(f"  {s.name}: removed " + ", ".join(f"class {k}: {v} vox" for k, v in removed.items()))
    print(f">> wrote {dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocessing for SegTHOR predictions")
    parser.add_argument("--src", required=True, help="Prediction .nii.gz file or folder")
    parser.add_argument("--dest", required=True, help="Output file or folder")
    parser.add_argument(
        "--steps", nargs="+", default=["lcc"], choices=list(POSTPROCESSING),
        help="Postprocessing steps, applied in order",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
