#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
from pathlib import Path

import numpy as np
from skimage.io import imread


def get_z(image: Path) -> int:
    return int(image.stem.split("_")[-1])


def load_patient_slices(folder: Path, id_: str) -> np.ndarray:
    slices = sorted(folder.glob(f"{id_}_*.png"), key=get_z)
    if not slices:
        raise FileNotFoundError(f"No PNG slices found for {id_} in {folder}")

    first_slice = imread(slices[0])
    volume = np.empty((*first_slice.shape, len(slices)), dtype=first_slice.dtype)
    for path in slices:
        volume[:, :, get_z(path)] = imread(path)

    return volume


def stitch_patient_from_slices(
    id_: str,
    image_folder: Path,
    gt_folder: Path,
    dest_folder: Path,
) -> None:
    """Save one patient's CT and labels as 3D NumPy arrays."""
    image_volume = load_patient_slices(image_folder, id_)
    gt_volume = load_patient_slices(gt_folder, id_)

    patient_folder = dest_folder / id_
    patient_folder.mkdir(parents=True, exist_ok=True)
    patient_number = id_.split("_")[-1]
    np.save(patient_folder / f"patient_{patient_number}.npy", image_volume)
    np.save(patient_folder / f"GT_{patient_number}.npy", gt_volume // 63)


def find_slice_folders(data_root: Path) -> list[tuple[Path, Path]]:
    """Find matching image and ground-truth folders below the SEGTHOR root."""
    folders = [
        (image_folder, image_folder.parent / "gt")
        for image_folder in sorted(data_root.rglob("img"))
        if (image_folder.parent / "gt").is_dir()
    ]
    if not folders:
        raise FileNotFoundError(f"No matching img/gt folders found in {data_root}")
    return folders


def main(args) -> None:
    slice_folders = (
        [(args.data_folder, args.gt_folder)]
        if args.gt_folder is not None
        else find_slice_folders(args.data_folder)
    )

    for image_folder, gt_folder in slice_folders:
        patient_ids = (
            [args.patient]
            if args.patient is not None
            else sorted(
                {
                    path.stem.rsplit("_", 1)[0]
                    for path in image_folder.glob("*.png")
                }
            )
        )
        for patient_id in patient_ids:
            if not list(image_folder.glob(f"{patient_id}_*.png")):
                continue
            stitch_patient_from_slices(
                patient_id,
                image_folder,
                gt_folder,
                args.dest_folder,
            )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merging slices parameters")
    parser.add_argument(
        "--data_folder",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "SEGTHOR",
        help="SEGTHOR root folder; defaults to data/SEGTHOR",
    )
    parser.add_argument(
        "--gt_folder",
        type=Path,
        help="Optional single ground-truth folder; otherwise all SEGTHOR splits are used",
    )
    parser.add_argument(
        "--patient",
        type=str,
        help="Process only this patient in slice-only NumPy mode",
    )
    parser.add_argument(
        "--dest_folder",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "segthor_numpy",
        help="Destination folder; defaults to data/segthor_numpy",
    )
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
