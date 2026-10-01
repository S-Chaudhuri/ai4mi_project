# MIT License

# Copyright (c) 2025 Hoel Kervadec

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

from pathlib import Path
import random
import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, Union, List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != "test":
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    if len(images) != len(full_labels):
        raise ValueError("Not the same number of images and labels in dataset")

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(
        self,
        subset,
        root_dir,
        img_transform,
        gt_transform,
        augment=False,
        equalize=False,
        debug=False,
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == "test"

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))

        data_dict = {"images": img, "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict


def make_3d_dataset(root_dir: Path, subset: str) -> List[Dict[str, Any]]:
    # 1. Resolve image and ground-truth directories
    subset_dir = root_dir / subset
    img_dir = (
        subset_dir / "img" if (subset_dir / "img").exists() else subset_dir / "images"
    )
    gt_dir = subset_dir / "gt" if (subset_dir / "gt").exists() else subset_dir / "masks"

    if not img_dir.exists() or not gt_dir.exists():
        print(f"Error: Expected image/gt folders not found in {subset_dir}")
        return []

    # 2. Group PNG files by Patient ID (e.g., 'Patient_01_042.png' -> key 'Patient_01')
    img_groups = defaultdict(list)
    gt_groups = defaultdict(list)

    for img_path in sorted(img_dir.glob("*.png")):
        # Extract patient ID prefix (e.g., "Patient_01" from "Patient_01_045.png")
        pid = "_".join(img_path.stem.split("_")[:2])
        img_groups[pid].append(img_path)

    for gt_path in sorted(gt_dir.glob("*.png")):
        pid = "_".join(gt_path.stem.split("_")[:2])
        gt_groups[pid].append(gt_path)

    # 3. Pair 3D patient volumes
    items = []
    for pid in sorted(img_groups.keys()):
        if pid not in gt_groups:
            print(f"[Warning] Skipping {pid}: No matching GT slices found.")
            continue

        img_slices = sorted(img_groups[pid])
        gt_slices = sorted(gt_groups[pid])

        if len(img_slices) != len(gt_slices):
            print(
                f"[Warning] Mismatch for {pid}: {len(img_slices)} images vs {len(gt_slices)} GTs."
            )
            continue

        items.append({"stem": pid, "images": img_slices, "gts": gt_slices})

    print(
        f"Successfully matched {len(items)} 3D patient volumes between images and GT."
    )
    return items


"""Currently this only gives a single box per sample, so we should update this in"""


class BoxDataset(Dataset):
    def __init__(
        self,
        subset: str,
        root_dir: Path,
        img_transform=None,
        gt_transform=None,
        sub_box_size: tuple[int, int, int] = (128, 132, 132),  # e.g., (128, 132, 132)
        debug: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.sub_box_size = sub_box_size
        self.debug = debug

        self.items = make_3d_dataset(self.root_dir, self.subset)
        if self.debug:
            self.items = self.items[:10]

        print(f">> Created {subset} dataset with {len(self.items)} 3D patient volumes.")
        if self.sub_box_size:
            print(f"   Using sub-box extraction size: {self.sub_box_size}")

    def __len__(self) -> int:
        """Returns total number of 3D patient volume samples in dataset."""
        return len(self.items)

    def _load_volume(self, slice_paths: List[Path]) -> np.ndarray:
        """Loads a list of 2D PNG file paths and stacks them along the depth axis (axis=0).

        Output shape: (D, H, W)
        """

        slices = [np.array(Image.open(p)) for p in slice_paths]
        return np.stack(slices, axis=0)

    def _extract_sub_box(
        self, img: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crops a sub-box of shape (D_sub, H_sub, W_sub) from full (1, D_full, H_full, W_full)."""
        _, D_full, H_full, W_full = img.shape
        D_sub, H_sub, W_sub = self.sub_box_size

        d_start = random.randint(0, max(0, D_full - D_sub))
        h_start = random.randint(0, max(0, H_full - H_sub))
        w_start = random.randint(0, max(0, W_full - W_sub))

        img_crop = img[
            :,
            d_start : d_start + D_sub,
            h_start : h_start + H_sub,
            w_start : w_start + W_sub,
        ]
        gt_crop = gt[
            :,
            d_start : d_start + D_sub,
            h_start : h_start + H_sub,
            w_start : w_start + W_sub,
        ]

        return img_crop, gt_crop

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        img_np = self._load_volume(item["images"])
        gt_np = self._load_volume(item["gts"])

        img = (
            self.img_transform(img_np)
            if self.img_transform
            else torch.from_numpy(img_np)
        )
        gt = self.gt_transform(gt_np) if self.gt_transform else torch.from_numpy(gt_np)

        assert img.shape[1:] == gt.shape[1:], (
            f"Full volume shape mismatch: img {img.shape[1:]} vs gt {gt.shape[1:]}"
        )

        if self.sub_box_size is not None:
            img, gt = self._extract_sub_box(img, gt)
            assert img.shape[1:] == self.sub_box_size, (
                f"Sub-box crop shape {img.shape[1:]} does not match expected {self.sub_box_size}"
            )

        return {"images": img, "gts": gt, "stems": item["stem"]}

