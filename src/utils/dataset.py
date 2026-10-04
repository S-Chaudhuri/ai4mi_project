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
import torch.nn.functional as F


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


class BoxDataset(Dataset):
    def __init__(
        self,
        subset: str,
        root_dir: Path,
        img_transform=None,
        gt_transform=None,
        sub_box_size: Optional[Tuple[int, int, int]] = None,
        fg_prob: float = 0.5,  # chance a box is forced to contain foreground
        debug: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.sub_box_size = sub_box_size
        self.fg_prob = fg_prob
        self.debug = debug

        self.items = make_3d_dataset(self.root_dir, self.subset)
        if self.debug:
            self.items = self.items[:10]

        print(f">> Created {subset} dataset with {len(self.items)} 3D patient volumes.")
        if self.sub_box_size:
            print(
                f"   Sub-box size: {self.sub_box_size}, foreground probability: {self.fg_prob:.2f}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def _load_volume(self, slice_paths: List[Path]) -> np.ndarray:
        """Stacks 2D PNGs along axis 0. Output shape: (D, H, W)"""
        slices = [np.array(Image.open(p)) for p in slice_paths]
        return np.stack(slices, axis=0)

    def _foreground_masks(self, gt: torch.Tensor) -> List[torch.Tensor]:
        """Boolean (D, H, W) mask for each foreground class present in this volume."""
        if gt.shape[0] > 1:  # one-hot: channel 0 = background
            masks = [gt[k] > 0 for k in range(1, gt.shape[0])]
        else:  # label map (1, D, H, W)
            masks = [gt[0] == l for l in torch.unique(gt[0]) if l != 0]
        return [m for m in masks if m.any()]

    def _pick_start(self, vol_shape, box_size, gt) -> Tuple[int, int, int]:
        """Corner of the box; foreground-aware with probability fg_prob."""
        masks = self._foreground_masks(gt) if random.random() < self.fg_prob else []

        if masks:
            mask = random.choice(
                masks
            )  # random class first, so rare classes get equal chance
            coords = torch.nonzero(mask)  # voxels of that class, shape (N, 3)
            vox = coords[random.randrange(len(coords))].tolist()
            starts = []
            for v, size, full in zip(vox, box_size, vol_shape):
                s = v - random.randint(
                    0, size - 1
                )  # voxel lands somewhere inside the box
                starts.append(
                    min(max(s, 0), full - size)
                )  # keep the box inside the volume
            return tuple(starts)

        # no foreground requested (or none present): uniform random box
        return tuple(
            random.randint(0, full - size) for size, full in zip(box_size, vol_shape)
        )

    def _extract_sub_box(
        self, img: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Box of shape sub_box_size from (C, D, H, W) tensors; pads if the volume is smaller."""
        _, D, H, W = img.shape
        d, h, w = self.sub_box_size

        pd, ph, pw = max(0, d - D), max(0, h - H), max(0, w - W)
        if pd or ph or pw:
            pad = (0, pw, 0, ph, 0, pd)  # F.pad order: W, H, D
            valid = F.pad(torch.ones_like(img[:1]), pad, value=0)
            img = F.pad(img, pad, value=0)
            gt = F.pad(gt, pad, value=0)
            if gt.shape[0] > 1:
                gt[0][valid[0] == 0] = 1  # padded voxels count as background
            _, D, H, W = img.shape

        ds, hs, ws = self._pick_start((D, H, W), (d, h, w), gt)
        return (
            img[:, ds : ds + d, hs : hs + h, ws : ws + w],
            gt[:, ds : ds + d, hs : hs + h, ws : ws + w],
        )

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
            assert tuple(img.shape[1:]) == tuple(self.sub_box_size), (
                f"Sub-box shape {tuple(img.shape[1:])} does not match expected {self.sub_box_size}"
            )

        return {"images": img, "gts": gt, "stems": item["stem"]}
