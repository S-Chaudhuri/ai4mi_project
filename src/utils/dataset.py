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

def coord_grid(vol_shape, start, size):
    """(3, d, h, w) float32 in [0,1]: normalised (z, y, x) index of every voxel in the box."""
    axes = [np.clip(np.arange(s, s + n, dtype=np.float32) / max(full - 1, 1), 0, 1)
            for full, s, n in zip(vol_shape, start, size)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    return np.stack([z, y, x])

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

            _, H, W = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, H, W)

            data_dict["gts"] = gt

        return data_dict


# def make_3d_dataset(root_dir: Path, subset: str) -> List[Dict[str, Any]]:
#     # 1. Resolve image and ground-truth directories
#     subset_dir = root_dir / subset
#     img_dir = (
#         subset_dir / "img" if (subset_dir / "img").exists() else subset_dir / "images"
#     )
#     gt_dir = subset_dir / "gt" if (subset_dir / "gt").exists() else subset_dir / "masks"

#     if not img_dir.exists() or not gt_dir.exists():
#         print(f"Error: Expected image/gt folders not found in {subset_dir}")
#         return []

#     # 2. Group PNG files by Patient ID (e.g., 'Patient_01_042.png' -> key 'Patient_01')
#     img_groups = defaultdict(list)
#     gt_groups = defaultdict(list)

#     for img_path in sorted(img_dir.glob("*.png")):
#         # Extract patient ID prefix (e.g., "Patient_01" from "Patient_01_045.png")
#         pid = "_".join(img_path.stem.split("_")[:2])
#         img_groups[pid].append(img_path)

#     for gt_path in sorted(gt_dir.glob("*.png")):
#         pid = "_".join(gt_path.stem.split("_")[:2])
#         gt_groups[pid].append(gt_path)

#     # 3. Pair 3D patient volumes
#     items = []
#     for pid in sorted(img_groups.keys()):
#         if pid not in gt_groups:
#             print(f"[Warning] Skipping {pid}: No matching GT slices found.")
#             continue

#         img_slices = sorted(img_groups[pid])
#         gt_slices = sorted(gt_groups[pid])

#         if len(img_slices) != len(gt_slices):
#             print(
#                 f"[Warning] Mismatch for {pid}: {len(img_slices)} images vs {len(gt_slices)} GTs."
#             )
#             continue

#         items.append({"stem": pid, "images": img_slices, "gts": gt_slices})

#     print(
#         f"Successfully matched {len(items)} 3D patient volumes between images and GT."
#     )
#     return items

def make_3d_dataset(
    root_dir: Path,
    subset: str,
) -> List[Dict[str, Any]]:
    subset_dir = Path(root_dir) / subset

    if not subset_dir.is_dir():
        print(f"Error: Dataset directory not found: {subset_dir}")
        return []

    items = []

    for patient_dir in sorted(subset_dir.glob("Patient_*")):
        if not patient_dir.is_dir():
            continue

        img_path = patient_dir / "img.npy"
        gt_path = patient_dir / "gt.npy"

        if not img_path.is_file() or not gt_path.is_file():
            print(f"[Warning] Skipping {patient_dir.name}: missing image or GT")
            continue

        items.append({
            "stem": patient_dir.name,
            "images": img_path,
            "gts": gt_path,
        })

    print(
        f"Found {len(items)} patient volumes in {subset_dir}"
    )
    return items

def load_volume(slice_paths: List[Path]) -> np.ndarray:
    """Stacks 2D PNG slices along axis 0. Output shape: (D, H, W)"""
    slices = [np.array(Image.open(p)) for p in slice_paths]
    return np.stack(slices, axis=0)

class BoxDataset(Dataset):
    def __init__(
        self,
        subset: str,
        root_dir: Path,
        img_transform=None,
        gt_transform=None,
        sub_box_size: Optional[Tuple[int, int, int]] = None,
        num_classes: int = 5,
        fg_prob: float = 0.5,  # chance a box is forced to contain foreground
        debug: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.sub_box_size = sub_box_size
        self.num_classes = num_classes
        self.fg_prob = fg_prob
        self.debug = debug

        self.items = make_3d_dataset(self.root_dir, self.subset)
        if self.debug:
            self.items = self.items[:10]

        # Load every patient volume into RAM once; __getitem__ never touches disk.
        # Images are kept channel-first (C, D, H, W); labels are class indices (D, H, W).
        for item in self.items:
            img = np.load(item["images"])
            if img.ndim == 3:
                img = img[None]                      # -> (C, D, H, W)
            gt = np.load(item["gts"])
            if gt.ndim == 4:
                gt = gt[0]                           # labels are (D, H, W)

            assert img.shape[1:] == gt.shape, (item["stem"], img.shape, gt.shape)
            # Catches the old 0/63/126/189/252 label encoding
            assert gt.max() < num_classes, (item["stem"], np.unique(gt))

            item["img_vol"] = img
            item["gt_cls"] = gt.astype(np.int8)
            item["fg_coords"] = [
                np.argwhere(item["gt_cls"] == k) for k in range(1, num_classes)
            ]
            del item["images"], item["gts"]

        n_vox = sum(v["img_vol"].size for v in self.items)
        print(f">> Created {subset} dataset with {len(self.items)} 3D patient volumes.")
        print(f"   Loaded all volumes into RAM: {n_vox * 2 / 1e6:.0f} MB (img + gt, rough).")
        if self.items:
            first = self.items[0]
            print(
                f"   Example {first['stem']}: img {first['img_vol'].shape} {first['img_vol'].dtype}, "
                f"gt {first['gt_cls'].shape}, labels {np.unique(first['gt_cls'])}"
            )
        if self.sub_box_size:
            print(
                f"   Sub-box size: {self.sub_box_size}, foreground probability: {self.fg_prob:.2f}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def _pick_start(self, item: Dict[str, Any], vol_shape) -> Tuple[int, int, int]:
        """Corner of the box; foreground-aware with probability fg_prob."""
        d, h, w = self.sub_box_size

        if item["fg_coords"] and random.random() < self.fg_prob:
            nonempty = [c for c in item["fg_coords"] if len(c) > 0]
            if nonempty:
                # random class first, so rare classes get equal chance
                coords = random.choice(nonempty)  # voxels of that class, (N, 3)
                vox = coords[random.randrange(len(coords))]
                starts = []
                for v, size, full in zip(vox, (d, h, w), vol_shape):
                    s = int(v) - random.randint(0, size - 1)
                    starts.append(
                        min(max(s, 0), max(0, full - size))
                    )  # keep the box inside the volume
                return tuple(starts)

        # no foreground requested (or none present): uniform random box
        return tuple(
            random.randint(0, max(0, full - size))
            for size, full in zip((d, h, w), vol_shape)
        )

    def _pad_to_box(self, img: Tensor, gt: Tensor) -> Tuple[Tensor, Tensor]:
        """Pad at the end if the volume is smaller than the box on some axis."""
        d, h, w = self.sub_box_size
        pd, ph, pw = d - img.shape[1], h - img.shape[2], w - img.shape[3]
        if pd or ph or pw:
            pad = (0, pw, 0, ph, 0, pd)  # F.pad order: W, H, D
            valid = F.pad(torch.ones_like(img[:1]), pad, value=0)
            img = F.pad(img, pad, value=0)
            gt = F.pad(gt, pad, value=0)
            if gt.shape[0] > 1:
                gt[0][valid[0] == 0] = 1  # padded voxels count as background
        return img, gt

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        if self.sub_box_size is None:  # no sub-boxing: return the full volumes
            img = (
                self.img_transform(item["img_vol"])
                if self.img_transform
                else torch.from_numpy(item["img_vol"])
            )
            gt = (
                self.gt_transform(item["gt_cls"])
                if self.gt_transform
                else torch.from_numpy(item["gt_cls"].astype(np.int64, copy=False))
            )
            return {"images": img, "gts": gt, "stems": item["stem"]}

        d, h, w = self.sub_box_size
        vol_shape = item["img_vol"].shape[1:]  

        ds, hs, ws = self._pick_start(item, vol_shape)

        # Crop the small box out of the in-RAM volumes first (cheap uint8/int8
        # views), and only then run the transforms on the box, not the volume.
        img_box = item["img_vol"][:, ds : ds + d, hs : hs + h, ws : ws + w]
        gt_box = item["gt_cls"][ds : ds + d, hs : hs + h, ws : ws + w]

        img = (
            self.img_transform(img_box)
            if self.img_transform
            else torch.from_numpy(np.ascontiguousarray(img_box))
        )
        gt = (
            self.gt_transform(gt_box)
            if self.gt_transform
            else torch.from_numpy(gt_box.astype(np.int64, copy=False))
        )

        if img.shape[1:] != (d, h, w):
            img, gt = self._pad_to_box(img, gt)

        assert tuple(img.shape[1:]) == tuple(self.sub_box_size), (
            f"Sub-box shape {tuple(img.shape[1:])} does not match expected {tuple(self.sub_box_size)}"
        )

        return {"images": img, "gts": gt, "stems": item["stem"]}


def window_starts(full: int, size: int, step: int) -> List[int]:
    """Start positions along one axis; the last box is pushed back so it touches the end."""
    if full <= size:
        return [0]
    starts = list(range(0, full - size + 1, step))
    if starts[-1] != full - size:
        starts.append(full - size)
    return starts


class GridBoxDataset(BoxDataset):
    """Deterministic validation boxes: a fixed grid over every patient, with overlap."""

    def __init__(
        self,
        subset: str,
        root_dir: Path,
        img_transform=None,
        gt_transform=None,
        sub_box_size: Tuple[int, int, int] = (128, 128, 128),
        num_classes: int = 5,
        overlap: float = 0.5,  # 0.5 = each box overlaps its neighbour by half
        debug: bool = False,
    ):
        super().__init__(
            subset,
            root_dir,
            img_transform,
            gt_transform,
            sub_box_size=sub_box_size,
            num_classes=num_classes,
            fg_prob=0.0,
            debug=debug,
        )
        self.overlap = overlap

        step = [max(1, int(s * (1 - overlap))) for s in sub_box_size]
        d, h, w = sub_box_size

        # (patient index, depth start, height start, width start) for every box
        self.boxes: List[Tuple[int, int, int, int]] = []
        for pi, item in enumerate(self.items):
            _, D, H_img, W_img = item["img_vol"].shape       # always (C, D, H, W) now
            for ds in window_starts(D, d, step[0]):
                for hs in window_starts(H_img, h, step[1]):
                    for ws in window_starts(W_img, w, step[2]):
                        self.boxes.append((pi, ds, hs, ws))

        print(
            f"   Grid: overlap {overlap:.0%}, {len(self.boxes)} boxes from {len(self.items)} patients"
        )

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, idx: int) -> dict:
        pi, ds, hs, ws = self.boxes[idx]
        item = self.items[pi]
        d, h, w = self.sub_box_size

        # Crop the box from the in-RAM volumes (copy so it is contiguous)
        img_np = item["img_vol"][:, ds : ds + d, hs : hs + h, ws : ws + w].copy()
        gt_np = item["gt_cls"][ds : ds + d, hs : hs + h, ws : ws + w].copy()

        img = (
            self.img_transform(img_np)
            if self.img_transform
            else torch.from_numpy(img_np)
        )
        gt = (
            self.gt_transform(gt_np)
            if self.gt_transform
            else torch.from_numpy(gt_np.astype(np.int64, copy=False))
        )

        # Pad at the end if the volume is smaller than the box on some axis
        _, D, H, W = img.shape
        pd, ph, pw = max(0, d - D), max(0, h - H), max(0, w - W)
        if pd or ph or pw:
            pad = (0, pw, 0, ph, 0, pd)
            valid = F.pad(torch.ones_like(img[:1]), pad, value=0)
            img = F.pad(img, pad, value=0)
            gt = F.pad(gt, pad, value=0)
            if gt.shape[0] > 1:
                gt[0][valid[0] == 0] = 1  # padded voxels count as background

        assert tuple(img.shape[1:]) == tuple(self.sub_box_size), (
            f"Box shape {tuple(img.shape[1:])} does not match expected {self.sub_box_size}"
        )

        return {
            "images": img,
            "gts": gt,
            "stems": f"{item['stem']}_d{ds}_h{hs}_w{ws}",
        }


class CoarseDataset(BoxDataset):
    """Dataset that returns the full 3D volume downsampled to a target size."""

    def __init__(
        self,
        subset: str,
        root_dir: Path,
        img_transform=None,
        gt_transform=None,
        target_size: Tuple[int, int, int] = (64, 64, 64),
        num_classes: int = 5,
        debug: bool = False,
    ):
        super().__init__(
            subset=subset,
            root_dir=root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            sub_box_size=None,
            num_classes=num_classes,
            fg_prob=0.0,
            debug=debug,
        )
        self.target_size = target_size

        # Pre-compute the coarse versions
        for item in self.items:
            # (1, 1, D, H, W) for F.interpolate
            img_t = torch.from_numpy(item["img_vol"]).unsqueeze(0).float()
            gt_t = torch.from_numpy(item["gt_cls"]).unsqueeze(0).unsqueeze(0).float()

            img_coarse = F.interpolate(
                img_t, size=self.target_size, mode="trilinear", align_corners=False
            )
            gt_coarse = F.interpolate(gt_t, size=self.target_size, mode="nearest")

            # Squeeze twice to get back proper dimensions
            img_np = img_coarse.squeeze(0).numpy()
            item["img_vol"] = np.clip(np.round(img_np), 0, 255).astype(
                item["img_vol"].dtype
            )
            item["gt_cls"] = (
                gt_coarse.squeeze(0).squeeze(0).numpy().astype(item["gt_cls"].dtype)
            )

        print(f"   Target coarse size: {self.target_size} (pre-computed in memory)")

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        img = (
            self.img_transform(item["img_vol"])
            if self.img_transform
            else torch.from_numpy(np.ascontiguousarray(item["img_vol"]))
        )
        gt = (
            self.gt_transform(item["gt_cls"])
            if self.gt_transform
            else torch.from_numpy(item["gt_cls"].astype(np.int64, copy=False))
        )

        return {
            "images": img,
            "gts": gt,
            "stems": item["stem"],
        }
