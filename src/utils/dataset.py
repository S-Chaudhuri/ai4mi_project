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
from typing import Callable, Union


import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torch.version import debug
from typing import Callable, Optional, Union
import numpy as np


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
    
 
def make_3d_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    """
    3D counterpart of make_dataset(). Same directory layout convention:
        root/subset/img/*.npy
        root/subset/gt/*.npy
 
    Assumes each *.npy file holds one full volumetric box (e.g. 132x132x128).
    If your boxes are stored as NIfTI (.nii.gz) instead, just change the glob
    pattern below (e.g. "*.nii.gz") and load with nibabel in the Dataset.
    """
    assert subset in ["train", "val", "test"]
 
    root = Path(root)
    print(f"> {root=}")
 
    img_path = root / subset / "img"
    full_path = root / subset / "gt"
 
    images: list[Path] = sorted(img_path.glob("*.npy"))
    full_labels: list[Path | None]
    if subset != "test":
        full_labels = sorted(full_path.glob("*.npy"))
    else:
        full_labels = [None] * len(images)
 
    if len(images) != len(full_labels):
        raise ValueError("Not the same number of images and labels in dataset")
 
    return list(zip(images, full_labels))
 
 
class BoxDataset(Dataset):
    """
    3D counterpart of SliceDataset. Yields fixed-size volumetric boxes
    (e.g. 132x132x128) instead of 2D slices, for use with a 3D-conv network.
 
    img_transform / gt_transform are expected to take a numpy array
    (D, H, W) and return a Tensor shaped (C, D, H, W) -- same role as
    the PIL-based transforms in the 2D version, just swapped for whatever
    3D-capable transform pipeline you're using (e.g. torchio, MONAI, or
    your own numpy/torch functions).
    """
 
    def __init__(
        self,
        subset,
        root_dir,
        img_transform,
        gt_transform,
        box_size: tuple[int, int, int] = (132, 132, 128),
        augment=False,
        equalize=False,
        debug=False,
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.box_size: tuple[int, int, int] = tuple(box_size)
 
        self.test_mode: bool = subset == "test"
 
        self.files = make_3d_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]
 
        print(
            f">> Created {subset} dataset with {len(self)} boxes "
            f"of size {self.box_size}..."
        )
 
    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
 
        img_np = np.load(img_path)
        img: Tensor = self.img_transform(img_np)
 
        data_dict = {"images": img, "stems": img_path.stem}
 
        if not self.test_mode:
            gt_np = self._load_volume(gt_path)
            gt: Tensor = self.gt_transform(gt_np)
 
            _, D, H, W = img.shape
            K, Dg, Hg, Wg = gt.shape
            assert (Dg, Hg, Wg) == (D, H, W)
            assert (D, H, W) == self.box_size, (
                f"Expected box size {self.box_size}, got {(D, H, W)}"
            )
 
            data_dict["gts"] = gt
 
        return data_dict


