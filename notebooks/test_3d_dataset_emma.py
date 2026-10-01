#!/usr/bin/env python3
"""
3D Volumetric Construction Test
Verifies that 2D PNG slice groups are correctly assembled into 3D volumes.
"""

from pathlib import Path
from functools import partial
import numpy as np
import torch
import autoroot  # noqa

# Import pipeline components
from main import img_transform_3d, gt_transform_3d
from src.utils.dataset import BoxDataset


def test_volumetric_structure():
    print("==================================================")
    print("      TESTING 3D VOLUMETRIC DATA CONSTRUCTION     ")
    print("==================================================\n")

    dataset_name = "SEGTHOR"
    data_dir = autoroot.root / "data" / dataset_name
    num_classes = 5

    if not data_dir.exists():
        print(f"Error: Path '{data_dir}' not found.")
        return

    # 1. Instantiate Dataset
    dataset = BoxDataset(
        subset="train",
        root_dir=data_dir,
        img_transform=img_transform_3d,
        gt_transform=partial(gt_transform_3d, num_classes),
        sub_box_size=(128, 128, 128), # Returns sub-box crop of shape (128, 128, 132)
    )

    if len(dataset) == 0:
        print("Error: No items found in dataset.")
        return

    print(f"Dataset loaded. Total 3D volumes found: {len(dataset)}\n")

    # 2. Inspect First Sample
    sample = dataset[0]
    img = sample["images"]
    gt = sample["gts"]
    stem = sample["stems"]

    print(f"Patient Identifier / Stem: '{stem}'")
    print(f"Image Tensor Shape       : {img.shape}  --> (Channels, Depth, Height, Width)")
    print(f"GT Tensor Shape          : {gt.shape}  --> (Classes, Depth, Height, Width)")

    # 3. Dimensionality Checks
    print("\n--- Dimensionality Checks ---")
    
    # Verify 4D Tensor Output (C, D, H, W)
    if img.ndim == 4:
        print("  [PASS] Image is a 4D Tensor (1, D, H, W)")
    else:
        print(f"  [FAIL] Expected 4D tensor, got {img.ndim}D shape: {img.shape}")

    # Verify Depth Dimension (D > 1)
    C, D, H, W = img.shape
    if D > 1:
        print(f"  [PASS] Successfully stacked {D} axial slices into 3D Depth dimension")
    else:
        print(f"  [FAIL] Depth dimension is {D}. Expected D > 1 slices stacked together.")

    # Verify 2D Spatial Consistency (H, W)
    print(f"  [INFO] Volume 2D Slice Resolution: {H} x {W}")

    # 4. Content Verification across Slices
    print("\n--- Slice Continuity Check ---")
    
    # Calculate slice-wise mean intensity along the Depth axis (dim 1)
    slice_means = img[0].mean(dim=(1, 2)).numpy()  # Mean intensity per slice
    
    print(f"  - First slice mean intensity : {slice_means[0]:.4f}")
    print(f"  - Middle slice mean intensity: {slice_means[D // 2]:.4f}")
    print(f"  - Last slice mean intensity  : {slice_means[-1]:.4f}")

    # Check if intensity varies across slices (proves distinct slices were loaded, not duplicate copies)
    if not np.allclose(slice_means[0], slice_means[D // 2]):
        print("  [PASS] Slices contain distinct volumetric spatial content across depth")
    else:
        print("  [WARNING] Slice intensities are identical across depth. Verify slice ordering.")

    # 5. One-Hot Spatial Consistency
    print("\n--- Ground Truth Class Channel Check ---")
    present_classes = torch.nonzero(gt.sum(dim=(1, 2, 3))).flatten().tolist()
    print(f"  - Organ classes present in this volume: {present_classes} / {list(range(num_classes))}")
    
    # Check sum along class dimension equals 1 for all voxels
    class_sum = gt.sum(dim=0)
    if torch.allclose(class_sum, torch.ones_like(class_sum)):
        print("  [PASS] One-hot encoding valid: Every voxel sums to 1 across class channels")
    else:
        print("  [FAIL] Invalid one-hot encoding: Voxels do not sum to 1 across classes")

    print("\n==================================================")
    print("          VOLUMETRIC VERIFICATION COMPLETE        ")
    print("==================================================")


if __name__ == "__main__":
    test_volumetric_structure()