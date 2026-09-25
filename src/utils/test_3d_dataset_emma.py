import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Import your actual dataset class and helper functions
from dataset import BoxDataset


def test_box_dataset():
    # 1. Create a temporary folder structure simulating 1 patient with 10 slices
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "train" / "img" / "patient_01"
        gt_dir = tmp_path / "train" / "gt" / "patient_01"
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy 100x100 PNG files
        for i in range(10):
            Image.new("L", (100, 100), color=100).save(img_dir / f"slice_{i:03d}.png")
            Image.new("L", (100, 100), color=1).save(gt_dir / f"slice_{i:03d}.png")

        # Dummy transform functions
        img_transform = lambda img: torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0) / 255.0
        gt_transform = lambda img: torch.from_numpy(np.array(img, dtype=np.int64)).unsqueeze(0)

        # 2. Instantiate BoxDataset with a specific box size (Depth=4, H=32, W=32)
        target_box = (4, 32, 32)
        dataset = BoxDataset(
            subset="train",
            root_dir=tmp_path,
            img_transform=img_transform,
            gt_transform=gt_transform,
            box_size=target_box,
        )

        # 3. Test single item output
        sample = dataset[0]
        img_tensor = sample["images"]
        gt_tensor = sample["gts"]

        assert img_tensor.shape == (1, 4, 32, 32), f"Expected (1, 4, 32, 32), got {img_tensor.shape}"
        assert gt_tensor.shape == (1, 4, 32, 32), f"Expected (1, 4, 32, 32), got {gt_tensor.shape}"
        print("✅ Single item test passed!")

        # 4. Test PyTorch DataLoader compatibility
        loader = DataLoader(dataset, batch_size=1)
        batch = next(iter(loader))
        assert batch["images"].shape == (1, 1, 4, 32, 32), "DataLoader batching failed"
        print("✅ DataLoader batching test passed!")


if __name__ == "__main__":
    test_box_dataset()

data_root = "/Users/emmaoosterhuis/Library/CloudStorage/OneDrive-UvA/Master/Medical Imaging/ai4mi_project/data"

real_dataset = BoxDataset(
    subset="train",
    root_dir=data_root,g
    img_transform=lambda img: torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0) / 255.0,
    gt_transform=lambda img: torch.from_numpy(np.array(img, dtype=np.int64)).unsqueeze(0),
    box_size=(4, 32, 32),
    debug=True
)   

sample = real_dataset[0]
print(f"Sample image shape: {sample['images'].shape}")  
print("3D Image Shape (C, D, H, W):", sample["images"].shape)
print("3D Label Shape (K, D, H, W):", sample["gts"].shape)