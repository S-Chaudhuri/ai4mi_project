import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch



# (window center in HU, window width in HU (Level, Width))
DEFAULT_WINDOWS = [
    (40,   400),   # channel 1 — soft tissue  (heart, esophagus)
    (-600, 1500),  # channel 2 — lung/air      (trachea)
    (100,  700),   # channel 3 — blood/aorta   (aorta)
    #Other possible windows:
    # (400, 1800), # channel 4 — bone
    # (200,  1000),  # channel 5 — fat
    # (50, 350),   # mediastinum — sharper contrast for esophagus borders
    # Link: https://radiopaedia.org/articles/windowing-ct, https://en.wikipedia.org/wiki/Hounsfield_scale, https://maxillofacial.org/reference/diagnostics/radiology/ct-windows-hounsfield
]


def apply_window(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply a single HU window to a 3D volume."""
    min_value = center - width / 2
    max_value = center + width / 2

    # Clip the values to the specified window
    volume = np.clip(volume, min_value, max_value)

    # Normalization to [0, 1]
    volume = (volume - min_value) / (max_value - min_value)

    return volume

def convert_to_multichannel(input_path, output_path, windows=DEFAULT_WINDOWS):
    source = nib.load(str(input_path))
    volume = source.get_fdata(dtype=np.float32)
    channels = [apply_window(volume, c, w) for c, w in windows]

    array = np.stack(channels, axis=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True) 
    nib.save(nib.Nifti1Image(array, source.affine, source.header), output_path)
    print(f"Saved {array.shape} to {output_path}")

def copy_gt(patient_dir: Path, out_dir: Path) -> bool:
    """Copy the unchanged ground truth into the output patient folder."""
    gt_path = next(patient_dir.glob("GT*.nii*"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(gt_path, out_dir / "GT_4label_v6.nii.gz")

def convert_all(data_root: Path):
    out_root = data_root.with_name(f"{data_root.name}_3channel")

    for path in sorted(data_root.rglob("*.nii*")):
        name = path.name.removesuffix(".gz").removesuffix(".nii")
        if name != path.parent.name:
            continue
        out_dir = out_root / path.parent.relative_to(data_root)
        convert_to_multichannel(path, out_dir / f"{name}.nii.gz")
        copy_gt(path.parent, out_dir)

def main() -> None:
    # Path to the data folder containing the CT scans and ground truth
    convert_all(Path(__file__).resolve().parents[2] / "data" / "segthor_midterm")


if __name__ == "__main__":
    main()