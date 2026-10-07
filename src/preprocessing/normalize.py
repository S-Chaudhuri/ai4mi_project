import argparse
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np


def normalize_channel(channel: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize a single channel of the scan."""
    if method == "zscore":
        mean = np.mean(channel)
        std = np.std(channel)
        if std > 0:
            return (channel - mean) / std
        return channel - mean
    elif method == "minmax":
        min_val = np.min(channel)
        max_val = np.max(channel)
        if max_val - min_val > 0:
            return (channel - min_val) / (max_val - min_val)
        return channel - min_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_volume(volume: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize the values in the scan.
    If the volume has multiple channels (windows), normalizes each channel independently.
    """
    normalized = np.zeros_like(volume, dtype=np.float32)

    # Check if the volume is multi-channel.
    # Usually medical 3D images have 3 spatial dimensions.
    # If ndim == 4, the last dimension is the channel (window).
    if volume.ndim == 4:
        for c in range(volume.shape[-1]):
            normalized[..., c] = normalize_channel(volume[..., c], method)
    else:
        normalized = normalize_channel(volume, method)

    return normalized


def process_file(input_path: Path, output_path: Path, method: str):
    source = nib.load(str(input_path))
    volume = source.get_fdata(dtype=np.float32)

    normalized_volume = normalize_volume(volume, method)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(normalized_volume, source.affine, source.header), output_path
    )
    print(f"Normalized {input_path.name} ({method}) and saved to {output_path}")


def normalize_folder(input_dir: Path, output_dir: Path, method: str = "zscore"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in input_dir.rglob("*.nii*"):
        if not path.is_file():
            continue

        out_path = output_dir / path.relative_to(input_dir)

        # Ground truth files should not be normalized
        is_gt = "gt" in path.name.lower() or "mask" in path.name.lower()
        if is_gt:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_path)
            print(f"Copied GT {path.name}")
        else:
            process_file(path, out_path, method)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize NIfTI scans in a preprocessing pipeline."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing NIfTI files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="zscore",
        choices=["zscore", "minmax"],
        help="Normalization method (zscore or minmax)",
    )
    args = parser.parse_args()

    normalize_folder(args.input_dir, args.output_dir, args.method)


if __name__ == "__main__":
    main()
