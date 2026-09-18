from pathlib import Path
import numpy as np
import SimpleITK as sitk

from Image_reshape import resample_image

def main() -> None:
    source_path = Path(
        "data/segthor_part1/train/Patient_02/Patient_02.nii.gz"
    )

    target_path = Path(
        "data/SEGTHOR_resampled/train/Patient_002/Patient_002.nii.gz"
    )

    target_spacing = (0.98, 0.98, 2.5)

    # Run the preprocessing function.
    resample_image(
        src_dir=source_path,
        tar_dir=target_path,
        target_spacing=target_spacing,
    )

    # Test 1: Did the new file get written?
    assert target_path.exists(), (
        f"Resampled file was not created: {target_path}"
    )

    # Test 2: Read the file from disk, rather than trusting the in-memory result.
    resampled_ct = sitk.ReadImage(str(target_path))

    # Test 3: Is the spacing correct?
    assert np.allclose(
        resampled_ct.GetSpacing(),
        target_spacing,
        atol=1e-6,
    ), (
        f"Incorrect spacing: {resampled_ct.GetSpacing()}"
    )

    print("Test passed.")
    print(f"Saved file: {target_path}")
    print(f"Output spacing: {resampled_ct.GetSpacing()}")
    print(f"Output size: {resampled_ct.GetSize()}")


if __name__ == "__main__":
    main()