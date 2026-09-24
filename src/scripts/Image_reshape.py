import SimpleITK as sitk
from pathlib import Path

TARGET_SPACING = (0.98, 0.98, 2.5)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "segthor_part1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "SEGTHOR_resampled"

def resample_image(src_dir: Path, tar_dir: Path, target_spacing = TARGET_SPACING):
    """
    Resample a 3D medical image to a specified target spacing.

    Args:
        src_dir (Path): Path to the source image file.
        tar_dir (Path): Path to save the resampled image.
        target_spacing: Desired spacing in (x, y, z) directions, Most common is (0.98, 0.98, 2.5).
    """
    # Read the image
    image = sitk.ReadImage(str(src_dir))

    # Get the original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate the new size based on the target spacing (Reference: Innolitics blog)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    # sitknearestneighbor is used for labels/masks and sitklinear for images (Reference: simpleitk documentation)
    filename = src_dir.name.lower()

    if filename in {"gt.nii", "gt.nii.gz"}:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)

    resampled_image = resampler.Execute(image)

    #create folder if it does not exist
    tar_dir.parent.mkdir(parents=True, exist_ok=True)

    # Save the resampled image
    sitk.WriteImage(resampled_image, str(tar_dir))

    print(f"Resampled: {src_dir.name}")
    print(f"  Original spacing: {original_spacing}")
    print(f"  Target spacing:   {resampled_image.GetSpacing()}")
    print(f"  Original size:    {original_size}")
    print(f"  New size:         {resampled_image.GetSize()}")

    # Return the resampled image
    return resampled_image


def resample_folder(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR, target_spacing=TARGET_SPACING,):
    """Resample every NIfTI image while preserving the input folder structure."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No NIfTI images found in {input_dir}")

    for source_path in image_paths:
        target_path = output_dir / source_path.relative_to(input_dir)
        resample_image(
            src_dir=source_path,
            tar_dir=target_path,
            target_spacing=target_spacing,
        )

    print(f"Resampled {len(image_paths)} image(s) to {output_dir}")


def main() -> None:
    resample_folder()


if __name__ == "__main__":
    main()