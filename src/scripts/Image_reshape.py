import SimpleITK as sitk
from pathlib import Path

TARGET_SPACING = (0.98, 0.98, 2.5)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "segthor_midterm"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "SEGTHOR_resampled"

def resample_image(
    src_dir: Path,
    tar_dir: Path,
    target_spacing=TARGET_SPACING,
    reference_image: sitk.Image = None,
    geometry_reference: sitk.Image = None,
):
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

    # sitknearestneighbor is used for labels/masks and sitklinear for images (Reference: simpleitk documentation)
    is_ground_truth = src_dir.name.lower() == "gt_4label_v6.nii.gz"
    interpolator = sitk.sitkNearestNeighbor if is_ground_truth else sitk.sitkLinear

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)

    if reference_image is not None:
        if geometry_reference is not None:
            image.CopyInformation(geometry_reference)
        resampler.SetReferenceImage(reference_image)
        resampler.SetSize(reference_image.GetSize())
        resampler.SetOutputDirection(reference_image.GetDirection())
        resampler.SetOutputOrigin(reference_image.GetOrigin())
    else:
        # Calculate the new size based on the target spacing
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

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

    # Group files by patient folder so we can process CT before GT
    from collections import defaultdict
    patient_files: dict = defaultdict(dict)
    # Differentiate CT and GT
    for path in image_paths:
        key = "gt" if path.name.lower() == "gt_4label_v6.nii.gz" else "ct"
        patient_files[path.parent][key] = path

    for patient_dir, files in patient_files.items():
        # CT resampling should be done first, and then GT resampling should use the resampled CT as a reference.
        resampled_ct = None
        if "ct" in files:
            ct_target = output_dir / files["ct"].relative_to(input_dir)
            resampled_ct = resample_image(files["ct"], ct_target, target_spacing)

        if "gt" in files:
            gt_target = output_dir / files["gt"].relative_to(input_dir)
            ct_geometry = sitk.ReadImage(str(files["ct"])) if "ct" in files else None
            resample_image(
                files["gt"],
                gt_target,
                target_spacing,
                reference_image=resampled_ct,
                geometry_reference=ct_geometry,
            )

    print(f"Resampled {len(image_paths)} image(s) to {output_dir}")



def main() -> None:
    resample_folder()


if __name__ == "__main__":
    main()