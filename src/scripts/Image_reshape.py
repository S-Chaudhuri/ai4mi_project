import SimpleITK as sitk
from pathlib import Path
import sys

target_spacing = (0.98, 0.98, 2.5)

def resample_image(src_dir: Path, tar_dir: Path, target_spacing = target_spacing):
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