"""Online intensity augmentation, applied to training images as they are loaded.

Geometry (affine, elastic) is done offline by src/preprocessing/augment_offline.py. 
The transforms here are cheap an pointwise so won't ever create tearing in a patient.

How coherent each transform is matches the physical thing it imitates:

- gamma and contrast stand in for acquisition and reconstruction settings, which
  are a property of a whole scan. One draw per sample, applied to every voxel of
  it equally. A different gamma per slice would put an intensity step along z
  that doesnt make any sense. And the 3D kernels would see a false gradient.
- noise stands in for quantum and electronic noise in the detector, which really
  is independent per voxel, so it is drawn per voxel.

Only the image is touched; labels are left alone (because they should stay the same 
on the augmented image) , which is why this can hook into
img_transform without the datasets needing to know about it.
"""

import torch
from torch import Tensor

from src.utils.config import AugmentationConfig

# Contrast pivots around this fixed value rather than the image mean, so the
# result does not depend on how much of the volume is visible when it runs.
# BoxDataset transforms the whole patient and crops after, GridBoxDataset crops
# first. A mean-based contrast would disagree between the two.
CONTRAST_PIVOT = 0.5


def _draw(low: float, high: float) -> float:
    """One sample from U(low, high), from torch's RNG so workers stay seeded."""
    return torch.empty(1).uniform_(low, high).item()


def random_gamma(img: Tensor, low: float, high: float) -> Tensor:
    """Non-linear intensity remap. One exponent for the whole sample."""
    return img.clamp(min=0.0) ** _draw(low, high)


def random_contrast(img: Tensor, low: float, high: float) -> Tensor:
    """Stretch intensities around CONTRAST_PIVOT. One factor for the whole sample."""
    return (img - CONTRAST_PIVOT) * _draw(low, high) + CONTRAST_PIVOT


def add_noise(img: Tensor, sigma: float) -> Tensor:
    """Gaussian noise, drawn independently per voxel."""
    return img + torch.randn_like(img) * sigma


def augment_intensity(img: Tensor, config: AugmentationConfig) -> Tensor:
    """Apply the enabled transforms to one sample, each with its own probability.

    Gamma and contrast go first and noise last, so the noise is not reshaped by
    the pointwise maps that follow it. The result is clamped back into [0, 1],
    which the training loop asserts.
    """
    if torch.rand(1).item() < config.gamma_prob:
        img = random_gamma(img, *config.gamma_range)

    if torch.rand(1).item() < config.contrast_prob:
        img = random_contrast(img, *config.contrast_range)

    if torch.rand(1).item() < config.noise_prob:
        img = add_noise(img, config.noise_level)

    return img.clamp(0.0, 1.0)


def make_img_transform(base_transform, config: AugmentationConfig):
    """Wrap an img_transform so its output is augmented. Returns it unchanged
    when augmentation is off, so the non-augmented path stays exactly as it was.

    Pass the result as the training set's img_transform only -- the validation
    set keeps the plain one.
    """
    if not config.enabled:
        return base_transform

    def transform(x):
        return augment_intensity(base_transform(x), config)

    print(
        f">> Online augmentation on: "
        f"gamma p={config.gamma_prob} {config.gamma_range}, "
        f"contrast p={config.contrast_prob} {config.contrast_range}, "
        f"noise p={config.noise_prob} sigma={config.noise_level}"
    )
    return transform
