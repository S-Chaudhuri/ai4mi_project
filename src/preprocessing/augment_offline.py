#!/usr/bin/env python3

"""Offline augmentation: write augmented copies of a patient as new pseudo-patients.

This produces PNG slices in the same layout as slice_segthor.py, so both
SliceDataset (2D) and BoxDataset (3D) pick them up with no code changes 


This script does the geometry only: affine (skew, rotation, scale, shift) and
elastic, the expensive transforms. Every slice of a pseudo-patient is warped
consistently, so stacking them back into a volume should not tear the anatomy
(I think...) The two are composed into a single sampling grid and resampled once, 
so the labels take only one nearest-neighbour pass.

The cheap per-sample transforms (noise, gamma, brightness, contrast) will be 
done online.

Usage:
    # two augmented variants of Patient_07, plus the un-augmented v6 reference
    uv run python src/preprocessing/augment_offline.py --variants 2

    # pure in-plane extruded warp (identical deformation at every depth)
    uv run python src/preprocessing/augment_offline.py \
        --elastic-alpha 0 12 12 --elastic-sigma inf 16 16

Output (default):
    data/SEGTHOR_aug/train/img/Patient_07v6_0000.png   un-augmented v6 labels
    data/SEGTHOR_aug/train/img/Patient_07a1_0000.png   variant 1
    data/SEGTHOR_aug/train/gt/...                      same stems, class * 63
"""

import gzip
import math
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import autoroot  # noqa     Do not remove, puts the project root on sys.path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from src.utils.dataset import load_volume

# slice_segthor.py stores the class index times 63 so the PNGs are visible
GT_SCALE = 63

CLASS_NAMES: tuple[str, ...] = ("esophagus", "heart", "trachea", "aorta")
CLASS_COLORS: tuple[str, ...] = ("gold", "crimson", "deepskyblue", "limegreen")

# Same call as slice_segthor.py uses, so our GT lands on the identical grid
resize_ = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


@dataclass
class Args:
    patient: str = "Patient_07"
    """Patient to augment, as named in the sliced dataset."""

    subset: str = "train"
    """Which split the patient lives in (train / val)."""

    sliced_root: Path = autoroot.root / "data" / "SEGTHOR"
    """Sliced dataset, read for the CT image PNGs."""

    gt_v6: Path = autoroot.root / "Patient_labels_GTv6.zip"
    """The 4-label ground truths. Either the .zip or an extracted directory."""

    out_root: Path = autoroot.root / "data" / "SEGTHOR_aug"
    """Where the pseudo-patients are written. Left separate from data/SEGTHOR."""

    variants: int = 2
    """Number of augmented copies to generate."""

    write_original: bool = True
    """Also write the un-augmented patient with v6 labels, as a reference."""

    seed: int = 0
    """Variant i uses seed + i, so any copy can be reproduced from its name."""

    num_classes: int = 5
    """Background plus the four organs."""

    # --- affine, one shared in-plane draw per patient -----------------------
    degrees: float = 10.0
    """Rotation about the scanner's long axis, in degrees."""

    translate: float = 0.05
    """Max in-plane shift, as a fraction of the image size."""

    scale: tuple[float, float] = (0.9, 1.1)
    """In-plane zoom range."""

    shear: float = 5.0
    """In-plane skew, in degrees."""

    # --- elastic, per-axis (z, y, x) ---------------------------------------
    elastic_alpha: tuple[float, float, float] = (4.0, 12.0, 12.0)
    """Max displacement in voxels per axis. Set the z entry to 0 for a purely
    in-plane deformation."""

    elastic_sigma: tuple[float, float, float] = (24.0, 16.0, 16.0)
    """Correlation length in voxels per axis: the distance over which the
    deformation changes. Large means the volume bends gently as one, small
    means many local wobbles (and risks folding). `inf` makes the field
    constant along that axis, i.e. the extruded 2D case."""

    preview: int = 4
    """Save a figure with this many z positions per variant (0 to skip)."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_img_volume(sliced_root: Path, subset: str, patient: str) -> np.ndarray:
    """CT image as (D, H, W) uint8, stacked from the sliced PNGs."""
    img_dir = sliced_root / subset / "img"
    paths = sorted(img_dir.glob(f"{patient}_*.png"))
    if not paths:
        raise SystemExit(f"No slices for {patient} in {img_dir}")
    return load_volume(paths)


def _read_v6_bytes(gt_v6: Path, patient: str) -> bytes:
    """The patient's GT_4label_v6.nii.gz, from either the zip or a directory."""
    name = f"{patient}/GT_4label_v6.nii.gz"

    if gt_v6.is_dir():
        path = gt_v6 / name
        if not path.exists():
            raise SystemExit(f"{path} not found")
        return path.read_bytes()

    with zipfile.ZipFile(gt_v6) as archive:
        if name not in archive.namelist():
            raise SystemExit(f"{name} not in {gt_v6}")
        return archive.read(name)


def load_v6_gt_volume(
    gt_v6: Path, patient: str, shape: tuple[int, int], num_classes: int
) -> np.ndarray:
    """v6 labels as (D, H, W) uint8, resliced onto the sliced dataset's grid.

    The v6 NIfTI is (X, Y, Z) at the native 512x512; slice_segthor.py resizes
    each z slice down with order=0, so we repeat exactly that.
    """
    raw = gzip.decompress(_read_v6_bytes(gt_v6, patient))
    volume = np.asarray(nib.Nifti1Image.from_bytes(raw).dataobj)

    labels = set(np.unique(volume).tolist())
    if not labels <= set(range(num_classes)):
        raise SystemExit(f"v6 GT for {patient} has labels {sorted(labels)}")

    # (X, Y, Z) -> (D, H, W), one resize per slice
    return np.stack(
        [
            resize_(volume[:, :, z], shape, order=0).astype(np.uint8)
            for z in range(volume.shape[2])
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# The deformation field
# ---------------------------------------------------------------------------


def elastic_offsets(
    shape: tuple[int, int, int],
    alpha: tuple[float, float, float],
    sigma: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Smooth displacement field as (3, D, H, W) in voxels, ordered (z, y, x).

    White noise per voxel, blurred so neighbouring voxels share nearly the same
    displacement, then scaled so the largest displacement is alpha. An infinite
    sigma collapses the field along that axis to its mean, which is the
    extruded case: the deformation no longer varies along it.
    """
    out = np.zeros((3, *shape), dtype=np.float32)

    for axis_i, (a, s) in enumerate(zip(alpha, sigma)):
        if a == 0:
            continue  # this component is switched off

        component = rng.standard_normal(shape, dtype=np.float32)

        # Collapse the axes with an infinite sigma, keeping the dims so the
        # blur below can ignore them and the result broadcasts back out.
        blur_sigma = []
        for ax, s_ax in enumerate(sigma):
            if math.isinf(s_ax):
                component = component.mean(axis=ax, keepdims=True)
                blur_sigma.append(0.0)
            else:
                blur_sigma.append(s_ax)

        component = gaussian_filter(component, sigma=blur_sigma, mode="nearest")

        # Normalise by the peak so alpha reads directly as voxels
        peak = np.abs(component).max()
        if peak > 0:
            component = component / peak * a

        out[axis_i] = np.broadcast_to(component, shape)

    return out


def jacobian_min(offsets: np.ndarray) -> float:
    """Smallest Jacobian determinant of the deformation.

    Values at or below zero mean the field folds over on itself: tissue passes
    through itself and the labels take on geometry that cannot occur. Comes out
    near 1.0 for a gentle warp.
    """
    if not offsets.any():
        return 1.0

    # d(z,y,x) / d(z,y,x), identity plus the gradient of the displacement
    jac = np.empty((3, 3, *offsets.shape[1:]), dtype=np.float32)
    for comp in range(3):
        for ax, grad in enumerate(np.gradient(offsets[comp])):
            jac[comp, ax] = grad + (1.0 if comp == ax else 0.0)

    # Subsample: the determinant of every voxel is slow and the field is smooth
    jac = jac[:, :, ::2, ::2, ::2]
    det = np.linalg.det(np.moveaxis(jac, (0, 1), (-2, -1)))
    return float(det.min())


def affine_matrix(args: Args, rng: np.random.Generator) -> np.ndarray:
    """One in-plane 2x3 matrix, in normalised [-1, 1] coordinates.

    Applied identically at every depth, so it is exactly a 3D affine whose
    rotation axis is z -- the patient lying at a different angle in the gantry.
    """
    theta = math.radians(rng.uniform(-args.degrees, args.degrees))
    shear = math.radians(rng.uniform(-args.shear, args.shear))
    zoom = rng.uniform(*args.scale)

    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    skew = np.array([[1.0, math.tan(shear)], [0.0, 1.0]])

    # Sampling coordinates, so the inverse zoom: a larger zoom samples a
    # smaller region and the anatomy appears bigger.
    linear = rotation @ skew / zoom

    shift = rng.uniform(-args.translate, args.translate, size=2) * 2.0
    return np.concatenate([linear, shift[:, None]], axis=1).astype(np.float32)


def sampling_grid(
    shape: tuple[int, int, int], offsets: np.ndarray, affine: np.ndarray
) -> torch.Tensor:
    """Combine the elastic field and the affine into one grid for grid_sample.

    grid_sample wants (N, D, H, W, 3) with the last axis ordered (x, y, z),
    i.e. (W, H, D) -- the reverse of the array axes.
    """
    D, H, W = shape

    zs, ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, D),
        torch.linspace(-1.0, 1.0, H),
        torch.linspace(-1.0, 1.0, W),
        indexing="ij",
    )

    # Voxel displacements -> normalised units, then added to the identity grid
    offs = torch.from_numpy(offsets)
    zs = zs + offs[0] * (2.0 / max(D - 1, 1))
    ys = ys + offs[1] * (2.0 / max(H - 1, 1))
    xs = xs + offs[2] * (2.0 / max(W - 1, 1))

    # In-plane affine on top; z is left alone
    a = torch.from_numpy(affine)
    xs, ys = (
        a[0, 0] * xs + a[0, 1] * ys + a[0, 2],
        a[1, 0] * xs + a[1, 1] * ys + a[1, 2],
    )

    return torch.stack([xs, ys, zs], dim=-1)[None]


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------


def warp(
    img: np.ndarray, gt: np.ndarray, grid: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Resample both volumes once: bilinear for the image, nearest for labels.

    Nearest keeps the labels exactly on {0..K-1}, which class2one_hot asserts.
    Sampling outside the volume gives 0, which is air for the image and
    background for the labels.
    """
    img_t = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]

    img_out = F.grid_sample(
        img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    gt_out = F.grid_sample(
        gt_t, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )

    return img_out[0, 0].numpy(), gt_out[0, 0].numpy().round().astype(np.uint8)


def write_patient(
    out_root: Path, subset: str, stem: str, img: np.ndarray, gt: np.ndarray
) -> None:
    """Write one pseudo-patient as PNG slices, named like slice_segthor.py.

    The stem must keep `Patient_<something>` as its first two underscore
    fields, because make_3d_dataset groups volumes on exactly that, and the
    slice index must stay four digits so the slices sort into depth order.
    """
    img_dir = out_root / subset / "img"
    gt_dir = out_root / subset / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    img_u8 = np.clip(img * 255.0, 0, 255).round().astype(np.uint8)
    gt_u8 = (gt * GT_SCALE).astype(np.uint8)

    for idz in range(img_u8.shape[0]):
        name = f"{stem}_{idz:04d}.png"
        Image.fromarray(img_u8[idz]).save(img_dir / name)
        Image.fromarray(gt_u8[idz]).save(gt_dir / name)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(gt_before: np.ndarray, gt_after: np.ndarray, num_classes: int) -> None:
    """Per-class voxel counts, to catch organs warped out of the volume."""
    print(f"    {'class':<12} {'before':>9} {'after':>9}   retained")
    for k in range(1, num_classes):
        before = int((gt_before == k).sum())
        after = int((gt_after == k).sum())
        name = CLASS_NAMES[k - 1] if k - 1 < len(CLASS_NAMES) else f"class {k}"
        pct = f"{after / before:6.1%}" if before else "     --"
        flag = "  <-- lost" if before and after / before < 0.9 else ""
        print(f"    {name:<12} {before:9d} {after:9d}   {pct}{flag}")


def save_preview(
    out_root: Path,
    stem: str,
    img_before: np.ndarray,
    gt_before: np.ndarray,
    img_after: np.ndarray,
    gt_after: np.ndarray,
    n_slices: int,
) -> None:
    """Original next to augmented at several depths.

    Spread over the labelled range, so a warp that varies along z the way it
    should -- or tears, the way it should not -- is visible down the column.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    labelled = np.where((gt_before > 0).any(axis=(1, 2)))[0]
    if len(labelled) == 0:
        print("    (no labelled slices, skipping preview)")
        return
    zs = np.linspace(labelled[0], labelled[-1], n_slices).round().astype(int)

    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(len(zs), 2, figsize=(6, 3.1 * len(zs)), squeeze=False)

    for row, z in enumerate(zs):
        for col, (image, labels, title) in enumerate(
            [(img_before, gt_before, "original"), (img_after, gt_after, "augmented")]
        ):
            ax = axes[row, col]
            ax.imshow(image[z], cmap="gray", vmin=0, vmax=1, interpolation="none")
            ax.imshow(
                np.ma.masked_equal(labels[z], 0),
                cmap=cmap,
                vmin=0.5,
                vmax=len(CLASS_NAMES) + 0.5,
                alpha=0.45,
                interpolation="none",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
        axes[row, 0].set_ylabel(f"z = {z}")

    fig.suptitle(stem)
    fig.legend(
        handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)],
        loc="lower center",
        ncol=len(CLASS_NAMES),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))

    out_dir = out_root / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    preview -> {out}")


# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    img = load_img_volume(args.sliced_root, args.subset, args.patient)
    gt = load_v6_gt_volume(
        args.gt_v6,
        args.patient,
        img.shape[1:],
        args.num_classes,  # type: ignore[arg-type]
    )

    if gt.shape != img.shape:
        raise SystemExit(f"image is {img.shape} but v6 GT reslices to {gt.shape}")

    print(f">> {args.patient}: {img.shape} (D, H, W)")
    print(f"   v6 labels present: {sorted(np.unique(gt).tolist())}")
    print(f"   writing to {args.out_root / args.subset}")

    if args.write_original:
        stem = f"{args.patient}v6"
        print(f"\n>> {stem} (un-augmented, v6 labels)")
        write_patient(
            args.out_root, args.subset, stem, img.astype(np.float32) / 255.0, gt
        )
        print(f"    wrote {img.shape[0]} slices")

    for i in range(1, args.variants + 1):
        stem = f"{args.patient}a{i}"
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        print(f"\n>> {stem} (seed {seed})")

        offsets = elastic_offsets(
            img.shape,
            args.elastic_alpha,
            args.elastic_sigma,
            rng,  # type: ignore[arg-type]
        )
        affine = affine_matrix(args, rng)
        grid = sampling_grid(img.shape, offsets, affine)  # type: ignore[arg-type]

        det = jacobian_min(offsets)
        note = "  <-- field folds, lower alpha or raise sigma" if det <= 0 else ""
        print(f"    min Jacobian determinant: {det:.3f}{note}")

        aug_img, aug_gt = warp(img, gt, grid)

        labels = set(np.unique(aug_gt).tolist())
        if not labels <= set(range(args.num_classes)):
            raise SystemExit(
                f"augmented labels escaped the class set: {sorted(labels)}"
            )

        report(gt, aug_gt, args.num_classes)
        write_patient(args.out_root, args.subset, stem, aug_img, aug_gt)
        print(f"    wrote {aug_img.shape[0]} slices")

        if args.preview:
            save_preview(
                args.out_root,
                stem,
                img.astype(np.float32) / 255.0,
                gt,
                aug_img,
                aug_gt,
                args.preview,
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
