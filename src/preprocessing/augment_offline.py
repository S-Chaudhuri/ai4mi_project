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

Both image and labels are read from the sliced dataset, so the train/val split
that slice_segthor.py decided is respected: only the subset asked for is
augmented, and a validation patient can never leak into training.

Usage:
    # two variants of every patient in data/SEGTHOR/train
    uv run python src/preprocessing/augment_offline.py

    # one patient only, with preview figures to check it by eye
    uv run python src/preprocessing/augment_offline.py \
        --patient Patient_07 --preview 4

    # pure in-plane extruded warp (identical deformation at every depth)
    uv run python src/preprocessing/augment_offline.py \
        --elastic-alpha 0 12 12 --elastic-sigma inf 16 16

Output (default):
    data/SEGTHOR_aug/train/img/Patient_01a1_0000.png   variant 1 of Patient_01
    data/SEGTHOR_aug/train/img/Patient_01a2_0000.png   variant 2
    data/SEGTHOR_aug/train/gt/...                      same stems, class * 63
"""

import math
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

import autoroot  # noqa     Do not remove, puts the project root on sys.path
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.utils.dataset import load_volume

# slice_segthor.py stores the class index times 63 so the PNGs are visible
GT_SCALE = 63

CLASS_NAMES: tuple[str, ...] = ("esophagus", "heart", "trachea", "aorta")
CLASS_COLORS: tuple[str, ...] = ("gold", "crimson", "deepskyblue", "limegreen")


@dataclass
class Args:
    patient: str = ""
    """One patient to augment, e.g. Patient_07. Empty means every patient in
    the subset."""

    subset: str = "train"
    """Which split to augment. Augmenting val would leak it into training."""

    sliced_root: Path = autoroot.root / "data" / "SEGTHOR"
    """Sliced dataset, read for both the image and the label PNGs."""

    out_root: Path = autoroot.root / "data" / "SEGTHOR_aug"
    """Where the pseudo-patients are written. Left separate from data/SEGTHOR."""

    variants: int = 2
    """Number of augmented copies to generate per patient."""

    write_original: bool = False
    """Also copy each patient through un-augmented. Off: the originals are
    already in sliced_root, so this would only duplicate them."""

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
    # These defaults were picked by sweeping alpha/sigma over 15 patients x 2
    # seeds and counting folds: (12,16) folded 3 times in 30, (8,16) none but
    # with a worst Jacobian of 0.09, (6,20) none with 0.40 to spare.
    elastic_alpha: tuple[float, float, float] = (2.0, 6.0, 6.0)
    """Max displacement in voxels per axis. Set the z entry to 0 for a purely
    in-plane deformation. Raising this without raising sigma is what makes the
    deformation fold -- it is the in-plane alpha/sigma ratio that matters, and
    0.3 is comfortable while 0.75 folds."""

    elastic_sigma: tuple[float, float, float] = (24.0, 20.0, 20.0)
    """Correlation length in voxels per axis: the distance over which the
    deformation changes. Large means the volume bends gently as one, small
    means many local wobbles (and risks folding). `inf` makes the field
    constant along that axis, i.e. the extruded 2D case -- which is no
    protection against folding on its own, the in-plane ratio still rules."""

    preview: int = 0
    """Save a figure with this many z positions per variant. 0 (the default)
    writes no figures, just the augmented slices."""

    copy_subsets: tuple[str, ...] = ("val",)
    """Subsets to copy through into out_root unchanged, so the output is a
    complete dataset on its own. BoxDataset needs a val/ folder to start,
    and val must stay real (un-augmented) images."""

    process: int = 1
    """Number of processes for the patient warps. 1 (the default) runs them
    sequentially; -1 uses all cores. The output is identical either way:
    each patient is seeded on its own name."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def discover_patients(sliced_root: Path, subset: str) -> list[str]:
    """Every patient in the subset, from the slice filenames.

    Same grouping make_3d_dataset uses, so what we augment is exactly what the
    3D pipeline will later read back as one volume.
    """
    img_dir = sliced_root / subset / "img"
    if not img_dir.is_dir():
        raise SystemExit(f"{img_dir} does not exist -- has the data been sliced?")

    patients = sorted({"_".join(p.stem.split("_")[:2]) for p in img_dir.glob("*.png")})
    if not patients:
        raise SystemExit(f"No slices found in {img_dir}")
    return patients


def load_img_volume(sliced_root: Path, subset: str, patient: str) -> np.ndarray:
    """CT image as (D, H, W) uint8, stacked from the sliced PNGs."""
    img_dir = sliced_root / subset / "img"
    paths = sorted(img_dir.glob(f"{patient}_*.png"))
    if not paths:
        raise SystemExit(f"No slices for {patient} in {img_dir}")
    return load_volume(paths)


def load_gt_volume(
    sliced_root: Path, subset: str, patient: str, num_classes: int
) -> np.ndarray:
    """Labels as (D, H, W) uint8 class indices, from the sliced gt PNGs.

    They are stored as the class index times 255/(K-1), so divide back out and
    round, exactly as gt_transform_* in train.py does.
    """
    gt_dir = sliced_root / subset / "gt"
    paths = sorted(gt_dir.glob(f"{patient}_*.png"))
    if not paths:
        raise SystemExit(f"No labels for {patient} in {gt_dir}")

    raw = load_volume(paths).astype(np.float32)
    labels = np.round(raw / (255.0 / (num_classes - 1))).astype(np.uint8)

    present = set(np.unique(labels).tolist())
    if not present <= set(range(num_classes)):
        raise SystemExit(
            f"{patient}: labels {sorted(present)} outside 0..{num_classes - 1}"
        )
    return labels


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


def retention(
    gt_before: np.ndarray, gt_after: np.ndarray, num_classes: int
) -> tuple[float, str]:
    """Worst per-class voxel retention, to catch organs warped out of frame.

    Returns the ratio and the name of the class that fared worst. Well under
    1.0 means the warp pushed part of an organ outside the volume; a little
    over is just the zoom draw.
    """
    worst, worst_name = float("inf"), "-"
    for k in range(1, num_classes):
        before = int((gt_before == k).sum())
        if not before:
            continue
        ratio = int((gt_after == k).sum()) / before
        if ratio < worst:
            worst = ratio
            worst_name = CLASS_NAMES[k - 1] if k - 1 < len(CLASS_NAMES) else f"c{k}"
    return (worst, worst_name) if worst != float("inf") else (1.0, "-")


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


def augment_patient(args: Args, patient: str) -> list[str]:
    """Write every variant of one patient. Returns any warnings raised."""
    img = load_img_volume(args.sliced_root, args.subset, patient)
    gt = load_gt_volume(args.sliced_root, args.subset, patient, args.num_classes)

    if gt.shape != img.shape:
        raise SystemExit(f"{patient}: image is {img.shape} but labels are {gt.shape}")

    warnings: list[str] = []

    if args.write_original:
        write_patient(
            args.out_root, args.subset, patient, img.astype(np.float32) / 255.0, gt
        )

    for i in range(1, args.variants + 1):
        # Seeded on the patient too, so one patient can be regenerated alone
        # and still come out identical to its copy from a whole-dataset run.
        # crc32, not hash(), which python randomises per process.
        seed = args.seed + i
        rng = np.random.default_rng([zlib.crc32(patient.encode()), seed])
        stem = f"{patient}a{i}"

        offsets = elastic_offsets(
            img.shape,
            args.elastic_alpha,
            args.elastic_sigma,
            rng,  # type: ignore[arg-type]
        )
        affine = affine_matrix(args, rng)
        grid = sampling_grid(img.shape, offsets, affine)  # type: ignore[arg-type]

        det = jacobian_min(offsets)
        aug_img, aug_gt = warp(img, gt, grid)

        labels = set(np.unique(aug_gt).tolist())
        if not labels <= set(range(args.num_classes)):
            raise SystemExit(f"{stem}: labels escaped the class set: {sorted(labels)}")

        kept, worst_class = retention(gt, aug_gt, args.num_classes)
        write_patient(args.out_root, args.subset, stem, aug_img, aug_gt)

        flag = ""
        if det <= 0:
            flag = "  FOLDS"
            warnings.append(f"{stem}: field folds (jac {det:.3f})")
        elif kept < 0.9:
            flag = f"  thin {worst_class}"
            warnings.append(f"{stem}: {worst_class} kept only {kept:.0%}")

        print(
            f"  {stem:<16} {aug_img.shape[0]:4d} slices"
            f"   jac {det:5.3f}   worst class kept {kept:5.0%}{flag}"
        )

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

    return warnings


def main(args: Args) -> None:
    patients = (
        [args.patient]
        if args.patient
        else discover_patients(args.sliced_root, args.subset)
    )

    print(f">> {len(patients)} patient(s) in {args.sliced_root / args.subset}")
    print(f"   {args.variants} variant(s) each -> {args.out_root / args.subset}\n")

    if args.process == 1:
        warnings: list[str] = []
        for patient in patients:
            warnings += augment_patient(args, patient)
    else:
        from multiprocessing import Pool, cpu_count

        n = cpu_count() if args.process == -1 else args.process
        with Pool(n) as pool:
            per_patient = pool.starmap(augment_patient, [(args, p) for p in patients])
        warnings = [line for sub in per_patient for line in sub]

    written = len(patients) * (args.variants + (1 if args.write_original else 0))
    print(f"\n>> wrote {written} pseudo-patients")

    for subset in args.copy_subsets:
        src = args.sliced_root / subset
        if not src.is_dir():
            print(f">> no {subset}/ folder in {args.sliced_root}, skipping the copy")
            continue
        dst = args.out_root / subset
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f">> copied {subset} through -> {dst}")

    if warnings:
        print(f">> {len(warnings)} warning(s):")
        for line in warnings:
            print(f"     {line}")


if __name__ == "__main__":
    main(tyro.cli(Args))
