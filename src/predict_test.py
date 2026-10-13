#!/usr/bin/env python3
"""Predict unlabelled CT scans (the SegTHOR test set) with a trained 3D model.

For every Patient_XX.nii.gz CT: the same preprocessing as the .npy training
data (resample to the working spacing, crop to the body, HU window(s)),
sliding-window inference, postprocessing, then back to the CT's own grid and
written as Patient_XX.nii.gz (uint8 labels 0-4, the CT's affine).

    python src/predict_test.py --weights results/<run>/bestweights.pt \
        --model ResUNet3D --input data/test --dest group-XX/test/pred

--input is searched recursively for Patient_XX.nii.gz (GT files are skipped).
The model options (--model, --kernels, --in_channels, --deep_supervision) must
match the checkpoint; --windows must match the training data (soft for
SEGTHOR_3D_1ch, soft lung blood for SEGTHOR_3D_3ch).
"""

import argparse
import re
import time
from pathlib import Path

import autoroot  # noqa
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

from src.eval_3D import predict_volume
from src.models.MedNeXt3D import MedNeXt3D
from src.models.ResUNet3D import ResUNet3D
from src.models.UNet3D import UNet3D
from src.models.VNet3D import VNet3D
from src.postprocess import POSTPROCESSING, postprocess
from src.preprocessing.hu_window_to_3channel import DEFAULT_WINDOWS, apply_window

MODELS = {"UNet3D": UNet3D, "VNet3D": VNet3D, "ResUNet3D": ResUNet3D, "MedNeXt3D": MedNeXt3D}
WINDOWS = dict(zip(["soft", "lung", "blood"], DEFAULT_WINDOWS))
NUM_CLASSES = 5
BODY_HU_THRESHOLD = -500


def resample(image: sitk.Image, spacing) -> sitk.Image:
    size = [int(round(n * s / t)) for n, s, t in zip(image.GetSize(), image.GetSpacing(), spacing)]
    r = sitk.ResampleImageFilter()
    r.SetSize(size)
    r.SetOutputSpacing(spacing)
    r.SetOutputDirection(image.GetDirection())
    r.SetOutputOrigin(image.GetOrigin())
    r.SetInterpolator(sitk.sitkLinear)
    r.SetDefaultPixelValue(-1000)
    return r.Execute(image)


def body_crop(ct: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    labels, n = ndimage.label(ct > BODY_HU_THRESHOLD)
    if n == 0:
        return 0, ct.shape[1], 0, ct.shape[2]
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, range(1, n + 1))
    body = labels == int(np.argmax(sizes)) + 1
    ys = np.where(body.any(axis=(0, 2)))[0]
    xs = np.where(body.any(axis=(0, 1)))[0]
    return (max(int(ys[0]) - margin, 0), min(int(ys[-1]) + 1 + margin, ct.shape[1]),
            max(int(xs[0]) - margin, 0), min(int(xs[-1]) + 1 + margin, ct.shape[2]))


def preprocess(ct_img: sitk.Image, spacing, windows, margin):
    """(C, D, H, W) float in [0, 1] and the meta dict to_native_space needs."""
    ct_res = resample(ct_img, spacing)
    ct = sitk.GetArrayFromImage(ct_res)  # (z, y, x)
    y0, y1, x0, x1 = body_crop(ct, margin)
    ct = ct[:, y0:y1, x0:x1].astype(np.float32)
    chans = [np.round(apply_window(ct, *WINDOWS[w]) * 255).astype(np.uint8) for w in windows]
    img = torch.from_numpy(np.stack(chans).astype(np.float32) / 255.0)
    meta = {
        "crop_yx": [y0, y1, x0, x1],
        "resampled_size_zyx": list(ct_res.GetSize()[::-1]),
        "spacing_zyx": [spacing[2], spacing[1], spacing[0]],
        "original_spacing_xyz": list(ct_img.GetSpacing()),
        "original_size_xyz": list(ct_img.GetSize()),
    }
    return img, meta


def to_native_space(pred_zyx: np.ndarray, meta: dict) -> np.ndarray:
    """Cropped (D, h, w) prediction -> (X, Y, Z) on the CT's own grid: undo the
    crop, then nearest-neighbour from the working spacing to the original one.
    Kept here (not imported) so this script works with any evaluation.py version."""
    y0, _, x0, _ = meta["crop_yx"]
    Dz, Hs, Ws = meta["resampled_size_zyx"]
    sz, sy, sx = meta["spacing_zyx"]
    ox, oy, oz = meta["original_spacing_xyz"]
    X, Y, Z = meta["original_size_xyz"]
    D, hc, wc = pred_zyx.shape
    full = np.zeros((Dz, Hs, Ws), np.uint8)
    full[:D, y0:y0 + hc, x0:x0 + wc] = pred_zyx
    iz = np.clip(np.rint(np.arange(Z) * oz / sz).astype(int), 0, Dz - 1)
    iy = np.clip(np.rint(np.arange(Y) * oy / sy).astype(int), 0, Hs - 1)
    ix = np.clip(np.rint(np.arange(X) * ox / sx).astype(int), 0, Ws - 1)
    return full[iz][:, iy][:, :, ix].transpose(2, 1, 0)


def find_cts(root: Path) -> list[Path]:
    cts = [p for p in root.rglob("Patient_*.nii*") if re.fullmatch(r"Patient_\d+\.nii(\.gz)?", p.name)]
    return sorted(cts, key=lambda p: p.name)


def main(args: argparse.Namespace) -> None:
    unknown = [s for s in args.postprocess if s not in POSTPROCESSING]
    if unknown:
        raise SystemExit(f"Unknown --postprocess {unknown}, available: {list(POSTPROCESSING)}")
    if len(args.windows) != args.in_channels:
        raise SystemExit(f"--windows {args.windows} gives {len(args.windows)} channel(s) "
                         f"but --in_channels is {args.in_channels}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net = MODELS[args.model](args.in_channels, NUM_CLASSES, kernels=args.kernels,
                             deep_supervision=args.deep_supervision)
    net.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    net.to(device).eval()

    cts = find_cts(Path(args.input))
    if args.patients:
        cts = [p for p in cts if p.name.split(".")[0] in set(args.patients)]
    if not cts:
        raise SystemExit(f"No Patient_XX.nii.gz CT scans found under {args.input}")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f">> {args.model} on {len(cts)} scan(s) ({device}), windows {args.windows}, "
          f"postprocess {args.postprocess} -> {dest}")

    for ct_path in cts:
        t0 = time.time()
        pid = ct_path.name.split(".")[0]
        img, meta = preprocess(sitk.ReadImage(str(ct_path)), tuple(args.spacing), args.windows, args.margin)
        probs = predict_volume(net, img, tuple(args.patch), args.overlap, NUM_CLASSES, args.batch_size, device)
        pred = probs.argmax(0).cpu().numpy().astype(np.uint8)
        pred = postprocess(pred, args.postprocess)
        pred_xyz = to_native_space(pred, meta)

        ct_nib = nib.load(str(ct_path))
        if pred_xyz.shape != ct_nib.shape[:3]:
            raise SystemExit(f"{pid}: prediction {pred_xyz.shape} != CT {ct_nib.shape}")
        nib.save(nib.Nifti1Image(pred_xyz.astype(np.uint8), ct_nib.affine), str(dest / f"{pid}.nii.gz"))
        counts = {k: int((pred_xyz == k).sum()) for k in range(1, NUM_CLASSES)}
        print(f"  {pid}: {pred_xyz.shape} voxels per organ {counts} ({time.time() - t0:.0f}s)")
    print(f">> wrote {len(cts)} prediction(s) to {dest}")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict unlabelled CT scans with a trained 3D model")
    p.add_argument("--weights", required=True, help="bestweights.pt of the run")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--kernels", type=int, default=16)
    p.add_argument("--in_channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--deep_supervision", action="store_true", help="if the run was trained with it")
    p.add_argument("--windows", nargs="+", default=["soft"], choices=list(WINDOWS),
                   help="soft (1 channel) or soft lung blood (3 channels), as in training")
    p.add_argument("--input", required=True, help="folder with Patient_XX.nii.gz CT scans (searched recursively)")
    p.add_argument("--dest", required=True, help="output folder for Patient_XX.nii.gz predictions")
    p.add_argument("--patients", nargs="+", help="only these, e.g. Patient_41 Patient_42")
    p.add_argument("--postprocess", nargs="*", default=["lcc"], help="steps from src/postprocess.py")
    p.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 2.5], metavar=("X", "Y", "Z"))
    p.add_argument("--margin", type=int, default=4)
    p.add_argument("--patch", type=int, nargs=3, default=[132, 132, 128])
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())
