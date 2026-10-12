"""Compare postprocessing variants on the validation set for ONE model.

Inference runs once per patient and the raw prediction is cached on disk. Every
variant is then applied to that same prediction (in native space, so volumes in
mm^3 use the real voxel spacing) and scored with evaluate_patient.

Use the same Config as eval_3D, with a separate results_dir per model (the raw
prediction cache lives in results_dir/eval/raw_pred). Call it where run_eval is
called:

    from src.eval_postprocess import run_sweep
    run_sweep(config)

Outputs, in results_dir/eval/:
    postprocess_sweep_per_patient.csv   one row per (patient, variant)
    postprocess_sweep_summary.csv       mean over patients, one row per variant

To add a variant, add one line to VARIANTS.
"""

import autoroot  # noqa: F401
import numpy as np
import pandas as pd
import torch

from src.eval_3D import predict_volume
from src.postprocess import keep_largest_component, remove_small_esophagus_components
from src.train import get_model
from src.utils.config import Config
from src.utils.dataset import BoxDataset
from src.utils.evaluation import evaluate_patient, load_native_gt, to_native_space


# A variant is a function (label map, spacing) -> label map.
def esoph(min_mm3):
    return lambda p, sp: remove_small_esophagus_components(
        p, min_volume_mm3=min_mm3, spacing=sp
    )


def chain(*fns):
    def run(p, sp):
        for f in fns:
            p = f(p, sp)
        return p

    return run


lcc = lambda p, sp: keep_largest_component(p)  # noqa: E731
lcc26 = lambda p, sp: keep_largest_component(p, connectivity=26)  # noqa: E731

VARIANTS = {
    "none": lambda p, sp: p,
    "lcc": lcc,
    "lcc26": lcc26,
    "esoph200": esoph(200),
    "esoph500": esoph(500),
    "esoph1000": esoph(1000),
    "lcc+esoph200": chain(lcc, esoph(200)),
    "lcc+esoph500": chain(lcc, esoph(500)),
    "lcc+esoph1000": chain(lcc, esoph(1000)),
}


def flatten(res, prefix=""):
    """Turn whatever evaluate_patient returns (dicts, arrays, scalars) into
    {"name/sub": float}. Non-numeric entries are skipped."""
    if isinstance(res, dict):
        out = {}
        for k, v in res.items():
            out.update(flatten(v, f"{prefix}{k}/"))
        return out
    try:
        arr = np.asarray(res, dtype=float)
    except (TypeError, ValueError):
        return {}
    key = prefix.rstrip("/")
    if arr.ndim == 0:
        return {key: float(arr)}
    return {f"{key}/{i}": float(v) for i, v in enumerate(arr.ravel())}


def run_sweep(config: Config, variants: dict | None = None):
    variants = variants or VARIANTS
    if not config.model.is_3d:
        raise ValueError("run_sweep requires a 3D model: set --model.is-3d")
    if config.eval.weights is None:
        raise ValueError("pass --evaluation.weights PATH")
    if config.paths.results_dir is None:
        raise ValueError("paths.results_dir is required")

    device = torch.device(
        "cuda" if config.training.gpu and torch.cuda.is_available() else "cpu"
    )
    K = config.dataset.num_classes
    net = get_model(config)
    net.load_state_dict(
        torch.load(config.eval.weights, map_location=device, weights_only=True)
    )
    net.to(device).eval()

    out_dir = config.paths.results_dir / "eval"
    cache = out_dir / "raw_pred"
    cache.mkdir(parents=True, exist_ok=True)
    data_root = config.paths.data_path / config.dataset.name
    ds = BoxDataset(config.eval.split, data_root, sub_box_size=None, num_classes=K)
    raw_split = "test" if config.eval.split == "test" else "train"

    rows = []
    for item in ds.items:
        stem = item["stem"]
        pid = config.eval.patient_id
        if pid is not None and stem != f"Patient_{pid:02d}":
            continue

        f = cache / f"{stem}.npy"
        if f.exists():
            raw = np.load(f)
        else:
            img = torch.from_numpy(item["img_vol"]).float().unsqueeze(0) / 255.0
            probs = predict_volume(
                net,
                img,
                config.dataset.box_size,
                config.eval.overlap,
                K,
                config.training.batch_size,
                device,
                config.model.temperature,
            )
            raw = probs.max(dim=0).indices.cpu().numpy().astype(np.uint8)
            np.save(f, raw)

        gt_xyz, spacing, _ = load_native_gt(data_root, stem, split_dir=raw_split)
        pred_xyz = to_native_space(raw, gt_xyz.shape)

        for name, fn in variants.items():
            res = evaluate_patient(fn(pred_xyz, spacing), gt_xyz, spacing)
            rows.append({"patient": stem, "variant": name, **flatten(res)})
        print(f">> {stem}: {len(variants)} variants scored")

    if not rows:
        print(">> Warning: no patients evaluated; check --eval.split / --eval.patient-id")
        return None

    df = pd.DataFrame(rows)
    summary = (
        df.drop(columns="patient")
        .groupby("variant", sort=False)
        .mean(numeric_only=True)
    )
    df.to_csv(out_dir / "postprocess_sweep_per_patient.csv", index=False)
    summary.to_csv(out_dir / "postprocess_sweep_summary.csv")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.round(4))
    print(f">> Wrote {out_dir / 'postprocess_sweep_summary.csv'}")
    return df, summary

