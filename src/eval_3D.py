import autoroot
import numpy as np
import torch
import torch.nn.functional as F

from src.train import get_model
from src.utils.config import Config
from src.utils.dataset import BoxDataset, window_starts
from src.utils.evaluation import (
    evaluate_patient,
    load_native_gt,
    to_native_space,
    write_csv,
    write_submission,
)


def gaussian_window(size, sigma_scale=1 / 8):
    ws = []
    for s in size:
        x = torch.arange(s, dtype=torch.float32) - (s - 1) / 2
        ws.append(torch.exp(-0.5 * (x / (s * sigma_scale)) ** 2))
    w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return (w / w.max()).clamp_min(1e-3)  # centre-weighted, never 0


@torch.no_grad()
def predict_volume(net, img, box, overlap, K, batch_size, device, temperature=1.0):
    """img: (C, D, H, W) float [0,1]. Returns (K, D, H, W) stitched probabilities."""
    C, D, H, W = img.shape
    d, h, w = box
    img_p = F.pad(
        img, (0, max(0, w - W), 0, max(0, h - H), 0, max(0, d - D))
    )  # same as training pad
    _, Dp, Hp, Wp = img_p.shape
    step = [max(1, int(s * (1 - overlap))) for s in box]
    starts = [
        (a, b, c)
        for a in window_starts(Dp, d, step[0])
        for b in window_starts(Hp, h, step[1])
        for c in window_starts(Wp, w, step[2])
    ]
    win = gaussian_window(box).to(device)
    prob_sum = torch.zeros((K, Dp, Hp, Wp), device=device)
    w_sum = torch.zeros((Dp, Hp, Wp), device=device)

    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        xs = [img_p[:, a : a + d, b : b + h, c : c + w] for a, b, c in chunk]
        probs = F.softmax(temperature * net(torch.stack(xs).to(device)).float(), dim=1)
        for p, (a, b, c) in zip(probs, chunk):
            prob_sum[:, a : a + d, b : b + h, c : c + w] += p * win
            w_sum[a : a + d, b : b + h, c : c + w] += win
    return (prob_sum / w_sum)[:, :D, :H, :W]


def run_eval(config: Config):
    if not config.model.is_3d:
        raise ValueError("eval_3D requires a 3D model: set --model.is-3d")

    if config.eval.weights is None:
        raise ValueError(
            "evaluation.weights is required: pass --evaluation.weights PATH"
        )

    device = torch.device(
        "cuda" if config.training.gpu and torch.cuda.is_available() else "cpu"
    )
    K = config.dataset.num_classes
    net = get_model(config)
    net.load_state_dict(
        torch.load(config.eval.weights, map_location=device, weights_only=True)
    )
    net.to(device).eval()

    if config.paths.results_dir is None:
        raise
    eval_results_dir = config.paths.results_dir / "eval"
    data_root_dir = config.paths.data_path / config.dataset.name

    ds = BoxDataset(
        config.eval.split, data_root_dir, sub_box_size=None, num_classes=K
    )  # whole volumes, no transforms

    # Raw NIfTIs of both the train and val splits live under segthor_part1/train
    # (slice_segthor.py only reads from train/ and test/), so map the split.
    raw_split = "test" if config.eval.split == "test" else "train"

    results = {}
    for item in ds.items:
        patient_id = config.eval.patient_id
        if patient_id is not None and item["stem"] != f"Patient_{patient_id:02d}":
            continue
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
        pred = probs.max(dim=0).indices.cpu().numpy().astype(np.uint8)  # (D, H, W)

        gt_xyz, spacing, affine = load_native_gt(
            config.paths.data_path, item["stem"], split_dir=raw_split
        )
        pred_xyz = to_native_space(pred, gt_xyz.shape)  # back to native grid
        results[item["stem"]] = evaluate_patient(pred_xyz, gt_xyz, spacing)
        print(item["stem"], results[item["stem"]])

        if config.eval.submission:
            write_submission(
                pred_xyz, affine, item["stem"], eval_results_dir / "submission"
            )

    if not results:
        print(
            ">> Warning: no patients evaluated; "
            "check --evaluation.split and --evaluation.patient-id"
        )

    write_csv(results, eval_results_dir / "metrics.csv")
    print(f">> Wrote metrics to {eval_results_dir / 'metrics.csv'}")
