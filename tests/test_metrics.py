import autoroot  # noqa

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from src.utils.utils import (
    dice_coef,
    ahd_coef,
    hausdorff_coef,
    hd95_coef,
    nsd_coef,
    biou_coef,
    class2one_hot,
    hd95_batch_maps,
)


def make_disk(shape, center, radius):
    yy, xx = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij"
    )
    return ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2).long()


# --- Test 1: identical circles -> should get DSC=1, all distances=0, NSD=1, BIoU=1
gt = make_disk((64, 64), (32, 32), 15)
pred_perfect = gt.clone()

# --- Test 2: circle shifted by 3 pixels -> partial overlap, small distances
pred_shifted = make_disk((64, 64), (35, 32), 15)

# --- Test 3: circle with wrong radius (smaller) -> DSC drops more than NSD would with tolerance
pred_small = make_disk((64, 64), (32, 32), 10)

K = 2  # background + 1 foreground class
for name, pred in [
    ("perfect", pred_perfect),
    ("shifted_3px", pred_shifted),
    ("smaller_radius", pred_small),
]:
    label_oh = class2one_hot(gt.unsqueeze(0), K)  # (1, K, H, W)
    pred_oh = class2one_hot(pred.unsqueeze(0), K)

    label_b = label_oh.bool()
    pred_b = pred_oh.bool()

    print(f"\n--- {name} ---")
    print("DSC:  ", dice_coef(label_oh, pred_oh)[0, 1].item())
    print("AHD:  ", ahd_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("HD:   ", hausdorff_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("HD95: ", hd95_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("NSD:  ", nsd_coef(label_b, pred_b, spacing_mm=(1, 1))[0, 1].item())
    print("BIoU: ", biou_coef(label_b, pred_b)[0, 1].item())


def _make_blobs_3d(B, S, K, seed, pred_shift=2):
    """Random class maps (B, D, H, W) uint8 with a few spheres per class.
    Some samples/classes are intentionally left empty (both gt and pred)."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((B, *S), dtype=np.uint8)
    pred = np.zeros((B, *S), dtype=np.uint8)
    zz, yy, xx = np.mgrid[0 : S[0], 0 : S[1], 0 : S[2]]
    for b in range(B):
        for k in range(1, K):
            if rng.random() < 0.25:  # class absent from this sample
                continue
            c = rng.integers(12, min(S) - 12, 3)
            r = int(rng.integers(4, 9))
            ball = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r * r
            gt[b][ball] = k
            rp = r + int(rng.integers(-2, 3))
            if b % 3 == 0 and k == 2 and rp > 0:  # one guaranteed missed organ
                continue
            pred[b][
                (zz - (c[0] + pred_shift)) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
                <= rp * rp
            ] = k
    return gt, pred


def test_hd95_class_maps_match_one_hot():
    B, S, K = 4, (48, 48, 48), 5
    sp = (1.5, 0.7, 0.7)
    gt, pred = _make_blobs_3d(B, S, K, seed=0)

    gt_oh = class2one_hot(torch.from_numpy(gt).long(), K)
    pred_oh = class2one_hot(torch.from_numpy(pred).long(), K)
    ref = hd95_coef(gt_oh, pred_oh, spacing_mm=sp).numpy()

    maps = hd95_batch_maps(gt, pred, [sp] * B, K)

    assert maps.shape == (B, K)
    assert np.allclose(maps, ref, rtol=0, atol=1e-6)
    # absent classes stay 0, present ones are positive
    for b in range(B):
        for k in range(1, K):
            if (gt[b] == k).any():
                assert maps[b, k] > 0 or (pred[b] == k).sum() == 0
            else:
                assert maps[b, k] == 0.0


def test_hd95_class_maps_threadpool_matches_sequential():
    B, S, K = 4, (48, 48, 48), 5
    sp = (1.5, 0.7, 0.7)
    gt, pred = _make_blobs_3d(B, S, K, seed=1)

    seq = hd95_batch_maps(gt, pred, [sp] * B, K)
    with ThreadPoolExecutor(max_workers=4) as pool:
        par = hd95_batch_maps(gt, pred, [sp] * B, K, pool=pool)

    assert np.array_equal(seq, par)
