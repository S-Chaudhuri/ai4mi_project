import math

import autoroot
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from src.eval3d import organ_metrics, sliding_window, to_original_space
from src.postprocess import keep_largest_component


def box(shape, z, y, x):
    m = np.zeros(shape, bool)
    m[z[0] : z[1], y[0] : y[1], x[0] : x[1]] = True
    return m


# --- Perfect prediction: Dice 1, every distance 0, NSD 1
gt = box((30, 40, 40), (10, 20), (10, 30), (10, 30))
m = organ_metrics(gt, gt.copy(), (2.5, 1.0, 1.0), nsd_tau=2.0)
assert m["dice"] == 1 and m["iou"] == 1 and m["precision"] == 1 and m["recall"] == 1, m
assert m["hd"] == 0 and m["hd95"] == 0 and m["assd"] == 0 and m["nsd"] == 1, m
assert m["rel_vol_diff"] == 0, m

# --- Shift by 2 slices along z with 2.5 mm slices: HD is 2 * 2.5 = 5 mm
shifted = np.roll(gt, 2, axis=0)
m = organ_metrics(gt, shifted, (2.5, 1.0, 1.0), nsd_tau=2.0)
assert math.isclose(m["hd"], 5.0, abs_tol=1e-4), m["hd"]
assert math.isclose(m["dice"], 8 / 10, abs_tol=1e-6), m["dice"]  # 8 of 10 slices overlap

# --- Missed organ: Dice 0, distances fall back to a finite worst case
m = organ_metrics(gt, np.zeros_like(gt), (1.0, 1.0, 1.0), nsd_tau=2.0)
assert m["dice"] == 0 and np.isfinite(m["hd"]) and m["hd"] > 0, m


# --- Sliding window reproduces a label volume exactly, with padding and odd sizes
class Echo(torch.nn.Module):
    """Logits = scaled one-hot of the (integer-coded) input: a perfect model."""

    def forward(self, x):
        labels = (x[:, 0] * 4).round().long()
        return F.one_hot(labels, 5).permute(0, 4, 1, 2, 3).float() * 10


labels = torch.randint(0, 5, (37, 50, 45))
vol = (labels.float() / 4)[None]
probs = sliding_window(Echo(), vol, 5, (16, 32, 64), 0.5, 3, torch.device("cpu"))
assert probs.shape == (5, 37, 50, 45), probs.shape
assert torch.equal(probs.argmax(0), labels)
assert torch.allclose(probs.sum(0), torch.ones(37, 50, 45), atol=1e-4)

# --- Largest component: a stray heart blob is removed, the esophagus is untouched
pred = np.zeros((20, 30, 30), np.uint8)
pred[2:10, 2:12, 2:12] = 2  # heart
pred[15:17, 20:22, 20:22] = 2  # stray heart voxels
pred[15:17, 2:4, 2:4] = 1  # esophagus piece 1
pred[2:4, 25:27, 25:27] = 1  # esophagus piece 2
out = keep_largest_component(pred)
assert (out[15:17, 20:22, 20:22] == 0).all() and (out[2:10, 2:12, 2:12] == 2).all()
assert (out == 1).sum() == (pred == 1).sum()

# --- Back to the original grid: identity geometry gives the same array back
meta = {
    "resampled_size_zyx": [20, 30, 30],
    "crop_yx": [5, 25, 3, 28],
    "spacing_zyx": [2.5, 0.9, 0.9],
    "original_size_xyz": [30, 30, 20],
    "original_spacing_xyz": [0.9, 0.9, 2.5],
    "original_origin_xyz": [-10.0, 5.0, 100.0],
    "original_direction": [1, 0, 0, 0, 1, 0, 0, 0, 1],
}
crop = pred[:, 5:25, 3:28]
back = sitk.GetArrayFromImage(to_original_space(crop, meta))
expected = np.zeros_like(pred)
expected[:, 5:25, 3:28] = crop
assert np.array_equal(back, expected)

print("OK")
