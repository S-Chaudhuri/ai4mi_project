import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                       # saves to file, works without a display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from src.utils.dataset import BoxDataset
from src.utils.utils import class2one_hot

# ---- settings -------------------------------------------------------------
DATA_ROOT = Path("data/SEGTHOR")
BOX_SIZE = (128, 128, 128)
NUM_CLASSES = 5
N_SHOW = 6                  # boxes shown in the picture
N_STATS = 10                # boxes drawn per setting for the statistics (raise for smoother numbers)
SEED = 0

# ---- same transforms as in your training script ---------------------------
def img_transform_3d(vol):
    vol = vol.astype(np.float32) / 255.0
    t = torch.from_numpy(vol)
    return t.unsqueeze(0) if t.ndim == 3 else t

def gt_transform_3d(K, vol):
    vol = np.round(np.array(vol, dtype=np.float32) / 63.0).astype(np.int64)
    return class2one_hot(torch.from_numpy(vol)[None, ...], K=K)[0]

gt_t = partial(gt_transform_3d, NUM_CLASSES)

def make_ds(fg_prob):
    return BoxDataset("train", DATA_ROOT, img_transform=img_transform_3d,
                      gt_transform=gt_t, sub_box_size=BOX_SIZE, fg_prob=fg_prob)

# ---- 1. statistics: fg_prob 0.0 vs 0.5 vs 1.0 ----------------------------
random.seed(SEED)
settings = [0.0, 0.5, 1.0]
results = {}

for p in settings:
    ds = make_ds(p)
    present = np.zeros(NUM_CLASSES)
    for _ in tqdm(range(N_STATS), desc=f"Stats  fg_prob={p:.1f}", unit="box"):
        box = ds[random.randrange(len(ds))]
        assert tuple(box["images"].shape) == (1, *BOX_SIZE)
        assert tuple(box["gts"].shape) == (NUM_CLASSES, *BOX_SIZE)
        present += (box["gts"].flatten(1).sum(1) > 0).numpy()
    results[p] = present / N_STATS

print(f"\nFraction of boxes that contain each class ({N_STATS} random boxes per setting)")
print(f"{'fg_prob':>8} | " + " | ".join(f"class {k}" for k in range(1, NUM_CLASSES)))
for p in settings:
    print(f"{p:>8.1f} | " + " | ".join(f"{results[p][k]:7.0%}" for k in range(1, NUM_CLASSES)))

# ---- 2. picture of random boxes (fg_prob = 0.5) ---------------------------
random.seed(SEED)
ds = make_ds(0.5)
colors = ["none", "tab:red", "tab:green", "tab:blue", "tab:orange"]
cmap = ListedColormap(colors[:NUM_CLASSES])

fig, axes = plt.subplots(N_SHOW, 2, figsize=(7, 3.2 * N_SHOW))
for r in tqdm(range(N_SHOW), desc="Picture", unit="box"):
    box = ds[random.randrange(len(ds))]
    img = box["images"][0].numpy()                      # (D, H, W)
    lab = box["gts"].argmax(0).numpy()                  # (D, H, W) class indices

    fg_per_slice = (lab > 0).sum(axis=(1, 2))
    z = int(fg_per_slice.argmax()) if fg_per_slice.max() > 0 else BOX_SIZE[0] // 2

    classes = [int(c) for c in np.unique(lab) if c > 0]
    axes[r, 0].imshow(img[z], cmap="gray", vmin=0, vmax=1)
    axes[r, 0].set_title(f"{box['stems']}  slice {z}", fontsize=9)
    axes[r, 1].imshow(img[z], cmap="gray", vmin=0, vmax=1)
    axes[r, 1].imshow(np.ma.masked_equal(lab[z], 0), cmap=cmap, vmin=0,
                      vmax=NUM_CLASSES - 1, alpha=0.6, interpolation="nearest")
    axes[r, 1].set_title(f"classes in box: {classes if classes else 'none'}", fontsize=9)
    for a in axes[r]:
        a.axis("off")

fig.suptitle("Random boxes (fg_prob=0.5). Red=1, green=2, blue=3, orange=4", fontsize=10)
fig.tight_layout()
out = Path("box_samples.png")
fig.savefig(out, dpi=120)
print(f"\nSaved {out.resolve()}")