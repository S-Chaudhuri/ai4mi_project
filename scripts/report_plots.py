import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy import ndimage

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
MODELS = ["UNet3D", "VNet3D", "ResUNet3D", "MedNeXt3D"]
FIRST_BASELINES = {"UNet3D": "SEGTHOR_3D_1ch/11-10-2026_00-02-13", "VNet3D": "SEGTHOR_3D_1ch/11-10-2026_00-02-46",
                   "MedNeXt3D": "SEGTHOR_3D_1ch/11-10-2026_00-21-01"}
TWO_D = {"ShallowNet": {"dice": [0.000, 0.018, 0.028, 0.038], "mean": 0.021, "hd95": 179.47},
         "ENet": {"dice": [0.586, 0.373, 0.804, 0.756], "mean": 0.630, "hd95": 36.09}}
BOX = dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.75)
RNG = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "axes.axisbelow": True,
})


class Runs:
    def __init__(self, root: Path, runs_csv: Path):
        self.root = root
        self.local_runs = root / "runs"
        self.snellius = root / "snellius_eval" / "results"
        self.rows = {(r["model"], r["label"]): r for r in csv.DictReader(open(runs_csv))}

    def value(self, model, label, key):
        return float(self.rows[(model, label)][key])

    def has(self, model, label):
        return (model, label) in self.rows

    def metrics_file(self, model, label):
        job = self.rows[(model, label)]["job"]
        if job == "local":
            return self.local_runs / "ResUNet3D_skel1_local" / "eval" / "metrics.csv"
        if job == "kaggle":
            return self.local_runs / "ResUNet3D_skel1_kaggle_metrics.csv"
        if label == "1ch baseline" and model in FIRST_BASELINES:
            return self.snellius / FIRST_BASELINES[model] / "eval" / "metrics.csv"
        return next(self.snellius.glob(f"*/*job{job}/eval/metrics.csv"))

    def patients(self, model, label, metric, organ=None):
        rs = [r for r in csv.DictReader(open(self.metrics_file(model, label))) if r["patient"] != "MEAN"]
        pats = sorted({r["patient"] for r in rs})
        return [float(np.mean([float(r[metric]) for r in rs
                               if r["patient"] == p and (organ is None or r["organ"] == organ)])) for p in pats]


def save(fig, out: Path, name):
    for ext in ("png", "svg"):
        p = out / f"{name}.{ext}"
        if p.exists():
            raise FileExistsError(f"{p} exists: move it first")
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def dots(ax, x, pts, width, first, ymax=None):
    xs = x + RNG.uniform(-width / 4, width / 4, len(pts))
    inside = [(a, b) for a, b in zip(xs, pts) if ymax is None or b <= ymax]
    ax.scatter([a for a, _ in inside], [b for _, b in inside], s=11, color=INK, alpha=0.5, zorder=3,
               linewidths=0, label="one patient" if first else None)
    for a, b in zip(xs, pts):
        if ymax is not None and b > ymax:
            ax.scatter(a, ymax * 0.985, marker="^", s=22, color=INK, zorder=3, linewidths=0)
            ax.annotate(f"{b:.0f}", (a, ymax * 0.985), xytext=(6, -3), textcoords="offset points", ha="left",
                        fontsize=7, color=INK)
    ax.plot([x - width / 2.2, x + width / 2.2], [min(np.median(pts), ymax or np.inf)] * 2, color=INK, lw=1.6,
            zorder=4, label="median" if first else None)


def grouped(ax, groups, series, values, colors, fmt, width=0.8, fontsize=8.5, labels=True, points=None, ymax=None):
    n = len(series)
    w = width / n
    done = set()
    for g in range(len(groups)):
        present = [i for i in range(n) if values[series[i]][g] is not None]
        for j, i in enumerate(present):
            x = g + (j - (len(present) - 1) / 2) * w
            v = values[series[i]][g]
            ax.bar(x, v, w * 0.9, color=colors[i], label=series[i] if i not in done else None, zorder=2)
            done.add(i)
            pts = points[series[i]][g] if points is not None else []
            if points is not None:
                dots(ax, x, pts, w * 0.9, (g, j) == (0, 0), ymax)
            if labels:
                top = max([v, *(q for q in pts if q <= (ymax or np.inf))])
                ax.annotate(fmt.format(v), (x, top), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=fontsize, color=INK)
    ax.set_xticks(range(len(groups)), groups)


def plot_2d(runs, out):
    nets = list(TWO_D)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    cols = [o.capitalize() for o in ORGANS] + ["Mean"]
    grouped(axes[0], cols, nets, {n: TWO_D[n]["dice"] + [TWO_D[n]["mean"]] for n in nets}, [ORANGE, BLUE], "{:.3f}")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("validation Dice (higher is better)")
    axes[0].set_title("Dice per organ", loc="left", fontsize=11)
    grouped(axes[1], nets, ["HD95"], {"HD95": [TWO_D[n]["hd95"] for n in nets]}, [INK2], "{:.1f}", width=0.5)
    axes[1].set_ylim(0, 200)
    axes[1].set_ylabel("validation HD95 in mm (lower is better)")
    axes[1].set_title("HD95, mean of 4 organs", loc="left", fontsize=11)
    axes[0].legend(ncol=2, frameon=False, loc="upper left")
    fig.suptitle("2D baselines: ShallowNet vs ENet (50 epochs, Dice + CE, no augmentation)",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.03, "Per-slice validation metrics from the training logs at epoch 45 (ShallowNet) and 40 (ENet), "
             "the epochs nearest the best that computed HD95. Not comparable with the 3D whole-volume scores.",
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, out, "report_2d_baselines")


def plot_3d(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, metric, ylabel, fmt, ymax in [(axes[0], "dice", "Dice (higher is better)", "{:.2f}", 1.08),
                                          (axes[1], "hd95", "HD95 in mm (lower is better)", "{:.0f}", 80)]:
        values = {o.capitalize(): [runs.value(m, "1ch baseline", f"{o}_{metric}") for m in MODELS] for o in ORGANS}
        pts = {o.capitalize(): [runs.patients(m, "1ch baseline", metric, o) for m in MODELS] for o in ORGANS}
        grouped(ax, MODELS, list(values), values, [BLUE, ORANGE, AQUA, YELLOW], fmt, fontsize=7.5,
                points=pts, ymax=ymax)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
    axes[0].set_title("Dice per organ", loc="left", fontsize=11)
    axes[1].set_title("HD95 per organ", loc="left", fontsize=11)
    axes[0].legend(ncol=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.suptitle("3D baselines: 1 channel, no augmentation, Dice + CE, LCC; 5 validation patients",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "report_3d_baselines")


def plot_loss(runs, out):
    groups = [("ResUNet3D", "1ch"), ("ResUNet3D", "1ch + aug + DS"), ("ResUNet3D", "3ch + aug"),
              ("ResUNet3D", "3ch + aug + DS"), ("MedNeXt3D", "1ch + aug + DS"), ("MedNeXt3D", "3ch + aug"),
              ("MedNeXt3D", "3ch + aug + DS")]
    losses = ["Dice + CE", "Tversky", "Dice + CE + skeleton"]
    skeleton_runs = [f"1ch + skeleton ({p})" for p in ("laptop", "Kaggle")]

    def label(setup, loss):
        if loss == "Dice + CE":
            return "1ch baseline" if setup == "1ch" else setup
        return setup.replace("aug", "aug Tversky", 1)

    def get(model, setup, loss, key):
        if loss == "Dice + CE + skeleton":
            if (model, setup) != ("ResUNet3D", "1ch"):
                return None
            return float(np.mean([runs.value(model, s, key) for s in skeleton_runs]))
        if loss == "Tversky" and setup == "1ch":
            return None
        return runs.value(model, label(setup, loss), key) if runs.has(model, label(setup, loss)) else None

    def get_pts(model, setup, loss, metric):
        if get(model, setup, loss, "mean_dice") is None:
            return None
        if loss == "Dice + CE + skeleton":
            return [v for s in skeleton_runs for v in runs.patients(model, s, metric)]
        return runs.patients(model, label(setup, loss), metric)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    xlabels = [f"{m}\n{s}" for m, s in groups]
    for ax, key, ylabel, fmt, ymax in [(axes[0], "mean_dice", "mean Dice (higher is better)", "{:.3f}", 1.08),
                                       (axes[1], "mean_hd95", "mean HD95 in mm (lower is better)", "{:.1f}", 120)]:
        values = {l: [get(m, s, l, key) for m, s in groups] for l in losses}
        pts = {l: [get_pts(m, s, l, key.split("_")[1]) for m, s in groups] for l in losses}
        grouped(ax, xlabels, losses, values, [BLUE, ORANGE, AQUA], fmt, points=pts, ymax=ymax)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
    axes[0].legend(ncol=5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13))
    axes[1].text(0, -0.32, "aug = offline augmented copies, DS = deep supervision. Skeleton loss: mean of 2 runs "
                 "(laptop 0.886 / 7.2 mm, Kaggle 0.877 / 8.2 mm), dots of both. Dot = one patient (mean of 4 organs). "
                 "MedNeXt3D + Tversky collapses the trachea (Dice < 0.01) in 2 of 3 setups.",
                 transform=axes[1].transAxes, fontsize=8.5, color=INK2, wrap=True)
    fig.suptitle("Loss ablation: same model and setup, only the loss changes", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "report_loss_ablation")


def plot_postprocessing(runs, out):
    variants = {"No postprocessing": "eval_val", "LCC (heart, trachea, aorta)": "eval_val_lcc"}

    def per_patient(model, folder, metric):
        rs = list(csv.DictReader(open(runs.local_runs / f"{model}_1ch" / folder / "metrics.csv")))
        pats = sorted({r["patient"] for r in rs})
        return [float(np.mean([float(r[metric]) for r in rs if r["patient"] == p])) for p in pats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, ylabel, fmt in [(axes[0], "dice", "mean Dice (higher is better)", "{:.3f}"),
                                    (axes[1], "hd95", "mean HD95 in mm (lower is better)", "{:.1f}")]:
        values = {v: [float(np.mean(per_patient(m, d, metric))) for m in MODELS] for v, d in variants.items()}
        grouped(ax, MODELS, list(variants), values, [BLUE, AQUA], fmt, labels=False)
        for i, d in enumerate(variants.values()):
            for g, m in enumerate(MODELS):
                pts = per_patient(m, d, metric)
                x = g + (i - 0.5) * 0.4 + RNG.uniform(-0.06, 0.06, len(pts))
                ax.scatter(x, pts, s=14, color=INK, alpha=0.55, zorder=3, linewidths=0,
                           label="one patient" if (i, g) == (0, 0) else None)
        before, after = values.values()
        ax.set_xticks(range(len(MODELS)), [f"{m}\n{fmt.format(a)} \u2192 {fmt.format(b)}"
                                           for m, a, b in zip(MODELS, before, after)])
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.suptitle("Postprocessing: keeping the largest connected component (same predictions, earlier local runs)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "report_postprocessing")


def plot_best_vs_worst(runs, out):
    best = ("ResUNet3D", "1ch + skeleton (laptop)")
    worst = ("MedNeXt3D", "1ch + aug Tversky")
    names = {best: "Best: ResUNet3D, Dice + CE + skeleton\n(1ch, no aug)",
             worst: "Worst: MedNeXt3D, Tversky\n(1ch + offline aug, no DS)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, ylabel, fmt, ymax in [(axes[0], "dice", "Dice (higher is better)", "{:.2f}", 1.08),
                                          (axes[1], "hd95", "HD95 in mm (lower is better)", "{:.0f}", 360)]:
        values = {names[r]: [runs.value(*r, f"{o}_{metric}") for o in ORGANS] for r in (best, worst)}
        pts = {names[r]: [runs.patients(*r, metric, o) for o in ORGANS] for r in (best, worst)}
        grouped(ax, [o.capitalize() for o in ORGANS], list(values), values, [AQUA, ORANGE], fmt,
                points=pts, ymax=ymax)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
    axes[0].set_title("Dice per organ", loc="left", fontsize=11)
    axes[1].set_title("HD95 per organ", loc="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0.01, -0.1))
    b, w = (runs.value(*r, "mean_dice") for r in (best, worst))
    bh, wh = (runs.value(*r, "mean_hd95") for r in (best, worst))
    fig.suptitle(f"Best vs worst 3D run: mean Dice {b:.3f} vs {w:.3f}, mean HD95 {bh:.1f} vs {wh:.1f} mm "
                 "(5 validation patients)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "report_best_vs_worst")


def plot_mistake(runs, out):
    root = runs.root
    patient = root / "segthor_train_full" / "train" / "Patient_30"
    ct_img = nib.load(patient / "Patient_30.nii.gz")
    ct = np.asarray(ct_img.dataobj).astype(np.float32)
    sx, _, sz = ct_img.header.get_zooms()[:3]
    gt = np.asarray(nib.load(patient / "GT.nii.gz").dataobj) == 1

    def mask(folder):
        return np.asarray(nib.load(root / "predictions" / folder / "Patient_30.nii.gz").dataobj) == 1

    masks = {"Ground truth": gt,
             "UNet3D (job 27892954)\nDice 0.656, HD95 261.4 mm": mask("unet_base_patient30"),
             "Best: ResUNet3D + skeleton\nDice 0.762, HD95 13.9 mm": mask("best_patient30")}
    weak = masks[list(masks)[1]]
    y0 = int(ndimage.center_of_mass(gt)[1])
    bg = np.clip((ct[:, y0, :] + 160) / 400, 0, 1).T

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), facecolor=SURFACE)
    for ax, (title, m) in zip(axes, masks.items()):
        proj = m.any(axis=1).T
        ax.imshow(bg, cmap="gray", origin="lower", aspect=sz / sx)
        ax.imshow(np.ma.masked_where(~proj, proj), cmap=ListedColormap([BLUE]), origin="lower", aspect=sz / sx,
                  alpha=0.9, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        lab, n = ndimage.label(m)
        big = int((ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1)) >= 100).sum())
        ax.text(0.02, 0.02, f"{big} piece{'s' if big != 1 else ''} of 100+ voxels ({n - big} specks)",
                transform=ax.transAxes, color="white", fontsize=10, va="bottom", bbox=BOX)

    gz = np.where(gt.any(axis=(0, 1)))[0]
    gx = ndimage.center_of_mass(gt)[0]
    lab, n = ndimage.label(weak)
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
    for i in range(n):
        c = ndimage.center_of_mass(lab == i + 1)
        if sizes[i] > 100 and (c[2] > gz.max() + 5 or abs(c[0] - gx) > 60):
            axes[1].add_patch(Ellipse((c[0], c[2]), 60, 50 * sx / sz, fill=False, ec=RED, lw=2.5))
            axes[1].annotate("false positive far from\nthe esophagus (HD95)", (c[0], c[2]),
                             xytext=(c[0] + 20, c[2] - 45), color="white", fontsize=9,
                             arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), bbox=BOX)
    covered = set(np.where(weak.any(axis=(0, 1)))[0])
    gaps = [z for z in range(gz.min(), gz.max()) if z not in covered]
    if gaps:
        zc = (min(gaps) + max(gaps)) / 2
        axes[1].annotate("gap: middle\nof the tube missed", (gx, zc), xytext=(gx + 95, zc - 25), color="white",
                         fontsize=9, va="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), bbox=BOX)
    fig.suptitle("Patient 30, esophagus (coronal view, mask projected through the body): the weak model breaks the "
                 "tube and adds a distant piece", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, out, "report_mistake_patient30")


FIGURES = {"2d": plot_2d, "3d": plot_3d, "loss": plot_loss, "pp": plot_postprocessing, "bw": plot_best_vs_worst,
           "mistake": plot_mistake}


def main():
    parser = argparse.ArgumentParser(description="Report figures")
    parser.add_argument("figures", nargs="*", help=f"any of {', '.join(FIGURES)} (default: all)")
    parser.add_argument("--root", type=Path, default=Path("C:/Users/soham/ai4mi_data"))
    parser.add_argument("--runs_csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    unknown = [f for f in args.figures if f not in FIGURES]
    if unknown:
        parser.error(f"unknown figure(s) {unknown}, choose from {list(FIGURES)}")
    out = args.out or args.root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = Runs(args.root, args.runs_csv or args.root / "figures" / "source" / "runs_curated.csv")
    for name in args.figures or FIGURES:
        FIGURES[name](runs, out)


if __name__ == "__main__":
    main()
