#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import dataclasses
from datetime import datetime
import warnings
from typing import Any, Optional
from pathlib import Path
from pprint import pprint
import traceback

import torch
from torch.optim.lr_scheduler import LRScheduler
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler

from functools import partial
import autoroot  # noqa     Do not remove

from src import eval_3D
from src.utils.config import Config, get_config
from src.utils.dataset import GridBoxDataset, SliceDataset, BoxDataset, CoarseDataset
from src.utils.augmentations_online import make_img_transform
from src.models.ShallowNet import shallowCNN
from src.models.ENet import ENet
from src.models.UNet3D import UNet3D
from src.models.VNet3D import VNet3D
from src.models.ResUNet3D import ResUNet3D
from src.models.MedNeXt3D import MedNeXt3D
from src.utils.utils import (
    Dcm,
    class2one_hot,
    class_index_to_name,
    probs2one_hot,
    probs2class,
    seed_all,
    tqdm_,
    dice_coef,
    masked_mean,
    hd95_coef,
    save_images,
    patient_key,
    load_spacing,
)
from src.utils.losses import (
    CrossEntropy,
    CrossEntropyPlusDice,
)


def img_transform_2d(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img


def gt_transform_2d(K, img):
    img = np.array(img, dtype=np.float32)[...]
    # Classes are stored as multiples of 255/(K-1) (e.g. {0, 63, 126, 189, 252} for K=5).
    # Round to the nearest class index so boundary values are never misassigned.
    img = np.round(img / (255.0 / (K - 1))).astype(np.int64)
    img = torch.tensor(img, dtype=torch.int64)[
        None, ...
    ]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


def img_transform_3d(vol: np.ndarray) -> Tensor:
    """
    Input:  vol is a uint8 3D numpy array (box or volume) with shape (D, H, W)
    Output: 4D float Tensor with shape (1, D, H, W) normalized to [0, 1]
    """
    vol = vol.astype(np.float32) / 255.0  # Normalize PNG values to [0, 1]
    tensor = torch.from_numpy(vol)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Shape: (1, D, H, W)
    return tensor


def gt_transform_3d(K: int, vol: np.ndarray) -> Tensor:
    """
    Input:  vol is a 3D numpy array (D, H, W) of class indices {0, ..., K-1}.
            The dataset decodes the 255-quantized PNG values once per volume,
            so here we only one-hot the (already cropped) box.
    Output: 4D int32 Tensor (K, D, H, W) one-hot encoded
    """
    vol_tensor = torch.from_numpy(vol.astype(np.int64))[None, ...]

    # Return one-hot encoded tensor: (K, D, H, W)
    return class2one_hot(vol_tensor, K=K)[0]


# def img_transform_3d(vol: np.ndarray) -> Tensor:
#     vol = vol.astype(np.float32)
#     if vol.max() > 1.0:  # heuristic: looks like it wasn't pre-normalized
#         vol = vol / 255.0
#     return torch.from_numpy(vol[np.newaxis, ...])  # (1, D, H, W)
#
#
# def gt_transform_3d(K: int, vol: np.ndarray) -> Tensor:
#     vol = torch.from_numpy(np.asarray(vol)).long()[None, ...]  # fake batch dim
#     return class2one_hot(vol, K=K)[0]  # (K, D, H, W)


def get_model(config: Config):
    num_classes: int = config.dataset.num_classes
    kernels: int = config.model.kernels
    factor: int = config.model.factor

    # NOTE Gonna rewrite this into a BaseModel which can load any subclass from str
    if config.model.is_3d:
        models_3d = {
            "UNet3D": UNet3D,
            "VNet3D": VNet3D,
            "ResUNet3D": ResUNet3D,
            "MedNeXt3D": MedNeXt3D,
        }
        if config.model.name not in models_3d:
            raise ValueError(
                f"is_3d=True requires a 3D model {list(models_3d)}, got {config.model.name!r}"
            )
        return models_3d[config.model.name](
            1,
            num_classes,
            kernels=kernels,
            factor=factor,
            dropoutRate=config.model.dropout,
        )
    elif config.model.name == "ENet":
        return ENet(
            1,
            num_classes,
            kernels=kernels,
            factor=factor,
            dropoutRate=config.model.dropout,
        )
    elif config.model.name == "shallowCNN":
        return shallowCNN(
            1,
            num_classes,
            kernels=kernels,
            factor=factor,
            dropoutRate=config.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model.name {config.model.name!r} for is_3d=False")


def build_dataloaders(config: Config):

    # Dataset part
    batch_size = config.training.batch_size
    num_classes = config.dataset.num_classes
    data_root_dir = config.paths.data_path / config.dataset.name

    dataset_cls: type[Dataset]
    val_dataset_cls: type[Dataset]  # NEW: validation can use another class
    dataset_kwargs: dict[str, Any] = {}
    train_kwargs: dict[str, Any] = {}  # NEW: only for the training set
    val_kwargs: dict[str, Any] = {}  # NEW: only for the validation set
    if config.model.is_3d:
        img_transform = img_transform_3d
        gt_transform = partial(gt_transform_3d, num_classes)
        dataset_kwargs["num_classes"] = (
            num_classes  # BoxDataset decodes the quantized GT once per volume
        )
        if config.dataset.coarse:
            # Whole volume downsampled to coarse_size, no sub-boxing. The
            # model learns an organ-location map to use as a prior for boxes.
            dataset_cls = CoarseDataset
            val_dataset_cls = CoarseDataset
            dataset_kwargs["target_size"] = config.dataset.coarse_size
        else:
            dataset_cls = BoxDataset
            val_dataset_cls = GridBoxDataset
            dataset_kwargs["sub_box_size"] = config.dataset.box_size
            train_kwargs["fg_prob"] = config.dataset.fg_prob  # NEW
            # val_kwargs["overlap"] = config.val_overlap  # NEW
    else:
        dataset_cls = SliceDataset
        val_dataset_cls = SliceDataset  # NEW
        img_transform = img_transform_2d
        gt_transform = partial(gt_transform_2d, num_classes)

    # Online intensity augmentation wraps the image transform, so only the
    # training set sees it; val_set below keeps the plain one.
    train_img_transform = make_img_transform(img_transform, config.augmentation)

    train_set = dataset_cls(
        "train",
        data_root_dir,
        img_transform=train_img_transform,
        gt_transform=gt_transform,
        debug=config.runtime.debug,
        **dataset_kwargs,
        **train_kwargs,  # NEW
    )

    # NEW: for 3D, an epoch is a fixed number of random batches (drawn with replacement)
    train_loader_kwargs: dict[str, Any]
    if config.model.is_3d:
        train_loader_kwargs = {
            "sampler": RandomSampler(
                train_set,
                replacement=True,
                num_samples=config.training.batches_per_epoch * batch_size,
            )
        }
    else:
        # NOTE why do we set shuffle to false
        train_loader_kwargs = {"shuffle": False}

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=config.runtime.num_workers,
        pin_memory=True,
        persistent_workers=config.runtime.num_workers > 0,
        **train_loader_kwargs,  # NEW: replaces shuffle=False
    )

    val_set = val_dataset_cls(  # NEW: was dataset_cls
        "val",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=config.runtime.debug,
        **dataset_kwargs,
        **val_kwargs,  # NEW
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=config.runtime.num_workers,
        pin_memory=True,
        persistent_workers=config.runtime.num_workers > 0,
        shuffle=False,
    )

    if config.wandb.store_artifect:
        # Store the checksum of the dataset
        artifect = wandb.Artifact(name=config.dataset.name, type="dataset")
        artifect.add_reference(f"file://{data_root_dir}")
        wandb.log_artifact(artifect)

    return train_loader, val_loader


def setup(
    config: Config,
) -> tuple[nn.Module, Any, LRScheduler | None, Any, DataLoader, DataLoader]:
    # Networks and scheduler
    device = torch.device("cuda") if config.training.gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    net = get_model(config)

    net.init_weights()
    net.to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
    )

    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    elif config.training.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler {config.training.scheduler!r}")

    train_loader, val_loader = build_dataloaders(config)

    return (net, optimizer, scheduler, device, train_loader, val_loader)


def get_loss_func(config: Config):
    if config.training.mode == "full":
        idk = list(
            range(config.dataset.num_classes)
        )  # Supervise both background and foreground
    elif config.training.mode == "partial" and config.dataset.name == "SEGTHOR":
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(config.training.mode, config.dataset.name)

    if config.training.loss == "ce":
        return CrossEntropy(idk=idk)
    elif config.training.loss == "dice_ce":
        dice_idk = [c for c in idk if c != 0]
        return CrossEntropyPlusDice(
            ce_idk=idk,
            dice_idk=dice_idk,
            is_3d=config.model.is_3d,
            dice_weight=config.training.dice_weight,
        )
    else:
        raise ValueError(config.training.loss)


# ---------------------------------------------------------------------------
# Saving predictions -- PNGs for 2D slices, .npy volumes for 3D boxes
# ---------------------------------------------------------------------------


def save_predictions(
    predicted_class: Tensor, stems, dest: Path, is_3d: bool, mult: int
) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if is_3d:
        for vol, stem in zip(predicted_class.cpu().numpy(), stems):
            np.save(dest / f"{stem}.npy", vol.astype(np.uint8))
    else:
        save_images(predicted_class * mult, stems, dest)


def create_profiler(config: Config, out_dir: Path):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if config.training.gpu:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def on_trace_ready(prof: torch.profiler.profile):
        trace_path = out_dir / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        prof.export_chrome_trace(str(trace_path))
        print(f">> Profiler: saved trace to {trace_path}")

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.profiler.wait,
            warmup=config.profiler.warmup,
            active=config.profiler.active,
        ),
        on_trace_ready=on_trace_ready,
        record_shapes=config.profiler.record_shapes,
        profile_memory=config.profiler.profile_memory,
        with_stack=config.profiler.with_stack,
        with_flops=config.profiler.with_flops,
        with_modules=config.profiler.with_modules,
    )


def runTraining(config: Config):
    print(
        f">>> Setting up to train on {config.dataset.name} "
        f"({'3D' if config.model.is_3d else '2D'}) with {config.training.mode}"
    )

    net, optimizer, scheduler, device, train_loader, val_loader = setup(config)

    num_classes = config.dataset.num_classes
    data_spacing = (1, 1, 1) if config.model.is_3d else (1, 1)
    result_dir = config.paths.results_dir
    # Guaranteed by get_config to be a path
    if result_dir is None:
        raise

    device_type = "cuda" if config.training.gpu else "cpu"

    profiler: torch.profiler.profile | None = None
    if config.profiler.enabled:
        prof_out_dir = config.profiler.output_dir or (result_dir / "profiler")
        prof_out_dir.mkdir(parents=True, exist_ok=True)
        profiler = create_profiler(config, prof_out_dir)
        profiler.start()
        print(f">> Profiler enabled, traces will be saved to {prof_out_dir}")

    # Adds histogram of the gradients and parameters
    # NOTE Does add a lot of info to our project, need to see if we want that
    if config.wandb.watch:
        wandb.watch(net, log="all", log_freq=100)

    loss_fn = get_loss_func(config)

    # bf16 needs no GradScaler (no fp16 range issues); fp16 keeps the old
    # scaler-based behavior.
    amp_dtype = torch.bfloat16
    if config.training.gpu:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[
            config.runtime.amp_dtype
        ]
    scaler = torch.amp.GradScaler(
        device_type,
        enabled=config.training.gpu and config.runtime.amp_dtype == "fp16",
    )

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros(
        (config.training.epochs, len(train_loader)), device=device
    )
    inter_tra: Tensor = torch.zeros(
        (config.training.epochs, num_classes), device=device
    )
    union_tra: Tensor = torch.zeros(
        (config.training.epochs, num_classes), device=device
    )
    log_loss_val: Tensor = torch.zeros(
        (config.training.epochs, len(val_loader)), device=device
    )
    inter_val: Tensor = torch.zeros(
        (config.training.epochs, num_classes), device=device
    )
    union_val: Tensor = torch.zeros(
        (config.training.epochs, num_classes), device=device
    )
    # j counts samples seen in the epoch; with the 3D replacement sampler an
    # epoch holds batches_per_epoch * batch_size samples, which exceeds
    # len(dataset) for small datasets. Size by the loader, which always
    # bounds the sample count.
    log_hd95_tra: Tensor = torch.zeros(
        (
            config.training.epochs,
            len(train_loader) * config.training.batch_size,
            num_classes,
        ),
        device=device,
    )
    log_hd95_val: Tensor = torch.zeros(
        (
            config.training.epochs,
            len(val_loader) * config.training.batch_size,
            num_classes,
        ),
        device=device,
    )
    log_present_tra: Tensor = torch.zeros(
        (
            config.training.epochs,
            len(train_loader) * config.training.batch_size,
            num_classes,
        ),
        dtype=torch.bool,
        device=device,
    )
    log_present_val: Tensor = torch.zeros(
        (
            config.training.epochs,
            len(val_loader) * config.training.batch_size,
            num_classes,
        ),
        dtype=torch.bool,
        device=device,
    )

    best_dice: float = 0

    # NOTE Just need a total rewrite of this, split it up into functions
    # Also not handy bc train and val are in this same loop
    for e in range(config.training.epochs):
        # Only run hd95 on these intervals, including first epoch to make sure graph looks nice
        calculate_hd95 = e % config.runtime.hd95_interval == 0

        for m in ["train", "val"]:
            match m:
                case "train":
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    inter = inter_tra
                    union = union_tra
                    log_hd95 = log_hd95_tra
                    log_present = log_present_tra
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    inter = inter_val
                    union = union_val
                    log_hd95 = log_hd95_val
                    log_present = log_present_val
                case _:
                    raise  # Should never be reached, but needed to silence ide warn

            with (
                cm()
            ):  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    # non_blocking=True: the DataLoader uses pin_memory, so this
                    # is an async H2D DMA copy instead of a blocking one.
                    img = data["images"].to(device, non_blocking=True)
                    gt = data["gts"].to(device, non_blocking=True)

                    if opt is not None:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    batch_size = img.shape[0]  # works for (B,C,W,H) and (B,C,D,W,H)

                    # HD95 takes seconds per batch. So we only run it on validation or explicitly set
                    compute_hd95 = (
                        m == "val" or config.runtime.hd95_in_train
                    ) and calculate_hd95

                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        pred_logits = net(img)
                        pred_probs = F.softmax(
                            config.model.temperature * pred_logits.float(), dim=1
                        )

                        # Metrics computation, not used for training
                        pred_class = probs2class(pred_probs)
                        gt_class = probs2class(gt)
                        for k in range(num_classes):
                            inter[e, k] += ((pred_class == k) & (gt_class == k)).sum()
                            union[e, k] += (pred_class == k).sum() + (
                                gt_class == k
                            ).sum()

                        if compute_hd95:
                            spacing_map = (
                                load_spacing(
                                    config.paths.data_path / config.dataset.name
                                )
                                if config.model.is_3d
                                else None
                            )
                            pred_seg = class2one_hot(pred_class, num_classes)
                            for b in range(batch_size):
                                sp = (
                                    spacing_map[patient_key(data["stems"][b])]
                                    if spacing_map
                                    else data_spacing
                                )
                                log_hd95[e, j + b, :] = hd95_coef(
                                    gt[b : b + 1], pred_seg[b : b + 1], spacing_mm=sp
                                )[0]
                        log_present[e, j : j + batch_size, :] = (
                            gt.sum(dim=tuple(range(2, gt.ndim))) > 0
                        )  # Per-sample, per-class: is the class in the gt?

                    # Computed outside autocast: under CUDA fp16 autocast the
                    # einsum in the loss is promoted to fp16 and overflows
                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.detach()  # One loss value per batch

                    if opt is not None:  # Only for training
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

                    if (
                        m == "val" and False
                    ):  # Turn off for now, might just be something we only want to do during eval
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = (
                                63 if num_classes == 5 else int(255 / (num_classes - 1))
                            )
                            save_predictions(
                                predicted_class,
                                data["stems"],
                                result_dir / f"iter{e:03d}" / m,
                                config.model.is_3d,
                                mult,
                            )

                    j += batch_size  # Keep in mind that _in theory_, each batch might have a different size
                    if profiler is not None:
                        profiler.step()
                    # For the DSC average: do not take the background class (0) into account:
                    # HD95 and gated dice are only averaged over samples where the class is in the gt
                    present = log_present[e, :j, 1:]

                    # Necessary for gated dice: only average over samples where the class is present in the gt
                    seen = log_present[e, :j, 1:].any(dim=0)
                    d = (2 * inter[e, 1:] + 1e-8) / (union[e, 1:] + 1e-8)

                    postfix_dict: dict[str, str] = {
                        "Dice": f"{((2 * inter[e, 1:] + 1e-8) / (union[e, 1:] + 1e-8)).mean():05.3f}",
                        "GDice": f"{d[seen].mean():05.3f}" if seen.any() else "n/a",
                        "Loss": f"{log_loss[e, : i + 1].mean():5.2e}",
                    }
                    if compute_hd95:
                        postfix_dict["HD95"] = (
                            f"{masked_mean(log_hd95[e, :j, 1:], present):05.2f}"
                        )
                    if num_classes > 2:
                        postfix_dict |= {
                            f"Dice-{class_index_to_name(k)}": f"{((2 * inter[e, k] + 1e-8) / (union[e, k] + 1e-8)):05.3f}"
                            for k in range(1, num_classes)
                        }
                    tq_iter.set_postfix(postfix_dict)

        metrics = {
            "epoch": e,
            "train/loss": log_loss_tra[e].mean().item(),
            "train/dice": ((2 * inter_tra[e, 1:] + 1e-8) / (union_tra[e, 1:] + 1e-8))
            .mean()
            .item(),
            # "train/acc": acc_tra,
            "val/loss": log_loss_val[e].mean().item(),
            "val/dice": ((2 * inter_val[e, 1:] + 1e-8) / (union_val[e, 1:] + 1e-8))
            .mean()
            .item(),
            "val/hd95": masked_mean(
                log_hd95_val[e, :, 1:], log_present_val[e, :, 1:]
            ).item(),
            # "val/acc": acc_val,
        }
        if config.runtime.hd95_in_train:
            metrics["train/hd95"] = masked_mean(
                log_hd95_tra[e, :, 1:], log_present_tra[e, :, 1:]
            ).item()

        if num_classes > 2:
            for k in range(1, num_classes):
                class_name = class_index_to_name(k)
                metrics[f"train/dice_{class_name}"] = (
                    (2 * inter_tra[e, k] + 1e-8) / (union_tra[e, k] + 1e-8)
                ).item()
                metrics[f"val/dice_{class_name}"] = (
                    (2 * inter_val[e, k] + 1e-8) / (union_val[e, k] + 1e-8)
                ).item()

                # Only on correct intervals
                if calculate_hd95:
                    if config.runtime.hd95_in_train:
                        metrics[f"train/hd95_{class_name}"] = masked_mean(
                            log_hd95_tra[e, :, k], log_present_tra[e, :, k]
                        ).item()
                    metrics[f"val/hd95_{class_name}"] = masked_mean(
                        log_hd95_val[e, :, k], log_present_val[e, :, k]
                    ).item()
        wandb.log(metrics)

        # Scheduler at the end of each epoch
        if scheduler is not None:
            scheduler.step()

        current_dice: float = (
            ((2 * inter_val[e, 1:] + 1e-8) / (union_val[e, 1:] + 1e-8)).mean().item()
        )
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(result_dir / "best_epoch.txt", "a") as f:
                f.write(message)

            torch.save(net.state_dict(), result_dir / "bestweights.pt")

    if profiler is not None:
        profiler.stop()


def main():
    config = get_config()

    # Seed everything right at the beginning
    seed_all(config.training.seed, config.training.gpu, config.runtime.cudnn_benchmark)

    # Setup wandb
    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=dataclasses.asdict(config),
        dir=config.wandb.dir or (autoroot.root / "results" / "wandb"),
        notes=config.wandb.notes,
    )

    pprint(dataclasses.asdict(config))

    try:
        runTraining(config)
    except:
        # Log traceback in wandb logs
        tb = traceback.format_exc()
        wandb.log(
            {
                "exceptions/traceback": tb,
            }
        )
        wandb.finish(exit_code=1)

        raise  # Re-raise so the traceback is printed and the job exits non zero

    wandb.finish()

    if config.evaluation.enabled:
        config.evaluation.weights = config.paths.results_dir / "bestweights.pt"

        eval_3D.run_eval(config)


if __name__ == "__main__":
    main()
