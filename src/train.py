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
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
from torch.optim.lr_scheduler import LRScheduler
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from functools import partial
import autoroot  # noqa     Do not remove

from src.utils.config import Config, get_config
from src.utils.dataset import SliceDataset, BoxDataset
from src.models.ShallowNet import shallowCNN
from src.models.ENet import ENet
from src.models.UNet3D import UNet3D
from src.utils.utils import (
    Dcm,
    class2one_hot,
    probs2one_hot,
    probs2class,
    seed_all,
    tqdm_,
    dice_coef,
    gated_dice,
    hd95_coef,
    save_images,
)
from src.utils.losses import (
    CrossEntropy,
    CrossEntropyPlusDice,
    CrossEntropy2D,
    CrossEntropyPlusDice2D,
)


def img_transform_2d(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img


def noised_img_transform(img, p: float, sigma: float):
    img = img_transform_2d(img)
    if np.random.random() < p:
        img = img + torch.randn_like(img) * sigma
    return img.clamp(0.0, 1.0)


def gt_transform_2d(K, img):
    img = np.array(img)[...]
    # The idea is that the classes are mapped to {0, 255} for binary cases
    # {0, 85, 170, 255} for 4 classes
    # {0, 51, 102, 153, 204, 255} for 6 classes
    # Very sketchy but that works here and that simplifies visualization
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[
        None, ...
    ]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


def img_transform_3d(vol: np.ndarray) -> Tensor:
    """
    Input:  vol is a 3D numpy array from stacked PNGs with shape (D, H, W)
    Output: 4D float Tensor with shape (1, D, H, W) normalized to [0, 1]
    """
    vol = vol.astype(np.float32) / 255.0  # Normalize PNG values to [0, 1]
    tensor = torch.from_numpy(vol)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Shape: (1, D, H, W)
    return tensor


def gt_transform_3d(K: int, vol: np.ndarray) -> Tensor:
    """
    Input:  vol is a 3D numpy array from stacked PNG masks (D, H, W)
            containing values in {0, 63, 126, 189, 252}
    Output: 4D float Tensor (K, D, H, W) one-hot encoded
    """
    vol = np.array(vol, dtype=np.float32)

    # Convert intensity values {0, 63, 126, 189, 252} -> class indices {0, 1, 2, 3, 4}
    # Using 63.0 step for SEGTHOR 5-class masks
    vol = np.round(vol / 63.0).astype(np.int64)

    # Add channel dimension: (1, D, H, W)
    vol_tensor = torch.from_numpy(vol)[None, ...]

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
    if config.is_3d:
        if config.model.name != "UNet3D":
            raise ValueError(
                f"dims='3d' requires model.name='UNet3D', got {config.model.name!r}"
            )
        return UNet3D(
            1, num_classes, kernels=kernels, factor=factor, dropoutRate=config.dropout
        )
    elif config.model.name == "ENet":
        return ENet(
            1, num_classes, kernels=kernels, factor=factor, dropoutRate=config.dropout
        )
    elif config.model.name == "shallowCNN":
        return shallowCNN(
            1, num_classes, kernels=kernels, factor=factor, dropoutRate=config.dropout
        )
    else:
        raise ValueError(f"Unknown model.name {config.model.name!r} for dims='2d'")


def build_dataloaders(config: Config):

    # Dataset part
    batch_size = config.batch_size
    num_classes = config.dataset.num_classes
    data_root_dir = autoroot.root / "data" / config.dataset.name

    dataset_cls: type[Dataset]
    dataset_kwargs: dict[str, Any] = {}
    if config.is_3d:
        dataset_cls = BoxDataset
        img_transform = img_transform_3d
        gt_transform = partial(gt_transform_3d, num_classes)
        dataset_kwargs["sub_box_size"] = config.dataset.box_size
    else:
        dataset_cls = SliceDataset
        img_transform = img_transform_2d
        gt_transform = partial(gt_transform_2d, num_classes)

    train_set = dataset_cls(
        "train",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=config.debug,
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    val_set = dataset_cls(
        "val",
        data_root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=config.debug,
        **dataset_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    # Store the checksum of the dataset
    artifect = wandb.Artifact(name=config.dataset.name, type="dataset")
    artifect.add_reference(f"file://{data_root_dir}")
    wandb.log_artifact(artifect)

    return train_loader, val_loader


def setup(
    config: Config,
) -> tuple[nn.Module, Any, LRScheduler, Any, DataLoader, DataLoader]:
    # Networks and scheduler
    device = torch.device("cuda") if config.gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    net = get_model(config)

    net.init_weights()
    net.to(device)

    lr = config.lr
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=lr, weight_decay=config.weight_decay, betas=config.betas
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    train_loader, val_loader = build_dataloaders(config)

    return (net, optimizer, scheduler, device, train_loader, val_loader)


def get_loss_func(config: Config):
    if config.mode == "full":
        idk = list(
            range(config.dataset.num_classes)
        )  # Supervise both background and foreground
    elif config.mode == "partial" and config.dataset.name == "SEGTHOR":
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(config.mode, config.dataset.name)

    if config.is_3d:
        ce_cls, ce_dice_cls = CrossEntropy, CrossEntropyPlusDice
    else:
        ce_cls, ce_dice_cls = CrossEntropy2D, CrossEntropyPlusDice2D

    if config.loss == "ce":
        return ce_cls(idk=idk)
    elif config.loss == "dice_ce":
        dice_idk = [c for c in idk if c != 0]
        return ce_dice_cls(
            ce_idk=idk, dice_idk=dice_idk, dice_weight=config.dice_weight
        )
    else:
        raise ValueError(config.loss)


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


def runTraining(config: Config):
    print(
        f">>> Setting up to train on {config.dataset.name} ({'3D' if config.is_3d else '2D'}) with {config.mode}"
    )

    net, optimizer, scheduler, device, train_loader, val_loader = setup(config)

    num_classes = config.dataset.num_classes

    result_dir = config.dest or Path(
        f"results/{config.dataset.name}/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    device_type = "cuda" if config.gpu else "cpu"

    # Adds histogram of the gradients and parameters
    # NOTE Does add a lot of info to our project, need to see if we want that
    if config.wandb_watch:
        wandb.watch(net, log="all", log_freq=100)

    loss_fn = get_loss_func(config)
    scaler = torch.amp.GradScaler(device_type, enabled=config.gpu)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((config.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros(
        (config.epochs, len(train_loader.dataset), num_classes)  # type: ignore
    )
    log_loss_val: Tensor = torch.zeros((config.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros(
        (config.epochs, len(val_loader.dataset), num_classes)  # type: ignore
    )
    log_hd95_tra: Tensor = torch.zeros(
        (config.epochs, len(train_loader.dataset), num_classes)  # type: ignore
    )
    log_hd95_val: Tensor = torch.zeros(
        (config.epochs, len(val_loader.dataset), num_classes)  # type: ignore
    )
    log_present_tra: Tensor = torch.zeros(
        (config.epochs, len(train_loader.dataset), num_classes),  # type: ignore
        dtype=torch.bool,
    )
    log_present_val: Tensor = torch.zeros(
        (config.epochs, len(val_loader.dataset), num_classes),  # type: ignore
        dtype=torch.bool,
    )

    best_dice: float = 0

    # NOTE Just need a total rewrite of this, split it up into functions
    # Also not handy bc train and val are in this same loop
    for e in range(config.epochs):
        for m in ["train", "val"]:
            match m:
                case "train":
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                    log_hd95 = log_hd95_tra
                    log_present = log_present_tra
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
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
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if opt is not None:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    batch_size = img.shape[0]  # works for (B,C,W,H) and (B,C,D,W,H)

                    with torch.autocast(device_type=device_type):
                        pred_logits = net(img)
                        pred_probs = F.softmax(
                            config.temperature * pred_logits.float(), dim=1
                        )  # 1 is the temperature parameter

                        # Metrics computation, not used for training
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j : j + batch_size, :] = dice_coef(
                            pred_seg, gt
                        )  # One DSC value per sample and per class
                        log_hd95[e, j : j + batch_size, :] = hd95_coef(
                            pred_seg, gt, spacing_mm=(1, 1)
                        )
                        log_present[e, j : j + batch_size, :] = (
                            gt.sum(dim=(-2, -1)) > 0
                        )  # Per-sample, per-class: is the class in the gt?

                        loss = loss_fn(pred_probs, gt)
                        log_loss[e, i] = (
                            loss.item()
                        )  # One loss value per batch (averaged in the loss)

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
                                config.is_3d,
                                mult,
                            )

                    j += batch_size  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    # HD95 and gated dice are only averaged over samples where the class is in the gt
                    present = log_present[e, :j, 1:]

                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "GDice": f"{gated_dice(log_dice[e, :j, 1:], present):05.3f}",
                        "HD95": f"{gated_dice(log_hd95[e, :j, 1:], present):05.2f}",
                        "Loss": f"{log_loss[e, : i + 1].mean():5.2e}",
                    }
                    if num_classes > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                            for k in range(1, num_classes)
                        }
                    tq_iter.set_postfix(postfix_dict)

        metrics = {
            "epoch": e,
            "train/loss": log_loss_tra[e].mean().item(),
            "train/dice": log_dice_tra[e, :, 1:].mean().item(),
            "train/gated_dice": gated_dice(
                log_dice_tra[e, :, 1:], log_present_tra[e, :, 1:]
            ).item(),
            "train/hd95": gated_dice(
                log_hd95_tra[e, :, 1:], log_present_tra[e, :, 1:]
            ).item(),
            # "train/acc": acc_tra,
            "val/loss": log_loss_val[e].mean().item(),
            "val/dice": log_dice_val[e, :, 1:].mean().item(),
            "val/gated_dice": gated_dice(
                log_dice_val[e, :, 1:], log_present_val[e, :, 1:]
            ).item(),
            "val/hd95": gated_dice(
                log_hd95_val[e, :, 1:], log_present_val[e, :, 1:]
            ).item(),
            # "val/acc": acc_val,
        }

        if num_classes > 2:
            for k in range(1, num_classes):
                metrics[f"train/dice_{k}"] = log_dice_tra[e, :, k].mean().item()
                metrics[f"val/dice_{k}"] = log_dice_val[e, :, k].mean().item()
                metrics[f"train/hd95_{k}"] = gated_dice(
                    log_hd95_tra[e, :, k], log_present_tra[e, :, k]
                ).item()
                metrics[f"val/hd95_{k}"] = gated_dice(
                    log_hd95_val[e, :, k], log_present_val[e, :, k]
                ).item()
        wandb.log(metrics)

        # Scheduler at the end of each epoch
        scheduler.step()

        # np.save(result_dir / "loss_tra.npy", log_loss_tra)
        # np.save(result_dir / "dice_tra.npy", log_dice_tra)
        # np.save(result_dir / "loss_val.npy", log_loss_val)
        # np.save(result_dir / "dice_val.npy", log_dice_val)
        #

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(result_dir / "best_epoch.txt", "a") as f:
                f.write(message)

            # best_folder = result_dir / "best_epoch"
            # if best_folder.exists():
            #     rmtree(best_folder)
            # copytree(result_dir / f"iter{e:03d}", Path(best_folder))
            #
            torch.save(net.state_dict(), result_dir / "bestweights.pt")


def main():
    config = get_config()

    # Seed everything right at the beginning
    seed_all(config.seed, config.gpu)

    pprint(dataclasses.asdict(config))

    # Setup wandb
    wandb.init(
        entity="ai-for-medical-imaging",
        project=f"{config.dataset.name}-{'3D' if config.is_3d else '2D'}",
        config=dataclasses.asdict(config),
        dir=autoroot.root / "results" / "wandb",
        notes=config.notes,
    )

    try:
        runTraining(config)
    except:
        wandb.finish()


if __name__ == "__main__":
    main()
