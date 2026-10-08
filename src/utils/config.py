import argparse
from pathlib import Path
from typing import Literal, Optional, Union, get_args, get_origin, get_type_hints
import autoroot
import yaml

import dataclasses
from dataclasses import dataclass, field

import torch
import tyro


@dataclass
class PathConfig:
    """Filesystem locations for data and results."""

    # Root directory containing the datasets, one subfolder per dataset name.
    data_path: Path = autoroot.root / "data"

    # Destination directory to save the results (predictions and weights).
    # If None, defaults to results/<Dataset>/<timestamp>.
    dest: Optional[Path] = None


@dataclass
class DatasetConfig:
    """Which dataset to train on and how it is sampled."""

    # Name of the dataset, must match a folder under paths.data_path.
    name: Literal["TOY2", "SEGTHOR"] = "SEGTHOR"

    # Number of segmentation classes (including background).
    num_classes: int = 5

    # In-plane spatial size (H, W) that 2D slices are resized to.
    shape: tuple[int, int] = (256, 256)

    # Number of patients to retain per fold.
    retains: int = 5

    # Cross-validation fold to use.
    fold: int = 0

    # Spatial dimensions for 3D sub-box crops (Depth, Height, Width).
    box_size: tuple[int, int, int] = (132, 132, 128)

    # Probability that a sampled 3D training box contains foreground.
    fg_prob: float = 0.5


@dataclass
class ModelConfig:
    """Which network to build and its architecture settings."""

    # Model to use. ShallowNet/ENet are 2D, UNet3D/VNet3D/ResUNet3D/MedNeXt3D are 3D.
    name: Literal[
        "ShallowNet", "ENet", "UNet3D", "VNet3D", "ResUNet3D", "MedNeXt3D"
    ] = "ENet"

    # Toggle 3D volumetric sub-box pipeline. ShallowNet/ENet are 2D,
    # UNet3D/VNet3D/ResUNet3D/MedNeXt3D are 3D.
    is_3d: bool = False

    # Number of kernels in the first convolutional layer (doubles per stage).
    kernels: int = 8

    # Down-sampling factor between stages.
    factor: int = 2

    # Dropout rate used in the model.
    dropout: float = 0.01

    # Temperature applied to the logits before softmax (1 = no scaling).
    temperature: float = 1.0


@dataclass
class TrainingConfig:
    """Optimization, scheduling and loss settings."""

    # Number of training epochs.
    epochs: int = 20

    # Use the GPU if available.
    gpu: bool = True

    # Random seed, applied to python/numpy/torch/cuda.
    seed: int = 42

    # Batch size used for both train and validation loaders.
    batch_size: int = 8

    # For 3D, an epoch is a fixed number of batches drawn with replacement.
    batches_per_epoch: int = 20

    # Initial learning rate for AdamW.
    lr: float = 0.0005

    # Weight decay for AdamW.
    weight_decay: float = 0

    # AdamW momentum coefficients.
    betas: tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule: "cosine" (CosineAnnealingLR over epochs)
    # or "none" (constant lr).
    scheduler: Literal["cosine", "none"] = "cosine"

    # "full" supervises all classes, "partial" skips the heart class on
    # SEGTHOR.
    mode: Literal["partial", "full"] = "full"

    # Loss function to use.
    loss: Literal["ce", "dice_ce"] = "ce"

    # Weight of the dice term when loss is "dice_ce".
    dice_weight: float = 1.0


@dataclass
class RuntimeConfig:
    """Data-loading and execution settings."""

    # Keep only a fraction (10 samples) of the datasets, to test the logics
    # around epochs and logging easily.
    debug: bool = False

    # Number of worker processes per DataLoader.
    num_workers: int = 5

    # Also compute HD95 on training batches (expensive: scipy per sample/class).
    # Off by default: the train pass logs loss and dice only, validation still
    # reports HD95 every epoch.
    hd95_in_train: bool = False

    # Mixed-precision autocast dtype on GPU. "bf16" (default) is faster on
    # Hopper, needs no GradScaler and has no fp16 range issues. "fp16" keeps
    # the old behavior (GradScaler enabled, loss computed outside autocast).
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    # Let cuDNN benchmark convolution algorithms per shape. Faster, but
    # slightly non-deterministic; off by default to keep runs reproducible.
    cudnn_benchmark: bool = False


@dataclass
class AugmentationConfig:
    """Data augmentation settings."""

    # Probability of adding gaussian noise to a training image.
    noise_prob: float = 0.5

    # Standard deviation of the gaussian noise.
    noise_level: float = 0.05


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    # W&B entity (team or user) to log to.
    entity: str = "ai-for-medical-imaging"

    # W&B project name. If None, defaults to "<Dataset>-<2D/3D>".
    project: Optional[str] = None

    # Directory wandb writes its local files to.
    # If None, defaults to <root>/results/wandb.
    dir: Optional[Path] = None

    # Log gradient and parameter histograms via wandb.watch.
    watch: bool = False

    # Store the checksum of the dataset
    store_artifect: bool = True

    # Free-form notes attached to the wandb run.
    notes: Optional[str] = None


@dataclass
class ProfilerConfig:
    """Optional torch.profiler settings for the training loop (off by default)."""

    # Enable the torch profiler.
    enabled: bool = False

    # Record the shapes of tensors passed to each operator.
    record_shapes: bool = False

    # Record memory usage of each operator.
    profile_memory: bool = False

    # Record the stack traces that triggered each operator.
    with_stack: bool = False

    # Estimate the FLOPS of each operator.
    with_flops: bool = False

    # Record the module hierarchy of each operator.
    with_modules: bool = False

    # Schedule in batch steps, cycling wait -> warmup -> active.
    wait: int = 1

    # Schedule in batch steps, cycling wait -> warmup -> active.
    warmup: int = 1

    # Schedule in batch steps, cycling wait -> warmup -> active.
    active: int = 3

    # Where to write chrome traces (None -> <result_dir>/profiler).
    output_dir: Optional[Path] = None


@dataclass
class Config:
    """Top-level training configuration, composed from the sub-configs below."""

    # Filesystem locations (data root, result destination).
    paths: PathConfig = field(default_factory=PathConfig)

    # Dataset selection and sampling settings.
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Network architecture and model-related settings.
    model: ModelConfig = field(default_factory=ModelConfig)

    # Optimization, scheduling and loss settings.
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Data-loading and execution settings.
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Data augmentation settings.
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Weights & Biases logging settings.
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # torch.profiler settings.
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


def _unwrap_optional(expected):
    # Optional[X] is Union[X, None]; reduce it to X when unambiguous
    if get_origin(expected) is Union:
        non_none = [a for a in get_args(expected) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return expected


def _coerce_value(value, expected):
    # Cast plain values loaded from yaml to the annotated type
    if value is None:
        return value

    origin = get_origin(expected)
    if origin is tuple:
        return tuple(value)
    if origin is list:
        return list(value)
    if expected is Path:
        return Path(value)
    if expected is bool:
        return bool(value)
    if expected is int:
        return int(value)
    if expected is float:
        return float(value)
    if expected is str:
        return str(value)
    return value


def instantiate_dataclass(cls, data: dict):
    if not dataclasses.is_dataclass(cls):
        return data

    field_names = {f.name for f in dataclasses.fields(cls)}
    field_types = get_type_hints(cls)
    kwargs = {}

    for key, value in data.items():
        # Ignore keys in yaml that don't exist in the dataclass
        if key not in field_names:
            continue

        expected_type = _unwrap_optional(field_types[key])

        # If the expected field is a nested dataclass, instantiate it recursively
        if dataclasses.is_dataclass(expected_type) and isinstance(value, dict):
            kwargs[key] = instantiate_dataclass(expected_type, value)
        else:
            kwargs[key] = _coerce_value(value, expected_type)

    return cls(**kwargs)  # type: ignore


def _load_config_file():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="")
    args, remaining_argv = parser.parse_known_args()

    config_path = Path(args.config)
    if args.config != "" and config_path.exists():
        with open(config_path, "r") as fp:
            return yaml.safe_load(fp) or {}, remaining_argv

    return {}, remaining_argv


def get_config() -> Config:
    yaml_dict, remaining_argv = _load_config_file()

    base_inst = Config()
    if yaml_dict:
        base_inst = instantiate_dataclass(Config, yaml_dict)

    config = tyro.cli(Config, default=base_inst, args=remaining_argv)
    if isinstance(config, dict):
        raise Exception("config was loaded as a dict")

    # Make sure a gpu is available if configured to use
    config.training.gpu = config.training.gpu and torch.cuda.is_available()

    return config
