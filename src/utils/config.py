import argparse
from pathlib import Path
from typing import Literal, Optional, Union, get_args, get_origin, get_type_hints
import yaml

import dataclasses
from dataclasses import dataclass, field

import torch
import tyro


@dataclass
class DatasetConfig:
    name: Literal["TOY2", "SEGTHOR"] = "SEGTHOR"

    num_classes: int = 5

    shape: tuple[int, int] = (256, 256)
    retains: int = 5
    fold: int = 0

    box_size: tuple[int, int, int] = (132, 132, 128)


@dataclass
class ModelConfig:
    name: Literal["ShallowNet", "ENet", "UNet3D"] = "ENet"

    kernels: int = 8
    factor: int = 2


@dataclass
class ProfilerConfig:
    """Optional torch.profiler settings for the training loop (off by default)."""

    enabled: bool = False

    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False

    # Schedule in batch steps, cycling wait -> warmup -> active
    wait: int = 1
    warmup: int = 1
    active: int = 3

    # Where to write chrome traces (None -> <result_dir>/profiler)
    output_dir: Optional[Path] = None


@dataclass
class Config:
    # Destination directory to save the results (predictions and weights).
    dest: Optional[Path] = None

    # Toggle 3D Volumetric Sub-Box Pipeline
    is_3d: bool = False

    # Spatial dimensions for 3D sub-box crops (Depth, Height, Width)
    sub_box_size: tuple[int, int, int] = (128, 128, 128)

    # The dataset to train on
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    model: ModelConfig = field(default_factory=ModelConfig)

    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    epochs: int = 20

    mode: Literal["partial", "full"] = "full"

    loss: Literal["ce", "dice_ce"] = "ce"

    dice_weight: float = 1.0

    gpu: bool = True

    num_workers: int = 5

    # Keep only a fraction (10 samples) of the datasets, to test the logics around epochs and logging easily.
    debug: bool = False

    wandb_watch: bool = False

    seed: int = 42

    lr: float = 0.0005
    weight_decay: float = 0
    batch_size: int = 8
    betas: tuple[float, float] = (0.9, 0.999)
    dropout: float = 0.01
    temperature: float = 1

    noise_prob: float = 0.5
    noise_level: float = 0.05

    notes: Optional[str] = None


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
    config.gpu = config.gpu and torch.cuda.is_available()

    return config
