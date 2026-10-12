import argparse
import dataclasses
import math
import os
from pathlib import Path
import random
import re
import secrets
from typing import Union, get_args, get_origin, get_type_hints

import yaml


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


_SWEEP_SPEC_RE = re.compile(r"^<(\w+)\s+(.+)>$")


def parse_sweep_spec(value):
    """Return (dist, args) if value is a sweep spec like '<loguniform 1e-5 5e-4>'.

    Returns None for everything else, so plain values pass through unchanged.
    """
    if not isinstance(value, str):
        return None
    match = _SWEEP_SPEC_RE.match(value.strip())
    if not match:
        return None
    return match.group(1), match.group(2).split()


def sample_sweep_spec(dist: str, args: list[str], field_name: str, rng):
    """Draw one concrete value from the distribution named by dist."""
    if dist == "uniform":
        if len(args) != 2:
            raise ValueError(f"{field_name}: <uniform low high> expects 2 args")
        return rng.uniform(float(args[0]), float(args[1]))
    if dist in ("loguniform", "uniformexp"):
        if len(args) != 2:
            raise ValueError(f"{field_name}: <{dist} low high> expects 2 args")
        low, high = float(args[0]), float(args[1])
        if low <= 0 or high <= 0:
            raise ValueError(
                f"{field_name}: <{dist}> requires positive bounds, got {low} {high}"
            )
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    if dist == "intuniform":
        if len(args) != 2:
            raise ValueError(f"{field_name}: <intuniform low high> expects 2 args")
        return rng.randint(int(args[0]), int(args[1]))
    if dist == "gaussian":
        if len(args) != 2:
            raise ValueError(f"{field_name}: <gaussian mean std> expects 2 args")
        return rng.gauss(float(args[0]), float(args[1]))
    if dist == "choice":
        if len(args) < 1:
            raise ValueError(f"{field_name}: <choice ...> needs at least 1 option")
        return rng.choice(args)
    raise ValueError(
        f"{field_name}: unknown sweep distribution {dist!r} "
        f"(supported: uniform, loguniform, intuniform, gaussian, choice)"
    )


def instantiate_dataclass(cls, data: dict, rng=None, record=None, path=""):
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = get_type_hints(cls)
    kwargs = {}

    # Iterate in dataclass declaration order for reproducability (instead of yaml)
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        dotted = f"{path}{f.name}"
        expected_type = _unwrap_optional(field_types[f.name])

        if rng is not None and (spec := parse_sweep_spec(value)) is not None:
            if dataclasses.is_dataclass(expected_type):
                raise ValueError(f"{dotted}: sweep specs only apply to scalar fields")
            dist, args = spec
            kwargs[f.name] = _coerce_value(
                sample_sweep_spec(dist, args, dotted, rng), expected_type
            )
            if record is not None:
                record[dotted] = {
                    "spec": f"<{dist} {' '.join(args)}>",
                    "value": kwargs[f.name],
                }
            continue

        # If the expected field is a nested dataclass, instantiate it recursively
        if dataclasses.is_dataclass(expected_type) and isinstance(value, dict):
            kwargs[f.name] = instantiate_dataclass(
                expected_type, value, rng=rng, record=record, path=f"{dotted}."
            )
        else:
            kwargs[f.name] = _coerce_value(value, expected_type)

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


def _make_sweep_rng():
    # Seed the sweep rng from the SWEEP_SEED env var when set (e.g.
    # SWEEP_SEED=$SLURM_ARRAY_TASK_ID), otherwise draw a random one.
    seed_env = os.environ.get("SWEEP_SEED")
    if seed_env is not None:
        return random.Random(int(seed_env)), int(seed_env)
    seed = secrets.randbits(32)
    return random.Random(seed), seed
