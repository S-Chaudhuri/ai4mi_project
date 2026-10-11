"""Tests for sweep specs in yaml configs: <uniform/loguniform/intuniform/gaussian/choice ...>."""

import os
import random
import sys
import tempfile
from pathlib import Path

import autoroot  # noqa

from src.utils.config import get_config, parse_sweep_spec, sample_sweep_spec

# --- parse_sweep_spec ---------------------------------------------------------

assert parse_sweep_spec("<loguniform 1e-5 5e-4>") == ("loguniform", ["1e-5", "5e-4"])
assert parse_sweep_spec("<choice a b>") == ("choice", ["a", "b"])
assert parse_sweep_spec("<choice 8 16 32>") == ("choice", ["8", "16", "32"])
assert parse_sweep_spec("<uniform  0.0  0.1  >") == ("uniform", ["0.0", "0.1"])
assert parse_sweep_spec("cosine") is None
assert parse_sweep_spec("0.0005") is None
assert parse_sweep_spec(0.0005) is None
assert parse_sweep_spec("<bad>") is None  # no args
assert parse_sweep_spec("foo bar") is None  # no brackets
print("parse_sweep_spec: ok")


# --- sample_sweep_spec --------------------------------------------------------

rng = random.Random(0)
for _ in range(100):
    assert 0.0 <= sample_sweep_spec("uniform", ["0.0", "0.1"], "f", rng) <= 0.1
for _ in range(100):
    v = sample_sweep_spec("loguniform", ["1e-5", "5e-4"], "f", rng)
    assert 1e-5 <= v <= 5e-4
    v = sample_sweep_spec("uniformexp", ["1e-5", "5e-4"], "f", rng)
    assert 1e-5 <= v <= 5e-4
for _ in range(100):
    v = sample_sweep_spec("intuniform", ["10", "30"], "f", rng)
    assert isinstance(v, int) and 10 <= v <= 30
assert isinstance(sample_sweep_spec("gaussian", ["0.1", "0.05"], "f", rng), float)
for _ in range(100):
    assert sample_sweep_spec("choice", ["ce", "dice_ce"], "f", rng) in ("ce", "dice_ce")

try:
    sample_sweep_spec("uniformexp", ["0", "1"], "f", rng)
    raise AssertionError("expected ValueError for non-positive loguniform bounds")
except ValueError:
    pass
try:
    sample_sweep_spec("poisson", ["1"], "f", rng)
    raise AssertionError("expected ValueError for unknown distribution")
except ValueError:
    pass
try:
    sample_sweep_spec("uniform", ["0.0"], "f", rng)
    raise AssertionError("expected ValueError for wrong arg count")
except ValueError:
    pass
print("sample_sweep_spec: ok")


# --- end-to-end through get_config -------------------------------------------

SWEEP_YAML = """
training:
  lr: <loguniform 0.00001 0.0005>
  epochs: <intuniform 10 30>
  scheduler: <choice cosine none>
model:
  kernels: <choice 8 16 32>
"""

SWEEP_YAML_SHUFFLED = """
model:
  kernels: <choice 8 16 32>
training:
  scheduler: <choice cosine none>
  epochs: <intuniform 10 30>
  lr: <loguniform 0.00001 0.0005>
"""


def load_config(yaml_text, extra_argv=(), seed="1234", use_config=True):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fp:
        fp.write(yaml_text)
        path = fp.name

    old_argv = sys.argv
    old_seed = os.environ.get("SWEEP_SEED")
    argv = ["prog"] + (["--config", path] if use_config else []) + list(extra_argv)
    sys.argv = argv
    if seed is not None:
        os.environ["SWEEP_SEED"] = seed
    else:
        os.environ.pop("SWEEP_SEED", None)
    try:
        return get_config()
    finally:
        sys.argv = old_argv
        if old_seed is None:
            os.environ.pop("SWEEP_SEED", None)
        else:
            os.environ["SWEEP_SEED"] = old_seed
        Path(path).unlink()


cfg1 = load_config(SWEEP_YAML, seed="1234")
cfg2 = load_config(SWEEP_YAML, seed="1234")
cfg3 = load_config(SWEEP_YAML, seed="9999")
cfg_shuffled = load_config(SWEEP_YAML_SHUFFLED, seed="1234")

# Same seed -> same draws
assert cfg1.training.lr == cfg2.training.lr
assert cfg1.training.epochs == cfg2.training.epochs
assert cfg1.training.scheduler == cfg2.training.scheduler
assert cfg1.model.kernels == cfg2.model.kernels

# Different seed -> different draws
assert cfg1.training.lr != cfg3.training.lr

# Key order in the yaml file must not change the draws
assert cfg1.training.lr == cfg_shuffled.training.lr
assert cfg1.training.epochs == cfg_shuffled.training.epochs

# Ranges and types
assert isinstance(cfg1.training.lr, float) and 1e-5 <= cfg1.training.lr <= 5e-4
assert isinstance(cfg1.training.epochs, int) and 10 <= cfg1.training.epochs <= 30
assert cfg1.training.scheduler in ("cosine", "none")
assert isinstance(cfg1.model.kernels, int) and cfg1.model.kernels in (8, 16, 32)

# Record + seed exposed on the config
assert cfg1.sweep_seed == 1234
assert set(cfg1.sweep) == {
    "training.lr",
    "training.epochs",
    "training.scheduler",
    "model.kernels",
}
assert cfg1.sweep["training.lr"]["spec"] == "<loguniform 0.00001 0.0005>"
assert cfg1.sweep["training.lr"]["value"] == cfg1.training.lr
print("get_config sweep: ok")

# A CLI flag beats a swept yaml value
cfg_cli = load_config(SWEEP_YAML, extra_argv=["--training.lr", "0.01"], seed="1234")
assert cfg_cli.training.lr == 0.01
assert cfg_cli.training.epochs == cfg1.training.epochs  # untouched fields still sweep
print("CLI override wins: ok")

# No config file -> no sweep at all
cfg_noyaml = load_config("", use_config=False, seed=None)
assert cfg_noyaml.sweep == {}
assert cfg_noyaml.sweep_seed is None
assert cfg_noyaml.training.lr == 0.0005  # plain dataclass default
print("no-config run: ok")

print("\nAll sweep tests passed.")
