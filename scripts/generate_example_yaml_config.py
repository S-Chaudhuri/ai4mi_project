import dataclasses
from pathlib import Path

import autoroot  # noqa

import yaml
from src.utils.config import Config


class _PlainDumper(yaml.SafeDumper):
    """Dump Path and tuple in plain yaml so the file stays safe_load-able."""


def _represent_path(dumper: yaml.Dumper, path: Path) -> yaml.ScalarNode:
    return dumper.represent_str(str(path))


def _represent_tuple(dumper: yaml.Dumper, tup: tuple) -> yaml.SequenceNode:
    return dumper.represent_list(list(tup))


_PlainDumper.add_multi_representer(Path, _represent_path)
_PlainDumper.add_representer(tuple, _represent_tuple)


def generate_yaml_config(config: Config, path: Path):
    with open(path, "w") as fp:
        yaml.dump(dataclasses.asdict(config), fp, Dumper=_PlainDumper)


def main():
    generate_yaml_config(Config(), autoroot.root / "configs/example_config.yaml")


if __name__ == "__main__":
    main()
