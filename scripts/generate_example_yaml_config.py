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


def main():
    config = Config()

    with open(autoroot.root / "configs/example.yaml", "w") as fp:
        yaml.dump(dataclasses.asdict(config), fp, Dumper=_PlainDumper)


if __name__ == "__main__":
    main()
