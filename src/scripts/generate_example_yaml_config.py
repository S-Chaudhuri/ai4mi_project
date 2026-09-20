import dataclasses

import autoroot  # noqa

import yaml
from src.utils.config import Config


def main():
    config = Config()

    with open(autoroot.root / "configs/example.yaml", "w") as fp:
        yaml.dump(dataclasses.asdict(config), fp)


if __name__ == "__main__":
    main()
