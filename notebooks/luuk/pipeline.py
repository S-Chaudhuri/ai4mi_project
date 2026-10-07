from dataclasses import dataclass
import hashlib

import autorootcwd  # noqa
from pathlib import Path
import json


def load_json(path: Path):
    with open(path, "r") as fp:
        return json.load(fp)


def update_json(path: Path, hashmap: dict):
    with open(path, "w") as fp:
        json.dump(hashmap, fp)


class CacheMap:
    def __init__(self) -> None:
        self.json_path = Path("data/cache_map.json")
        self.hashmap: dict[str, str] = load_json(self.json_path)

    def get(self, hash: str) -> Path | None:
        value = self.hashmap.get(hash)

        if value is None:
            return None

        return Path(value)

    def add(self, key: str, value: Path):
        self.hashmap[key] = str(value)
        update_json(self.json_path, self.hashmap)


@dataclass
class Preprocessing:
    num_extra_patients: int = 10
    seed: int = 21


@dataclass
class Config:
    preprocessing: Preprocessing
    dataset: str = "SEGTHOR"
    seed: int = 42


def hash_config(preprocessing: Preprocessing):
    return hashlib.md5(str(preprocessing).encode(), usedforsecurity=False).hexdigest()


def run_pipeline(config: Config) -> Path:
    hash = hash_config(config.preprocessing)

    dataset_dir = Path("data/prepared_datasets") / hash

    if dataset_dir.exists(follow_symlinks=False):
        print(f"{dataset_dir} exists!")
        return dataset_dir

    dataset_dir.mkdir(parents=True)

    dataset = config.preprocessing.num_extra_patients * config.preprocessing.seed

    (dataset_dir / f"{dataset}").touch(exist_ok=False)

    print(dataset_dir)

    return dataset_dir


def main():
    import tyro

    run_pipeline(tyro.cli(Config))


if __name__ == "__main__":
    main()
