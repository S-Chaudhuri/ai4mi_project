import autoroot  # noqa: F401  (keep first, it makes `from src...` imports work)

from src.utils.config import get_config
from src.eval_postprocess import run_sweep


def main():
    config = get_config()
    run_sweep(config)


if __name__ == "__main__":
    main()