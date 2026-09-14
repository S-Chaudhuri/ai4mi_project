import json
from argparse import Namespace
from pathlib import Path


def store_args(
    step_name: str,
    args: Namespace,
    dest_path: Path,
    config_filename: str = "config.json",
):
    args_dict = vars(args)

    # Get the path to the config record and make sure for open(..., "r") that it exists
    config_path = Path(dest_path) / config_filename
    config_path.touch(exist_ok=True)

    with open(config_path, "r") as f:
        try:
            current_config = json.load(f)

            if step_name in current_config:
                # Means that the preprocessing step has already been
                # done on this, probably just raise but need to
                raise ValueError("Step already recorded in config record")

            current_config[step_name] = args_dict

        except json.JSONDecodeError:
            # If there is no config record, just build it
            current_config = {step_name: args_dict}

    with open(config_path, "w") as f:
        json.dump(current_config, f)
