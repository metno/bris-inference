import importlib
import json
import sys
from argparse import ArgumentParser

from .checkpoint import Checkpoint
from .forcings import anemoi_dynamic_forcings
from .model import get_variable_indices


def clean_version_name(name: str) -> str:
    """Filter wout these weird version names like
    torch==2.6.0+cu124
    """
    if "+" in name:
        return name.split("+")[0]
    return name


def check_module_versions(checkpoint: Checkpoint, debug: bool = False) -> list:
    """List installed module versions that doesn't match versions in the checkpoint."""
    modules_with_wrong_version = []
    for module in checkpoint.metadata.provenance_training.module_versions:
        if debug:
            print(
                f"  {module} is version\t{checkpoint.metadata.provenance_training.module_versions[module]}"
            )

        if "_remote_module_non_scriptable" in module:
            continue

        try:
            m = importlib.import_module(module)
            if clean_version_name(
                checkpoint.metadata.provenance_training.module_versions[module]
            ) != clean_version_name(m.__version__):
                if debug:
                    print(
                        f"  Warning: Installed version of {module} is <{m.__version__}>, while "
                        f"checkpoint was created with <{checkpoint.metadata.provenance_training.module_versions[module]}>."
                    )
                modules_with_wrong_version.append(
                    f"{module}=={clean_version_name(checkpoint.metadata.provenance_training.module_versions[module])}"
                )
        except AttributeError:
            print(f"  Error: Could not find version for module <{module}>.")
            modules_with_wrong_version.append(
                f"{module}=={clean_version_name(checkpoint.metadata.provenance_training.module_versions[module])}"
            )
        except ModuleNotFoundError:
            if debug:
                print(f"  Warning: Could not find module <{module}>.")
            modules_with_wrong_version.append(
                f"{module}=={clean_version_name(checkpoint.metadata.provenance_training.module_versions[module])}"
            )
    return modules_with_wrong_version


def get_required_variables(checkpoint: Checkpoint) -> list:
    """Get list of required variables from a checkpoint."""

    required_prognostic_variables = [
        name
        for name, index in checkpoint.name_to_index[0].items()
        if index in checkpoint.metadata.data_indices.internal_model.input.prognostic
    ]
    required_forcings = [
        name
        for name, index in checkpoint.name_to_index[0].items()
        if index in checkpoint.metadata.data_indices.internal_model.input.forcing
    ]
    required_static_forcings = [
        forcing
        for forcing in required_forcings
        if forcing not in anemoi_dynamic_forcings()
    ]

    return required_prognostic_variables + required_static_forcings


def inspect(checkpoint_path: str, debug: bool = False) -> bool:
    """Inspect a checkpoint and check if all modules are installed with correct versions."""

    # Load checkpoint
    checkpoint = Checkpoint(checkpoint_path)

    print(
        f"Checkpoint created with Python {checkpoint.metadata.provenance_training.python}"
    )
    print("Checkpoint version\t", checkpoint.metadata.version)
    print("checkpoint run_id\t", checkpoint.metadata.run_id)
    print("checkpoint timestamp\t", checkpoint.metadata.timestamp)
    print("checkpoint multistep\t", checkpoint.multistep)

    print(
        "checkpoint variables", json.dumps(get_required_variables(checkpoint), indent=4)
    )

    print("\nFor each module, checking if we have matching version installed...")
    modules_with_wrong_version = check_module_versions(checkpoint, debug)

    if len(modules_with_wrong_version) > 0:
        print("Done.\n\nTo install correct versions, run:\n")
        print(f"pip install {' '.join(modules_with_wrong_version)}")
        print("\nThen test again to make sure.")
        return False
    print("Done.\n\nAll modules are correct version.")
    return True


def main():
    """Parse arguments and run inspect."""
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        dest="checkpoint_path",
        required=True,
        help="Path to checkpoint",
    )
    args, _ = parser.parse_known_args()
    sys.exit(inspect(args.checkpoint_path, args.debug))


if __name__ == "__main__":
    main()
