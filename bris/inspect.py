import importlib
import json
import sys
from argparse import ArgumentParser

from .checkpoint import Checkpoint
from .forcings import anemoi_dynamic_forcings


def clean_version_name(name: str) -> tuple[str, str]:
    """Filter out these weird version names that can't be compared, like torch==2.6.0+cu124.

    anemoi-models 0.1.dev92+g80c9fbf comes from https://github.com/metno/anemoi-models/commit/80c9fbf,
    but there's no automatic way to figure out which fork it's from.
    """
    clean_version = name
    version_hash = ""
    if "+g" in name:
        clean_version = name.split("+")[0]
        version_hash = name.split("+g")[1]
    if "+" in name:
        clean_version = name.split("+")[0]
    return clean_version, version_hash


def get_pip_name(name: str) -> str:
    """Workaround for modules with legacy name, like PIL is really Pillow. If
    not found, return input."""
    names = {"PIL": "pillow", "attr": "attrs"}
    return names.get(name, name)


def check_module_versions(
    checkpoint: Checkpoint, debug: bool = False
) -> tuple[list, list]:
    """List installed module versions that doesn't match versions in the checkpoint.
    First list are pip-installed modules. Second list have hashes."""
    pip_modules = []
    hash_modules = []

    for module in checkpoint.metadata.provenance_training.module_versions:
        if debug:
            print(
                f"  {module} in checkpoint was version\t{checkpoint.metadata.provenance_training.module_versions[module]}"
            )

        # Skip non-scriptable modules, modules from standard library.
        if module in [
            "_remote_module_non_scriptable",
            "hydra_plugins.anemoi_searchpath",
            "distutils",  # Standard library
        ]:
            continue

        try:  # Import each module to check version
            m = importlib.import_module(module)
            clean_version, version_hash = clean_version_name(
                checkpoint.metadata.provenance_training.module_versions[module]
            )
            if clean_version != clean_version_name(m.__version__)[0]:
                if debug:
                    print(
                        f"  Warning: Installed version of {module} is <{m.__version__}>, while "
                        f"checkpoint was created with <{checkpoint.metadata.provenance_training.module_versions[module]}>."
                    )

                if version_hash:
                    hash_modules.append(f"{get_pip_name(module)} hash: {version_hash}")
                else:
                    pip_modules.append(f"{get_pip_name(module)}=={clean_version}")

        except AttributeError:
            print(
                f"  Error: Could not find version for module <{get_pip_name(module)}>."
            )
            pip_modules.append(f"{module}=={clean_version}")
        except ModuleNotFoundError:
            if debug:
                print(f"  Warning: Could not find module <{module}>, please install.")
            pip_modules.append(f"{module}=={clean_version}")
    return pip_modules, hash_modules


def get_required_variables(checkpoint: Checkpoint) -> dict:
    """Get dict of datasets with list of required variables for each dataset."""

    # If simple checkpoint
    if len(checkpoint.data_indices) == 1:
        data_indices = checkpoint.data_indices[0]
        required_prognostic_variables = [
            name
            for name, index in data_indices.internal_model.input.name_to_index.items()
            if index in data_indices.internal_model.input.prognostic
        ]
        required_forcings = [
            name
            for name, index in data_indices.internal_model.input.name_to_index.items()
            if index in data_indices.internal_model.input.forcing
        ]
        required_static_forcings = [
            forcing
            for forcing in required_forcings
            if forcing not in anemoi_dynamic_forcings()
        ]
        return {0: required_prognostic_variables + required_static_forcings}

    # If Multiencdec checkpoint
    datasets = {}
    for i, data_indices in enumerate(checkpoint.data_indices):
        required_prognostic_variables = [
            name
            for name, index in data_indices.internal_model.input.name_to_index.items()
            if index in data_indices.internal_model.input.prognostic
        ]
        required_forcings = [
            name
            for name, index in data_indices.internal_model.input.name_to_index.items()
            if index in data_indices.internal_model.input.forcing
        ]
        required_static_forcings = [
            forcing
            for forcing in required_forcings
            if forcing not in anemoi_dynamic_forcings()
        ]
        datasets[i] = required_prognostic_variables + required_static_forcings
    return datasets


def inspect(checkpoint_path: str, debug: bool = False) -> int:
    """Inspect a checkpoint and check if all modules are installed with correct versions. Return exit status."""

    # Load checkpoint
    checkpoint = Checkpoint(checkpoint_path)

    print(
        f"Checkpoint created with\tPython {checkpoint.metadata.provenance_training.python}\n"
        f"Checkpoint version\t{checkpoint.metadata.version}\n"
        f"Checkpoint run_id\t{checkpoint.metadata.run_id}\n"
        f"Checkpoint timestamp\t{checkpoint.metadata.timestamp}\n"
        f"Checkpoint multistep\t{checkpoint.multistep}\n"
        f"Checkpoint required variables:\t{json.dumps(get_required_variables(checkpoint), indent=4)}"
    )

    pip_modules, hash_modules = check_module_versions(checkpoint, debug)
    print(
        "\nThe important module is <anemoi-models>, but showing all modules that differs."
    )
    if pip_modules:
        print(
            f"To install correct versions, run:\n   pip install {' '.join(pip_modules)}"
        )

    if hash_modules:
        print(
            "The following modules were not installed from a package. Visit "
            "the repo for each module and search for the hash. Install "
            "directly from hash using for example `pip install "
            "git+https://github.com/metno/example-module.git@80c9fbf`"
        )
        print("  " + "\n  ".join(hash_modules))

    if pip_modules or hash_modules:
        return 1
    print("\nAll modules are correct version.")
    return 0


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
