import importlib
import json
from argparse import ArgumentParser

from .checkpoint import Checkpoint


def clean_version_name(name):
    """Filter wout these weird version names like
    torch==2.6.0+cu124
    """
    if "+" in name:
        return name.split("+")[0]
    return name


def inspect():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "-c", type=str, dest="checkpoint_path", required=True, help="Path to checkpoint"
    )
    args, _ = parser.parse_known_args()

    # Load checkpoint
    checkpoint = Checkpoint(args.checkpoint_path)

    # Print data
    print(
        f"Checkpoint created with Python {checkpoint.metadata.provenance_training.python}"
    )
    print("Checkpoint version\t", checkpoint.metadata.version)
    print("checkpoint run_id\t", checkpoint.metadata.run_id)
    print("checkpoint timestamp\t", checkpoint.metadata.timestamp)
    print("checkpoint multistep\t", checkpoint.multistep)

    print("checkpoint variables", json.dumps(checkpoint.index_to_name, indent=4))

    print("\nFor each module, checking if we have matching version installed...")
    modules_with_wrong_version = []
    for module in checkpoint.metadata.provenance_training.module_versions:
        if args.debug:
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
            print(f"  Warning: Could not find module <{module}>.")
            modules_with_wrong_version.append(
                f"{module}=={clean_version_name(checkpoint.metadata.provenance_training.module_versions[module])}"
            )

    if len(modules_with_wrong_version) > 0:
        print("Done.\n\nTo install correct versions, run:\n")
        print(f"pip install {' '.join(modules_with_wrong_version)}")
        print("\nThen test again to make sure.")
    else:
        print("Done.\n\nAll modules are correct version.")


if __name__ == "__main__":
    inspect()
