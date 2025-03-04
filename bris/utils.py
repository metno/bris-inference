import logging
import json
import jsonschema
import yaml
import numbers
import os
import re
import time
import uuid
from argparse import ArgumentParser

import numpy as np
from anemoi.utils.config import DotDict
from omegaconf import OmegaConf


LOGGER = logging.getLogger(__name__)


def expand_time_tokens(filename: str, unixtime: int):
    """Expand time tokens in a filename and return absolute path."""
    if not isinstance(unixtime, numbers.Number):
        raise ValueError(f"Unixtime but be numeric not {unixtime}")

    return os.path.abspath(time.strftime(filename, time.gmtime(unixtime)))


def create_directory(filename: str):
    """Creates all sub directories necessary to be able to write filename"""
    directory = os.path.dirname(filename)
    if directory != "":
        os.makedirs(directory, exist_ok=True)


def is_number(value):
    """Check if value is a number."""
    return isinstance(value, numbers.Number)


def get_workdir(path: str) -> str:
    """If SLURM_PROCID is set, return path/SLURM_JOB_ID, else return path/<a uuid>."""
    if "SLURM_PROCID" in os.environ:
        return f"{path}/{os.environ["SLURM_JOB_ID"]}"
    return f"{path}/{uuid.uuid4()}"


def check_anemoi_training(metadata: DotDict) -> bool:
    assert isinstance(
        metadata, DotDict
    ), f"Expected metadata to be a DotDict, got {type(metadata)}"
    return hasattr(metadata.provenance_training, "module_versions") and \
        hasattr(metadata.provenance_training.module_versions, "anemoi.training")


def create_config(parser: ArgumentParser) -> OmegaConf:
    args, _ = parser.parse_known_args()

    validate(args.config, raise_on_error=True)

    try:
        config = OmegaConf.load(args.config)
        LOGGER.debug("config file from %s is loaded", args.config)
    except Exception as e:
        raise e

    parser.add_argument(
        "-c", type=str, dest="checkpoint_path", default=config.checkpoint_path
    )
    parser.add_argument("-sd", type=str, dest="start_date", required=False,
        default=config.start_date if "start_date" in config else None)
    parser.add_argument("-ed", type=str, dest="end_date", default=config.end_date)
    parser.add_argument(
        "-p", type=str, dest="dataset_path", help="Path to dataset", default=None
    )
    parser.add_argument(
        "-wd", type=str, dest="workdir", help="Path to work directory", required=False,
        default=config.workdir if "workdir" in config else None
    )

    parser.add_argument(
        "-pc",
        type=str,
        dest="dataset_path_cutout",
        nargs="*",
        help="List of paths for the input datasets in a cutout dataset",
        default=None,
        const=None,
    )
    # TODO: Logic that can add dataset or cutout dataset to the dataloader config

    parser.add_argument("-f", type=str, dest="frequency", default=config.frequency)
    parser.add_argument("-l", type=int, dest="leadtimes", default=config.leadtimes)
    args = parser.parse_args()

    args_dict = vars(args)

    # TODO: change start_date and end_date to numpy datetime, https://github.com/metno/bris-inference/issues/53
    return OmegaConf.merge(config, OmegaConf.create(args_dict))


def datetime_to_unixtime(dt: np.datetime64) -> np.typing.NDArray[int]:
    """Convert a np.datetime64 object or list of objects to unixtime"""
    return np.array(dt).astype("datetime64[s]").astype("int")


def unixtime_to_datetime(ut: int) -> np.datetime64:
    """Convert unixtime to a np.datetime64 object."""
    return np.datetime64(ut, "s")


def validate(filename: str, raise_on_error: bool = False) -> None:
    """Validate config file against a json schema."""
    schema_filename = os.path.dirname(os.path.abspath(__file__)) + "/schema/schema.json"
    with open(schema_filename, encoding="utf-8") as file:
        schema = json.load(file)

    with open(filename, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        if raise_on_error:
            raise
        print("WARNING: Schema does not validate")
        print(e)


def recursive_list_to_tuple(data):
    if isinstance(data, list):
        return tuple(recursive_list_to_tuple(item) for item in data)
    return data


def get_usable_indices(
    missing_indices: set[int] | None,
    series_length: int,
    rollout: int,
    multistep: int,
    timeincrement: int = 1,
) -> np.ndarray:
    """Get the usable indices of a series with missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Dataset to be used.
    series_length : int
        Length of the series.
    rollout : int
        Number of steps to roll out.
    multistep : int
        Number of previous indices to include as predictors.
    timeincrement : int
        Time increment, by default 1.

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    prev_invalid_dates = (multistep - 1) * timeincrement
    next_invalid_dates = rollout * timeincrement

    usable_indices = np.arange(series_length)  # set of all indices

    if missing_indices is None:
        missing_indices = set()

    missing_indices |= {-1, series_length}  # to filter initial and final indices

    # Missing indices
    for i in missing_indices:
        usable_indices = usable_indices[
            (usable_indices < i - next_invalid_dates) + (usable_indices > i + prev_invalid_dates)
        ]

    return usable_indices


def get_base_seed(env_var_list=("AIFS_BASE_SEED", "SLURM_JOB_ID")) -> int:
    """Gets the base seed from the environment variables.

    Option to manually set a seed via export AIFS_BASE_SEED=xxx in job script
    """
    base_seed = None
    for env_var in env_var_list:
        if env_var in os.environ:
            base_seed = int(os.environ.get(env_var, default=-1))
            break
    else: # No break from for loop
        raise AssertionError (f"Base seed not found in environment variables {env_var_list}")

    if base_seed < 1000:
        base_seed = base_seed * 1000  # make it (hopefully) big enough

    return base_seed
