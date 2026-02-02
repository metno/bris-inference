import json
import logging
import numbers
import os
import sys
import time
import uuid
from argparse import ArgumentParser
from collections.abc import Iterable
from typing import Any

import jsonschema
import numpy as np
import torch
import yaml
from anemoi.models.data_indices.index import DataIndex, ModelIndex
from anemoi.utils.config import DotDict
from omegaconf import DictConfig, ListConfig, OmegaConf

from .forcings import anemoi_dynamic_forcings, get_dynamic_forcings

LOGGER = logging.getLogger("bris")


def expand_time_tokens(filename: str, unixtime: int) -> str:
    """Expand time tokens in a filename and return absolute path."""
    if not isinstance(unixtime, numbers.Number):
        raise ValueError(f"Unixtime must be numeric not {unixtime}")

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
        return f"{path}/{os.environ['SLURM_JOB_ID']}"
    return f"{path}/{uuid.uuid4()}"


def check_anemoi_training(metadata: DotDict) -> bool:
    assert isinstance(metadata, DotDict), (
        f"Expected metadata to be a DotDict, got {type(metadata)}"
    )
    return hasattr(metadata.provenance_training, "module_versions") and hasattr(
        metadata.provenance_training.module_versions, "anemoi.training"
    )


# def check_anemoi_dataset_version(metadata) -> tuple[bool, str]:
#     """Not currently in use, but can be handy for testing, debugging."""
#     assert isinstance(metadata, DotDict), (
#         f"Expected metadata to be a DotDict, got {type(metadata)}"
#     )
#     if hasattr(metadata.provenance_training, "module_versions"):
#         try:
#             _version = metadata.provenance_training.module_versions["anemoi.datasets"]
#             _version = re.match(r"^\d+\.\d+\.\d+", _version).group()
#             if _version < "0.5.0":
#                 return True, _version
#             return False, _version
#         except AttributeError as e:
#             raise e
#     else:
#         raise RuntimeError("metadata.provenance_training does not module_versions")


def parse_args(arg_list: list[str] | None) -> ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument(
        "-sd",
        type=str,
        dest="start_date",
        required=False,
    )
    parser.add_argument("-ed", type=str, dest="end_date", required=False)
    parser.add_argument(
        "-wd",
        type=str,
        dest="workdir",
        help="Path to work directory",
        required=False,
    )
    parser.add_argument("-f", type=str, dest="frequency", required=False)
    parser.add_argument(
        "-l",
        type=int,
        dest="checkpoints.forecaster.leadtimes",
    )

    # If passed a list, will parse it. If passed None, will parse sys.argv
    args, _ = parser.parse_known_args(arg_list)

    # Don't return None values, as they will override the config file
    return {k: v for k, v in args.__dict__.items() if v is not None}


def create_config(config_path: str, overrides: dict) -> DictConfig | ListConfig:
    # Validate config file
    validate(config_path, raise_on_error=True)

    config = OmegaConf.load(config_path)
    # Set hydra defaults
    config.defaults = [
        {"override hydra/job_logging": "none"},  # disable config parsing logs
        {"override hydra/hydra_logging": "none"},  # disable config parsing logs
        "_self_",
    ]
    return OmegaConf.merge(config, OmegaConf.create(overrides))


def setup_logging(config: DotDict) -> None:
    # Set up logging
    console_handler = logging.StreamHandler(sys.stdout)
    LOGGER.setLevel(logging.DEBUG)
    # Create a formatter for the logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    if "loglevel" in config and (
        "debug" in config.loglevel.lower() or "verbose" in config.loglevel.lower()
    ):
        console_handler.setLevel(logging.DEBUG)

    # Add the handler to the LOGGER
    LOGGER.addHandler(console_handler)
    LOGGER.debug(
        f"config file from {config.config} loaded with log level {console_handler.level}"
    )


def datetime_to_unixtime(
    dt: list[np.datetime64],
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Convert a np.datetime64 object or list of objects to unixtime"""
    return np.array(dt).astype("datetime64[s]").astype("int")


def unixtime_to_datetime(ut: int) -> np.datetime64:
    """Convert unixtime to a np.datetime64 object."""
    return np.datetime64(ut, "s")


def timedelta64_from_timestep(timestep):
    if isinstance(timestep, str) and timestep[-1] in ("h", "m", "s"):
        return np.timedelta64(timestep[0:-1], timestep[-1])

    LOGGER.warning(
        "could not decode model timestep from checkpoint, trying to assume hours"
    )
    return np.timedelta64(timestep, "h")


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
        print("WARNING: Schema does not validate")
        print(e)
        if raise_on_error:
            raise


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
            (usable_indices < i - next_invalid_dates)
            + (usable_indices > i + prev_invalid_dates)
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
    else:  # No break from for loop
        raise AssertionError(
            f"Base seed not found in environment variables {env_var_list}"
        )

    if base_seed < 1000:
        base_seed = base_seed * 1000  # make it (hopefully) big enough

    return base_seed


def set_encoder_decoder_num_chunks(chunks: int = 1) -> None:
    assert isinstance(chunks, int), (
        f"Expecting chunks to be int, got: {chunks}, {type(chunks)}"
    )
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(chunks)
    LOGGER.info("Encoder and decoder are chunked to %s", chunks)


def set_base_seed() -> None:
    """
    Sets os environment variables ANEMOI_BASE_SEED and AIFS_BASE_SEED.
    """
    os.environ["ANEMOI_BASE_SEED"] = "1234"
    os.environ["AIFS_BASE_SEED"] = "1234"
    LOGGER.info("ANEMOI_BASE_SEED and ANEMOI_BASE_SEED set to 1234")


def get_all_leadtimes(
    leadtimes_forecaster: int,
    timestep_forecaster: int,
    leadtimes_interpolator: int = 0,
    timestep_interpolator: int = 3600,
) -> np.ndarray:
    """
    Calculates all the leadtimes in the output with combined forecaster and interpolator.
    """
    high_res = (
        np.arange(leadtimes_interpolator * timestep_forecaster // timestep_interpolator)
        * timestep_interpolator
    )
    low_res = np.arange(
        (leadtimes_interpolator * timestep_forecaster),
        (leadtimes_forecaster * timestep_forecaster),
        timestep_forecaster,
    )

    return np.concatenate([high_res, low_res])

def get_dataset_config(config: DictConfig) -> DictConfig:
    """Convert dataset config to the correct format."""
    if "dataset" in config:
        ds_cfg = {
            "data": {
                "dataset": config.dataset,
                "start": config.start_date,
                "end": config.end_date,
                "frequency": config.frequency,
            }
        }
    elif "datasets" in config:
        ds_cfg = {}
        for dataset_name, dataset_recipe in config.datasets.items():
            ds_cfg[dataset_name] = {
                "dataset": dataset_recipe,
                "start": config.start_date,
                "end": config.end_date,
                "frequency": config.frequency,
            }
    else:
        raise ValueError("Config must contain either 'dataset' or 'datasets' key.")
    
    return OmegaConf.create(ds_cfg)