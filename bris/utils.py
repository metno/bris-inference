import json
import logging
import numbers
import os
import time
import uuid
from argparse import ArgumentParser

import jsonschema
import numpy as np
import torch
import yaml
from anemoi.models.data_indices.index import DataIndex, ModelIndex
from anemoi.utils.config import DotDict
from omegaconf import DictConfig, ListConfig, OmegaConf

from .forcings import anemoi_dynamic_forcings, get_dynamic_forcings

LOGGER = logging.getLogger(__name__)


def expand_time_tokens(filename: str, unixtime: int) -> str:
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


def create_config(parser: ArgumentParser) -> DictConfig | ListConfig:
    args, _ = parser.parse_known_args()

    validate(args.config, raise_on_error=True)

    config = OmegaConf.load(args.config)
    LOGGER.debug("config file from %s is loaded", args.config)

    parser.add_argument(
        "-sd",
        type=str,
        dest="start_date",
        required=False,
        default=config.start_date if "start_date" in config else None,
    )
    parser.add_argument("-ed", type=str, dest="end_date", default=config.end_date)
    parser.add_argument(
        "-p", type=str, dest="dataset_path", help="Path to dataset", default=None
    )
    parser.add_argument(
        "-wd",
        type=str,
        dest="workdir",
        help="Path to work directory",
        required=False,
        default=config.workdir if "workdir" in config else None,
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
    parser.add_argument(
        "-l",
        type=int,
        dest="checkpoints.forecaster.leadtimes",
        default=config.checkpoints.forecaster.leadtimes,
    )
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


def timedelta64_from_timestep(timestep):
    if isinstance(timestep, str) and timestep[-1] in ("h", "m", "s"):
        return np.timedelta64(timestep[0:-1], timestep[-1])

    print(
        "WARNING: could not decode model timestep from checkpoint, trying to assume hours"
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


def get_variable_indices(
    required_variables: list,
    datamodule_variables: list,
    internal_data: DataIndex,
    internal_model: ModelIndex,
    decoder_index: int,
) -> tuple[dict, dict]:
    # Set up indices for the variables we want to write to file
    variable_indices_input = []
    variable_indices_output = []
    for name in required_variables:
        variable_indices_input.append(internal_data.input.name_to_index[name])
        variable_indices_output.append(internal_model.output.name_to_index[name])

    # Set up indices that can map from the variable order in the input data to the input variable order expected by the model
    full_ordered_variable_list = [
        var
        for var, _ in sorted(
            internal_data.input.name_to_index.items(), key=lambda item: item[1]
        )
    ]

    required_prognostic_variables = [
        name
        for name, index in internal_model.input.name_to_index.items()
        if index in internal_model.input.prognostic
    ]
    required_forcings = [
        name
        for name, index in internal_model.input.name_to_index.items()
        if index in internal_model.input.forcing
    ]
    required_dynamic_forcings = [
        forcing for forcing in anemoi_dynamic_forcings() if forcing in required_forcings
    ]
    required_static_forcings = [
        forcing
        for forcing in required_forcings
        if forcing not in anemoi_dynamic_forcings()
    ]

    missing_vars = [
        var
        for var in required_prognostic_variables + required_static_forcings
        if var not in datamodule_variables
    ]
    if len(missing_vars) > 0:
        raise ValueError(
            f"Missing the following required variables in dataset {decoder_index}: {missing_vars}"
        )

    indices_prognostic_dataset = torch.tensor(
        [
            index
            for index, var in enumerate(datamodule_variables)
            if var in required_prognostic_variables
        ],
        dtype=torch.int64,
    )
    indices_static_forcings_dataset = torch.tensor(
        [
            index
            for index, var in enumerate(datamodule_variables)
            if var in required_static_forcings
        ],
        dtype=torch.int64,
    )

    indices_prognostic_input = torch.tensor(
        [
            full_ordered_variable_list.index(var)
            for var in datamodule_variables
            if var in required_prognostic_variables
        ],
        dtype=torch.int64,
    )
    indices_static_forcings_input = torch.tensor(
        [
            full_ordered_variable_list.index(var)
            for var in datamodule_variables
            if var in required_static_forcings
        ],
        dtype=torch.int64,
    )
    indices_dynamic_forcings_input = torch.tensor(
        [
            full_ordered_variable_list.index(var)
            for var in datamodule_variables
            if var in required_dynamic_forcings
        ],
        dtype=torch.int64,
    )

    indices = {
        "variables_input": variable_indices_input,
        "variables_output": variable_indices_output,
        "prognostic_dataset": indices_prognostic_dataset,
        "static_forcings_dataset": indices_static_forcings_dataset,
        "prognostic_input": indices_prognostic_input,
        "static_forcings_input": indices_static_forcings_input,
        "dynamic_forcings_input": indices_dynamic_forcings_input,
    }
    variables = {
        "all": full_ordered_variable_list,
        "dynamic_forcings": required_dynamic_forcings,
    }

    return indices, variables
