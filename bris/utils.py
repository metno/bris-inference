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


def expand_time_tokens(filename, unixtime):
    """Expand time tokens in a filename"""
    if not isinstance(unixtime, numbers.Number):
        raise ValueError(f"Unixtime but be numeric not {unixtime}")

    return os.path.abspath(time.strftime(filename, time.gmtime(unixtime)))


def create_directory(filename):
    """Creates all sub directories necessary to be able to write filename"""
    dir = os.path.dirname(filename)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def is_number(value):
    return isinstance(value, numbers.Number)


def get_workdir(path):
    multiple_processes = "SLURM_PROCID" in os.environ
    if multiple_processes:
        v = os.environ["SLURM_JOB_ID"]
    else:
        v = uuid.uuid4()
    return path + "/" + str(v)


def check_anemoi_training(metadata) -> bool:
    assert isinstance(
        metadata, DotDict
    ), f"Expected metadata to be a DotDict, got {type(metadata)}"
    if hasattr(metadata.provenance_training, "module_versions"):
        if hasattr(metadata.provenance_training.module_versions, "anemoi.training"):
            return True
        else:
            return False


def check_anemoi_dataset_version(metadata) -> tuple[bool, str]:
    assert isinstance(
        metadata, DotDict
    ), f"Expected metadata to be a DotDict, got {type(metadata)}"
    if hasattr(metadata.provenance_training, "module_versions"):
        try:
            _version = metadata.provenance_training.module_versions["anemoi.datasets"]
            _version = re.match(r"^\d+\.\d+\.\d+", _version).group()
            if _version < "0.5.0":
                return True, _version
            else:
                return False, _version
        except Exception as e:
            raise e
    else:
        raise RuntimeError("metadata.provenance_training does not module_versions")


def create_config(parser: ArgumentParser) -> OmegaConf:
    args, _ = parser.parse_known_args()

    validate(args.config, raise_on_error=True)

    try:
        config = OmegaConf.load(args.config)
        LOGGER.debug(f"config file from {args.config} is loaded")
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

    # TODO: change start_date and end_date to numpy datetime
    return OmegaConf.merge(config, OmegaConf.create(args_dict))


def datetime_to_unixtime(dt):
    """Converts a np.datetime64 object or list of objects to unixtime"""
    return np.array(dt).astype("datetime64[s]").astype("int")


def unixtime_to_datetime(ut):
    return np.datetime64(ut, "s")


def validate(filename, raise_on_error=False):
    schema_filename = os.path.dirname(os.path.abspath(__file__)) + "/schema/schema.json"
    with open(schema_filename) as file:
        schema = json.load(file)

    with open(filename) as file:
        config = yaml.safe_load(file)
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        if raise_on_error:
            raise
        else:
            print("WARNING: Schema does not validate")
            print(e)


def recursive_list_to_tuple(data):
    if isinstance(data, list):
        return tuple(recursive_list_to_tuple(item) for item in data)
    return data
