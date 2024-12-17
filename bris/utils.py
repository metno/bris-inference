import numbers
import os
import re
import time
import uuid

from anemoi.utils.config import DotDict


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
    v = uuid.uuid4()
    return path + "/" + str(v)


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
