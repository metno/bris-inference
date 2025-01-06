"""
This utils submodule contains legacy functions which
legacy dataset.py needs to run aifs-mono
"""

import logging
import os
import sys
import time

import numpy as np
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


def get_code_logger(name: str, debug: bool = True) -> logging.Logger:
    """Returns a logger with a custom level and format.

    We use ISO8601 timestamps and UTC times.

    Parameters
    ----------
    name : str
        Name of logger object
    debug : bool, optional
        set logging level to logging.DEBUG; else set to logging.INFO, by default True

    Returns
    -------
    logging.Logger
        Logger object
    """
    # create logger object
    logger = logging.getLogger(name=name)
    if not logger.hasHandlers():
        # logging level
        level = logging.DEBUG if debug else logging.INFO
        # logging format
        datefmt = "%Y-%m-%dT%H:%M:%SZ"
        msgfmt = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName).30s] [%(levelname)s] %(message)s"
        # handler object
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(msgfmt, datefmt=datefmt)
        # record UTC time
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)

    return logger


def get_base_seed(env_var_list=("AIFS_BASE_SEED", "SLURM_JOB_ID")) -> int:
    """Gets the base seed from the environment variables.

    Option to manually set a seed via export AIFS_BASE_SEED=xxx in job script
    """
    base_seed = None
    for env_var in env_var_list:
        if env_var in os.environ:
            base_seed = int(os.environ.get(env_var))
            break

    assert (
        base_seed is not None
    ), f"Base seed not found in environment variables {env_var_list}"

    if base_seed < 1000:
        base_seed = base_seed * 1000  # make it (hopefully) big enough

    return base_seed


def get_usable_indices(
    missing_indices: set[int],
    series_length: int,
    rollout: int,
    multistep: int,
    timeincrement: int = 1,
) -> np.ndarray:
    """Get the usable indices of a series whit missing indices.

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

    # No missing indices
    if missing_indices is None:
        return usable_indices[prev_invalid_dates : series_length - next_invalid_dates]

    missing_indices |= {-1, series_length}  # to filter initial and final indices

    # Missing indices
    for i in missing_indices:
        usable_indices = usable_indices[
            (usable_indices < i - next_invalid_dates)
            + (usable_indices > i + prev_invalid_dates)
        ]

    return usable_indices


def _legacy_slurm_proc_id(config: DotDict) -> None:
    """
    Set up model communication groups, rank and id
    based on slurm proccess ID's

    args:
        config (DotDict): configuration file (Yaml)

    return:
    model_comm_group_rank (int), model_comm_group_id (int),
    model_comm_num_groups (int)

    """
    global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # global rank
    model_comm_group_id = (
        global_rank // config.hardware.num_gpus_per_model
    )  # id of the model communication group the rank is participating in
    model_comm_group_rank = (
        global_rank % config.hardware.num_gpus_per_model
    )  # rank within one model communication group
    total_gpus = config.hardware.num_gpus_per_node * config.hardware.num_nodes
    assert (
        total_gpus
    ) % config.hardware.num_gpus_per_model == 0, f"GPUs per model {config.hardware.num_gpus_per_model} does not divide total GPUs {total_gpus}"
    model_comm_num_groups = (
        config.hardware.num_gpus_per_node
        * config.hardware.num_nodes
        // config.hardware.num_gpus_per_model
    )  # number of model communication groups

    LOGGER.debug(
        "Rank %d model communication group number %d, with local model communication group rank %d",
        global_rank,
        model_comm_group_id,
        model_comm_group_rank,
    )

    return model_comm_group_rank, model_comm_group_id, model_comm_num_groups
