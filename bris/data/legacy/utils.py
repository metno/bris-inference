"""
This utils submodule contains legacy functions which
legacy dataset.py needs to run aifs-mono
"""

import logging
import os
import sys
import time

import numpy as np


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
