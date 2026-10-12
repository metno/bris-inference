from collections.abc import Iterable

import numpy as np
import torch
from anemoi.models.data_indices.index import DataIndex, ModelIndex
from anemoi.utils.config import DotDict

from ..checkpoint import Checkpoint
from ..forcings import anemoi_dynamic_forcings
from ..utils import timedelta64_from_timestep


def get_model_static_forcings(
    selection: list,
    data_normalized: torch.Tensor,
    internal_data: DataIndex,
) -> dict:
    """Static forcings of a model, taken from a state.

    Every forcing in ``selection`` that is not computed on the fly (see
    :func:`bris.forcings.anemoi_dynamic_forcings`) is read from ``data_normalized`` as it is:
    cos/sin of latitude/longitude, lsm, z and any other time-constant input of the model (e.g.
    physiography fields). ``data_normalized`` should be the starting state of the forecast, not
    the first time step of the dataset: z (and in principle the other fields) change with model
    updates in long datasets.

    Args:
        selection (list): Forcing names of the model (the dynamic ones are skipped).

        data_normalized (torch.Tensor): Normalized state in the data-index variable order,
            shape (..., points, variables); leading dimensions (e.g. batch, ensemble) are kept.

        internal_data (DataIndex): Data index object, from checkpoint.data_indices

    Returns:
        dict: forcing name -> tensor of shape (..., points), a compact copy of the state values
    """
    if selection is None:
        return None
    dynamic = anemoi_dynamic_forcings()
    return {
        name: data_normalized[..., internal_data.input.name_to_index[name]]
        .contiguous()
        .float()
        for name in selection
        if name not in dynamic
    }


def static_forcing_time_index(dates: np.ndarray, time: np.datetime64) -> int:
    """Index of the time step of a dataset whose static forcings are valid at ``time``.

    That is the latest date <= time (static fields change at model updates and stay valid
    until the next one), or 0 if time is before the first date.

    Args:
        dates (np.ndarray): The dataset dates (datetime64), sorted.
        time (np.datetime64): Start time of the forecast.
    """
    dates = np.asarray(dates, dtype="datetime64[s]")
    index = int(np.searchsorted(dates, np.datetime64(time, "s"), side="right")) - 1
    return max(index, 0)


def get_variable_indices(
    required_variables: list,
    datamodule_variables: list,
    internal_data: DataIndex,
    internal_model: ModelIndex,
    decoder_name: str,
) -> tuple[dict, dict]:
    """
    Helper function for BrisPredictor, get indices for variables in input data and model. This is used to map the
    variables in the input data to the variables in the model.
    Args:
        required_variables (list): List of required variables.
        datamodule_variables (list): List of variables in the input data.
        internal_data (DataIndex): Data index object, from checkpoint.data_indices
        internal_model (ModelIndex): Model index object, from checkpoint.data_indices.model
        decoder_name (int): Name of dataset associated with decoder.
    Returns:
        tuple[dict, dict]:
            - indices: A dictionary containing the indices for the variables in the input data and the model.
            - variables: A dictionary containing the variables in the input data and the model.
    """
    # Set up indices for the variables we want to write to file
    variable_indices_input = []
    variable_indices_output = []
    for name in required_variables:
        variable_indices_input.append(internal_data.input.name_to_index[name])
        variable_indices_output.append(internal_model.output.name_to_index[name])

    # Set up indices that can map from the variable order in the input data to the input variable order expected by the
    # model
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
            f"Missing the following required variables in dataset {decoder_name}: {missing_vars}"
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
        "static_forcings": required_static_forcings,
    }

    return indices, variables


def get_data_config(config: DotDict) -> DotDict:
    """Get data configuration from the model configuration.
    Backwards compatibility to checkpoints with old data config structure.

    Args:
        config (DotDict): Model configuration.
    Returns:
        DotDict: Data configuration.
    """

    if "dataset_specific" in config.data:
        return config.data.dataset_specific.datasets
    elif "datasets" in config.data:
        return config.data.datasets

    cfg = DotDict({"data": config.data})
    return cfg


def get_interpolator_interp_times(checkpoint: Checkpoint) -> list[int]:
    try:
        return checkpoint.config.training.explicit_times.target
    except AttributeError:
        pass

    try:
        input_td = timedelta64_from_timestep(checkpoint.config.task.input_timestep)
        output_td = timedelta64_from_timestep(checkpoint.config.task.output_timestep)
        input_seconds = input_td / np.timedelta64(1, "s")
        output_seconds = output_td / np.timedelta64(1, "s")

        n = int(input_seconds // output_seconds)

        indices = list(range(0, n + 1))

        if not checkpoint.config.task.output_left_boundary:
            indices = indices[1:]
        if not checkpoint.config.task.output_right_boundary:
            indices = indices[:-1]
        return indices
    except AttributeError:
        pass

    raise AttributeError("Could not find interp_times in Interpolator checkpoint.")


def get_interpolator_boundary_times(checkpoint: Checkpoint) -> list[int]:
    try:
        return checkpoint.config.training.explicit_times.input
    except AttributeError:
        pass

    try:
        input_td = timedelta64_from_timestep(checkpoint.config.task.input_timestep)
        output_td = timedelta64_from_timestep(checkpoint.config.task.output_timestep)
        input_seconds = input_td / np.timedelta64(1, "s")
        output_seconds = output_td / np.timedelta64(1, "s")

        n = int(input_seconds // output_seconds)

        return [0, n]

    except AttributeError:
        pass

    raise AttributeError("Could not find boundary_times in Interpolator checkpoint.")
