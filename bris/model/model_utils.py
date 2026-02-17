import numpy as np
import torch
from anemoi.datasets.data.dataset import Dataset
from anemoi.models.data_indices.index import DataIndex, ModelIndex
from anemoi.utils.config import DotDict

from ..forcings import anemoi_dynamic_forcings, get_dynamic_forcings


def get_model_static_forcings(
    selection: list,
    data_reader: Dataset,
    data_normalized,
    internal_data: DataIndex,
) -> dict:
    """Get static forcings from the model.

    Args:
        selection (list): List of static forcings to get.

        data_reader (Dataset): Data reader object.

        data_normalized: Normalized data.

        internal_data (DataIndex): Data index object, from checkpoint.data_indices

    """
    static_forcings = {}
    if selection is None:
        return

    if "cos_latitude" in selection:
        static_forcings["cos_latitude"] = torch.from_numpy(
            np.cos(data_reader.latitudes * np.pi / 180.0)
        ).float()

    if "sin_latitude" in selection:
        static_forcings["sin_latitude"] = torch.from_numpy(
            np.sin(data_reader.latitudes * np.pi / 180.0)
        ).float()

    if "cos_longitude" in selection:
        static_forcings["cos_longitude"] = torch.from_numpy(
            np.cos(data_reader.longitudes * np.pi / 180.0)
        ).float()

    if "sin_longitude" in selection:
        static_forcings["sin_longitude"] = torch.from_numpy(
            np.sin(data_reader.longitudes * np.pi / 180.0)
        ).float()

    if "lsm" in selection:
        static_forcings["lsm"] = (
            data_normalized[..., internal_data.input.name_to_index["lsm"]]
        ).float()

    if "z" in selection:
        static_forcings["z"] = (
            data_normalized[..., internal_data.input.name_to_index["z"]].float()
        )

    return static_forcings


def get_variable_indices(
    required_variables: list,
    datamodule_variables: list,
    internal_data: DataIndex,
    internal_model: ModelIndex,
    decoder_name: int,
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