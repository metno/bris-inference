import numpy as np
import torch
from anemoi.models.data_indices.index import DataIndex, ModelIndex
from anemoi.datasets.data.dataset import Dataset


def get_model_static_forcings(
    selection: list, data_reader: Dataset, data_normalized, internal_data: DataIndex, dataset_no: int = 0
) -> dict:
    """Get static forcings from the model."""
    static_forcings = {}

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
        static_forcings["lsm"] = data_normalized[
            ..., internal_data.input.name_to_index["lsm"]
        ].float()

    if "z" in selection:
        static_forcings["z"] = data_normalized[
            ..., internal_data.input.name_to_index["z"]
        ].float()

    return static_forcings
