import logging
import math
import os
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Union

import numpy as np
import pytorch_lightning as pl
import torch
from anemoi.models.data_indices.index import DataIndex, ModelIndex
from torch.distributed.distributed_c10d import ProcessGroup

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..forcings import anemoi_dynamic_forcings, get_dynamic_forcings
from ..utils import (
    check_anemoi_training,
    get_model_static_forcings,
    timedelta64_from_timestep,
)
from .basepredictor import BasePredictor

LOGGER = logging.getLogger(__name__)


class BrisPredictor(BasePredictor):
    """
    Custom Bris predictor.

    Methods
    -------

    __init__

    set_static_forcings: Set static forcings for the model.

    forward: Forward pass through the model.

    advance_input_predict: Advance the input tensor for the next prediction step.

    predict_step: Predicts the next time step using the model.

    allgather_batch
    """

    def __init__(
        self,
        *args,
        checkpoints: dict[str, Checkpoint],
        datamodule: DataModule,
        forecast_length: int,
        required_variables: dict,
        release_cache: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the BrisPredictor.

        Args:
            checkpoints
                Example: {"forecaster": checkpoint_object}

            datamodule
                Data loader containing the dataset, from one or more datasets. Loaded in config as for example:

                    dataset: /home/larsfp/nobackup/bris_random_data.zarr
                    dataloader:
                        datamodule:
                            _target_: bris.data.dataset.NativeGridDataset

            forecast_length
                Length of the forecast in timesteps.

            required_variables
                Dictionary of datasets with list of required variables for each dataset. Example:
                    {0: ['2d', '2t']}

            release_cache
                Release cache (torch.cuda.empty_cache()) after each prediction step. This is useful for large models,
                but may slow down the prediction.
        """

        super().__init__(*args, checkpoints=checkpoints, **kwargs)

        checkpoint = checkpoints["forecaster"]
        self.model = checkpoint.model
        self.data_indices = checkpoint.data_indices[0]
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.forecast_length = forecast_length
        self.latitudes = datamodule.data_reader.latitudes
        self.longitudes = datamodule.data_reader.longitudes

        # Backwards compatibility with older anemoi-models versions,
        # for example legendary-gnome.
        if hasattr(self.data_indices, "internal_model") and hasattr(
            self.data_indices, "internal_data"
        ):
            self.internal_model = self.data_indices.internal_model
            self.internal_data = self.data_indices.internal_data
        else:
            self.internal_model = self.data_indices.model
            self.internal_data = self.data_indices.data

        self.indices, self.variables = get_variable_indices(
            required_variables=required_variables[0],
            datamodule_variables=datamodule.data_reader.variables,
            internal_data=self.internal_data,
            internal_model=self.internal_model,
            decoder_index=0,
        )
        self.set_static_forcings(datamodule.data_reader, self.metadata.config.data)

        self.model.eval()
        self.release_cache = release_cache

    def set_static_forcings(self, data_reader: Iterable, data_config: dict) -> None:
        """
        Set static forcings for the model. Done by reading from the data reader, reshape, store as a tensor. Tensor is
        populated with prognostic and static forcing variables based on predefined indices. Then normalized.

        The static forcings are the variables that are not prognostic and not dynamic forcings, e.g., cos_latitude,
        sin_latitude, cos_longitude, sin_longitude, lsm, z

        Args:
            data_reader (Iterable): Data reader containing the dataset.
            data_config (dict): Configuration dictionary containing forcing information.
        """
        data = torch.from_numpy(data_reader[0].squeeze(axis=1).swapaxes(0, 1))
        data_input = torch.zeros(
            data.shape[:-1] + (len(self.variables["all"]),),
            dtype=data.dtype,
            device=data.device,
        )
        data_input[..., self.indices["prognostic_input"]] = data[
            ..., self.indices["prognostic_dataset"]
        ]
        data_input[..., self.indices["static_forcings_input"]] = data[
            ..., self.indices["static_forcings_dataset"]
        ]

        self.static_forcings = get_model_static_forcings(
            selection=data_config["forcing"],
            data_reader=data_reader,
            data_normalized=self.model.pre_processors(data_input, in_place=True),
            internal_data=self.internal_data,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor after processing by the model.
        """
        return self.model(x, self.model_comm_group)

    def advance_input_predict(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        """
        Advance the input tensor for the next prediction step.
        Args:
            x (torch.Tensor): Input tensor to be advanced.
            y_pred (torch.Tensor): Predicted output tensor.
            time (np.datetime64): Current time.
        Returns:
            torch.Tensor: Advanced input tensor for the next prediction step.
        """
        # Shift the input tensor to the next time step
        x = x.roll(-1, dims=1)

        # Get prognostic variables:
        x[:, -1, :, :, self.internal_model.input.prognostic] = y_pred[
            ..., self.internal_model.output.prognostic
        ]

        forcings = get_dynamic_forcings(
            time, self.latitudes, self.longitudes, self.variables["dynamic_forcings"]
        )
        forcings.update(self.static_forcings)

        for forcing, value in forcings.items():
            if isinstance(value, np.ndarray):
                x[:, -1, :, :, self.internal_model.input.name_to_index[forcing]] = (
                    torch.from_numpy(value).to(dtype=x.dtype)
                )
            else:
                x[:, -1, :, :, self.internal_model.input.name_to_index[forcing]] = value
        return x

    @torch.inference_mode
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Perform a prediction step using the model.
        Args:
            batch (tuple): Input batch containing the data.
            batch_idx (int): Index of the batch.
        Returns:
            dict: Dictionary containing the predicted output, time stamps, group rank, and ensemble member.
        """
        multistep = self.metadata.config.training.multistep_input

        batch = self.allgather_batch(batch)

        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0])
        times = [time]
        y_preds = torch.empty(
            (
                batch.shape[0],
                self.forecast_length,
                batch.shape[-2],
                len(self.indices["variables_output"]),
            ),
            dtype=batch.dtype,
            device="cpu",
        )

        # Set up data_input with variable order expected by the model.
        # Prognostic and static forcings come from batch, dynamic forcings
        # are calculated and diagnostic variables are filled with 0.
        data_input = torch.zeros(
            batch.shape[:-1] + (len(self.variables["all"]),),
            dtype=batch.dtype,
            device=batch.device,
        )
        data_input[..., self.indices["prognostic_input"]] = batch[
            ..., self.indices["prognostic_dataset"]
        ]
        data_input[..., self.indices["static_forcings_input"]] = batch[
            ..., self.indices["static_forcings_dataset"]
        ]

        # Calculate dynamic forcings
        for time_index in range(multistep):
            toi = time - (multistep - 1 - time_index) * self.timestep
            forcings = get_dynamic_forcings(
                toi, self.latitudes, self.longitudes, self.variables["dynamic_forcings"]
            )

            for forcing, value in forcings.items():
                if isinstance(value, np.ndarray):
                    data_input[
                        :,
                        time_index,
                        :,
                        :,
                        self.internal_data.input.name_to_index[forcing],
                    ] = torch.from_numpy(value).to(dtype=data_input.dtype)
                else:
                    data_input[
                        :,
                        time_index,
                        :,
                        :,
                        self.internal_data.input.name_to_index[forcing],
                    ] = value

        y_preds[:, 0, ...] = data_input[
            :, multistep - 1, ..., self.indices["variables_input"]
        ].cpu()

        # Possibly have to extend this to handle imputer, see _step in forecaster.
        data_input = self.model.pre_processors(data_input, in_place=True)
        x = data_input[..., self.internal_data.input.full]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for fcast_step in range(self.forecast_length - 1):
                y_pred = self(x)
                time += self.timestep
                x = self.advance_input_predict(x, y_pred, time)
                y_preds[:, fcast_step + 1] = self.model.post_processors(
                    y_pred, in_place=True
                )[:, 0, :, self.indices["variables_output"]].cpu()

                times.append(time)
                if self.release_cache:
                    del y_pred
                    torch.cuda.empty_cache()
        return {
            "pred": [y_preds.to(torch.float32).numpy()],
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": 0,
        }

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Allgather the batch-shards across the reader group.
        """
        return batch  # Not implemented properly, https://github.com/metno/bris-inference/issues/123


def get_variable_indices(
    required_variables: list,
    datamodule_variables: list,
    internal_data: DataIndex,
    internal_model: ModelIndex,
    decoder_index: int,
) -> tuple[dict, dict]:
    """
    Helper function for BrisPredictor, get indices for variables in input data and model. This is used to map the
    variables in the input data to the variables in the model.
    Args:
        required_variables (list): List of required variables.
        datamodule_variables (list): List of variables in the input data.
        internal_data (DataIndex): Data index object, from checkpoint.data_indices
        internal_model (ModelIndex): Model index object, from checkpoint.data_indices.model
        decoder_index (int): Index of decoder, always zero for brispredictor.
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
