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
from ..forcings import (
    anemoi_dynamic_forcings,
    get_dynamic_forcings,
)
from ..utils import (
    check_anemoi_training,
    timedelta64_from_timestep,
)
from .basepredictor import BasePredictor
from .model_utils import get_model_static_forcings, get_variable_indices

LOGGER = logging.getLogger(__name__)


class BrisLamPredictor(BasePredictor):
    """
    Custom Bris LAM predictor.

    Methods
    -------

    __init__

    set_static_forcings: Set static forcings for the model.

    TODO: Implement other forecast forcings and boundary forcings.

    forward: Forward pass through the model.

    advance_input_predict: Advance the input tensor for the next prediction step.
    TODO: adapt to support forecast forcings ++

    predict_step: Predicts the next time step using the model.

    allgather_batch:
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
        # TODO: read other forcings from file

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
        # TODO: Implement forecast forcings (from file) and boundary forcings.

        self.model.eval()
        self.release_cache = release_cache

        self.batch_info = {}

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
            dataset_no=None,
        )

    def update_batch_info(self, time):
        if time not in self.batch_info:
            self.batch_info[time] = 1
        else:
            self.batch_info[time] += 1

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor after processing by the model.
        """
        return self.model(x, model_comm_group=self.model_comm_group, **kwargs)

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

        # TODO: also handel forecast forcings and boundary forcings.

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
            for forecast_step in range(self.forecast_length - 1):
                # Backwards compatibility to older models without kwargs
                try:
                    y_pred = self(x, fcstep=forecast_step)
                except TypeError:
                    y_pred = self(x)
                time += self.timestep
                x = self.advance_input_predict(x, y_pred, time)
                y_preds[:, forecast_step + 1] = self.model.post_processors(
                    y_pred, in_place=True
                )[:, 0, :, self.indices["variables_output"]].cpu()

                times.append(time)
                if self.release_cache:
                    del y_pred
                    torch.cuda.empty_cache()
        self.update_batch_info(time)
        # Save info about which batches has been processed before, update ensemble member based on this.
        return {
            "pred": [y_preds.to(torch.float32).numpy()],
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": self.member_id
            + self.num_members_in_parallel * (self.batch_info[time] - 1),
        }

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Allgather the batch-shards across the reader group.
        """
        return batch  # Not implemented properly, https://github.com/metno/bris-inference/issues/123
