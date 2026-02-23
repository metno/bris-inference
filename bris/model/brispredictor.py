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
    LOGGER,
    check_anemoi_training,
    timedelta64_from_timestep,
)
from .basepredictor import BasePredictor
from .model_utils import (
    get_model_static_forcings,
    get_variable_indices,
    get_data_config,
)


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

    allgather_batch:
    """

    def __init__(
        self,
        *args,
        checkpoints: dict[str, Checkpoint],
        datamodule: DataModule,
        checkpoints_config: dict,
        required_variables: dict,
        release_cache: bool = False,
        fcstep_const: bool = False,
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

            fcstep_const
                For inference on non-rollout trained ensemble models. Keep fcstep constant at 0 as in training if True.
        """

        super().__init__(*args, checkpoints=checkpoints, **kwargs)

        checkpoint = checkpoints["forecaster"]
        self.model = checkpoint.model
        self.data_indices = checkpoint.data_indices
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
        self.latitudes = datamodule.latitudes
        self.longitudes = datamodule.longitudes
        self.fcstep_const = fcstep_const
        if hasattr(checkpoint.model.model, "inputs"):
            self.dataset_names = checkpoint.model.model.inputs
        elif hasattr(checkpoint.model.model, "dataset_names"): # Compatilbility with anemoi core main
            self.dataset_names = checkpoint.model.model.dataset_names
        else: # Legacy compatibility
            self.dataset_names = ["data"]

        assert self.dataset_names == datamodule.dataset_names, (
            f" Dataset names of the input data {datamodule.dataset_names} do not match expected dataset names {self.dataset_names} of the model."
        )

        self.internal_model = {}
        self.internal_data = {}
        self.indices = {}
        self.variables = {}

        for ds in self.dataset_names:
            self.internal_model[ds] = self.data_indices[ds].model
            self.internal_data[ds] = self.data_indices[ds].data

            self.indices[ds], self.variables[ds] = get_variable_indices(
                required_variables=required_variables.get(ds, []),
                datamodule_variables=datamodule.data_readers[ds].variables,
                internal_data=self.internal_data[ds],
                internal_model=self.internal_model[ds],
                decoder_name=ds,
            )

        data_cfg = get_data_config(self.metadata.config)
        self.set_static_forcings(datamodule.data_readers, data_cfg)

        self.model.eval()
        self.release_cache = release_cache

        self.batch_info = {}

    def set_static_forcings(self, data_readers: dict[str, Iterable], data_config: dict) -> None:
        """
        Set static forcings for the model. Done by reading from the data reader, reshape, store as a tensor. Tensor is
        populated with prognostic and static forcing variables based on predefined indices. Then normalized.

        The static forcings are the variables that are not prognostic and not dynamic forcings, e.g., cos_latitude,
        sin_latitude, cos_longitude, sin_longitude, lsm, z

        Args:
            data_readers (dict[str: Iterable]): Dictionary with data readers for the datasets.
            data_config (dict): Configuration dictionary containing forcing information.
        """
        self.static_forcings = {}
        for ds in self.dataset_names:
            data = torch.from_numpy(data_readers[ds][0].squeeze(axis=1).swapaxes(0, 1))
            data_input = torch.zeros(
                data.shape[:-1] + (len(self.variables[ds]["all"]),),
                dtype=data.dtype,
                device=data.device,
            )
            data_input[..., self.indices[ds]["prognostic_input"]] = data[
                ..., self.indices[ds]["prognostic_dataset"]
            ]
            data_input[..., self.indices[ds]["static_forcings_input"]] = data[
                ..., self.indices[ds]["static_forcings_dataset"]
            ]

            self.static_forcings[ds] = get_model_static_forcings(
                selection=data_config[ds]["forcing"],
                data_reader=data_readers[ds],
                data_normalized=self.model.pre_processors[ds](data_input, in_place=True),
                internal_data=self.internal_data[ds],
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
        for ds in x.keys():
            _x = x[ds]
            _x = _x.roll(-1, dims=1)

            # Get prognostic variables:
            _x[:, -1, :, :, self.internal_model[ds].input.prognostic] = y_pred[ds][
                ..., self.internal_model[ds].output.prognostic
            ]

            forcings = get_dynamic_forcings(
                time, self.latitudes[ds], self.longitudes[ds], self.variables[ds]["dynamic_forcings"]
            )
            forcings.update(self.static_forcings[ds])

            for forcing, value in forcings.items():
                if isinstance(value, np.ndarray):
                    _x[:, -1, :, :, self.internal_model[ds].input.name_to_index[forcing]] = (
                        torch.from_numpy(value).to(dtype=_x.dtype)
                    )
                else:
                    _x[:, -1, :, :, self.internal_model[ds].input.name_to_index[forcing]] = value

            x[ds] = _x
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

        y_preds = {}
        data_input = {}
        x = {}
        for ds in self.dataset_names:
            y_preds[ds] = torch.empty(
                (
                    batch[ds].shape[0],
                    self.forecast_length,
                    batch[ds].shape[-2],
                    len(self.indices[ds]["variables_output"]),
                ),
                dtype=batch[ds].dtype,
                device="cpu",
            )

            # Set up data_input with variable order expected by the model.
            # Prognostic and static forcings come from batch, dynamic forcings
            # are calculated and diagnostic variables are filled with 0.
            data_input[ds] = torch.full(
                batch[ds].shape[:-1] + (len(self.variables[ds]["all"]),),
                float("nan"),
                dtype=batch[ds].dtype,
                device=batch[ds].device,
            )
            data_input[ds][..., self.indices[ds]["prognostic_input"]] = batch[ds][
                ..., self.indices[ds]["prognostic_dataset"]
            ]
            data_input[ds][..., self.indices[ds]["static_forcings_input"]] = batch[ds][
                ..., self.indices[ds]["static_forcings_dataset"]
            ]

        # Calculate dynamic forcings
            for time_index in range(multistep):
                toi = time - (multistep - 1 - time_index) * self.timestep
                forcings = get_dynamic_forcings(
                    toi, self.latitudes[ds], self.longitudes[ds], self.variables[ds]["dynamic_forcings"]
                )

                for forcing, value in forcings.items():
                    if isinstance(value, np.ndarray):
                        data_input[ds][
                            :,
                            time_index,
                            :,
                            :,
                            self.internal_data[ds].input.name_to_index[forcing],
                        ] = torch.from_numpy(value).to(dtype=data_input[ds].dtype)
                    else:
                        data_input[ds][
                            :,
                            time_index,
                            :,
                            :,
                            self.internal_data[ds].input.name_to_index[forcing],
                        ] = value

            y_preds[ds][:, 0, ...] = data_input[ds][
                :, multistep - 1, ..., self.indices[ds]["variables_input"]
            ].cpu()

            # Possibly have to extend this to handle imputer, see _step in forecaster.
            data_input[ds] = self.model.pre_processors[ds](data_input[ds], in_place=True)
            x[ds] = data_input[ds][..., self.internal_data[ds].input.full]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for forecast_step in range(self.forecast_length - 1):
                # TODO: Need backwards compatibility with models where batch is tensor
                try:
                    if self.fcstep_const:
                       y_pred = self(x, fcstep=0)
                    else:
                        y_pred = self(x, fcstep=forecast_step)
                except TypeError:
                    y_pred = self(x)
                time += self.timestep
                x = self.advance_input_predict(x, y_pred, time)
                for ds in self.dataset_names:
                    y_preds[ds][:, forecast_step + 1] = self.model.post_processors[ds](
                    y_pred[ds], in_place=True
                    )[:, 0, :, self.indices[ds]["variables_output"]].cpu()

                times.append(time)
                if self.release_cache:
                    del y_pred
                    torch.cuda.empty_cache()
        self.update_batch_info(time)
        # Save info about which batches has been processed before, update ensemble member based on this.
        return {
            "pred": {ds: y_pred.to(torch.float32).numpy() for ds, y_pred in y_preds.items()},
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
