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
    get_variable_indices,
    timedelta64_from_timestep,
)
from .basepredictor import BasePredictor

LOGGER = logging.getLogger(__name__)


class BrisPredictor(BasePredictor):
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
        super().__init__(*args, checkpoints=checkpoints, **kwargs)

        checkpoint = checkpoints["forecaster"]
        self.model = checkpoint.model
        self.data_indices = checkpoint.data_indices[0]
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.forecast_length = forecast_length
        self.latitudes = datamodule.data_reader.latitudes
        self.longitudes = datamodule.data_reader.longitudes

        # this makes it backwards compatible with older
        # anemoi-models versions. I.e legendary gnome, etc..
        if hasattr(self.data_indices, "internal_model") and hasattr(
            self.data_indices, "internal_data"
        ):
            self.internal_model = self.data_indices.internal_model
            self.internal_data = self.data_indices.internal_data
        else:
            self.internal_model = self.data_indices.model
            self.internal_data = self.data_indices.data

        self.indices, self.variables = get_variable_indices(
            required_variables[0],
            datamodule.data_reader.variables,
            self.internal_data,
            self.internal_model,
            0,
        )
        self.set_static_forcings(datamodule.data_reader, self.metadata.config.data)

        self.model.eval()
        self.release_cache = release_cache

    def set_static_forcings(self, data_reader: Iterable, data_config: dict) -> None:
        selection = data_config["forcing"]
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

        data_normalized = self.model.pre_processors(data_input, in_place=True)

        self.static_forcings = {}
        if "cos_latitude" in selection:
            self.static_forcings["cos_latitude"] = torch.from_numpy(
                np.cos(data_reader.latitudes * np.pi / 180.0)
            ).float()

        if "sin_latitude" in selection:
            self.static_forcings["sin_latitude"] = torch.from_numpy(
                np.sin(data_reader.latitudes * np.pi / 180.0)
            ).float()

        if "cos_longitude" in selection:
            self.static_forcings["cos_longitude"] = torch.from_numpy(
                np.cos(data_reader.longitudes * np.pi / 180.0)
            ).float()

        if "sin_longitude" in selection:
            self.static_forcings["sin_longitude"] = torch.from_numpy(
                np.sin(data_reader.longitudes * np.pi / 180.0)
            ).float()

        if "lsm" in selection:
            self.static_forcings["lsm"] = data_normalized[
                ..., self.internal_data.input.name_to_index["lsm"]
            ].float()

        if "z" in selection:
            self.static_forcings["z"] = data_normalized[
                ..., self.internal_data.input.name_to_index["z"]
            ].float()

        del data_normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    def advance_input_predict(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
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
        return batch  # Not implemented properly
