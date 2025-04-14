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

from .basepredictor import BasePredictor
from .checkpoint import Checkpoint
from .data.datamodule import DataModule
from .forcings import anemoi_dynamic_forcings, get_dynamic_forcings
from .utils import (
    check_anemoi_training,
    get_variable_indices,
    timedelta64_from_timestep,
)

LOGGER = logging.getLogger(__name__)


class MultiEncDecPredictor(BasePredictor):
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
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.forecast_length = forecast_length
        self.latitudes = datamodule.data_reader.latitudes
        self.longitudes = datamodule.data_reader.longitudes
        self.data_indices = checkpoint.data_indices

        self.indices = ()
        self.variables = ()
        for dec_index, required_vars_dec in required_variables.items():
            _indices, _variables = get_variable_indices(
                required_vars_dec,
                datamodule.data_reader.datasets[dec_index].variables,
                self.data_indices[dec_index].internal_data,
                self.data_indices[dec_index].internal_model,
                dec_index,
            )
            self.indices += (_indices,)
            self.variables += (_variables,)

        self.set_static_forcings(
            datamodule.data_reader, self.metadata["config"]["data"]["zip"]
        )
        self.model.eval()

    def set_static_forcings(self, data_reader: Iterable, data_config: dict):
        data = data_reader[0]
        num_dsets = len(data)
        data_input = []
        for dec_index in range(num_dsets):
            _batch = torch.from_numpy(data[dec_index].squeeze(axis=1).swapaxes(0, 1))
            _data_input = torch.zeros(
                _batch.shape[:-1] + (len(self.variables[dec_index]["all"]),),
                dtype=_batch.dtype,
                device=_batch.device,
            )
            _data_input[..., self.indices[dec_index]["prognostic_input"]] = _batch[
                ..., self.indices[dec_index]["prognostic_dataset"]
            ]
            _data_input[..., self.indices[dec_index]["static_forcings_input"]] = _batch[
                ..., self.indices[dec_index]["static_forcings_dataset"]
            ]
            data_input += [_data_input]

        data_normalized = self.model.pre_processors(data_input, in_place=True)

        self.static_forcings = [{} for _ in range(num_dsets)]
        for dset in range(num_dsets):
            selection = data_config[dset]["forcing"]
            if "cos_latitude" in selection:
                self.static_forcings[dset]["cos_latitude"] = torch.from_numpy(
                    np.cos(data_reader.latitudes[dset] * np.pi / 180.0)
                ).float()

            if "sin_latitude" in selection:
                self.static_forcings[dset]["sin_latitude"] = torch.from_numpy(
                    np.sin(data_reader.latitudes[dset] * np.pi / 180.0)
                ).float()

            if "cos_longitude" in selection:
                self.static_forcings[dset]["cos_longitude"] = torch.from_numpy(
                    np.cos(data_reader.longitudes[dset] * np.pi / 180.0)
                ).float()

            if "sin_longitude" in selection:
                self.static_forcings[dset]["sin_longitude"] = torch.from_numpy(
                    np.sin(data_reader.longitudes[dset] * np.pi / 180.0)
                ).float()

            if "lsm" in selection:
                self.static_forcings[dset]["lsm"] = data_normalized[dset][
                    ...,
                    self.data_indices[dset].internal_data.input.name_to_index["lsm"],
                ].float()

            if "z" in selection:
                self.static_forcings[dset]["z"] = data_normalized[dset][
                    ..., self.data_indices[dset].internal_data.input.name_to_index["z"]
                ].float()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x, self.model_comm_group)

    def advance_input_predict(
        self, x: list[torch.Tensor], y_pred: list[torch.Tensor], time: np.datetime64
    ):
        for i in range(len(x)):
            x[i] = x[i].roll(-1, dims=1)
            # Get prognostic variables:
            x[i][:, -1, :, :, self.data_indices[i].internal_model.input.prognostic] = (
                y_pred[i][..., self.data_indices[i].internal_model.output.prognostic]
            )

            forcings = get_dynamic_forcings(
                time,
                self.latitudes[i],
                self.longitudes[i],
                self.metadata["config"]["data"]["zip"][i]["forcing"],
            )
            forcings.update(self.static_forcings[i])

            for forcing, value in forcings.items():
                if np.ndarray is type(value):
                    x[i][
                        :,
                        -1,
                        :,
                        :,
                        self.data_indices[i].internal_model.input.name_to_index[
                            forcing
                        ],
                    ] = torch.from_numpy(value)
                else:
                    x[i][
                        :,
                        -1,
                        :,
                        :,
                        self.data_indices[i].internal_model.input.name_to_index[
                            forcing
                        ],
                    ] = value

        return x

    @torch.inference_mode
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        num_dsets = len(batch)
        multistep = self.metadata["config"]["training"]["multistep_input"]

        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0])
        times = [time]
        y_preds = [
            torch.empty(
                (
                    batch[i].shape[0],
                    self.forecast_length,
                    batch[i].shape[-2],
                    len(self.indices[i]["variables_input"]),
                ),
                dtype=batch[i].dtype,
                device="cpu",
            )
            for i in range(num_dsets)
        ]
        data_input = []
        for dec_index in range(num_dsets):
            _data_input = torch.zeros(
                batch[dec_index].shape[:-1] + (len(self.variables[dec_index]["all"]),),
                dtype=batch[dec_index].dtype,
                device=batch[dec_index].device,
            )
            _data_input[..., self.indices[dec_index]["prognostic_input"]] = batch[
                dec_index
            ][..., self.indices[dec_index]["prognostic_dataset"]]
            _data_input[..., self.indices[dec_index]["static_forcings_input"]] = batch[
                dec_index
            ][..., self.indices[dec_index]["static_forcings_dataset"]]

            # Calculate dynamic forcings and add these to data_input
            for time_index in range(multistep):
                toi = time - (multistep - 1 - time_index) * self.timestep
                forcings = get_dynamic_forcings(
                    toi,
                    self.latitudes[dec_index],
                    self.longitudes[dec_index],
                    self.variables[dec_index]["dynamic_forcings"],
                )

                for forcing, value in forcings.items():
                    if isinstance(value, np.ndarray):
                        _data_input[
                            :,
                            time_index,
                            :,
                            :,
                            self.data_indices[
                                dec_index
                            ].internal_data.input.name_to_index[forcing],
                        ] = torch.from_numpy(value).to(dtype=_data_input.dtype)
                    else:
                        _data_input[
                            :,
                            time_index,
                            :,
                            :,
                            self.data_indices[
                                dec_index
                            ].internal_data.input.name_to_index[forcing],
                        ] = value
            data_input += [_data_input]

            y_preds[dec_index][:, 0, :, :] = data_input[dec_index][
                :, multistep - 1, ..., self.indices[dec_index]["variables_input"]
            ].cpu()

        data_input = self.model.pre_processors(data_input, in_place=True)
        x = [
            data_input[i][..., self.data_indices[i].internal_data.input.full]
            for i in range(num_dsets)
        ]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for fcast_step in range(self.forecast_length - 1):
                y_pred = self(x)
                time += self.timestep
                x = self.advance_input_predict(x, y_pred, time)
                y_pp = self.model.post_processors(y_pred, in_place=False)
                for i in range(num_dsets):
                    y_preds[i][:, fcast_step + 1, ...] = y_pp[i][
                        :, 0, ..., self.indices[i]["variables_output"]
                    ].cpu()
                times.append(time)

        return {
            "pred": y_preds,
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": 0,
        }
