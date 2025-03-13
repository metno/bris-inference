import logging
import math
import os
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from .forcings import get_dynamic_forcings, anemoi_dynamic_forcings
from .checkpoint import Checkpoint
from .utils import check_anemoi_training, timedelta64_from_timestep
from .data.datamodule import DataModule


LOGGER = logging.getLogger(__name__)


class BasePredictor(pl.LightningModule):
    def __init__(
        self, *args: Any, checkpoint: Checkpoint, hardware_config, **kwargs: Any
    ):
        """
        Base predictor class, overwrite all the class methods

        """

        super().__init__(*args, **kwargs)
        # Lazy init
        self.model_comm_group = None
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1

        if check_anemoi_training(checkpoint.metadata):
            self.legacy = False
        else:
            self.legacy = True

        if self.legacy:
            self.model_comm_group = None
            self.model_comm_group_id = (
                int(os.environ.get("SLURM_PROCID", "0"))
                // hardware_config["num_gpus_per_model"]
            )
            self.model_comm_group_rank = (
                int(os.environ.get("SLURM_PROCID", "0"))
                % hardware_config["num_gpus_per_model"]
            )
            self.model_comm_num_groups = math.ceil(
                hardware_config["num_gpus_per_node"]
                * hardware_config["num_nodes"]
                / hardware_config["num_gpus_per_model"],
            )
        else:
            # Lazy init
            self.model_comm_group = None
            self.model_comm_group_id = 0
            self.model_comm_group_rank = 0
            self.model_comm_num_groups = 1

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int = None,
        model_comm_group_rank: int = None,
        model_comm_num_groups: int = None,
        model_comm_group_size: int = None,
    ) -> None:
        self.model_comm_group = model_comm_group
        if not self.legacy:
            self.model_comm_group_id = model_comm_group_id
            self.model_comm_group_rank = model_comm_group_rank
            self.model_comm_num_groups = model_comm_num_groups
            self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    @abstractmethod
    def get_static_forcings(self, datareader):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def advance_input_predict(
        self, x: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def set_variable_indices(self, required_variables: list):
        pass


class BrisPredictor(BasePredictor):
    def __init__(
        self,
        *args,
        checkpoint: Checkpoint,
        datamodule: DataModule,
        forecast_length: int,
        required_variables: list,
        release_cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

        self.model = checkpoint.model
        self.data_indices = self.model.data_indices
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
        self.set_variable_indices(required_variables, datamodule)
        self.set_static_forcings(
            datamodule.data_reader, self.metadata.config.data.forcing
        )

        self.model.eval()
        self.release_cache = release_cache

    def set_static_forcings(self, data_reader, selection) -> None:
        self.static_forcings = {}
        batch = torch.from_numpy(data_reader[0].squeeze(axis=1).swapaxes(0, 1))
        data_input = torch.zeros(
            batch.shape[:-1] + (len(self.full_ordered_variable_list),),
            dtype=batch.dtype,
            device=batch.device,
        )
        data_input[..., self.indices_prognostic_input] = batch[
            ..., self.indices_prognostic_dataset
        ]
        data_input[..., self.indices_static_forcings_input] = batch[
            ..., self.indices_static_forcings_dataset
        ]
        data_normalized = self.model.pre_processors(data_input, in_place=True)

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

    def set_variable_indices(
        self, required_variables: list, datamodule: DataModule
    ) -> None:
        #
        required_variables = required_variables[0]  # Assume one decoder
        self.variable_indices_input = list()
        self.variable_indices_output = list()
        for name in required_variables:
            index_input = self.internal_data.input.name_to_index[name]
            self.variable_indices_input += [index_input]
            index_output = self.internal_model.output.name_to_index[name]
            self.variable_indices_output += [index_output]

        # Set up indices that can map from the variable order in the input data to the input variable order expected by the model
        # self.full_ordered_variable_list = self.metadata.dataset.variables
        self.full_ordered_variable_list = [
            var
            for var, _ in sorted(
                self.internal_data.input.name_to_index.items(), key=lambda item: item[1]
            )
        ]

        required_prognostic_variables = [
            name
            for name, index in self.internal_model.input.name_to_index.items()
            if index in self.internal_model.input.prognostic
        ]
        required_forcings = [
            name
            for name, index in self.internal_model.input.name_to_index.items()
            if index in self.internal_model.input.forcing
        ]
        self.required_dynamic_forcings = [
            forcing
            for forcing in anemoi_dynamic_forcings()
            if forcing in required_forcings
        ]
        required_static_forcings = [
            forcing
            for forcing in required_forcings
            if forcing not in anemoi_dynamic_forcings()
        ]

        # Check that all required variables are in the dataset
        missing_vars = [
            var
            for var in required_prognostic_variables + required_static_forcings
            if var not in datamodule.data_reader.variables
        ]
        if len(missing_vars) > 0:
            raise ValueError(
                f"Missing the following required variables in dataset: {missing_vars}"
            )

        self.indices_prognostic_dataset = torch.tensor(
            [
                index
                for index, var in enumerate(datamodule.data_reader.variables)
                if var in required_prognostic_variables
            ],
            dtype=torch.int64,
        )
        self.indices_static_forcings_dataset = torch.tensor(
            [
                index
                for index, var in enumerate(datamodule.data_reader.variables)
                if var in required_static_forcings
            ],
            dtype=torch.int64,
        )

        self.indices_prognostic_input = torch.tensor(
            [
                self.full_ordered_variable_list.index(var)
                for var in datamodule.data_reader.variables
                if var in required_prognostic_variables
            ],
            dtype=torch.int64,
        )
        self.indices_static_forcings_input = torch.tensor(
            [
                self.full_ordered_variable_list.index(var)
                for var in datamodule.data_reader.variables
                if var in required_static_forcings
            ],
            dtype=torch.int64,
        )
        self.indices_dynamic_forcings_input = torch.tensor(
            [
                self.full_ordered_variable_list.index(var)
                for var in datamodule.data_reader.variables
                if var in self.required_dynamic_forcings
            ],
            dtype=torch.int64,
        )

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
            time, self.latitudes, self.longitudes, self.required_dynamic_forcings
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
        y_preds = torch.zeros(
            (
                batch.shape[0],
                self.forecast_length,
                batch.shape[-2],
                len(self.variable_indices_output),
            ),
            dtype=batch.dtype,
            device="cpu",
        )

        # data_input has the variable order expected by the model. Prognostic and static forcings come from batch, dynamic forcings are calculated and
        # diagnostic variables are filled with 0.
        data_input = torch.zeros(
            batch.shape[:-1] + (len(self.full_ordered_variable_list),),
            dtype=batch.dtype,
            device=batch.device,
        )
        data_input[..., self.indices_prognostic_input] = batch[
            ..., self.indices_prognostic_dataset
        ]
        data_input[..., self.indices_static_forcings_input] = batch[
            ..., self.indices_static_forcings_dataset
        ]

        # Calculate dynamic forcings and add these to data_input
        for time_index in range(multistep):
            _time_stamp = time - (multistep - 1 - time_index) * self.timestep
            forcings = get_dynamic_forcings(
                _time_stamp,
                self.latitudes,
                self.longitudes,
                self.required_dynamic_forcings,
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

        y_preds[:, 0, :, :] = data_input[
            :, multistep - 1, ..., self.variable_indices_input
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
                )[:, 0, :, self.variable_indices_output].cpu()

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


class MultiEncDecPredictor(BasePredictor):
    def __init__(
        self,
        *args,
        checkpoint: Checkpoint,
        datamodule: DataModule,
        forecast_length: int,
        required_variables: list,
        release_cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

        self.model = checkpoint.model
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.forecast_length = forecast_length
        self.latitudes = datamodule.data_reader.latitudes
        self.longitudes = datamodule.data_reader.longitudes
        self.data_indices = self.model.data_indices
        self.set_variable_indices(required_variables, datamodule)

        self.set_static_forcings(
            datamodule.data_reader, self.metadata["config"]["data"]["zip"]
        )
        self.model.eval()

    def set_static_forcings(self, data_reader, zip_config):
        data = data_reader[0]
        num_dsets = len(data)
        data_input = []
        for dec_index in range(num_dsets):
            _batch = torch.from_numpy(data[dec_index].squeeze(axis=1).swapaxes(0, 1))
            _data_input = torch.zeros(
                _batch.shape[:-1] + (len(self.full_ordered_variable_list[dec_index]),),
                dtype=_batch.dtype,
                device=_batch.device,
            )
            _data_input[..., self.indices_prognostic_input[dec_index]] = _batch[
                ..., self.indices_prognostic_dataset[dec_index]
            ]
            _data_input[..., self.indices_static_forcings_input[dec_index]] = _batch[
                ..., self.indices_static_forcings_dataset[dec_index]
            ]
            data_input += [_data_input]

        data_normalized = self.model.pre_processors(data_input, in_place=False)

        self.static_forcings = [{} for _ in range(num_dsets)]
        for dset in range(num_dsets):
            selection = zip_config[dset]["forcing"]
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

    def set_variable_indices(
        self, required_variables: list, datamodule: DataModule
    ) -> None:
        self.variable_indices_input = [[] for _ in required_variables]
        self.variable_indices_output = [[] for _ in required_variables]
        self.full_ordered_variable_list = [[] for _ in required_variables]
        self.required_dynamic_forcings = [[] for _ in required_variables]
        self.indices_prognostic_dataset = [[] for _ in required_variables]
        self.indices_static_forcings_dataset = [[] for _ in required_variables]
        self.indices_prognostic_input = [[] for _ in required_variables]
        self.indices_static_forcings_input = [[] for _ in required_variables]
        self.indices_dynamic_forcings_input = [[] for _ in required_variables]
        required_prognostic_variables = [[] for _ in required_variables]
        required_forcings = [[] for _ in required_variables]
        required_static_forcings = [[] for _ in required_variables]

        for dec_index, required_vars_dec in required_variables.items():
            _variable_indices_input = list()
            _variable_indices_output = list()
            for name in required_vars_dec:
                index_input = self.data_indices[
                    dec_index
                ].internal_data.input.name_to_index[name]
                _variable_indices_input += [index_input]
                index_output = self.data_indices[
                    dec_index
                ].internal_model.output.name_to_index[name]
                _variable_indices_output += [index_output]
            self.variable_indices_input[dec_index] = _variable_indices_input
            self.variable_indices_output[dec_index] = _variable_indices_output

            # Set up indices that can map from the variable order in the input data to the input variable order expected by the model
            self.full_ordered_variable_list[dec_index] = [
                var
                for var, _ in sorted(
                    self.data_indices[
                        dec_index
                    ].internal_data.input.name_to_index.items(),
                    key=lambda item: item[1],
                )
            ]
            required_prognostic_variables[dec_index] = [
                name
                for name, index in self.data_indices[
                    dec_index
                ].internal_model.input.name_to_index.items()
                if index in self.data_indices[dec_index].internal_model.input.prognostic
            ]
            required_forcings[dec_index] = [
                name
                for name, index in self.data_indices[
                    dec_index
                ].internal_model.input.name_to_index.items()
                if index in self.data_indices[dec_index].internal_model.input.forcing
            ]
            self.required_dynamic_forcings[dec_index] = [
                forcing
                for forcing in anemoi_dynamic_forcings()
                if forcing in required_forcings[dec_index]
            ]
            required_static_forcings[dec_index] = [
                forcing
                for forcing in required_forcings[dec_index]
                if forcing not in anemoi_dynamic_forcings()
            ]

            # Check that all required variables are in the dataset
            missing_vars = [
                var
                for var in required_prognostic_variables[dec_index]
                + required_static_forcings[dec_index]
                if var not in datamodule.data_reader.datasets[dec_index].variables
            ]
            if len(missing_vars) > 0:
                raise ValueError(
                    f"Missing the following required variables in dataset {dec_index}: {missing_vars}"
                )

            self.indices_prognostic_dataset[dec_index] = torch.tensor(
                [
                    index
                    for index, var in enumerate(
                        datamodule.data_reader.datasets[dec_index].variables
                    )
                    if var in required_prognostic_variables[dec_index]
                ],
                dtype=torch.int64,
            )
            self.indices_static_forcings_dataset[dec_index] = torch.tensor(
                [
                    index
                    for index, var in enumerate(
                        datamodule.data_reader.datasets[dec_index].variables
                    )
                    if var in required_static_forcings[dec_index]
                ],
                dtype=torch.int64,
            )

            self.indices_prognostic_input[dec_index] = torch.tensor(
                [
                    self.full_ordered_variable_list[dec_index].index(var)
                    for var in datamodule.data_reader.datasets[dec_index].variables
                    if var in required_prognostic_variables[dec_index]
                ],
                dtype=torch.int64,
            )
            self.indices_static_forcings_input[dec_index] = torch.tensor(
                [
                    self.full_ordered_variable_list[dec_index].index(var)
                    for var in datamodule.data_reader.datasets[dec_index].variables
                    if var in required_static_forcings[dec_index]
                ],
                dtype=torch.int64,
            )
            self.indices_dynamic_forcings_input[dec_index] = torch.tensor(
                [
                    self.full_ordered_variable_list[dec_index].index(var)
                    for var in datamodule.data_reader.datasets[dec_index].variables
                    if var in self.required_dynamic_forcings[dec_index]
                ],
                dtype=torch.int64,
            )

        self.variable_indices_input = variable_indices_input
        self.variable_indices_output = variable_indices_output

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x, self.model_comm_group)

    def advance_input_predict(self, x, y_pred, time):
        data_indices = self.model.data_indices

        for i in range(len(x)):
            x[i] = x[i].roll(-1, dims=1)
            # Get prognostic variables:
            x[i][:, -1, :, :, data_indices[i].internal_model.input.prognostic] = y_pred[
                i
            ][..., data_indices[i].internal_model.output.prognostic]

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
                        data_indices[i].internal_model.input.name_to_index[forcing],
                    ] = torch.from_numpy(value)
                else:
                    x[i][
                        :,
                        -1,
                        :,
                        :,
                        data_indices[i].internal_model.input.name_to_index[forcing],
                    ] = value

        return x

    @torch.inference_mode
    def predict_step(self, batch: list, batch_idx: int) -> list:
        num_dsets = len(batch)
        data_indices = self.model.data_indices
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
                    len(self.variable_indices_input[i]),
                ),
                dtype=batch[i].dtype,
                device="cpu",
            )
            for i in range(num_dsets)
        ]
        # Insert analysis for t=0
        for i in range(num_dsets):
            y_analysis = batch[i][:, multistep - 1, 0, ...]
            y_analysis[..., data_indices[i].internal_data.output.diagnostic] = 0.0
            y_preds[i][:, 0, ...] = y_analysis[..., self.variable_indices_input[i]]

        batch = self.model.pre_processors(batch, in_place=True)
        x = [
            batch[i][..., data_indices[i].internal_data.input.full]
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
                        :, 0, ..., self.variable_indices_output[i]
                    ].cpu()
                times.append(time)

        return {
            "pred": y_preds,
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": 0,
        }
