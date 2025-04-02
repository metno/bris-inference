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

from .checkpoint import Checkpoint
from .data.datamodule import DataModule
from .forcings import anemoi_dynamic_forcings, get_dynamic_forcings
from .utils import check_anemoi_training, timedelta64_from_timestep

LOGGER = logging.getLogger(__name__)


class BasePredictor(pl.LightningModule):
    def __init__(
        self,
        *args: Any,
        checkpoints: dict[str, Checkpoint],
        hardware_config: dict,
        **kwargs: Any,
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

        if check_anemoi_training(checkpoints["forecaster"].metadata):
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
    def set_static_forcings(
        self,
        datareader: Iterable,
    ) -> None:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        pass

    @abstractmethod
    def advance_input_predict(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        y_pred: Union[torch.Tensor, list[torch.Tensor]],
        time: np.datetime64,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        pass


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


def get_variable_indices(
    required_variables: list,
    datamodule_variables: list,
    internal_data: DataIndex,
    internal_model: ModelIndex,
    decoder_index: int,
) -> tuple[dict, dict]:
    # Set up indices for the variables we want to write to file
    variable_indices_input = []
    variable_indices_output = []
    for name in required_variables:
        variable_indices_input.append(internal_data.input.name_to_index[name])
        variable_indices_output.append(internal_model.output.name_to_index[name])

    # Set up indices that can map from the variable order in the input data to the input variable order expected by the model
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
