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
from anemoi.datasets import open_dataset
from torch.distributed.distributed_c10d import ProcessGroup

from .checkpoint import Checkpoint
from .data.datamodule import DataModule
from .forcings import anemoi_dynamic_forcings, get_dynamic_forcings
from .utils import check_anemoi_training, timedelta64_from_timestep, get_all_leadtimes

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
        """
        Set model comm groups for model sharding over multiple gpus.
        """
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
        checkpoints_config: dict,
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
        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
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
        checkpoints_config: dict,
        required_variables: dict,
        release_cache: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, checkpoints=checkpoints, **kwargs)

        checkpoint = checkpoints["forecaster"]
        self.model = checkpoint.model
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
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
    

class Interpolator(BasePredictor):
    def __init__(
       self,
       *args,
       checkpoints: dict[str, Checkpoint],
       datamodule: DataModule,
       checkpoints_config: dict,
       required_variables: dict,
       release_cache: bool = False,
       **kwargs,
    ) -> None:
        super().__init__(*args, checkpoints=checkpoints, **kwargs)

        self.forecaster = checkpoints["forecaster"].model
        self.interpolator = checkpoints["interpolator"].model
        self.data_indices = {"forecaster": self.forecaster.data_indices,
                            "interpolator": self.interpolator.data_indices}
        self.multistep = checkpoints["forecaster"].metadata.config.training.multistep_input
        self.timestep_forecaster = timedelta64_from_timestep(checkpoints["forecaster"].metadata.config.data.timestep)
        self.timestep_interpolator = timedelta64_from_timestep(checkpoints["interpolator"].metadata.config.data.timestep)
        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
        self.interpolation_length = checkpoints_config["interpolator"]["leadtimes"]
        self.leadtimes = get_all_leadtimes(
            checkpoints_config["forecaster"]["leadtimes"],
            checkpoints_config["forecaster"]["timestep_seconds"],
            checkpoints_config["interpolator"]["leadtimes"],
            checkpoints_config["interpolator"]["timestep_seconds"],
        )
        self.interpolator_steps = 5 #TODO: figure out where to get this
        self.latitudes = datamodule.data_reader.latitudes
        self.longitudes = datamodule.data_reader.longitudes
        self.forcing_dataset_interp = open_dataset(checkpoints_config["interpolator"]["static_forcings_dataset"])
      
        # Set up variables and indices for both models
        # Use get_variables_indices for both, use internal_model.output to get the variable order "datamodule_variables"
        self.indices = {}
        self.variables = {}
        self.indices["forecaster"], self.variables["forecaster"] = get_variable_indices(
            required_variables[0], #Assume one decoder
            datamodule.data_reader.variables,
            self.data_indices["forecaster"].internal_data,
            self.data_indices["forecaster"].internal_model,
            0
        )
        self.indices["interpolator"], self.variables["interpolator"] = get_variable_indices(
            required_variables[0], #Assume decoder
            list(self.data_indices["forecaster"].internal_model.input.name_to_index.keys()),
            self.data_indices["interpolator"].internal_data,
            self.data_indices["interpolator"].internal_model,
            0,
            require_all_variables=False
        )
        
        self.indices["interpolator_forcings"], self.variables["interpolator_forcings"] = get_variable_indices(
            required_variables[0],
            self.forcing_dataset_interp.variables,
            self.data_indices["interpolator"].internal_data,
            self.data_indices["interpolator"].internal_model,
            0,
            require_all_variables=False
        )

        self.static_forcings_forecaster = self.get_static_forcings(
            datamodule.data_reader,
            checkpoints["forecaster"].metadata["config"]["data"],
            self.forecaster,
            self.variables["forecaster"],
            self.indices["forecaster"],
            self.data_indices["forecaster"].internal_data
        )
        
        self.static_forcings_interpolator = self.get_static_forcings(
            self.forcing_dataset_interp, 
            checkpoints["interpolator"].metadata["config"]["data"],
            self.interpolator,
            self.variables["interpolator_forcings"],
            self.indices["interpolator_forcings"],
            self.data_indices["interpolator"].internal_data
            )

        
        #Get these from config

        self.boundary_times = [0,6]
        self.interp_times = [1,2,3,4,5]
        self.target_forcings = ["insolation"]
        self.use_time_fraction = True

    # TODO: Move this to base class (should be able to write one function that works for interp, forecaster and multi-enc/dec)
    def get_static_forcings(
        self, 
        data_reader: Iterable, 
        data_config: dict, 
        model: torch.nn.Module, 
        variables: dict, 
        indices: dict, 
        internal_data: DataIndex,
        normalize: bool = True
    ) -> dict:
        selection = data_config["forcing"]
        data = torch.from_numpy(data_reader[0].squeeze(axis=1).swapaxes(0, 1))
        data_input = torch.zeros(
            data.shape[:-1] + (len(variables["all"]),),
            dtype=data.dtype,
            device=data.device,
        )

        data_input[..., indices["static_forcings_input"]] = data[
            ..., indices["static_forcings_dataset"]
        ]
        if normalize:
            data_input = model.pre_processors(data_input, in_place=True)

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
            static_forcings["lsm"] = data_input[
                ..., internal_data.input.name_to_index["lsm"]
            ].float()

        if "z" in selection:
            static_forcings["z"] = data_input[
                ..., internal_data.input.name_to_index["z"]
            ].float()

        return static_forcings
      
    @torch.inference_mode
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
           
        # Do the regular prediction (copy BrisForecaster)
        batch = self.allgather_batch(batch)
        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0])
        times = [time]
        y_preds = torch.empty(
            (
                batch.shape[0],
                len(self.leadtimes),
                batch.shape[-2],
                len(self.indices["forecaster"]["variables_output"]) #TODO: Should be both output from forecaster and interp
            ),
            dtype = batch.dtype,
            device = "cpu",
        )

        data_input = torch.zeros(
            batch.shape[:-1] + (len(self.variables["forecaster"]["all"]),),
            dtype = batch.dtype,
            device = batch.device,
        )
        data_input[..., self.indices["forecaster"]["prognostic_input"]] = batch[
            ..., self.indices["forecaster"]["prognostic_dataset"]
        ]
        data_input[..., self.indices["forecaster"]["static_forcings_input"]] = batch[
            ..., self.indices["forecaster"]["static_forcings_dataset"]
        ]

        for time_index in range(self.multistep):
            toi = time - (self.multistep - 1 - time_index) * self.timestep_forecaster
            forcings = get_dynamic_forcings(
                toi, self.latitudes, self.longitudes, self.variables["forecaster"]["dynamic_forcings"]
            )

            for forcing, value in forcings.items():
                if isinstance(value, np.ndarray):
                    data_input[
                        :,
                        time_index,
                        :,
                        :,
                        self.data_indices["forecaster"].internal_data.input.name_to_index[forcing],
                    ] = torch.from_numpy(value).to(dtype=data_input.dtype)
                else:
                    data_input[
                        :,
                        time_index,
                        :,
                        :,
                        self.data_indices["forecaster"].internal_data.input.name_to_index[forcing],
                    ] = value

        y_preds[:,0, ...] = data_input[
            :, self.multistep - 1, ..., self.indices["forecaster"]["variables_input"]
        ].cpu()

        x = self.forecaster.pre_processors(data_input, in_place=False)
        x = x[..., self.data_indices["forecaster"].internal_data.input.full]

        # Keep a non-normalized version of x for the interpolator - updated with output from forecaster (physical space) and interpolator forcings 
        # in advance_input_interpolator 
        x_interp = data_input[..., self.data_indices["forecaster"].internal_data.input.full]

        fcast_index = 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for fcast_step in range(self.forecast_length - 1):
                if self.model_comm_group_rank == 0:
                    print("time", time)
                y_pred = self.forecaster(x, self.model_comm_group)
                x = self.advance_input_predict(x, y_pred, time + self.timestep_forecaster)
                x_interp = self.advance_input_interpolator(x_interp, self.forecaster.post_processors(y_pred, in_place=False), time + self.timestep_forecaster)

                if fcast_step < self.interpolation_length:
                    
                    # Set up interpolator input
                    interpolator_input = torch.empty(
                        x_interp.shape[0],
                        len(self.boundary_times),
                        x_interp.shape[2],
                        x_interp.shape[3],
                        len(self.variables["interpolator"]["all"]),
                        dtype = batch.dtype,
                        device = batch.device,
                    )

                    interpolator_input[..., self.indices["interpolator"]["prognostic_input"]] =  x_interp[
                        :,-len(self.boundary_times):,:,:,self.indices["interpolator"]["prognostic_dataset"]
                    ]

                    for time_index, boundary_time in enumerate(self.boundary_times):
                        toi = time + boundary_time*self.timestep_interpolator
                        forcings = get_dynamic_forcings(
                            toi, self.latitudes, self.longitudes, self.variables["interpolator"]["dynamic_forcings"]
                        )
                        forcings.update(self.static_forcings_interpolator)

                        for forcing, value in forcings.items():
                            if isinstance(value, np.ndarray):
                                interpolator_input[
                                    :,
                                    time_index,
                                    :,
                                    :,
                                    self.data_indices["interpolator"].internal_data.input.name_to_index[forcing]
                                ] = torch.from_numpy(value).to(dtype=interpolator_input.dtype)
                            else:
                                interpolator_input[
                                    :,
                                    time_index,
                                    :,
                                    :,
                                    self.data_indices["interpolator"].internal_data.input.name_to_index[forcing]
                                ] = value
                    
                    # Do interpolator predictions
                    interpolator_input = self.interpolator.pre_processors(interpolator_input, in_place=True)
                    interpolator_input = interpolator_input[..., self.data_indices["interpolator"].internal_data.input.full]

                    # Setup target forcings
                    num_tfi = len(self.target_forcings)
                    target_forcing = torch.empty(
                        interpolator_input.shape[0],
                        interpolator_input.shape[2],
                        interpolator_input.shape[3],
                        num_tfi + self.use_time_fraction,
                        device=interpolator_input.device,
                        dtype=interpolator_input.dtype
                    )

                    for interp_index, interp_step in enumerate(self.interp_times):
                        time_interp = time + interp_step * self.timestep_interpolator
                        dynamic_target_forcings = get_dynamic_forcings(time_interp, self.longitudes, self.latitudes, self.target_forcings)
                        for forcing_index, forcing in enumerate(self.target_forcings):
                            if np.ndarray is type(dynamic_target_forcings[forcing]):
                                target_forcing[..., forcing_index] = torch.from_numpy(dynamic_target_forcings[forcing])
                            else:
                                target_forcing[..., forcing_index] = torch.from_numpy(dynamic_target_forcings[forcing])
                        if self.use_time_fraction:
                            target_forcing[..., -1] = (interp_step - self.boundary_times[1]) / (
                                self.boundary_times[1] - self.boundary_times[0]
                            )
                        if self.model_comm_group_rank == 0:
                            print("time_interp", time_interp)
                            print("time_frac", (interp_step - self.boundary_times[1]) / (
                                self.boundary_times[1] - self.boundary_times[0]
                            ))
                            print("index", fcast_index + interp_index)

                        y_pred_interp = self.interpolator(interpolator_input, target_forcing)
                        y_preds[:,fcast_index + interp_index] = self.interpolator.post_processors(
                            y_pred_interp, in_place=True
                        )[:, 0, :, self.indices["interpolator"]["variables_output"]].cpu()
                        times.append(time_interp)

                    fcast_index += self.interpolator_steps
                if self.model_comm_group_rank == 0:
                    print("fcast_index", fcast_index)
                y_preds[:,fcast_index] = self.forecaster.post_processors(
                    y_pred, in_place=True
                )[:, 0, :, self.indices["forecaster"]["variables_output"]].cpu()
                time += self.timestep_forecaster
                times.append(time)
                fcast_index += 1

        return {
            "pred": [y_preds.to(torch.float32).numpy()],
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": 0,
        }

    def advance_input_predict(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables:
        x[:, -1, :, :, self.data_indices["forecaster"].internal_model.input.prognostic] = y_pred[
            ..., self.data_indices["forecaster"].internal_model.output.prognostic
        ]

        forcings = get_dynamic_forcings(
            time, self.latitudes, self.longitudes, self.variables["forecaster"]["dynamic_forcings"]
        )
        forcings.update(self.static_forcings_forecaster)

        for forcing, value in forcings.items():
            if isinstance(value, np.ndarray):
                x[:, -1, :, :, self.data_indices["forecaster"].internal_model.input.name_to_index[forcing]] = (
                    torch.from_numpy(value).to(dtype=x.dtype)
                )
            else:
                x[:, -1, :, :, self.data_indices["forecaster"].internal_model.input.name_to_index[forcing]] = value
        return x
    
    def advance_input_interpolator(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables:
        x[:, -1, :, :, self.data_indices["forecaster"].internal_model.input.prognostic] = y_pred[
            ..., self.data_indices["forecaster"].internal_model.output.prognostic
        ]

        forcings = get_dynamic_forcings(
            time, self.latitudes, self.longitudes, self.variables["forecaster"]["dynamic_forcings"]
        )
        forcings.update(self.static_forcings_interpolator)

        for forcing, value in forcings.items():
            if isinstance(value, np.ndarray):
                x[:, -1, :, :, self.data_indices["forecaster"].internal_model.input.name_to_index[forcing]] = (
                    torch.from_numpy(value).to(dtype=x.dtype)
                )
            else:
                x[:, -1, :, :, self.data_indices["forecaster"].internal_model.input.name_to_index[forcing]] = value
        return x

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch  # Not implemented properly

        
def get_variable_indices(
    required_variables: list,
    datamodule_variables: list,
    internal_data: DataIndex,
    internal_model: ModelIndex,
    decoder_index: int,
    require_all_variables: bool = True
) -> tuple[dict, dict]:
    """
    Get variables and indices needed to map between data input and model input. 
    
    Parameters
    ----------
    required_variables: list
        List of required output variables (given by ouputs in routing config)
    datamodule_variables: list
        List of variables in the input data (datamodule)
    internal_data: DataIndex
        Indexing for variables in input dataset - gives the correct variable order when mapping from input to model
    internal_model:
        Indexing for variables in the model - gives correct variable output order
    decoder_index:
        Decoder index, used to throw a better error for missing variables
    require_all_variables:
        Whether to throw an error if there are missing variables


    Returns:
    --------
    variables: dict
        all: full ordered list of variables in input dataset
        dynamic forcings: dynamic forcings required by the model
    indices: dict
        variables_input: Indices for required variables in input dataset (used to create analysis in output)
        variables_output: Indices for required variables in model output 
        prognostic_dataset: Correct order of indices of prognostic variables in input dataset (used to map from batch)
        static_forcings_dataset: Correct order of indices of prognostic variables in input dataset (used to map from batch)
        prognostic_input: Model input variable order for prognostic variables
        static_forcings_input: Model input variable order for static forcings
        dynamic_forcings_input: Model input variable order for dynamic forcings

    """

    # Set up indices for the variables we want to write to file
    variable_indices_input = list()
    variable_indices_output = list()
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
        if require_all_variables:
            raise ValueError(
                f"Missing the following required variables in dataset {decoder_index}: {missing_vars}"
            )
        else:
            print(f"WARNING: Missing the following required variables in dataset {decoder_index}: {missing_vars}")

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
