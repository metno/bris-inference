import logging
from collections.abc import Iterable

import numpy as np
import torch
from anemoi.models.data_indices.index import DataIndex
from anemoi.datasets import open_dataset

from ..checkpoint import Checkpoint
from ..data.data_module import DataModule
from ..forcings import get_dynamic_forcings


from ..utils import (
    LOGGER,
    timedelta64_from_timestep,
    get_all_leadtimes,
)

from .basepredictor import BasePredictor
from .model_utils import get_variable_indices




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
        self.timestep_interpolator = timedelta64_from_timestep("1h") #checkpoints["interpolator"].metadata.config.data.timestep)
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
                y_pred = self.forecaster(x, model_comm_group = self.model_comm_group)
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
                        dynamic_target_forcings = get_dynamic_forcings(time_interp, self.latitudes, self.longitudes, self.target_forcings)
                        for forcing_index, forcing in enumerate(self.target_forcings):
                            if np.ndarray is type(dynamic_target_forcings[forcing]):
                                target_forcing[..., forcing_index] = torch.from_numpy(dynamic_target_forcings[forcing])
                            else:
                                target_forcing[..., forcing_index] = dynamic_target_forcings[forcing]
                        if self.use_time_fraction:
                            target_forcing[..., -1] = (interp_step - self.boundary_times[-2]) / (
                                self.boundary_times[-1] - self.boundary_times[-2]
                            )
                        if self.model_comm_group_rank == 0:
                            # print("time_interp", time_interp)
                            # print("time_frac", (interp_step - self.boundary_times[-2]) / (
                            #     self.boundary_times[-1] - self.boundary_times[-2]
                            # ))
                            # #print("index", fcast_index + interp_index)
                            pass

                        y_pred_interp = self.interpolator(interpolator_input, target_forcing = target_forcing, model_comm_group=self.model_comm_group)
                        y_preds[:,fcast_index + interp_index] = self.interpolator.post_processors(
                            y_pred_interp, in_place=True
                        )[:, 0, :, self.indices["interpolator"]["variables_output"]].cpu()
                        times.append(time_interp)

                    fcast_index += self.interpolator_steps
                if self.model_comm_group_rank == 0:
                    pass
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