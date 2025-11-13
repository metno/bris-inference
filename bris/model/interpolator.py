import logging
from collections.abc import Iterable

import numpy as np
import torch
from anemoi.datasets import open_dataset
from anemoi.models.data_indices.index import DataIndex

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..forcings import get_dynamic_forcings
from ..utils import (
    LOGGER,
    get_all_leadtimes,
    timedelta64_from_timestep,
)
from .basepredictor import BasePredictor
from .model_utils import get_variable_indices


class Interpolator(BasePredictor):
    """
    Combined Forecaster and Interpolator model.

    Methods
    -------

    __init__

    get_static_forcings: Get static forcings for the model.

    predict_step: Perform prediction step combining forecaster and interpolator.

    advance_input_predict: Advance input for forecaster predictions.

    advance_input_interpolator: Advance input for interpolator predictions.

    allgather_batch: Gather batch across distributed processes (not implemented).
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
        Initialize Interpolator.

        Args:
            checkpoints: Dictionary containing 'forecaster' and 'interpolator' Checkpoint objects.
            datamodule: DataModule object for data handling.
            checkpoints_config: Configuration dictionary for checkpoints.
            required_variables: Required variables for the model.
            release_cache: Boolean flag to release cache (default: False).
            fcstep_const: Boolean flag for constant forecast step (default: False).
        """

        super().__init__(*args, checkpoints=checkpoints, **kwargs)
        self.forecaster = checkpoints["forecaster"].model
        self.interpolator = checkpoints["interpolator"].model
        self.data_indices = {
            "forecaster": self.forecaster.data_indices,
            "interpolator": self.interpolator.data_indices,
        }

        # Backwards compatibility fix
        for data_indices in self.data_indices.values():
            if hasattr(data_indices, "internal_data") and hasattr(
                data_indices, "internal_model"
            ):
                continue
            else:
                data_indices.internal_data = data_indices.data
                data_indices.internal_model = data_indices.model

        self.multistep = checkpoints[
            "forecaster"
        ].metadata.config.training.multistep_input
        self.timestep_forecaster = np.timedelta64(
            checkpoints_config["forecaster"]["timestep_seconds"], "s"
        )
        self.timestep_interpolator = np.timedelta64(
            checkpoints_config["interpolator"]["timestep_seconds"], "s"
        )

        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
        self.interpolation_length = checkpoints_config["interpolator"]["leadtimes"]
        self.leadtimes = get_all_leadtimes(
            checkpoints_config["forecaster"]["leadtimes"],
            checkpoints_config["forecaster"]["timestep_seconds"],
            checkpoints_config["interpolator"]["leadtimes"],
            checkpoints_config["interpolator"]["timestep_seconds"],
        )
        self.latitudes = datamodule.data_reader.latitudes
        self.longitudes = datamodule.data_reader.longitudes
        self.forcing_dataset_interp = open_dataset(
            checkpoints_config["interpolator"]["static_forcings_dataset"]
        )

        # Set up variables and indices for both models
        # Use get_variables_indices for both, use internal_model.output to get the variable order "datamodule_variables"
        self.indices = {}
        self.variables = {}
        self.indices["forecaster"], self.variables["forecaster"] = get_variable_indices(
            required_variables[0],  # Assume one decoder
            datamodule.data_reader.variables,
            self.data_indices["forecaster"].internal_data,
            self.data_indices["forecaster"].internal_model,
            0,
        )
        self.indices["interpolator"], self.variables["interpolator"] = (
            get_variable_indices(
                required_variables[0],  # Assume one decoder
                list(
                    self.data_indices[
                        "forecaster"
                    ].internal_model.input.name_to_index.keys()
                ),
                self.data_indices["interpolator"].internal_data,
                self.data_indices["interpolator"].internal_model,
                0,
            )
        )

        (
            self.indices["interpolator_forcings"],
            self.variables["interpolator_forcings"],
        ) = get_variable_indices(
            required_variables[0],
            self.forcing_dataset_interp.variables,
            self.data_indices["interpolator"].internal_data,
            self.data_indices["interpolator"].internal_model,
            0,
        )

        self.static_forcings_forecaster = self.get_static_forcings(
            datamodule.data_reader,
            checkpoints["forecaster"].metadata["config"]["data"],
            self.forecaster,
            self.variables["forecaster"],
            self.indices["forecaster"],
            self.data_indices["forecaster"].internal_data,
        )

        self.static_forcings_interpolator = self.get_static_forcings(
            self.forcing_dataset_interp,
            checkpoints["interpolator"].metadata["config"]["data"],
            self.interpolator,
            self.variables["interpolator_forcings"],
            self.indices["interpolator_forcings"],
            self.data_indices["interpolator"].internal_data,
        )

        self.boundary_times = checkpoints[
            "interpolator"
        ].metadata.config.training.explicit_times.input
        self.interp_times = checkpoints[
            "interpolator"
        ].metadata.config.training.explicit_times.target
        self.interpolator_steps = len(self.interp_times)
        self.target_forcings = checkpoints[
            "interpolator"
        ].metadata.config.training.target_forcing.data
        self.use_time_fraction = checkpoints[
            "interpolator"
        ].metadata.config.training.target_forcing.time_fraction

        self.reforcast_last = self.boundary_times[-1] == self.interp_times[-1]

        self.batch_info = {}
        self.fcstep_const = fcstep_const

    def update_batch_info(self, time):
        if time not in self.batch_info:
            self.batch_info[time] = 1
        else:
            self.batch_info[time] += 1

    def get_static_forcings(
        self,
        data_reader: Iterable,
        data_config: dict,
        model: torch.nn.Module,
        variables: dict,
        indices: dict,
        internal_data: DataIndex,
        normalize: bool = True,
    ) -> dict:
        """
        Get static forcings for the model.

        Args:
            data_reader: Data reader or dataset to extract static forcings from.
            data_config: Configuration dictionary for data.
            model: The model for which static forcings are being retrieved.
            variables: Dictionary of variables required by the model.
            indices: Dictionary of indices mapping variables to their positions.
            internal_data: Internal data index for the model.
            normalize: Boolean flag to indicate if normalization is needed (default: True).
        Returns:
            Dictionary of static forcings.
        """
        selection = data_config["forcing"]
        data = torch.from_numpy(data_reader[0].squeeze(axis=1).swapaxes(0, 1))
        data_input = torch.full(
            data.shape[:-1] + (len(variables["all"]),),
            float("nan"),
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
        """
        Perform prediction step combining forecaster and interpolator.

        Args:
            batch: Input batch for prediction.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing predictions, times, group rank, and ensemble member.
        """

        batch = self.allgather_batch(batch)
        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0])
        times = [time]
        y_preds = torch.empty(
            (
                batch.shape[0],
                len(self.leadtimes),
                batch.shape[-2],
                len(
                    self.indices["forecaster"]["variables_output"]
                ),  # TODO: Should be both output from forecaster and interp
            ),
            dtype=batch.dtype,
            device="cpu",
        )

        data_input = torch.zeros(
            batch.shape[:-1] + (len(self.variables["forecaster"]["all"]),),
            dtype=batch.dtype,
            device=batch.device,
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
                toi,
                self.latitudes,
                self.longitudes,
                self.variables["forecaster"]["dynamic_forcings"],
            )

            for forcing, value in forcings.items():
                if isinstance(value, np.ndarray):
                    data_input[
                        :,
                        time_index,
                        :,
                        :,
                        self.data_indices[
                            "forecaster"
                        ].internal_data.input.name_to_index[forcing],
                    ] = torch.from_numpy(value).to(dtype=data_input.dtype)
                else:
                    data_input[
                        :,
                        time_index,
                        :,
                        :,
                        self.data_indices[
                            "forecaster"
                        ].internal_data.input.name_to_index[forcing],
                    ] = value

        y_preds[:, 0, ...] = data_input[
            :, self.multistep - 1, ..., self.indices["forecaster"]["variables_input"]
        ].cpu()

        x = self.forecaster.pre_processors(data_input, in_place=False)
        x = x[..., self.data_indices["forecaster"].internal_data.input.full]

        # Keep a non-normalized version of x for the interpolator - updated with output from forecaster (physical space) and interpolator forcings
        # in advance_input_interpolator
        x_interp = data_input[
            ..., self.data_indices["forecaster"].internal_data.input.full
        ]

        fcast_index = 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for fcast_step in range(self.forecast_length - 1):
                try:
                    if self.fcstep_const:
                        y_pred = self.forecaster(
                            x, model_comm_group=self.model_comm_group, fcstep=0
                        )
                    else:
                        y_pred = self.forecaster(
                            x, model_comm_group=self.model_comm_group, fcstep=fcast_step
                        )
                except TypeError:
                    y_pred = self.forecaster(x, model_comm_group=self.model_comm_group)

                x = self.advance_input_predict(
                    x, y_pred, time + self.timestep_forecaster
                )
                x_interp = self.advance_input_interpolator(
                    x_interp,
                    self.forecaster.post_processors(y_pred, in_place=False),
                    time + self.timestep_forecaster,
                )

                if fcast_step < self.interpolation_length:
                    # Set up interpolator input
                    interpolator_input = torch.empty(
                        x_interp.shape[0],
                        len(self.boundary_times),
                        x_interp.shape[2],
                        x_interp.shape[3],
                        len(self.variables["interpolator"]["all"]),
                        dtype=batch.dtype,
                        device=batch.device,
                    )

                    interpolator_input[
                        ..., self.indices["interpolator"]["prognostic_input"]
                    ] = x_interp[
                        :,
                        -len(self.boundary_times) :,
                        :,
                        :,
                        self.indices["interpolator"]["prognostic_dataset"],
                    ]

                    for time_index, boundary_time in enumerate(self.boundary_times):
                        toi = time + boundary_time * self.timestep_interpolator
                        forcings = get_dynamic_forcings(
                            toi,
                            self.latitudes,
                            self.longitudes,
                            self.variables["interpolator"]["dynamic_forcings"],
                        )
                        forcings.update(self.static_forcings_interpolator)

                        for forcing, value in forcings.items():
                            if isinstance(value, np.ndarray):
                                interpolator_input[
                                    :,
                                    time_index,
                                    :,
                                    :,
                                    self.data_indices[
                                        "interpolator"
                                    ].internal_data.input.name_to_index[forcing],
                                ] = torch.from_numpy(value).to(
                                    dtype=interpolator_input.dtype
                                )
                            else:
                                interpolator_input[
                                    :,
                                    time_index,
                                    :,
                                    :,
                                    self.data_indices[
                                        "interpolator"
                                    ].internal_data.input.name_to_index[forcing],
                                ] = value

                    # Do interpolator predictions
                    interpolator_input = self.interpolator.pre_processors(
                        interpolator_input, in_place=True
                    )
                    interpolator_input = interpolator_input[
                        ..., self.data_indices["interpolator"].internal_data.input.full
                    ]

                    # Setup target forcings
                    num_tfi = len(self.target_forcings)
                    target_forcing = torch.empty(
                        interpolator_input.shape[0],
                        interpolator_input.shape[2],
                        interpolator_input.shape[3],
                        num_tfi + self.use_time_fraction,
                        device=interpolator_input.device,
                        dtype=interpolator_input.dtype,
                    )

                    for interp_index, interp_step in enumerate(self.interp_times):
                        time_interp = time + interp_step * self.timestep_interpolator
                        dynamic_target_forcings = get_dynamic_forcings(
                            time_interp,
                            self.latitudes,
                            self.longitudes,
                            self.target_forcings,
                        )
                        for forcing_index, forcing in enumerate(self.target_forcings):
                            if np.ndarray is type(dynamic_target_forcings[forcing]):
                                target_forcing[..., forcing_index] = torch.from_numpy(
                                    dynamic_target_forcings[forcing]
                                )
                            else:
                                target_forcing[..., forcing_index] = (
                                    dynamic_target_forcings[forcing]
                                )
                        if self.use_time_fraction:
                            target_forcing[..., -1] = (
                                interp_step - self.boundary_times[-2]
                            ) / (self.boundary_times[-1] - self.boundary_times[-2])

                        y_pred_interp = self.interpolator(
                            interpolator_input,
                            target_forcing=target_forcing,
                            model_comm_group=self.model_comm_group,
                        )
                        y_preds[:, fcast_index + interp_index] = (
                            self.interpolator.post_processors(
                                y_pred_interp, in_place=True
                            )[
                                :,
                                0,
                                :,
                                self.indices["interpolator"]["variables_output"],
                            ].cpu()
                        )
                        times.append(time_interp)

                    fcast_index += self.interpolator_steps
                if not self.reforcast_last:    
                    y_preds[:, fcast_index] = self.forecaster.post_processors(
                        y_pred, in_place=True
                    )[:, 0, :, self.indices["forecaster"]["variables_output"]].cpu()
                    time += self.timestep_forecaster
                    times.append(time)
                    fcast_index += 1
                else:
                    time += self.timestep_forecaster
                    
        self.update_batch_info(time)
        return {
            "pred": [y_preds.to(torch.float32).numpy()],
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": self.member_id
            + self.num_members_in_parallel * (self.batch_info[time] - 1),
        }

    def advance_input_predict(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        """
        Advance input for forecaster predictions.
        Args:
            x: Current input tensor for the forecaster.
            y_pred: Predictions from the forecaster model.
            time: Current time as a numpy datetime64 object.
        Returns:
            Updated input tensor for the forecaster.
        """

        x = x.roll(-1, dims=1)

        # Get prognostic variables:
        x[
            :, -1, :, :, self.data_indices["forecaster"].internal_model.input.prognostic
        ] = y_pred[
            ..., self.data_indices["forecaster"].internal_model.output.prognostic
        ]

        forcings = get_dynamic_forcings(
            time,
            self.latitudes,
            self.longitudes,
            self.variables["forecaster"]["dynamic_forcings"],
        )
        forcings.update(self.static_forcings_forecaster)

        for forcing, value in forcings.items():
            if isinstance(value, np.ndarray):
                x[
                    :,
                    -1,
                    :,
                    :,
                    self.data_indices["forecaster"].internal_model.input.name_to_index[
                        forcing
                    ],
                ] = torch.from_numpy(value).to(dtype=x.dtype)
            else:
                x[
                    :,
                    -1,
                    :,
                    :,
                    self.data_indices["forecaster"].internal_model.input.name_to_index[
                        forcing
                    ],
                ] = value
        return x

    def advance_input_interpolator(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        """
        Advance input for interpolator predictions.

        Args:
            x: Current input tensor for the interpolator.
            y_pred: Predictions from the forecaster model.
            time: Current time as a numpy datetime64 object.
        Returns:
            Updated input tensor for the interpolator.
        """
        x = x.roll(-1, dims=1)

        # Get prognostic variables:
        x[
            :, -1, :, :, self.data_indices["forecaster"].internal_model.input.prognostic
        ] = y_pred[
            ..., self.data_indices["forecaster"].internal_model.output.prognostic
        ]

        forcings = get_dynamic_forcings(
            time,
            self.latitudes,
            self.longitudes,
            self.variables["forecaster"]["dynamic_forcings"],
        )
        forcings.update(self.static_forcings_interpolator)

        for forcing, value in forcings.items():
            if isinstance(value, np.ndarray):
                x[
                    :,
                    -1,
                    :,
                    :,
                    self.data_indices["forecaster"].internal_model.input.name_to_index[
                        forcing
                    ],
                ] = torch.from_numpy(value).to(dtype=x.dtype)
            else:
                x[
                    :,
                    -1,
                    :,
                    :,
                    self.data_indices["forecaster"].internal_model.input.name_to_index[
                        forcing
                    ],
                ] = value
        return x

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch  # Not implemented
