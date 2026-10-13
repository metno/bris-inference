import numpy as np
import torch
from anemoi.datasets import open_dataset

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..utils import LOGGER, get_all_leadtimes
from .basepredictor import BasePredictor
from .model_utils import (
    get_interpolator_boundary_times,
    get_interpolator_interp_times,
    static_forcing_time_index,
)
from .wrapper import ModelWrapper


class Interpolator(BasePredictor):
    """
    Combined forecaster and temporal interpolator.

    The forecaster is rolled out as in BrisPredictor; after each of the first
    ``interpolation_length`` forecaster steps the interpolator fills in the lead times between
    the boundary states (the last forecaster states, in physical units).

    Methods
    -------

    __init__

    update_static_forcings: Set the static forcings of both models for a forecast.

    interpolate: Run the interpolator between the latest boundary states.

    predict_step: Perform prediction step combining forecaster and interpolator.
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
            checkpoints_config: config.checkpoints; both models need "timestep_seconds" and "leadtimes",
                the interpolator also "static_forcings_dataset".
            required_variables: Required variables per dataset.
            release_cache: Boolean flag to release cache (default: False).
            fcstep_const: Boolean flag for constant forecast step (default: False).
        """

        super().__init__(
            *args,
            checkpoints=checkpoints,
            release_cache=release_cache,
            fcstep_const=fcstep_const,
            **kwargs,
        )
        self.forecaster = ModelWrapper(checkpoints["forecaster"])
        self.interpolator = ModelWrapper(checkpoints["interpolator"])
        assert len(self.forecaster.dataset_names) == 1, (
            "Interpolator is currently only compatible with single dataset checkpoints"
        )
        assert len(self.interpolator.dataset_names) == 1, (
            "Interpolator is currently only compatible with single dataset checkpoints"
        )
        self.ds_name_fc = self.forecaster.dataset_names[0]
        self.ds_name_interp = self.interpolator.dataset_names[0]

        self.multistep = self.forecaster.multistep
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

        data_reader = datamodule.data_readers[self.ds_name_fc]
        self.forecaster_inputs = self.forecaster.add_dataset(
            self.ds_name_fc,
            required_variables=required_variables[self.ds_name_fc],
            dataset_variables=data_reader.variables,
            latitudes=data_reader.latitudes,
            longitudes=data_reader.longitudes,
        )
        # The interpolator input is assembled from the forecaster's model input (physical units)
        self.interpolator_inputs = self.interpolator.add_dataset(
            self.ds_name_interp,
            required_variables=required_variables[self.ds_name_interp],
            dataset_variables=list(
                self.forecaster.model_index[self.ds_name_fc].input.name_to_index.keys()
            ),
            latitudes=data_reader.latitudes,
            longitudes=data_reader.longitudes,
        )

        # The interpolator's static forcings come from their own dataset, the time step valid at
        # the start of each forecast (see update_static_forcings)
        self.forcing_dataset_interp = open_dataset(
            checkpoints_config["interpolator"]["static_forcings_dataset"]
        )
        missing = [
            var
            for var in self.interpolator_inputs.mapping.static_forcings
            if var not in self.forcing_dataset_interp.variables
        ]
        if len(missing) > 0:
            raise ValueError(
                f"Missing the following static forcings in the interpolator's static_forcings_dataset: {missing}"
            )
        self.static_forcings_forecaster = {}
        self.static_forcings_interpolator = {}
        self._static_forcings_interpolator_index = None

        self.boundary_times = get_interpolator_boundary_times(
            checkpoints["interpolator"]
        )
        self.interp_times = get_interpolator_interp_times(checkpoints["interpolator"])
        # Whether the interpolator also outputs the last boundary time (then the forecaster's
        # output at that time is not written)
        self.reforcast_last = self.boundary_times[-1] == self.interp_times[-1]

    def interpolator_static_forcings(self, time_index: int) -> dict:
        """The static forcings of the interpolator from a time step of static_forcings_dataset, physical units."""
        data = torch.from_numpy(
            self.forcing_dataset_interp[time_index].squeeze(axis=1).swapaxes(0, 1)
        )
        variables = self.forcing_dataset_interp.variables
        return {
            name: data[..., variables.index(name)].contiguous().float()
            for name in self.interpolator_inputs.mapping.static_forcings
        }

    def update_static_forcings(
        self, time: np.datetime64, state_normalized: torch.Tensor
    ) -> None:
        """
        Set the static forcings of both models for a forecast starting at ``time``.

        The forecaster's come from the normalized starting state of the forecast, the
        interpolator's from the time step of static_forcings_dataset valid at ``time`` (the
        latest date <= time), cached between forecasts that map to the same time step.

        Args:
            time: Start time of the forecast.
            state_normalized: Normalized forecaster input in data order,
                shape (batch, multistep, ensemble, points, variables).
        """
        self.static_forcings_forecaster = self.forecaster_inputs.static_forcings(
            state_normalized[:, self.multistep - 1]
        )

        index = static_forcing_time_index(self.forcing_dataset_interp.dates, time)
        if index != self._static_forcings_interpolator_index:
            LOGGER.info(
                "Interpolator static forcings from %s (time step %d of static_forcings_dataset) for forecasts from %s",
                self.forcing_dataset_interp.dates[index],
                index,
                time,
            )
            self.static_forcings_interpolator = self.interpolator_static_forcings(index)
            self._static_forcings_interpolator_index = index

    def interpolate(
        self,
        x_physical: torch.Tensor,
        time: np.datetime64,
        y_preds: torch.Tensor,
        times: list[np.datetime64],
        out_index: int,
    ) -> int:
        """
        Run the interpolator between the latest boundary states and write its output.

        Args:
            x_physical: Forecaster states in model order and physical units, the last
                len(boundary_times) of which are the boundaries of the interpolation.
            time: Time of the first boundary state.
            y_preds: Output array, written at out_index, out_index + 1, ...
            times: Output times, appended to.
            out_index: First output index to write.
        Returns:
            int: The next free output index.
        """
        inputs = self.interpolator_inputs
        boundary_times = [
            time + boundary_time * self.timestep_interpolator
            for boundary_time in self.boundary_times
        ]
        x = inputs.build_state(
            x_physical[:, -len(self.boundary_times) :],
            boundary_times,
            static_forcings=self.static_forcings_interpolator,
        )
        x = self.interpolator.pre_process(x, self.ds_name_interp, in_place=True)
        x = inputs.to_model_order(x)

        # fcstep is always 0 for the interpolator
        y_pred = self.interpolator(
            {self.ds_name_interp: x}, model_comm_group=self.model_comm_group, fcstep=0
        )[self.ds_name_interp]

        for interp_index, interp_step in enumerate(self.interp_times):
            y_preds[:, out_index + interp_index] = inputs.output(
                self.interpolator.post_process(
                    y_pred[:, interp_index], self.ds_name_interp, in_place=True
                )
            )
            times.append(time + interp_step * self.timestep_interpolator)
        return out_index + len(self.interp_times)

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
        assert len(batch) == 1, (
            "Interpolator is currently only compatible with single dataset checkpoints"
        )
        batch = batch[self.ds_name_fc]
        time = np.datetime64(time_stamp[0])
        times = [time]

        inputs = self.forecaster_inputs
        y_preds = torch.empty(
            (
                batch.shape[0],
                len(self.leadtimes),
                batch.shape[-2],
                len(
                    inputs.mapping.variables_output
                ),  # TODO: Should be both output from forecaster and interp
            ),
            dtype=batch.dtype,
            device="cpu",
        )

        input_times = [
            time - (self.multistep - 1 - time_index) * self.timestep_forecaster
            for time_index in range(self.multistep)
        ]
        state = inputs.build_state(batch, input_times)
        y_preds[:, 0] = inputs.initial_output(state, self.multistep - 1)

        # Forecaster states in physical units: the boundary values of the interpolator
        x_physical = inputs.to_model_order(state)

        state = self.forecaster.pre_process(state, self.ds_name_fc, in_place=True)
        self.update_static_forcings(time, state)
        x = inputs.to_model_order(state)

        out_index = 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for forecast_step in range(self.forecast_length - 1):
                y_pred = self.forecaster(
                    {self.ds_name_fc: x},
                    model_comm_group=self.model_comm_group,
                    fcstep=self.fcstep(forecast_step),
                )[self.ds_name_fc]
                y_physical = self.forecaster.post_process(
                    y_pred, self.ds_name_fc, in_place=False
                )

                next_time = time + self.timestep_forecaster
                x = inputs.advance(
                    x, y_pred, next_time, self.static_forcings_forecaster
                )
                x_physical = inputs.roll(x_physical, y_physical)

                if forecast_step < self.interpolation_length:
                    out_index = self.interpolate(
                        x_physical, time, y_preds, times, out_index
                    )
                if not self.reforcast_last:
                    y_preds[:, out_index] = inputs.output(y_physical)
                    times.append(next_time)
                    out_index += 1
                time = next_time

                if self.release_cache:
                    del y_pred, y_physical
                    torch.cuda.empty_cache()

        return self.prediction({self.ds_name_fc: y_preds}, times)
