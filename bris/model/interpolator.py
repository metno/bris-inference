import numpy as np
import torch
from anemoi.datasets import open_dataset

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from .basepredictor import BasePredictor, ForecastOutputs
from .model_utils import (
    get_interpolator_boundary_times,
    get_interpolator_interp_times,
)
from .runner import ModelRunner


class Interpolator(BasePredictor):
    """
    Combined forecaster and temporal interpolator.

    The forecaster is rolled out as in BrisPredictor; after each of the first
    ``interpolation_length`` forecaster steps the interpolator fills in the lead times between
    the boundary states (the last forecaster states, in physical units). An interpolator that
    also outputs the right boundary provides that lead time instead of the forecaster, unless
    ``reforecast_boundary`` is False in its config.

    The checkpoints' properties (time steps, input/output steps, dataset names, indices,
    processors) live on ``self.forecaster`` and ``self.interpolator`` (ModelRunners); this
    class holds the run configuration (forecast and interpolation length, release_cache,
    fcstep_const) and the combined rollout.

    Methods
    -------

    __init__

    update_static_forcings: Set the static forcings of both models for a forecast.

    forecast: The forecaster rollout with the interpolator filling in between its steps.
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
                the interpolator also "static_forcings_dataset" and optionally "reforecast_boundary"
                (default True, see the class docstring).
            required_variables: Variables the outputs require, per dataset.
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
        self.forecaster = ModelRunner(
            checkpoints["forecaster"],
            timestep=np.timedelta64(
                checkpoints_config["forecaster"]["timestep_seconds"], "s"
            ),
        )
        self.interpolator = ModelRunner(
            checkpoints["interpolator"],
            timestep=np.timedelta64(
                checkpoints_config["interpolator"]["timestep_seconds"], "s"
            ),
            input_steps=get_interpolator_boundary_times(checkpoints["interpolator"]),
            output_steps=get_interpolator_interp_times(checkpoints["interpolator"]),
        )
        assert len(self.forecaster.dataset_names) == 1, (
            "Interpolator is currently only compatible with single dataset checkpoints"
        )
        assert len(self.interpolator.dataset_names) == 1, (
            "Interpolator is currently only compatible with single dataset checkpoints"
        )
        self.ds_name_fc = self.forecaster.dataset_names[0]
        self.ds_name_interp = self.interpolator.dataset_names[0]

        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
        self.interpolation_length = checkpoints_config["interpolator"]["leadtimes"]

        data_reader = datamodule.data_readers[self.ds_name_fc]
        self.forecaster_builder = self.forecaster.add_dataset(
            self.ds_name_fc,
            output_variables=required_variables[self.ds_name_fc],
            dataset_variables=data_reader.variables,
            latitudes=data_reader.latitudes,
            longitudes=data_reader.longitudes,
        )
        # The interpolator input is assembled from the forecaster's model input (physical units)
        self.interpolator_builder = self.interpolator.add_dataset(
            self.ds_name_interp,
            output_variables=required_variables[self.ds_name_interp],
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

        # Whether the interpolator provides the right boundary of each interval: it outputs it
        # and reforecast_boundary is not switched off (then the forecaster's output at that time
        # is not written)
        self.reforcast_last = (
            self.interpolator.input_steps[-1] == self.interpolator.output_steps[-1]
        ) and bool(checkpoints_config["interpolator"].get("reforecast_boundary", True))

    def update_static_forcings(
        self, time: np.datetime64, state_normalized: torch.Tensor
    ) -> None:
        """
        Set the static forcings of both models for a forecast starting at ``time``: the
        forecaster's from the normalized starting state of the forecast, the interpolator's
        from the time step of static_forcings_dataset valid at ``time``.

        Args:
            time: Start time of the forecast.
            state_normalized: Normalized forecaster input in data order,
                shape (batch, multistep, ensemble, points, variables).
        """
        self.forecaster_builder.set_static_forcings_from_state(state_normalized[:, -1])
        self.interpolator_builder.set_static_forcings_from_dataset(
            self.forcing_dataset_interp, time
        )

    def forecast(
        self, batch: dict[str, torch.Tensor], time: np.datetime64
    ) -> ForecastOutputs:
        """
        The forecaster rollout with the interpolator filling in between its first
        ``interpolation_length`` steps, see BasePredictor.forecast.
        """
        assert len(batch) == 1, (
            "Interpolator is currently only compatible with single dataset checkpoints"
        )
        batch = batch[self.ds_name_fc]

        builder = self.forecaster_builder
        ds = self.ds_name_fc
        n_boundary = len(self.interpolator.input_steps)

        state = builder.build_state(batch, self.forecaster.input_times(time))
        # (valid time, {dataset: output}) per lead time
        outputs = [(time, {ds: builder.initial_output(state, -1)})]

        # Forecaster states in physical units: the boundary values of the interpolator
        x_physical = builder.to_model_order(state)

        state = self.forecaster.pre_process(state, ds, in_place=True)
        self.update_static_forcings(time, state)
        x = builder.to_model_order(state)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for forecast_step in range(self.forecast_length - 1):
                y_pred = self.forecaster(
                    {ds: x},
                    model_comm_group=self.model_comm_group,
                    fcstep=self.fcstep(forecast_step),
                )[ds]
                y_physical = self.forecaster.post_process(y_pred, ds, in_place=False)

                next_time = time + self.forecaster.timestep
                x = builder.advance(x, y_pred, next_time)
                x_physical = builder.roll(x_physical, y_physical)

                interpolated = forecast_step < self.interpolation_length
                if interpolated:
                    # The interpolator's output at the right boundary is dropped when the
                    # forecaster provides it
                    outputs += [
                        (output_time, {ds: output})
                        for output_time, output in self.interpolator.predict_from_states(
                            self.ds_name_interp,
                            x_physical[:, -n_boundary:],
                            time,
                            model_comm_group=self.model_comm_group,
                        )
                        if self.reforcast_last or output_time != next_time
                    ]
                if not (interpolated and self.reforcast_last):
                    outputs.append((next_time, {ds: builder.output(y_physical)}))
                time = next_time

                if self.release_cache:
                    del y_pred, y_physical
                    torch.cuda.empty_cache()

        return outputs
