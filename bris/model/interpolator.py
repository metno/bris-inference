import numpy as np
import torch
from anemoi.datasets import open_dataset

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..utils import get_all_leadtimes
from .basepredictor import BasePredictor
from .model_utils import (
    get_interpolator_boundary_times,
    get_interpolator_interp_times,
)
from .wrapper import ModelWrapper, interpolate


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
        self.forecaster = ModelWrapper(
            checkpoints["forecaster"],
            timestep=np.timedelta64(
                checkpoints_config["forecaster"]["timestep_seconds"], "s"
            ),
        )
        self.interpolator = ModelWrapper(
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
        self.leadtimes = get_all_leadtimes(
            checkpoints_config["forecaster"]["leadtimes"],
            checkpoints_config["forecaster"]["timestep_seconds"],
            checkpoints_config["interpolator"]["leadtimes"],
            checkpoints_config["interpolator"]["timestep_seconds"],
        )

        data_reader = datamodule.data_readers[self.ds_name_fc]
        self.forecaster_builder = self.forecaster.add_dataset(
            self.ds_name_fc,
            required_variables=required_variables[self.ds_name_fc],
            dataset_variables=data_reader.variables,
            latitudes=data_reader.latitudes,
            longitudes=data_reader.longitudes,
        )
        # The interpolator input is assembled from the forecaster's model input (physical units)
        self.interpolator_builder = self.interpolator.add_dataset(
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

        # Whether the interpolator also outputs the last boundary time (then the forecaster's
        # output at that time is not written)
        self.reforcast_last = (
            self.interpolator.input_steps[-1] == self.interpolator.output_steps[-1]
        )

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
                    outputs += [
                        (output_time, {ds: output})
                        for output_time, output in interpolate(
                            self.interpolator,
                            self.ds_name_interp,
                            x_physical[:, -n_boundary:],
                            time,
                            model_comm_group=self.model_comm_group,
                        )
                    ]
                if not (interpolated and self.reforcast_last):
                    outputs.append((next_time, {ds: builder.output(y_physical)}))
                time = next_time

                if self.release_cache:
                    del y_pred, y_physical
                    torch.cuda.empty_cache()

        return self.prediction(outputs)
