import numpy as np
import torch
from anemoi.datasets import open_dataset

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from .basepredictor import BasePredictor, ForecastOutputs
from .model_utils import (
    get_interpolator_boundary_times,
    get_interpolator_interp_times,
    resolve_reforecast_last,
)
from .runner import ModelRunner


class InterpolatorOnly(BasePredictor):
    """
    Temporal interpolator run between boundary states read from the dataset, without a forecaster.

    Each batch holds the boundary states of one forecast: the dataset state at the initialisation
    time and the following ones, ``boundary_timestep`` apart (the DataModule's window,
    utils.get_interpolator_boundary_states of them). The interpolator is run between consecutive boundary states exactly
    as in Interpolator, where they are the forecaster's states. This evaluates the interpolator on
    its own, e.g. from an analysis.

    Output: lead time 0 is the first boundary state (the output variables the dataset holds, NaN
    for the others); each interval then contributes the interpolator's outputs and, unless the
    interpolator re-forecasts the right boundary, that boundary state from the dataset. With the
    model option ``reforecast_last`` False the dataset state is written at the right boundary
    even then, in place of the interpolator's re-forecast.

    Methods
    -------

    __init__

    boundary_output: The required output variables of a dataset state.

    forecast: The interpolator between the boundary states of a batch.
    """

    def __init__(
        self,
        *args,
        checkpoints: dict[str, Checkpoint],
        datamodule: DataModule,
        checkpoints_config: dict,
        required_variables: dict,
        release_cache: bool = False,
        reforecast_last: bool | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize InterpolatorOnly.

        Args:
            checkpoints: Dictionary containing the 'interpolator' Checkpoint object.
            datamodule: DataModule object for data handling.
            checkpoints_config: config.checkpoints; the interpolator needs "timestep_seconds",
                "boundary_timestep_seconds", "leadtimes" (the number of boundary intervals
                interpolated per forecast) and "static_forcings_dataset".
            required_variables: Variables the outputs require, per dataset.
            release_cache: Boolean flag to release cache (default: False).
            reforecast_last: Whether the interpolator provides the right boundary of each
                interval (model_utils.resolve_reforecast_last); default: when it outputs it.
        """
        super().__init__(
            *args, checkpoints=checkpoints, release_cache=release_cache, **kwargs
        )
        checkpoint = checkpoints["interpolator"]
        config = checkpoints_config["interpolator"]
        self.interpolator = ModelRunner(
            checkpoint,
            timestep=np.timedelta64(config["timestep_seconds"], "s"),
            input_steps=get_interpolator_boundary_times(checkpoint),
            output_steps=get_interpolator_interp_times(checkpoint),
        )
        assert len(self.interpolator.dataset_names) == 1, (
            "InterpolatorOnly is currently only compatible with single dataset checkpoints"
        )
        self.ds_name = self.interpolator.dataset_names[0]
        self.boundary_timestep = np.timedelta64(
            config["boundary_timestep_seconds"], "s"
        )
        self.interpolation_length = config["leadtimes"]
        self._check_time_geometry()

        data_reader = datamodule.data_readers[self.ds_name]
        output_variables = required_variables[self.ds_name]
        self.builder = self.interpolator.add_dataset(
            self.ds_name,
            output_variables=output_variables,
            dataset_variables=data_reader.variables,
            latitudes=data_reader.latitudes,
            longitudes=data_reader.longitudes,
        )
        self.num_output_variables = len(output_variables)
        self.boundary_output_positions = self._boundary_positions(
            output_variables, data_reader.variables
        )

        # The static forcings come from their own dataset, the time step valid at the start of
        # each forecast (as in Interpolator)
        self.forcing_dataset = open_dataset(config["static_forcings_dataset"])

        # Whether the interpolator provides the right boundary of each interval (then the
        # dataset state is not written)
        self.reforecast_last = resolve_reforecast_last(
            self.interpolator.input_steps,
            self.interpolator.output_steps,
            reforecast_last,
        )

    @staticmethod
    def _boundary_positions(
        output_variables: list, dataset_variables: list
    ) -> tuple[list[int], list[int]]:
        """Positions of the output variables the dataset holds: in the output and in the dataset."""
        present = [v for v in output_variables if v in dataset_variables]
        return (
            [output_variables.index(v) for v in present],
            [dataset_variables.index(v) for v in present],
        )

    def _check_time_geometry(self) -> None:
        """The interpolator's input times must be consecutive boundary states."""
        input_times = [
            step * self.interpolator.timestep for step in self.interpolator.input_steps
        ]
        boundary_times = [j * self.boundary_timestep for j in range(len(input_times))]
        if input_times != boundary_times:
            raise ValueError(
                f"The interpolator's input times {input_times} are not consecutive boundary "
                f"states {self.boundary_timestep} apart; check boundary_timestep_seconds"
            )

    def boundary_output(self, state: torch.Tensor) -> torch.Tensor:
        """
        The required output variables of a dataset state, shape (batch, points, variables) as
        InputBuilder.output; NaN for the variables the dataset does not hold (e.g. diagnostic
        variables).

        Args:
            state (torch.Tensor): A dataset state, shape (batch, ensemble, points, dataset variables).
        """
        y = torch.full(
            state.shape[:-1] + (self.num_output_variables,),
            float("nan"),
            dtype=state.dtype,
            device=state.device,
        )
        output_positions, dataset_positions = self.boundary_output_positions
        y[..., output_positions] = state[..., dataset_positions]
        return self.builder.output(y, variables=list(range(self.num_output_variables)))

    def forecast(
        self, batch: dict[str, torch.Tensor], time: np.datetime64
    ) -> ForecastOutputs:
        """
        The interpolator between the consecutive boundary states of the batch, see
        BasePredictor.forecast. ``time`` is the time of the first boundary state.
        """
        assert len(batch) == 1, (
            "InterpolatorOnly is currently only compatible with single dataset checkpoints"
        )
        ds = self.ds_name
        states = batch[ds]
        n_boundary = len(self.interpolator.input_steps)
        expected = self.interpolation_length + n_boundary - 1
        if states.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} boundary states per batch, got {states.shape[1]}: the "
                "DataModule window must be get_interpolator_boundary_states"
            )

        self.builder.set_static_forcings_from_dataset(self.forcing_dataset, time)
        # (valid time, {dataset: output}) per lead time
        outputs = [(time, {ds: self.boundary_output(states[:, 0])})]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for interval in range(self.interpolation_length):
                boundary_time = time + interval * self.boundary_timestep
                right_boundary_time = boundary_time + self.boundary_timestep
                # The interpolator's output at the right boundary is dropped when the dataset
                # provides it
                outputs += [
                    (output_time, {ds: output})
                    for output_time, output in self.interpolator.predict_from_states(
                        ds,
                        states[:, interval : interval + n_boundary],
                        boundary_time,
                        model_comm_group=self.model_comm_group,
                    )
                    if self.reforecast_last or output_time != right_boundary_time
                ]
                if not self.reforecast_last:
                    outputs.append(
                        (
                            right_boundary_time,
                            {ds: self.boundary_output(states[:, interval + 1])},
                        )
                    )
                if self.release_cache:
                    torch.cuda.empty_cache()

        return outputs
