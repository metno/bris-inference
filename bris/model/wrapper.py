import numpy as np
import torch
from torch import nn

from ..checkpoint import Checkpoint
from ..forcings import get_dynamic_forcings
from ..utils import LOGGER, get_model_multistep_input
from .model_utils import (
    VariableMapping,
    get_dataset_names,
    get_internal_indices,
    get_model_static_forcings,
    insert_forcings,
    static_forcing_time_index,
)


class InputBuilder:
    """
    Assembles and advances the input of a model for one of its datasets, and holds the static
    forcings of the current forecast.

    A state is a tensor of shape (batch, time, ensemble, points, variables). A state "in data
    order" holds the model's input variables in data-index order (``mapping.all``, the order
    the pre-processors expect); a state "in model order" holds ``data_index.input.full`` only,
    which is what the model takes.

    Methods
    -------

    dynamic_forcings: The forcings computed on the fly (julian day, local time, insolation) at a time.

    build_state: Model input in data order (physical units) from dataset states.

    set_static_forcings_from_state / set_static_forcings_from_dataset: Set the static forcings
    of the current forecast.

    to_model_order: Select the model input variables of a state in data order.

    roll: Append a prediction to a model-order input (prognostic variables only).

    advance: roll, then insert the forcings valid at the new time.

    initial_output / output: The required output variables of a state / a post-processed prediction.
    """

    def __init__(
        self,
        mapping: VariableMapping,
        data_index,
        model_index,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> None:
        """
        Args:
            mapping: Variable mapping of the dataset onto the model (see VariableMapping.build).
            data_index: Data index of the model for this dataset (checkpoint.data_indices[ds].data).
            model_index: Model index of the model for this dataset (checkpoint.data_indices[ds].model).
            latitudes, longitudes: Grid coordinates, for the dynamic forcings.
        """
        self.mapping = mapping
        self.data_index = data_index
        self.model_index = model_index
        self.latitudes = latitudes
        self.longitudes = longitudes
        # Static forcings (cos/sin lat/lon, lsm, z, ...) of the current forecast, name -> tensor.
        # They change between forecasts (z changes with model updates in long datasets), so
        # they are set per forecast, from the starting state or from a dataset.
        self.static_forcings = {}
        self._static_forcings_dataset_index = None

    def dynamic_forcings(self, time: np.datetime64) -> dict:
        return get_dynamic_forcings(
            time, self.latitudes, self.longitudes, self.mapping.dynamic_forcings
        )

    def build_state(
        self,
        source: torch.Tensor,
        times: list[np.datetime64],
        static_forcings_from_source: bool = True,
    ) -> torch.Tensor:
        """
        Model input in data order (physical units) from dataset states.

        Prognostic variables come from ``source``, dynamic forcings are computed for each time and
        diagnostic variables are left NaN. The static forcings are copied from ``source`` or, if
        ``static_forcings_from_source`` is False, taken from ``self.static_forcings`` (an
        interpolator takes them from its own dataset).

        Args:
            source (torch.Tensor): States in the dataset variable order of the mapping, shape
                (batch, len(times), ensemble, points, dataset variables).
            times (list): Valid time of each time step of ``source``.
            static_forcings_from_source (bool): Where the static forcings come from.
        """
        x = torch.full(
            source.shape[:-1] + (len(self.mapping.all),),
            float("nan"),
            dtype=source.dtype,
            device=source.device,
        )
        x[..., self.mapping.prognostic_input] = source[
            ..., self.mapping.prognostic_dataset
        ]
        if static_forcings_from_source:
            x[..., self.mapping.static_forcings_input] = source[
                ..., self.mapping.static_forcings_dataset
            ]
        for time_index, time in enumerate(times):
            forcings = self.dynamic_forcings(time)
            if not static_forcings_from_source:
                forcings.update(self.static_forcings)
            insert_forcings(
                x, forcings, self.data_index.input.name_to_index, time_index
            )
        return x

    def set_static_forcings_from_state(self, state_normalized: torch.Tensor) -> None:
        """
        Set the static forcings from a normalized state in data order, shape
        (..., points, variables), e.g. the starting state of a forecast.
        """
        self.static_forcings = get_model_static_forcings(
            self.mapping.static_forcings, state_normalized, self.data_index
        )

    def set_static_forcings_from_dataset(self, dataset, time: np.datetime64) -> None:
        """
        Set the static forcings (physical units) from the time step of an anemoi dataset valid
        at ``time``: the latest date <= time. Cached between calls that map to the same time step.

        Args:
            dataset: An open anemoi dataset holding the static forcings.
            time (np.datetime64): Start time of the forecast.
        """
        index = static_forcing_time_index(dataset.dates, time)
        if index == self._static_forcings_dataset_index:
            return
        missing = [
            v for v in self.mapping.static_forcings if v not in dataset.variables
        ]
        if len(missing) > 0:
            raise ValueError(
                f"Missing the following static forcings in the static forcings dataset: {missing}"
            )
        LOGGER.info(
            "Static forcings from %s (time step %d of the static forcings dataset) for forecasts from %s",
            dataset.dates[index],
            index,
            time,
        )
        data = torch.from_numpy(dataset[index].squeeze(axis=1).swapaxes(0, 1))
        self.static_forcings = {
            name: data[..., dataset.variables.index(name)].contiguous().float()
            for name in self.mapping.static_forcings
        }
        self._static_forcings_dataset_index = index

    def to_model_order(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., self.data_index.input.full]

    def roll(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Shift a model-order input one time step and put the predicted prognostic variables last.

        Args:
            x (torch.Tensor): Model input, shape (batch, time, ensemble, points, variables).
            y_pred (torch.Tensor): Model output, shape (batch, ensemble, points, variables).
        """
        x = x.roll(-1, dims=1)
        x[:, -1, :, :, self.model_index.input.prognostic] = y_pred[
            ..., self.model_index.output.prognostic
        ]
        return x

    def advance(
        self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        """
        Advance a model-order input to the next step: roll in the prediction and insert the
        dynamic forcings valid at ``time`` and the static forcings of the forecast.
        """
        x = self.roll(x, y_pred)
        forcings = self.dynamic_forcings(time)
        forcings.update(self.static_forcings)
        insert_forcings(x, forcings, self.model_index.input.name_to_index, -1)
        return x

    def initial_output(self, state: torch.Tensor, time_index: int) -> torch.Tensor:
        """
        The required output variables of one time step of a state in data order, as ``output``.
        """
        return self.output(state[:, time_index], self.mapping.variables_input)

    def output(
        self, y_pred: torch.Tensor, variables: list[int] | None = None
    ) -> torch.Tensor:
        """
        The required output variables of a post-processed prediction, shape
        (batch, points, variables), float32 on CPU.

        Args:
            y_pred (torch.Tensor): Prediction of shape (batch, ..., points, variables); any
                dimensions between batch and points (ensemble, time) are singletons in inference
                and the first entry is taken.
            variables (list | None): Positions in the last dimension; default the model-output
                positions of the required variables.
        """
        y = y_pred[
            ..., self.mapping.variables_output if variables is None else variables
        ]
        while y.ndim > 3:
            y = y[:, 0]
        return y.to(torch.float32).cpu()


class ModelWrapper(nn.Module):
    """
    A checkpoint's model with its time geometry, its per-dataset input builders and the compatibility
    shims for calling it.

    Registered as a submodule of the predictor, so Lightning moves the model to the device.

    Attributes
    ----------

    timestep: The model's time step.

    input_steps: The input times of one model call, in time steps relative to the reference
    time: -(multistep - 1) .. 0 for a forecaster, the boundary times for an interpolator.

    output_steps: The output times of one model call, likewise: [1] for a forecaster, the
    interpolation times for an interpolator.

    Methods
    -------

    input_times / output_times: The valid times of the inputs / outputs of a call at a reference time.

    add_dataset: Set up the input builder (variable mapping, coordinates) of one dataset of the model.

    pre_process / post_process: The model's processors of a dataset, for both single and multi
    dataset models.

    forward: Forward pass; takes and returns a dict of tensors keyed by dataset name, for models
    of any vintage.

    predict_from_states: One model call from physical states, outputs per output time.
    """

    def __init__(
        self,
        checkpoint: Checkpoint,
        timestep: np.timedelta64,
        input_steps: list[int] | None = None,
        output_steps: list[int] | None = None,
    ) -> None:
        """
        Args:
            checkpoint (Checkpoint): The checkpoint.
            timestep (np.timedelta64): The model's time step.
            input_steps (list | None): See the class docstring; default from the checkpoint's multistep.
            output_steps (list | None): See the class docstring; default [1].
        """
        super().__init__()
        self.model = checkpoint.model
        self.model.eval()
        self.dataset_names = get_dataset_names(checkpoint)
        self.multistep = get_model_multistep_input(checkpoint)
        self.timestep = timestep
        self.input_steps = (
            list(range(1 - self.multistep, 1))
            if input_steps is None
            else list(input_steps)
        )
        self.output_steps = [1] if output_steps is None else list(output_steps)
        self.data_index = {}
        self.model_index = {}
        for ds in self.dataset_names:
            self.data_index[ds], self.model_index[ds] = get_internal_indices(
                checkpoint.data_indices[ds]
            )
        self.input_builders: dict[str, InputBuilder] = {}

    def input_times(self, time: np.datetime64) -> list[np.datetime64]:
        return [time + step * self.timestep for step in self.input_steps]

    def output_times(self, time: np.datetime64) -> list[np.datetime64]:
        return [time + step * self.timestep for step in self.output_steps]

    def add_dataset(
        self,
        dataset_name: str,
        required_variables: list,
        dataset_variables: list,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> InputBuilder:
        """
        Args:
            dataset_name (str): One of ``dataset_names``.
            required_variables (list): Variables to output for this dataset.
            dataset_variables (list): Variable order of the states the input is built from.
            latitudes, longitudes: Grid coordinates of the dataset.
        """
        mapping = VariableMapping.build(
            required_variables,
            dataset_variables,
            self.data_index[dataset_name],
            self.model_index[dataset_name],
            dataset_name,
        )
        self.input_builders[dataset_name] = InputBuilder(
            mapping,
            self.data_index[dataset_name],
            self.model_index[dataset_name],
            latitudes,
            longitudes,
        )
        return self.input_builders[dataset_name]

    def pre_process(
        self, x: torch.Tensor, dataset_name: str, in_place: bool = False
    ) -> torch.Tensor:
        try:
            return self.model.pre_processors[dataset_name](x, in_place=in_place)
        except TypeError:  # Single dataset models
            return self.model.pre_processors(x, in_place=in_place)

    def post_process(
        self, x: torch.Tensor, dataset_name: str, in_place: bool = False
    ) -> torch.Tensor:
        try:
            return self.model.post_processors[dataset_name](x, in_place=in_place)
        except TypeError:  # Single dataset models
            return self.model.post_processors(x, in_place=in_place)

    def forward(
        self, x: dict[str, torch.Tensor], model_comm_group=None, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x (dict): Model inputs keyed by dataset name.
            model_comm_group: Process group of the model shards.
            kwargs: Passed on to the model (e.g. fcstep).
        Returns:
            dict: Model outputs keyed by dataset name.
        """
        try:
            y = self.model(x, model_comm_group=model_comm_group, **kwargs)
        except AttributeError:
            # Models that take a tensor instead of a dict of tensors
            x = list(x.values())[0]
            try:
                y = self.model(x, model_comm_group=model_comm_group, **kwargs)
            except TypeError:
                # Models that do not take kwargs
                y = self.model(x, model_comm_group=model_comm_group)
        if isinstance(y, torch.Tensor):
            y = {self.dataset_names[0]: y}
        return y

    def predict_from_states(
        self,
        dataset_name: str,
        states: torch.Tensor,
        time: np.datetime64,
        model_comm_group=None,
        static_forcings_from_source: bool = False,
    ) -> list[tuple[np.datetime64, torch.Tensor]]:
        """
        One model call from physical states: build the input at the input times, run the model
        and return its outputs at the output times. This is how an interpolator is run between
        boundary states; for a forecaster it is one forecast step.

        Args:
            dataset_name (str): Dataset of the model to run.
            states (torch.Tensor): The input states in physical units, shape
                (batch, len(input_steps), ensemble, points, variables) in the dataset variable
                order of the model's input builder (e.g. a forecaster's model input, or a data
                reader).
            time (np.datetime64): Reference time of the states (input step 0).
            model_comm_group: Process group of the model shards.
            static_forcings_from_source (bool): Take the static forcings from ``states`` instead
                of those held by the input builder (see InputBuilder.build_state).
        Returns:
            list: (valid time, output) per output step; output as InputBuilder.output.
        """
        builder = self.input_builders[dataset_name]
        x = builder.build_state(
            states,
            self.input_times(time),
            static_forcings_from_source=static_forcings_from_source,
        )
        x = self.pre_process(x, dataset_name, in_place=True)
        x = builder.to_model_order(x)

        # fcstep is always 0 for an interpolator
        y_pred = self({dataset_name: x}, model_comm_group=model_comm_group, fcstep=0)[
            dataset_name
        ]

        outputs = []
        for index, output_time in enumerate(self.output_times(time)):
            y = self.post_process(y_pred[:, index], dataset_name, in_place=True)
            outputs.append((output_time, builder.output(y)))
        return outputs
