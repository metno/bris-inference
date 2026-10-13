import numpy as np
import torch
from torch import nn

from ..checkpoint import Checkpoint
from ..forcings import get_dynamic_forcings
from ..utils import get_model_multistep_input
from .model_utils import (
    VariableMapping,
    get_dataset_names,
    get_internal_indices,
    get_model_static_forcings,
    insert_forcings,
)


class DatasetInputs:
    """
    Assembles and advances the input of a model for one of its datasets.

    A state is a tensor of shape (batch, time, ensemble, points, variables). A state "in data
    order" holds the model's input variables in data-index order (``mapping.all``, the order
    the pre-processors expect); a state "in model order" holds ``data_index.input.full`` only,
    which is what the model takes.

    Methods
    -------

    dynamic_forcings: The forcings computed on the fly (julian day, local time, insolation) at a time.

    build_state: Model input in data order (physical units) from dataset states.

    static_forcings: The static forcings of the model, from a normalized state.

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

    def dynamic_forcings(self, time: np.datetime64) -> dict:
        return get_dynamic_forcings(
            time, self.latitudes, self.longitudes, self.mapping.dynamic_forcings
        )

    def build_state(
        self,
        source: torch.Tensor,
        times: list[np.datetime64],
        static_forcings: dict | None = None,
    ) -> torch.Tensor:
        """
        Model input in data order (physical units) from dataset states.

        Prognostic variables come from ``source``, dynamic forcings are computed for each time and
        diagnostic variables are left NaN. The static forcings are copied from ``source`` unless
        given explicitly (an interpolator takes them from its own dataset).

        Args:
            source (torch.Tensor): States in the dataset variable order of the mapping, shape
                (batch, len(times), ensemble, points, dataset variables).
            times (list): Valid time of each time step of ``source``.
            static_forcings (dict | None): Static forcings to insert instead of those of ``source``.
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
        if static_forcings is None:
            x[..., self.mapping.static_forcings_input] = source[
                ..., self.mapping.static_forcings_dataset
            ]
        for time_index, time in enumerate(times):
            forcings = self.dynamic_forcings(time)
            if static_forcings is not None:
                forcings.update(static_forcings)
            insert_forcings(
                x, forcings, self.data_index.input.name_to_index, time_index
            )
        return x

    def static_forcings(self, state_normalized: torch.Tensor) -> dict:
        """
        The static forcings of the model (cos/sin lat/lon, lsm, z, ...) from a normalized state
        in data order, shape (..., points, variables), e.g. the starting state of a forecast.
        """
        return get_model_static_forcings(
            self.mapping.static_forcings, state_normalized, self.data_index
        )

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
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        time: np.datetime64,
        static_forcings: dict,
    ) -> torch.Tensor:
        """
        Advance a model-order input to the next step: roll in the prediction and insert the
        dynamic forcings valid at ``time`` and the static forcings.
        """
        x = self.roll(x, y_pred)
        forcings = self.dynamic_forcings(time)
        forcings.update(static_forcings)
        insert_forcings(x, forcings, self.model_index.input.name_to_index, -1)
        return x

    def initial_output(self, state: torch.Tensor, time_index: int) -> torch.Tensor:
        """The required output variables of one time step of a state in data order, on CPU."""
        return state[:, time_index, ..., self.mapping.variables_input].cpu()

    def output(self, y_pred: torch.Tensor) -> torch.Tensor:
        """The required output variables of a post-processed prediction (first ensemble member), on CPU."""
        return y_pred[:, 0, ..., self.mapping.variables_output].cpu()


class ModelWrapper(nn.Module):
    """
    A checkpoint's model with its per-dataset inputs and the compatibility shims for calling it.

    Registered as a submodule of the predictor, so Lightning moves the model to the device.

    Methods
    -------

    add_dataset: Set up the inputs (variable mapping, coordinates) of one dataset of the model.

    pre_process / post_process: The model's processors of a dataset, for both single and multi
    dataset models.

    forward: Forward pass; takes and returns a dict of tensors keyed by dataset name, for models
    of any vintage.
    """

    def __init__(self, checkpoint: Checkpoint) -> None:
        super().__init__()
        self.model = checkpoint.model
        self.model.eval()
        self.dataset_names = get_dataset_names(checkpoint)
        self.multistep = get_model_multistep_input(checkpoint)
        self.data_index = {}
        self.model_index = {}
        for ds in self.dataset_names:
            self.data_index[ds], self.model_index[ds] = get_internal_indices(
                checkpoint.data_indices[ds]
            )
        self.inputs: dict[str, DatasetInputs] = {}

    def add_dataset(
        self,
        dataset_name: str,
        required_variables: list,
        dataset_variables: list,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> DatasetInputs:
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
        self.inputs[dataset_name] = DatasetInputs(
            mapping,
            self.data_index[dataset_name],
            self.model_index[dataset_name],
            latitudes,
            longitudes,
        )
        return self.inputs[dataset_name]

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
