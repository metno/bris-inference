from dataclasses import dataclass

import numpy as np
import torch
from anemoi.models.data_indices.index import DataIndex, ModelIndex

from ..checkpoint import Checkpoint
from ..forcings import anemoi_dynamic_forcings
from ..utils import (  # noqa: F401  (re-exported for the predictors)
    get_interpolator_boundary_times,
    get_interpolator_interp_times,
)


def get_model_static_forcings(
    selection: list,
    data_normalized: torch.Tensor,
    internal_data: DataIndex,
) -> dict:
    """Static forcings of a model, taken from a state.

    Every forcing in ``selection`` that is not computed on the fly (see
    :func:`bris.forcings.anemoi_dynamic_forcings`) is read from ``data_normalized`` as it is:
    cos/sin of latitude/longitude, lsm, z and any other time-constant input of the model (e.g.
    physiography fields). ``data_normalized`` should be the starting state of the forecast, not
    the first time step of the dataset: z (and in principle the other fields) change with model
    updates in long datasets.

    Args:
        selection (list): Forcing names of the model (the dynamic ones are skipped).

        data_normalized (torch.Tensor): Normalized state in the data-index variable order,
            shape (..., points, variables); leading dimensions (e.g. batch, ensemble) are kept.

        internal_data (DataIndex): Data index object, from checkpoint.data_indices

    Returns:
        dict: forcing name -> tensor of shape (..., points), a compact copy of the state values
    """
    if selection is None:
        return None
    dynamic = anemoi_dynamic_forcings()
    return {
        name: data_normalized[..., internal_data.input.name_to_index[name]]
        .contiguous()
        .float()
        for name in selection
        if name not in dynamic
    }


def static_forcing_time_index(dates: np.ndarray, time: np.datetime64) -> int:
    """Index of the time step of a dataset whose static forcings are valid at ``time``.

    That is the latest date <= time (static fields change at model updates and stay valid
    until the next one), or 0 if time is before the first date.

    Args:
        dates (np.ndarray): The dataset dates (datetime64), sorted.
        time (np.datetime64): Start time of the forecast.
    """
    dates = np.asarray(dates, dtype="datetime64[s]")
    index = int(np.searchsorted(dates, np.datetime64(time, "s"), side="right")) - 1
    return max(index, 0)


@dataclass
class VariableMapping:
    """How the variables of a dataset (or any ordered variable list) map onto a model.

    Built by :meth:`VariableMapping.build`. ``dataset_variables`` is the variable order of the
    source the model input is assembled from: a data reader for a forecaster, the forecaster's
    model input for an interpolator, or the static forcings dataset of an interpolator.

    Attributes:
        all: Model input variables in data-index order (the variable order of a pre-processed
            state before selecting ``data_index.input.full``).
        dynamic_forcings: Model forcings computed on the fly (see bris.forcings).
        static_forcings: The other model forcings, taken from a state or a dataset.
        variables_input: Data-index positions of the required output variables (lead time 0).
        variables_output: Model-output positions of the required output variables.
        prognostic_dataset / prognostic_input: Positions of the prognostic variables in the
            dataset order and in ``all``.
        static_forcings_dataset / static_forcings_input: The same for the static forcings.
        dynamic_forcings_input: Positions of the dynamic forcings in ``all``.
    """

    all: list[str]
    dynamic_forcings: list[str]
    static_forcings: list[str]
    variables_input: list[int]
    variables_output: list[int]
    prognostic_dataset: torch.Tensor
    static_forcings_dataset: torch.Tensor
    prognostic_input: torch.Tensor
    static_forcings_input: torch.Tensor
    dynamic_forcings_input: torch.Tensor

    @classmethod
    def build(
        cls,
        output_variables: list,
        dataset_variables: list,
        internal_data: DataIndex,
        internal_model: ModelIndex,
        decoder_name: str,
    ) -> "VariableMapping":
        """
        Args:
            output_variables (list): Variables to output, i.e. those the outputs require (must exist in the model output).
            dataset_variables (list): Variable order of the source the model input is built from.
            internal_data (DataIndex): Data index object, from checkpoint.data_indices
            internal_model (ModelIndex): Model index object, from checkpoint.data_indices.model
            decoder_name (str): Name of the dataset, for error messages.
        """
        variables_input = [
            internal_data.input.name_to_index[name] for name in output_variables
        ]
        variables_output = [
            internal_model.output.name_to_index[name] for name in output_variables
        ]

        all_variables = [
            var
            for var, _ in sorted(
                internal_data.input.name_to_index.items(), key=lambda item: item[1]
            )
        ]

        prognostic = [
            name
            for name, index in internal_model.input.name_to_index.items()
            if index in internal_model.input.prognostic
        ]
        forcings = [
            name
            for name, index in internal_model.input.name_to_index.items()
            if index in internal_model.input.forcing
        ]
        dynamic = anemoi_dynamic_forcings()
        dynamic_forcings = [f for f in dynamic if f in forcings]
        static_forcings = [f for f in forcings if f not in dynamic]

        missing = [
            var for var in prognostic + static_forcings if var not in dataset_variables
        ]
        if len(missing) > 0:
            raise ValueError(
                f"Missing the following required variables in dataset {decoder_name}: {missing}"
            )

        def dataset_positions(names):
            return torch.tensor(
                [i for i, var in enumerate(dataset_variables) if var in names],
                dtype=torch.int64,
            )

        def input_positions(names):
            return torch.tensor(
                [all_variables.index(var) for var in dataset_variables if var in names],
                dtype=torch.int64,
            )

        return cls(
            all=all_variables,
            dynamic_forcings=dynamic_forcings,
            static_forcings=static_forcings,
            variables_input=variables_input,
            variables_output=variables_output,
            prognostic_dataset=dataset_positions(prognostic),
            static_forcings_dataset=dataset_positions(static_forcings),
            prognostic_input=input_positions(prognostic),
            static_forcings_input=input_positions(static_forcings),
            dynamic_forcings_input=input_positions(dynamic_forcings),
        )


def insert_forcings(
    x: torch.Tensor, forcings: dict, name_to_index: dict, time_index: int
) -> torch.Tensor:
    """Write forcing fields into one time step of a model input tensor, in place.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, time, ensemble, points, variables).
        forcings (dict): Forcing name -> value: a scalar, a (points,) array, or a tensor
            broadcastable to (batch, ensemble, points), e.g. from get_dynamic_forcings and
            get_model_static_forcings.
        name_to_index (dict): Variable name -> position in the last dimension of ``x``.
        time_index (int): Time step of ``x`` to write to.
    """
    for name, value in forcings.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            value = value.to(dtype=x.dtype, device=x.device)
        x[:, time_index, :, :, name_to_index[name]] = value
    return x


def get_dataset_names(checkpoint: Checkpoint) -> list[str]:
    """Names of the datasets (encoders/decoders) of a checkpoint's model."""
    model = checkpoint.model.model
    if hasattr(model, "inputs"):
        return list(model.inputs)
    if hasattr(model, "dataset_names"):  # Compatibility with anemoi core main
        return list(model.dataset_names)
    return ["data"]  # Legacy compatibility


def get_internal_indices(data_indices) -> tuple[DataIndex, ModelIndex]:
    """The (data, model) index objects of one dataset of a checkpoint.

    Checkpoints with a remapper carry ``internal_data``/``internal_model`` (the indices of the
    pre-processed state); older checkpoints only ``data``/``model``.
    """
    if hasattr(data_indices, "internal_data") and hasattr(
        data_indices, "internal_model"
    ):
        return data_indices.internal_data, data_indices.internal_model
    return data_indices.data, data_indices.model


def resolve_reforecast_last(
    input_steps: list[int], output_steps: list[int], override: bool | None = None
) -> bool:
    """
    Whether an interpolator provides the right boundary of each interval: by default when it
    outputs it (its last input and output steps coincide), which the model option
    ``reforecast_last`` can switch off (the forecaster or the dataset provides the boundary) or
    insist on.

    Args:
        input_steps, output_steps: The interpolator's input and output steps (ModelRunner).
        override: The ``reforecast_last`` model option; None follows the checkpoint.
    """
    outputs_boundary = input_steps[-1] == output_steps[-1]
    if override is None:
        return outputs_boundary
    if override and not outputs_boundary:
        raise ValueError(
            "reforecast_last=True requires an interpolator that outputs the right boundary, "
            f"but its input steps are {list(input_steps)} and its output steps {list(output_steps)}"
        )
    return bool(override)
