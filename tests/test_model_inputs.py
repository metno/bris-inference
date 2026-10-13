"""CPU tests of the model input bookkeeping (VariableMapping, insert_forcings, DatasetInputs)."""

import numpy as np
import pytest
import torch

from bris.forcings import anemoi_dynamic_forcings
from bris.model.model_utils import (
    VariableMapping,
    get_variable_indices,
    insert_forcings,
)
from bris.model.wrapper import DatasetInputs


class _Index:
    def __init__(self, names, prognostic=None, forcing=None, full=None):
        self.name_to_index = {n: i for i, n in enumerate(names)}
        self.prognostic = prognostic
        self.forcing = forcing
        self.full = full


class _DataIndex:
    """Stand-in for checkpoint.data_indices[ds].data: all variables in data order."""

    def __init__(self, names, forcing, diagnostic):
        self.input = _Index(
            names, full=[i for i, n in enumerate(names) if n not in diagnostic]
        )
        self.input.full = torch.tensor(self.input.full)


class _ModelIndex:
    """Stand-in for checkpoint.data_indices[ds].model: input (prognostic + forcing), output
    (prognostic + diagnostic), in the data order."""

    def __init__(self, names, forcing, diagnostic):
        input_names = [n for n in names if n not in diagnostic]
        output_names = [n for n in names if n not in forcing]
        self.input = _Index(
            input_names,
            prognostic=[i for i, n in enumerate(input_names) if n not in forcing],
            forcing=[i for i, n in enumerate(input_names) if n in forcing],
        )
        self.output = _Index(
            output_names,
            prognostic=[i for i, n in enumerate(output_names) if n not in diagnostic],
        )


# Data order of a small model: the dataset below holds the same variables in another order
NAMES = ["10u", "2t", "cos_latitude", "insolation", "lsm", "tp", "z"]
FORCING = ["cos_latitude", "insolation", "lsm", "z"]
DIAGNOSTIC = ["tp"]
DATASET_VARIABLES = [
    "z",
    "tp",
    "2t",
    "lsm",
    "10u",
    "cos_latitude",
    "insolation",
    "extra",
]


@pytest.fixture
def indices():
    return _DataIndex(NAMES, FORCING, DIAGNOSTIC), _ModelIndex(
        NAMES, FORCING, DIAGNOSTIC
    )


def test_variable_mapping(indices):
    data_index, model_index = indices
    m = VariableMapping.build(
        ["2t", "tp"], DATASET_VARIABLES, data_index, model_index, "data"
    )

    assert m.all == NAMES
    assert m.dynamic_forcings == ["insolation"]
    assert m.static_forcings == ["cos_latitude", "lsm", "z"]
    # required output variables: positions in the data order and in the model output
    assert m.variables_input == [1, 5]
    assert m.variables_output == [1, 2]
    # dataset positions follow the dataset order, input positions the corresponding data order
    assert m.prognostic_dataset.tolist() == [2, 4]
    assert m.prognostic_input.tolist() == [1, 0]
    assert m.static_forcings_dataset.tolist() == [0, 3, 5]
    assert m.static_forcings_input.tolist() == [6, 4, 2]
    assert m.dynamic_forcings_input.tolist() == [3]
    assert all(f in anemoi_dynamic_forcings() for f in m.dynamic_forcings)

    # the dictionaries of the former interface carry the same values
    idx, var = get_variable_indices(
        ["2t"], DATASET_VARIABLES, data_index, model_index, "data"
    )
    assert var == {
        "all": NAMES,
        "dynamic_forcings": ["insolation"],
        "static_forcings": m.static_forcings,
    }
    assert torch.equal(idx["prognostic_input"], m.prognostic_input)
    assert idx["variables_output"] == [1]


def test_variable_mapping_missing_variable(indices):
    data_index, model_index = indices
    with pytest.raises(
        ValueError,
        match="Missing the following required variables in dataset data: \\['z'\\]",
    ):
        VariableMapping.build(
            [],
            [v for v in DATASET_VARIABLES if v != "z"],
            data_index,
            model_index,
            "data",
        )


def test_insert_forcings():
    x = torch.zeros((2, 3, 1, 4, 3))
    name_to_index = {"a": 0, "b": 1, "c": 2}
    forcings = {
        "a": np.float64(0.5),  # scalar (julian day)
        "b": np.arange(4, dtype=np.float64),  # per point (local time, insolation)
        "c": torch.tensor(
            [[[10.0, 11.0, 12.0, 13.0]], [[20.0, 21.0, 22.0, 23.0]]]
        ),  # (batch, ensemble, points) static forcing
    }
    insert_forcings(x, forcings, name_to_index, 1)
    assert torch.all(x[:, 1, :, :, 0] == 0.5)
    assert torch.equal(x[1, 1, 0, :, 1], torch.arange(4, dtype=torch.float32))
    assert torch.equal(
        x[:, 1, 0, :, 2],
        torch.tensor([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]),
    )
    # other time steps untouched, dtype of x kept
    assert torch.all(x[:, [0, 2]] == 0)
    assert x.dtype == torch.float32
    insert_forcings(x, {"a": 2.0}, name_to_index, -1)
    assert torch.all(x[:, -1, :, :, 0] == 2.0)


@pytest.fixture
def dataset_inputs(indices):
    data_index, model_index = indices
    mapping = VariableMapping.build(
        ["2t", "tp"], DATASET_VARIABLES, data_index, model_index, "data"
    )
    lats = np.array([60.0, 61.0, 62.0])
    lons = np.array([10.0, 11.0, 12.0])
    return DatasetInputs(mapping, data_index, model_index, lats, lons)


def test_build_state_and_static_forcings(dataset_inputs):
    multistep, points = 2, 3
    source = torch.zeros((1, multistep, 1, points, len(DATASET_VARIABLES)))
    for i in range(len(DATASET_VARIABLES)):
        source[..., i] = 10 * i + torch.arange(points)
    source[:, 1] += 100  # the second time step differs
    times = [np.datetime64("2022-06-21T06"), np.datetime64("2022-06-21T12")]

    state = dataset_inputs.build_state(source, times)
    assert state.shape == (1, multistep, 1, points, len(NAMES))
    # prognostic variables and static forcings are copied from the source
    assert torch.equal(
        state[0, 0, 0, :, NAMES.index("2t")],
        source[0, 0, 0, :, DATASET_VARIABLES.index("2t")],
    )
    assert torch.equal(
        state[0, 1, 0, :, NAMES.index("10u")],
        source[0, 1, 0, :, DATASET_VARIABLES.index("10u")],
    )
    assert torch.equal(
        state[0, 1, 0, :, NAMES.index("z")],
        source[0, 1, 0, :, DATASET_VARIABLES.index("z")],
    )
    # diagnostic variables are NaN
    assert torch.all(torch.isnan(state[..., NAMES.index("tp")]))
    # dynamic forcings are computed per time step (insolation differs between 06 and 12 UTC)
    ins = state[0, :, 0, :, NAMES.index("insolation")]
    assert not torch.isnan(ins).any()
    assert not torch.equal(ins[0], ins[1])

    # the static forcings of the model, from the last state
    sf = dataset_inputs.static_forcings(state[:, -1])
    assert set(sf) == {"cos_latitude", "lsm", "z"}
    assert torch.equal(sf["z"][0, 0], source[0, 1, 0, :, DATASET_VARIABLES.index("z")])

    # explicit static forcings override those of the source
    state2 = dataset_inputs.build_state(
        source,
        times,
        static_forcings={
            "z": torch.full((points,), -1.0),
            "lsm": 0.0,
            "cos_latitude": 0.5,
        },
    )
    assert torch.all(state2[..., NAMES.index("z")] == -1.0)
    assert torch.all(state2[..., NAMES.index("lsm")] == 0.0)
    assert torch.equal(state2[..., NAMES.index("2t")], state[..., NAMES.index("2t")])

    # lead time 0 output: the required variables of the last state
    out0 = dataset_inputs.initial_output(state, multistep - 1)
    assert out0.shape == (1, 1, points, 2)
    assert torch.equal(
        out0[0, 0, :, 0], source[0, 1, 0, :, DATASET_VARIABLES.index("2t")]
    )


def test_roll_and_advance(dataset_inputs):
    points = 3
    n_input = len(NAMES) - len(DIAGNOSTIC)
    x = torch.zeros((1, 2, 1, points, n_input))
    x[:, 0] = 1.0
    x[:, 1] = 2.0
    # model output: 10u, 2t, tp
    y_pred = torch.tensor([[[[5.0, 6.0, 7.0]] * points]])
    static = {
        "z": torch.full((1, 1, points), 300.0),
        "lsm": torch.ones((1, 1, points)),
        "cos_latitude": torch.full((1, 1, points), 0.5),
    }

    x2 = dataset_inputs.advance(
        x.clone(), y_pred, np.datetime64("2022-06-21T18"), static
    )
    model_in = dataset_inputs.model_index.input.name_to_index
    # the old last step moved to the front
    assert torch.all(x2[:, 0] == 2.0)
    # prognostic variables from the prediction
    assert torch.all(x2[:, -1, :, :, model_in["10u"]] == 5.0)
    assert torch.all(x2[:, -1, :, :, model_in["2t"]] == 6.0)
    # forcings: static as given, dynamic computed
    assert torch.all(x2[:, -1, :, :, model_in["z"]] == 300.0)
    assert torch.all(x2[:, -1, :, :, model_in["lsm"]] == 1.0)
    assert not torch.all(x2[:, -1, :, :, model_in["insolation"]] == 2.0)

    # roll only touches the prognostic variables
    x3 = dataset_inputs.roll(x.clone(), y_pred)
    assert torch.all(x3[:, -1, :, :, model_in["z"]] == 1.0)  # the old first step
    assert torch.all(x3[:, -1, :, :, model_in["2t"]] == 6.0)

    # output of a (post-processed) prediction: required variables, first ensemble member
    out = dataset_inputs.output(y_pred)
    assert out.shape == (1, points, 2)
    assert torch.equal(out[0, 0], torch.tensor([6.0, 7.0]))
