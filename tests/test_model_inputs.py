"""CPU tests of the model input bookkeeping (VariableMapping, insert_forcings, DatasetInputs)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bris.forcings import anemoi_dynamic_forcings
from bris.model.model_utils import (
    VariableMapping,
    get_variable_indices,
    insert_forcings,
)
from bris.model.wrapper import DatasetInputs, ModelWrapper, interpolate


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
    dataset_inputs.set_static_forcings_from_state(state[:, -1])
    sf = dataset_inputs.static_forcings
    assert set(sf) == {"cos_latitude", "lsm", "z"}
    assert torch.equal(sf["z"][0, 0], source[0, 1, 0, :, DATASET_VARIABLES.index("z")])

    # the owned static forcings are used instead of those of the source
    dataset_inputs.static_forcings = {
        "z": torch.full((points,), -1.0),
        "lsm": 0.0,
        "cos_latitude": 0.5,
    }
    state2 = dataset_inputs.build_state(
        source, times, static_forcings_from_source=False
    )
    assert torch.all(state2[..., NAMES.index("z")] == -1.0)
    assert torch.all(state2[..., NAMES.index("lsm")] == 0.0)
    assert torch.equal(state2[..., NAMES.index("2t")], state[..., NAMES.index("2t")])

    # lead time 0 output: the required variables of the last state
    out0 = dataset_inputs.initial_output(state, multistep - 1)
    assert out0.shape == (1, points, 2)
    assert torch.equal(out0[0, :, 0], source[0, 1, 0, :, DATASET_VARIABLES.index("2t")])


def test_roll_and_advance(dataset_inputs):
    points = 3
    n_input = len(NAMES) - len(DIAGNOSTIC)
    x = torch.zeros((1, 2, 1, points, n_input))
    x[:, 0] = 1.0
    x[:, 1] = 2.0
    # model output: 10u, 2t, tp
    y_pred = torch.tensor([[[[5.0, 6.0, 7.0]] * points]])
    dataset_inputs.static_forcings = {
        "z": torch.full((1, 1, points), 300.0),
        "lsm": torch.ones((1, 1, points)),
        "cos_latitude": torch.full((1, 1, points), 0.5),
    }

    x2 = dataset_inputs.advance(x.clone(), y_pred, np.datetime64("2022-06-21T18"))
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


class _Dataset:
    """Stand-in for an anemoi dataset: dates, variables and (variables, ensemble, points) per time."""

    def __init__(self, dates, variables, points):
        self.dates = np.array(dates, dtype="datetime64[s]")
        self.variables = variables
        self.points = points

    def __getitem__(self, index):
        data = np.zeros((len(self.variables), 1, self.points), dtype=np.float32)
        for i in range(len(self.variables)):
            data[i, 0] = 100 * index + 10 * i + np.arange(self.points)
        return data


def test_static_forcings_from_dataset(dataset_inputs):
    points = 3
    dataset = _Dataset(
        ["2020-01-01T00", "2021-01-01T00", "2022-01-01T00"],
        ["lsm", "extra", "z", "cos_latitude"],
        points,
    )
    dataset_inputs.set_static_forcings_from_dataset(
        dataset, np.datetime64("2021-06-01T00")
    )
    sf = dataset_inputs.static_forcings
    assert set(sf) == {"cos_latitude", "lsm", "z"}
    # time step 1 (latest date <= time), variables picked by name
    assert torch.equal(sf["z"], torch.tensor([120.0, 121.0, 122.0]))
    assert torch.equal(sf["lsm"], torch.tensor([100.0, 101.0, 102.0]))
    assert sf["z"].dtype == torch.float32

    # cached while the time step stays the same, reloaded when it changes
    sf["z"][:] = -1.0
    dataset_inputs.set_static_forcings_from_dataset(
        dataset, np.datetime64("2021-12-31T18")
    )
    assert torch.all(dataset_inputs.static_forcings["z"] == -1.0)
    dataset_inputs.set_static_forcings_from_dataset(
        dataset, np.datetime64("2022-01-01T00")
    )
    assert torch.equal(
        dataset_inputs.static_forcings["z"], torch.tensor([220.0, 221.0, 222.0])
    )

    with pytest.raises(ValueError, match="Missing the following static forcings"):
        dataset_inputs.set_static_forcings_from_dataset(
            _Dataset(["2020-01-01T00"], ["lsm", "z"], points),
            np.datetime64("2020-01-01T00"),
        )


class _Checkpoint:
    """Stand-in for a Checkpoint: a model with dataset names, data indices and a multistep config."""

    def __init__(self, names, forcing, diagnostic, multistep=2):
        model = torch.nn.Module()
        model.model = SimpleNamespace(dataset_names=["data"])
        model.pre_processors = {"data": lambda x, in_place=False: x}
        model.post_processors = {"data": lambda x, in_place=False: x}
        self.model = model
        self.data_indices = {
            "data": SimpleNamespace(
                data=_DataIndex(names, forcing, diagnostic),
                model=_ModelIndex(names, forcing, diagnostic),
            )
        }
        self.config = SimpleNamespace(
            training=SimpleNamespace(multistep_input=multistep)
        )


def test_model_wrapper_time_geometry():
    checkpoint = _Checkpoint(NAMES, FORCING, DIAGNOSTIC, multistep=3)
    time = np.datetime64("2022-06-21T12")

    forecaster = ModelWrapper(checkpoint, timestep=np.timedelta64(6, "h"))
    assert forecaster.dataset_names == ["data"]
    assert forecaster.input_steps == [-2, -1, 0] and forecaster.output_steps == [1]
    assert forecaster.input_times(time) == [
        np.datetime64("2022-06-21T00"),
        np.datetime64("2022-06-21T06"),
        time,
    ]
    assert forecaster.output_times(time) == [np.datetime64("2022-06-21T18")]

    interpolator = ModelWrapper(
        checkpoint,
        timestep=np.timedelta64(1, "h"),
        input_steps=[0, 6],
        output_steps=[1, 2, 3, 4, 5],
    )
    assert interpolator.input_times(time) == [time, np.datetime64("2022-06-21T18")]
    assert interpolator.output_times(time)[-1] == np.datetime64("2022-06-21T17")

    inputs = forecaster.add_dataset(
        "data", ["2t"], DATASET_VARIABLES, np.zeros(3), np.zeros(3)
    )
    assert forecaster.inputs["data"] is inputs
    assert inputs.mapping.variables_output == [1]
    # forward of a wrapped model takes and returns dicts
    forecaster.model.forward = lambda x, model_comm_group=None, **kwargs: {
        "data": x["data"] + 1
    }
    y = forecaster({"data": torch.zeros(2)}, model_comm_group=None, fcstep=0)
    assert torch.equal(y["data"], torch.ones(2))


def test_output_shapes(dataset_inputs):
    points = 3
    # model outputs with and without a leading time dimension give (batch, points, variables)
    y4 = torch.arange(1 * 1 * points * 3, dtype=torch.float32).reshape(1, 1, points, 3)
    y5 = y4.unsqueeze(1)
    out4 = dataset_inputs.output(y4)
    out5 = dataset_inputs.output(y5)
    assert out4.shape == (1, points, 2) and torch.equal(out4, out5)
    assert torch.equal(out4[0, :, 0], y4[0, 0, :, 1])


def test_interpolate_window():
    """interpolate() builds the boundary input from physical states and maps the model output
    to (time, output) pairs."""
    checkpoint = _Checkpoint(NAMES, FORCING, DIAGNOSTIC)
    model = ModelWrapper(
        checkpoint,
        timestep=np.timedelta64(3, "h"),
        input_steps=[0, 2],
        output_steps=[1, 2],
    )
    points = 3
    inputs = model.add_dataset(
        "data", ["2t", "tp"], DATASET_VARIABLES, np.zeros(points), np.zeros(points)
    )
    inputs.static_forcings = {
        "z": torch.zeros(points),
        "lsm": torch.zeros(points),
        "cos_latitude": torch.zeros(points),
    }

    seen = {}

    def fake_forward(x, model_comm_group=None, **kwargs):
        seen["x"] = x["data"]
        seen["fcstep"] = kwargs.get("fcstep")
        n_out = len(model.model_index["data"].output.name_to_index)
        y = torch.zeros(x["data"].shape[0], 2, x["data"].shape[2], points, n_out)
        y[:, 0, ..., 1] = 1.0  # 2t at the first output time
        y[:, 1, ..., 1] = 2.0  # 2t at the second
        return {"data": y}

    model.model.forward = fake_forward

    states = torch.zeros((1, 2, 1, points, len(DATASET_VARIABLES)))
    states[:, 1, ..., DATASET_VARIABLES.index("2t")] = 5.0
    time = np.datetime64("2022-06-21T00")
    outputs = interpolate(model, "data", states, time)

    assert [t for t, _ in outputs] == [
        np.datetime64("2022-06-21T03"),
        np.datetime64("2022-06-21T06"),
    ]
    assert seen["fcstep"] == 0
    # model input: both boundaries, model variable order (no diagnostics), 2t of the second boundary
    assert seen["x"].shape == (1, 2, 1, points, len(NAMES) - 1)
    assert torch.all(
        seen["x"][:, 1, ..., model.model_index["data"].input.name_to_index["2t"]] == 5.0
    )
    assert torch.equal(outputs[0][1][0, :, 0], torch.ones(points))
    assert torch.equal(outputs[1][1][0, :, 0], torch.full((points,), 2.0))
    assert outputs[0][1].shape == (1, points, 2)
