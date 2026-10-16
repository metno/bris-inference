"""InterpolatorOnly.forecast: the interpolator between boundary states read from the dataset, with a fake model."""

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from test_interpolator import POINTS, T0, H, _fake_interpolator
from test_model_inputs import (
    DATASET_VARIABLES,
    DIAGNOSTIC,
    FORCING,
    NAMES,
    _Checkpoint,
    _Dataset,
)

from bris.model.interpolator_only import InterpolatorOnly
from bris.model.model_utils import resolve_reforecast_last
from bris.model.runner import ModelRunner

OUTPUT_VARIABLES = ["2t", "tp"]


def make_predictor(
    interp_output_steps,
    interpolation_length=2,
    boundary_hours=6,
    reforecast_last=None,
):
    """An InterpolatorOnly with a fake 1 h interpolator (input steps 0 and 6), built without checkpoints."""
    predictor = InterpolatorOnly.__new__(InterpolatorOnly)
    pl.LightningModule.__init__(predictor)
    predictor.num_members_in_parallel = 1
    predictor.release_cache = False
    predictor.fcstep_const = False
    predictor.batch_info = {}
    predictor.model_comm_group = None
    predictor.model_comm_group_rank = 0
    predictor.member_id = 0

    predictor.interpolator = ModelRunner(
        _Checkpoint(NAMES, FORCING, DIAGNOSTIC),
        timestep=H,
        input_steps=[0, 6],
        output_steps=interp_output_steps,
    )
    predictor.interpolator.model.forward = _fake_interpolator(predictor.interpolator)
    predictor.ds_name = "data"
    predictor.boundary_timestep = boundary_hours * H
    predictor.interpolation_length = interpolation_length
    predictor._check_time_geometry()

    lats = np.full(POINTS, 60.0)
    lons = np.full(POINTS, 10.0)
    predictor.builder = predictor.interpolator.add_dataset(
        "data", OUTPUT_VARIABLES, DATASET_VARIABLES, lats, lons
    )
    predictor.num_output_variables = len(OUTPUT_VARIABLES)
    predictor.boundary_output_positions = InterpolatorOnly._boundary_positions(
        OUTPUT_VARIABLES, DATASET_VARIABLES
    )
    predictor.forcing_dataset = _Dataset(
        [T0 - np.timedelta64(1, "D")], ["lsm", "z", "cos_latitude"], POINTS
    )
    predictor.reforecast_last = resolve_reforecast_last(
        [0, 6], interp_output_steps, reforecast_last
    )
    return predictor


def run(predictor, n_states):
    """Boundary state k has 2t = 10 k and tp = 5 k; returns lead times (h) and (lead time, variable)."""
    states = torch.zeros((1, n_states, 1, POINTS, len(DATASET_VARIABLES)))
    for k in range(n_states):
        states[:, k, ..., DATASET_VARIABLES.index("2t")] = 10 * k
        states[:, k, ..., DATASET_VARIABLES.index("tp")] = 5 * k
    out = predictor.predict_step(({"data": states}, [str(T0)]), 0)
    times = [(t - T0) / H for t in out["times"]]
    return times, out["pred"]["data"][0, :, 0, :]


def test_boundaries_from_dataset_between_interpolations():
    """Output steps 1..5: the interpolator fills in between, the boundary states are written as is."""
    times, values = run(make_predictor([1, 2, 3, 4, 5]), 3)
    assert times == list(range(13))
    # lead time 0 and the boundaries at 6 and 12 h are the dataset states, diagnostic tp included
    assert list(values[[0, 6, 12], 0]) == [0, 10, 20]
    assert list(values[[0, 6, 12], 1]) == [0, 5, 10]
    # in between: the interpolator (2t = 1000 + output step, tp = 0 from the fake model)
    assert list(values[1:6, 0]) == [1001, 1002, 1003, 1004, 1005]
    assert list(values[7:12, 0]) == [1001, 1002, 1003, 1004, 1005]
    assert list(values[1:6, 1]) == [0, 0, 0, 0, 0]


def test_interpolator_reforecasts_last():
    """Output steps 1..6: the interpolator also outputs the right boundary, replacing the dataset's."""
    times, values = run(make_predictor([1, 2, 3, 4, 5, 6]), 3)
    assert times == list(range(13))
    assert list(values[0]) == [0, 0]
    assert list(values[1:7, 0]) == [1001, 1002, 1003, 1004, 1005, 1006]
    assert list(values[7:13, 0]) == [1001, 1002, 1003, 1004, 1005, 1006]


def test_reforecast_last_off_writes_the_dataset():
    """Output steps 1..6 with reforecast_last False: the right boundary comes from the
    dataset instead of the interpolator, the other lead times are unchanged."""
    times, values = run(make_predictor([1, 2, 3, 4, 5, 6], reforecast_last=False), 3)
    assert times == list(range(13))
    assert list(values[[0, 6, 12], 0]) == [0, 10, 20]
    assert list(values[[0, 6, 12], 1]) == [0, 5, 10]
    assert list(values[1:6, 0]) == [1001, 1002, 1003, 1004, 1005]
    assert list(values[7:12, 0]) == [1001, 1002, 1003, 1004, 1005]


def test_output_variable_missing_from_dataset_is_nan():
    predictor = make_predictor([1, 2, 3, 4, 5])
    predictor.boundary_output_positions = InterpolatorOnly._boundary_positions(
        OUTPUT_VARIABLES, [v if v != "tp" else "other" for v in DATASET_VARIABLES]
    )
    _, values = run(predictor, 3)
    assert np.isnan(values[0, 1]) and np.isnan(values[6, 1])
    assert values[0, 0] == 0 and values[6, 0] == 10


def test_wrong_number_of_boundary_states():
    with pytest.raises(ValueError, match="boundary states"):
        run(make_predictor([1, 2, 3, 4, 5]), 4)


def test_boundary_timestep_must_match_input_steps():
    with pytest.raises(ValueError, match="boundary_timestep_seconds"):
        make_predictor([1, 2, 3, 4, 5], boundary_hours=3)
