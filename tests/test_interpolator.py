"""The interleaving of forecaster and interpolator outputs in Interpolator.forecast, with fake models."""

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from test_model_inputs import (
    DATASET_VARIABLES,
    DIAGNOSTIC,
    FORCING,
    NAMES,
    _Checkpoint,
    _Dataset,
)

from bris.model.interpolator import Interpolator
from bris.model.model_utils import resolve_reforecast_last
from bris.model.runner import ModelRunner

POINTS = 3
T0 = np.datetime64("2022-06-21T00")
H = np.timedelta64(1, "h")


def _fake_forecaster(runner):
    """2t of the output is 100 + the forecast step the model was called with."""

    def forward(x, model_comm_group=None, fcstep=None, **kwargs):
        n_out = len(runner.model_index["data"].output.name_to_index)
        y = torch.zeros(x["data"].shape[0], x["data"].shape[2], POINTS, n_out)
        y[..., 1] = 100 + fcstep
        return {"data": y}

    return forward


def _fake_interpolator(runner):
    """2t of output step k is 1000 + k, counted from 1."""

    def forward(x, model_comm_group=None, fcstep=None, **kwargs):
        n_out = len(runner.model_index["data"].output.name_to_index)
        y = torch.zeros(
            x["data"].shape[0],
            len(runner.output_steps),
            x["data"].shape[2],
            POINTS,
            n_out,
        )
        for k in range(len(runner.output_steps)):
            y[:, k, ..., 1] = 1000 + k + 1
        return {"data": y}

    return forward


def make_interpolator(
    interp_output_steps,
    forecast_length=4,
    interpolation_length=2,
    reforecast_last=None,
):
    """An Interpolator with fake 6 h forecaster and 1 h interpolator, built without checkpoints."""
    predictor = Interpolator.__new__(Interpolator)
    pl.LightningModule.__init__(predictor)
    predictor.num_members_in_parallel = 1
    predictor.release_cache = False
    predictor.fcstep_const = False
    predictor.batch_info = {}
    predictor.model_comm_group = None
    predictor.model_comm_group_rank = 0
    predictor.member_id = 0

    predictor.forecaster = ModelRunner(
        _Checkpoint(NAMES, FORCING, DIAGNOSTIC, multistep=2), timestep=6 * H
    )
    predictor.interpolator = ModelRunner(
        _Checkpoint(NAMES, FORCING, DIAGNOSTIC),
        timestep=H,
        input_steps=[0, 6],
        output_steps=interp_output_steps,
    )
    predictor.forecaster.model.forward = _fake_forecaster(predictor.forecaster)
    predictor.interpolator.model.forward = _fake_interpolator(predictor.interpolator)
    predictor.ds_name_fc = predictor.ds_name_interp = "data"
    predictor.forecast_length = forecast_length
    predictor.interpolation_length = interpolation_length

    lats = np.full(POINTS, 60.0)
    lons = np.full(POINTS, 10.0)
    predictor.forecaster_builder = predictor.forecaster.add_dataset(
        "data", ["2t"], DATASET_VARIABLES, lats, lons
    )
    predictor.interpolator_builder = predictor.interpolator.add_dataset(
        "data",
        ["2t"],
        list(predictor.forecaster.model_index["data"].input.name_to_index.keys()),
        lats,
        lons,
    )
    predictor.forcing_dataset_interp = _Dataset(
        [T0 - np.timedelta64(1, "D")], ["lsm", "z", "cos_latitude"], POINTS
    )
    predictor.reforecast_last = resolve_reforecast_last(
        [0, 6], interp_output_steps, reforecast_last
    )
    return predictor


def run(predictor):
    batch = {"data": torch.ones((1, 2, 1, POINTS, len(DATASET_VARIABLES)))}
    out = predictor.predict_step((batch, [str(T0)]), 0)
    times = [(t - T0) / H for t in out["times"]]
    values = out["pred"]["data"][0, :, 0, 0]
    return times, values


def test_interpolator_does_not_reforecast_last():
    """Output steps 1..5: the interpolator fills in between, the forecaster provides every 6 h."""
    times, values = run(make_interpolator([1, 2, 3, 4, 5]))
    assert times == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
    # lead time 0 is the input state (2t = 1 from the batch)
    assert values[0] == 1
    # hours 1-5 and 7-11 from the interpolator (two calls), 6 and 12 from the forecaster
    assert list(values[1:6]) == [1001, 1002, 1003, 1004, 1005]
    assert values[6] == 100
    assert list(values[7:12]) == [1001, 1002, 1003, 1004, 1005]
    assert values[12] == 101
    # beyond the interpolation length the forecaster output is written as is
    assert values[13] == 102


def test_interpolator_reforecasts_last():
    """Output steps 1..6: the interpolator also outputs the right boundary, replacing the
    forecaster's value there; later lead times still come from the forecaster."""
    times, values = run(make_interpolator([1, 2, 3, 4, 5, 6]))
    assert times == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
    assert values[0] == 1
    assert list(values[1:7]) == [1001, 1002, 1003, 1004, 1005, 1006]
    assert list(values[7:13]) == [1001, 1002, 1003, 1004, 1005, 1006]
    assert values[13] == 102


def test_reforecast_last_off_writes_the_forecaster():
    """Output steps 1..6 with reforecast_last False: the forecaster provides the right
    boundary, the interpolator's output there is dropped."""
    times, values = run(make_interpolator([1, 2, 3, 4, 5, 6], reforecast_last=False))
    assert times == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
    assert list(values[1:6]) == [1001, 1002, 1003, 1004, 1005]
    assert values[6] == 100
    assert list(values[7:12]) == [1001, 1002, 1003, 1004, 1005]
    assert values[12] == 101
    assert values[13] == 102


def test_interpolator_requires_static_forcings_dataset_variables():
    predictor = make_interpolator([1, 2, 3, 4, 5])
    predictor.forcing_dataset_interp = _Dataset([T0], ["lsm", "z"], POINTS)
    with pytest.raises(ValueError, match="Missing the following static forcings"):
        run(predictor)


def test_resolve_reforecast_last():
    # default: whether the interpolator outputs the right boundary
    assert resolve_reforecast_last([0, 6], [1, 2, 3, 4, 5, 6]) is True
    assert resolve_reforecast_last([0, 6], [1, 2, 3, 4, 5]) is False
    # the model option can switch it off, or insist on it when the interpolator outputs it
    assert resolve_reforecast_last([0, 6], [1, 2, 3, 4, 5, 6], False) is False
    assert resolve_reforecast_last([0, 6], [1, 2, 3, 4, 5, 6], True) is True
    with pytest.raises(ValueError, match="reforecast_last=True"):
        resolve_reforecast_last([0, 6], [1, 2, 3, 4, 5], True)
