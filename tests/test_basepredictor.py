"""The predict_step contract of BasePredictor: a subclass only implements forecast()."""

import numpy as np
import torch
from anemoi.utils.config import DotDict

from bris.model.basepredictor import BasePredictor


class _Checkpoint:
    metadata = DotDict(
        {"provenance_training": {"module_versions": {"anemoi.training": "0.0"}}}
    )


class _Predictor(BasePredictor):
    """Two lead times: the input state and the input state + 1, for one variable."""

    def __init__(self):
        super().__init__(
            checkpoints={"forecaster": _Checkpoint()},
            hardware_config={},
            num_members_in_parallel=1,
        )
        self.member_id = 0
        self.seen = []

    def forecast(self, batch, time):
        self.seen.append((set(batch), time))
        x = batch["data"][:, -1, 0, :, :1]  # (batch, points, 1)
        return [(time, {"data": x}), (time + np.timedelta64(6, "h"), {"data": x + 1})]


def test_predict_step_wraps_forecast():
    predictor = _Predictor()
    states = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(1, 2, 1, 3, 2)
    batch = ({"data": states}, ["2022-01-01T06:00:00"])

    out = predictor.predict_step(batch, 0)

    time = np.datetime64("2022-01-01T06")
    assert predictor.seen == [({"data"}, time)]
    assert out["times"] == [time, np.datetime64("2022-01-01T12")]
    # lead times stacked on axis 1, float32 numpy
    assert out["pred"]["data"].shape == (1, 2, 3, 1)
    assert out["pred"]["data"].dtype == np.float32
    np.testing.assert_array_equal(
        out["pred"]["data"][0, 0, :, 0], states[0, -1, 0, :, 0].numpy()
    )
    np.testing.assert_array_equal(
        out["pred"]["data"][0, 1, :, 0], states[0, -1, 0, :, 0].numpy() + 1
    )
    assert out["group_rank"] == 0

    # ensemble members run in sequence are numbered per initialisation time
    assert out["ensemble_member"] == 0
    assert predictor.predict_step(batch, 1)["ensemble_member"] == 1
    other = ({"data": states}, ["2022-01-02T06:00:00"])
    assert predictor.predict_step(other, 2)["ensemble_member"] == 0
