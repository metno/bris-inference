import os
import tempfile

import numpy as np

from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata


def get_test_pm() -> PredictMetadata:
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    return PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )


def test_num_members():
    pm = get_test_pm()
    pm.num_members = 3
    frt = 1672552800
    with tempfile.TemporaryDirectory() as temp_dir:
        i = Intermediate(predict_metadata=pm, workdir=temp_dir)
        assert i.num_members == 0
        for member in range(3):
            i.add_forecast([frt], member, np.zeros(pm.shape, np.float32))
            # Checking the num_members property
            assert i.num_members == member + 1


def test_missing_member():
    """The members are saved independently: one that never arrives is missing (NaN) in the
    ensemble and None on its own"""
    pm = get_test_pm()
    pm.num_members = 3
    frts = [1672552800, 1672552800 + 6 * 3600]
    rng = np.random.default_rng(0)
    pred = rng.random(pm.shape + [pm.num_members]).astype(np.float32)
    with tempfile.TemporaryDirectory() as temp_dir:
        workdir = f"{temp_dir}/work"
        i = Intermediate(predict_metadata=pm, workdir=workdir)
        for frt in frts:
            for member in [2, 0]:
                i.add_forecast([frt], member, pred[..., member])

        assert i.get_forecast_reference_times() == [
            np.datetime64(frt, "s") for frt in frts
        ]
        ensemble = i.get_forecast(frts[1])
        assert ensemble.shape == tuple(pm.shape) + (3,)
        assert np.array_equal(ensemble[..., [0, 2]], pred[..., [0, 2]])
        assert np.all(np.isnan(ensemble[..., 1]))
        assert np.array_equal(i.get_forecast(frts[0], 2), pred[..., 2])
        assert i.get_forecast(frts[0], 1) is None
        assert i.get_forecast(frts[0] + 1, 0) is None

        i.cleanup()
        assert not os.path.exists(workdir)
