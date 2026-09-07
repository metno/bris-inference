import os
import tempfile

import numpy as np
import pytest

from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata


def get_test_pm() -> PredictMetadata:
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    num_leadtimes = 4
    num_members = 1
    field_shape = [1, 2]
    return PredictMetadata(
        variables, lats, lons, altitudes, num_leadtimes, num_members, field_shape
    )


def test_num_members():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create test files
        for i in range(3):
            with open(f"{temp_dir}/_{str(i)}.npy", "w") as f:
                f.write("")

        i = Intermediate(predict_metadata=get_test_pm(), workdir=temp_dir)
        # Checking the num_members property
        assert i.num_members == 3


def test_ensemble_accumulator():
    from bris.outputs.intermediate import IntermediateEnsembleAccumulator

    reductions = {"sum": "sum", "lowest": "min", "highest": "max"}
    frt1 = np.datetime64("2023-01-01T06:00:00")
    frt2 = np.datetime64("2023-01-01T00:00:00")
    with tempfile.TemporaryDirectory() as temp_dir:
        workdir = f"{temp_dir}/workdir"
        acc = IntermediateEnsembleAccumulator(workdir, reductions)
        assert acc.get_forecast_reference_times() == []
        assert acc.get_num_members(frt1) == 0

        a = np.array([[1.0, 5.0], [3.0, -1.0]])
        b = np.array([[2.0, 1.0], [0.0, 4.0]])
        acc.accumulate(frt1, {"sum": a, "lowest": a, "highest": a}, 2)
        acc.accumulate(frt1, {"sum": b, "lowest": b, "highest": b}, 3)
        acc.accumulate(frt2, {"sum": b, "lowest": b, "highest": b}, 1)

        assert acc.get_num_members(frt1) == 5
        assert acc.get_num_members(frt2) == 1
        # Sorted, regardless of insertion order
        assert acc.get_forecast_reference_times() == [frt2, frt1]

        result = acc.get(frt1)
        np.testing.assert_array_equal(result["sum"], a + b)
        np.testing.assert_array_equal(result["lowest"], np.minimum(a, b))
        np.testing.assert_array_equal(result["highest"], np.maximum(a, b))

        # No temporary files left behind
        assert all(
            not f.endswith(".tmp.npy") for f in os.listdir(acc.get_directory(frt1))
        )

        acc.cleanup()
        assert not os.path.exists(workdir)


def test_ensemble_accumulator_invalid():
    from bris.outputs.intermediate import IntermediateEnsembleAccumulator

    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            IntermediateEnsembleAccumulator(temp_dir, {"sum": "mean"})
        with pytest.raises(ValueError):
            IntermediateEnsembleAccumulator(temp_dir, {"num_members": "sum"})
        acc = IntermediateEnsembleAccumulator(temp_dir, {"sum": "sum"})
        with pytest.raises(AssertionError):
            acc.accumulate(
                np.datetime64("2023-01-01T00:00:00"), {"other": np.ones(2)}, 1
            )
