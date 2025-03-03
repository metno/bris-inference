from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata
import pytest
import numpy as np
import os
import io
import tempfile


def get_test_pm() -> PredictMetadata:
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    num_leadtimes = 4
    num_members = 1
    field_shape = [1, 2]
    return PredictMetadata(variables, lats, lons, altitudes, num_leadtimes, num_members, field_shape)


def test_num_members():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create test files
        for i in range(3):
            with io.open(f"{temp_dir}/_{str(i)}.npy", 'w')as f:
                f.write("")
        
        i = Intermediate(predict_metadata=get_test_pm(), workdir=temp_dir)
        # Checking the num_members property
        assert i.num_members == 3


if __name__ == "__main__":
    test_num_members()
