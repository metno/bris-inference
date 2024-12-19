import os

import numpy as np
from bris import outputs
from bris.predict_metadata import PredictMetadata
from bris.sources import Verif


def test_instantiate():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    num_leadtimes = 4
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(variables, lats, lons, num_leadtimes, num_members, field_shape)

    filename = "%Y%m%d.nc"
    workdir = "test_dir"

    args = {"filename_pattern": filename}

    out = outputs.instantiate("netcdf", pm, workdir, args)


if __name__ == "__main__":
    test_instantiate()
