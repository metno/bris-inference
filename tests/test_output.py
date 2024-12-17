import numpy as np
import os


from bris.sources.verif import Verif
from bris import output
from bris.predict_metadata import PredictMetadata


def test_instantiate():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(variables, lats, lons, leadtimes, num_members, field_shape)

    filename = "%Y%m%d.nc"
    workdir = "test_dir"

    args = {"filename": filename}

    out = output.instantiate("netcdf", pm, workdir, args)


if __name__ == "__main__":
    test_instantiate()
