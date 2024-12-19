import numpy as np
from bris.outputs.netcdf import Netcdf
from bris.predict_metadata import PredictMetadata


def test_1():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(variables, lats, lons, len(leadtimes), num_members, field_shape)
    pattern = "test_%Y%m%dT00Z.nc"
    workdir = "test_gridded"
    output = Netcdf(pm, workdir, pattern, interp_res=0.2)

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes
    for member in range(num_members):
        output.add_forecast(times, member, pred)

    output.finalize()


if __name__ == "__main__":
    test_1()
