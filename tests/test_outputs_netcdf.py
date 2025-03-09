import os
import tempfile

import numpy as np

from bris.outputs.netcdf import Netcdf
from bris.predict_metadata import PredictMetadata


def test_1():
    variables = ["u_800", "u_600", "2t", "v_500", "10u", "tp"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT00Z.nc")
        workdir = os.path.join(temp_dir, "test_gridded")
        attrs = {"creator": "met.no"}
        output = Netcdf(pm, workdir, pattern, interp_res=0.2, global_attributes=attrs)

        pred = np.random.rand(*pm.shape)
        frt = 1672552800
        times = frt + leadtimes
        for member in range(num_members):
            output.add_forecast(times, member, pred)
            output.finalize()


def test_domain_name():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.random.rand(*lats.shape)
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test2_%Y%m%dT00Z.nc")
        workdir = os.path.join(temp_dir, "test_gridded")
        output = Netcdf(pm, workdir, pattern, domain_name="meps")

        pred = np.random.rand(*pm.shape)
        frt = 1672552800
        times = frt + leadtimes
        for member in range(num_members):
            output.add_forecast(times, member, pred)
            output.finalize()


if __name__ == "__main__":
    test_domain_name()
    test_1()
