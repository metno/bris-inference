import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from bris import outputs
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata


def test_instantiate():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    num_leadtimes = 4
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, num_leadtimes, num_members, field_shape
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "%Y%m%d.nc")
        workdir = os.path.join(temp_dir, "test_dir")

        args = {"filename_pattern": filename}

        _ = outputs.instantiate("netcdf", pm, workdir, args)


def test_get_required_variables():
    args = {"variables": ["2t", "w_500"], "extra_variables": ["ws", "wz_500", "wz_850"]}
    assert outputs.get_required_variables("netcdf", args) == [
        "10u",
        "10v",
        "2t",
        "q_500",
        "q_850",
        "t_500",
        "t_850",
        "w_500",
        "w_850",
    ]
    assert outputs.get_required_variables(
        "grib", args
    ) == outputs.get_required_variables("netcdf", args)
    assert outputs.get_required_variables("verif", {"variable": "wz_500"}) == [
        "w_500",
        "t_500",
        "q_500",
    ]
    assert outputs.get_required_variables("verif", {"variable": "2t"}) == ["2t"]

    with pytest.raises(ValueError):
        outputs.get_required_variables(
            "netcdf", {"variables": ["2t"], "extra_variables": ["foo"]}
        )


def test_extra_variables():
    """add_forecast appends derived variables to the prediction (checked through Intermediate)"""
    variables = ["10u", "10v", "w_500", "t_500", "q_500"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 2, 3600)
    num_leadtimes = len(leadtimes)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )
    pred = np.zeros([num_leadtimes, 2, len(variables)], np.float32)
    pred[..., 0] = 3  # 10u
    pred[..., 1] = 4  # 10v
    pred[..., 2] = -1  # w_500 (omega, Pa/s)
    pred[..., 3] = 250  # t_500
    pred[..., 4] = 0.001  # q_500

    times = 1672552800 + leadtimes
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Intermediate(pm, temp_dir, extra_variables=["ws", "wz_500"])
        assert output.pm.variables == variables + ["ws", "wz_500"]
        output.add_forecast(times, 0, pred)
        result = output.get_forecast(times[0], 0)

    assert result.shape == (num_leadtimes, 2, len(variables) + 2)
    np.testing.assert_allclose(result[..., 5], 5)
    np.testing.assert_allclose(result[..., 6], 0.146419, rtol=1e-5)


def test_netcdf_omega_attributes():
    """w is written as omega (Pa/s) and wz_<level> as upward_air_velocity (m/s)"""
    variables = ["w_500", "t_500", "q_500", "w_850", "t_850", "q_850"]
    lats = np.array([59.0, 60.0])
    lons = np.array([10.0, 11.0])
    altitudes = np.array([100.0, 200.0])
    leadtimes = np.arange(0, 3600 * 2, 3600)
    num_leadtimes = len(leadtimes)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )
    pred = np.zeros([num_leadtimes, 2, len(variables)], np.float32)
    pred[..., 0] = -1
    pred[..., 1] = 250
    pred[..., 2] = 0.0
    pred[..., 3] = -1
    pred[..., 4] = 280
    pred[..., 5] = 0.005
    times = 1672552800 + leadtimes

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "out.nc")
        args = {
            "filename_pattern": filename,
            "variables": ["w_500", "w_850"],
            "extra_variables": ["wz_500", "wz_850"],
        }
        output = outputs.instantiate("netcdf", pm, os.path.join(temp_dir, "work"), args)
        output.add_forecast(times, 0, pred)
        output.finalize()

        with xr.open_dataset(filename) as ds:
            omega = ds["lagrangian_tendency_of_air_pressure_pl"]
            assert omega.attrs["units"] == "Pa/s"
            assert omega.attrs["standard_name"] == "lagrangian_tendency_of_air_pressure"
            assert list(ds["pressure"].values) == [500, 850]
            np.testing.assert_allclose(omega.values, -1)

            wz = ds["upward_air_velocity_pl"]
            assert wz.attrs["units"] == "m/s"
            assert wz.attrs["standard_name"] == "upward_air_velocity"
            np.testing.assert_allclose(wz.sel(pressure=500).values, 0.146329, rtol=1e-5)
            np.testing.assert_allclose(wz.sel(pressure=850).values, 0.096699, rtol=1e-5)
            assert "vertical_velocity_pl" not in ds


if __name__ == "__main__":
    test_instantiate()
    test_get_required_variables()
    test_extra_variables()
    test_netcdf_omega_attributes()
