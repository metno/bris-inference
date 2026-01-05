import os
import tempfile

import numpy as np
import xarray as xr

from bris.outputs.netcdf import Netcdf
from bris.predict_metadata import PredictMetadata


def test_deterministic():
    variables = ["u_800", "u_600", "2t", "v_500", "10u", "tp", "skt"]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.nc")
        workdir = os.path.join(temp_dir, "test_gridded")
        attrs = {"creator": "met.no"}
        output = Netcdf(
            pm,
            workdir,
            pattern,
            accumulated_variables=["2t", "tp"],
            global_attributes=attrs,
        )

        for member in range(num_members):
            output.add_forecast(times, member, pred)
        output.finalize()

        output_filename = os.path.join(temp_dir, "test_20230101T06Z.nc")

        assert os.path.exists(output_filename)

        with xr.open_dataset(output_filename, decode_times=False) as file:
            # Check that global attributes are written
            for k, v in attrs.items():
                assert file.attrs[k] == v

            assert file.variables["forecast_reference_time"].dtype == "float64"
            assert file.variables["projection"].dtype == "int32"

            for variable in [
                "altitude",
                "air_temperature_2m",
                "air_temperature_0m",
                "x_wind_pl",
                "precipitation_amount_acc",
            ]:
                assert variable in file.variables, variable
                var = file.variables[variable]
                assert "units" in var.attrs
                assert "grid_mapping" in var.attrs

            # (time, height, y, x)
            assert file.variables["air_temperature_2m"].shape == (4, 1, 1, 2)

            height_dim = file.variables["air_temperature_2m"].dims[1]
            levels = file.variables[height_dim].values
            assert levels == [2]

            height_dim = file.variables["air_temperature_0m"].dims[1]
            levels = file.variables[height_dim].values
            assert levels == [0]

            # Test that accumulated values are correct
            assert "2t_acc" in file.variables
            tp = file["precipitation_amount"].data
            tp_acc = file["precipitation_amount_acc"].data
            assert np.allclose(
                np.cumsum(np.nan_to_num(tp, nan=0), axis=0), tp_acc, atol=1e-4
            )

    # Test interpolation
    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.nc")
        workdir = os.path.join(temp_dir, "test_gridded")
        attrs = {"creator": "met.no"}
        output = Netcdf(pm, workdir, pattern, interp_res=0.2)

        for member in range(num_members):
            output.add_forecast(times, member, pred)
        output.finalize()

        output_filename = os.path.join(temp_dir, "test_20230101T06Z.nc")
        with xr.open_dataset(output_filename) as file:
            assert "altitude" in file.variables
            assert np.max(file.variables["altitude"]) == 200
            assert np.min(file.variables["altitude"]) == 100


def test_ensemble():
    variables = [
        "u_500",
        "u_800",
        "u_600",
        "2t",
        "v_500",
        "10u",
        "tp",
        "skt",
        "unknown",
    ]
    lats = np.array([1, 2])
    lons = np.array([2, 4])
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 2
    field_shape = [1, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.nc")
        workdir = os.path.join(temp_dir, "test_gridded")
        attrs = {"creator": "met.no"}
        output = Netcdf(pm, workdir, pattern, global_attributes=attrs)

        for member in range(num_members):
            output.add_forecast(times, member, pred)
        output.finalize()

        output_filename = os.path.join(temp_dir, "test_20230101T06Z.nc")

        assert os.path.exists(output_filename)

        with xr.open_dataset(output_filename) as file:
            # Check that global attributes are written
            for k, v in attrs.items():
                assert file.attrs[k] == v

            for variable in [
                "altitude",
                "air_temperature_2m",
                "air_temperature_0m",
                "x_wind_pl",
            ]:
                assert variable in file.variables, variable
                var = file.variables[variable]
                assert "units" in var.attrs
                assert "grid_mapping" in var.attrs

            # (time, height, member, y, x)
            assert file.variables["air_temperature_2m"].shape == (4, 1, 2, 1, 2)

            height_dim = file.variables["air_temperature_2m"].dims[1]
            levels = file.variables[height_dim].values
            assert levels == [2]

            height_dim = file.variables["air_temperature_0m"].dims[1]
            levels = file.variables[height_dim].values
            assert levels == [0]

            # (time, pressure, member, y, x)
            assert file.variables["x_wind_pl"].shape == (4, 3, 2, 1, 2)

            height_dim = file.variables["x_wind_pl"].dims[1]
            levels = file.variables[height_dim].values
            assert len(levels) == 3
            assert levels[0] == 500
            assert levels[1] == 600
            assert levels[2] == 800
            assert file.variables["x_wind_pl"].values.shape[2] == num_members

    # Test interpolation
    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.nc")
        workdir = os.path.join(temp_dir, "test_gridded")
        attrs = {"creator": "met.no"}
        output = Netcdf(pm, workdir, pattern, interp_res=0.2)

        for member in range(num_members):
            output.add_forecast(times, member, pred)
        output.finalize()

        output_filename = os.path.join(temp_dir, "test_20230101T06Z.nc")
        with xr.open_dataset(output_filename) as file:
            # Check that altitude variable has attributes
            assert "altitude" in file.variables


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
    test_deterministic()
    test_ensemble()
