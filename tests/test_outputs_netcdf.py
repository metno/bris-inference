import os
import tempfile

import numpy as np
import pytest
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


def test_ensemble_streaming():
    """Members are written one at a time from the intermediate files; a member that was
    never added stays missing and the others keep their values"""
    variables = ["2t", "10u", "10v", "tp"]
    lats = np.array([59.0, 59.0, 60.0, 60.0])
    lons = np.array([10.0, 11.0, 10.0, 11.0])
    altitudes = np.array([10.0, 20.0, 30.0, 40.0])
    leadtimes = np.arange(0, 3600 * 3, 3600)
    num_members = 3
    field_shape = [2, 2]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    rng = np.random.default_rng(1)
    pred = rng.random(pm.shape + [num_members]).astype(np.float32)
    frt = 1672552800
    times = frt + leadtimes

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.nc")
        workdir = os.path.join(temp_dir, "work")
        output = Netcdf(pm, workdir, pattern, variables=["10u", "tp"])

        for member in [0, 2]:
            output.add_forecast(times, member, pred[..., member])
        output.finalize()

        # Intermediate files are removed after writing
        assert not os.path.exists(workdir)

        output_filename = os.path.join(temp_dir, "test_20230101T06Z.nc")
        with xr.open_dataset(output_filename) as file:
            u = file["x_wind_10m"]
            assert u.dims == ("time", "height", "ensemble_member", "y", "x")
            assert u.shape == (3, 1, 3, 2, 2)
            assert np.all(np.isnan(u.values[:, :, 1, ...]))
            for member in [0, 2]:
                expected = np.reshape(pred[:, :, 1, member], (3, 2, 2))
                assert np.array_equal(u.values[:, 0, member, ...], expected)
            assert u.attrs["units"] == "m/s"


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


def test_regular_grid():
    """Points of a lat-cropped (non-rectangular) projected grid, no 2-D field_shape: with
    regular_grid the netcdf output is the largest complete x/y rectangle of the points"""
    import pyproj

    from bris import projections

    proj4_str = projections.get_proj4_str("meps")
    nx, ny, spacing = 10, 8, 2500.0
    xx, yy = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
    # Remove the two upper corners -> largest rectangle is the lower 6 rows x 10 columns
    keep = ~((yi >= 6) & ((xi < 2) | (xi > 7))).ravel()
    xx, yy = xx.ravel()[keep], yy.ravel()[keep]
    to_lonlat = pyproj.Transformer.from_crs(
        pyproj.CRS.from_user_input(proj4_str),
        pyproj.CRS.from_epsg(4326),
        always_xy=True,
    )
    lons, lats = to_lonlat.transform(xx, yy)

    variables = ["2t", "10u", "10v"]
    altitudes = np.random.rand(len(lats))
    leadtimes = np.arange(0, 3600 * 3, 3600)
    num_members = 1
    pm = PredictMetadata(
        variables,
        lats,
        lons,
        altitudes,
        leadtimes,
        num_members,
        field_shape=(len(lats),),
    )
    pred = np.random.rand(*pm.shape).astype(np.float32)
    frt = 1672552800
    times = frt + leadtimes

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.nc")
        workdir = os.path.join(temp_dir, "test_regular_grid")
        output = Netcdf(
            pm,
            workdir,
            pattern,
            proj4_str=proj4_str,
            regular_grid={
                "x_spacing": spacing,
                "y_spacing": spacing,
                "absolute_tolerance": 1.0,
            },
        )
        assert output.grid_subset.shape == (6, nx)
        output.add_forecast(times, 0, pred)
        output.finalize()

        with xr.open_dataset(
            os.path.join(temp_dir, "test_20230101T06Z.nc"), decode_times=False
        ) as file:
            assert file["x"].shape == (nx,) and file["y"].shape == (6,)
            np.testing.assert_allclose(np.diff(file["x"].values), spacing)
            np.testing.assert_allclose(np.diff(file["y"].values), spacing)
            np.testing.assert_allclose(
                file["x"].values, np.arange(nx) * spacing, atol=1e-6
            )
            np.testing.assert_allclose(
                file["y"].values, np.arange(6) * spacing, atol=1e-6
            )
            assert (
                file["projection"].attrs["grid_mapping_name"]
                == "lambert_conformal_conic"
            )
            # (time, height, y, x)
            t2m = file["air_temperature_2m"]
            assert t2m.shape == (3, 1, 6, nx)
            # The written field is the rectangle subset of the points, in row-major x/y order
            idx = output.grid_subset.node_indices
            np.testing.assert_allclose(
                t2m.values[:, 0], pred[:, idx, 0].reshape(3, 6, nx), rtol=1e-6
            )
            np.testing.assert_allclose(
                file["latitude"].values, lats[idx].reshape(6, nx)
            )
            np.testing.assert_allclose(
                file["longitude"].values, lons[idx].reshape(6, nx)
            )
            np.testing.assert_allclose(
                file["altitude"].values, altitudes[idx].reshape(6, nx)
            )
            # The rotated wind rows are the same points
            assert file["x_wind_10m"].shape == (3, 1, 6, nx)

        # Inferred spacing (regular_grid: true) finds the same rectangle
        output2 = Netcdf(
            pm, workdir + "2", pattern, domain_name="meps", regular_grid=True
        )
        assert output2.grid_subset.shape == (6, nx)
        np.testing.assert_array_equal(
            output2.grid_subset.node_indices, output.grid_subset.node_indices
        )

        # The rectangle given as (row, column) indices (RegularGridIndices attribute of the
        # checkpoint graph) instead of recomputed: same output
        indices = np.full((len(lats), 2), -1, dtype=np.int64)
        rows, cols = np.divmod(np.arange(6 * nx), nx)
        indices[output.grid_subset.node_indices, 0] = rows
        indices[output.grid_subset.node_indices, 1] = cols
        output3 = Netcdf(
            pm, workdir + "3", pattern, domain_name="meps", grid_indices=indices
        )
        assert output3.grid_subset.shape == (6, nx)
        np.testing.assert_array_equal(
            output3.grid_subset.node_indices, output.grid_subset.node_indices
        )
        np.testing.assert_allclose(
            output3.grid_subset.x, output.grid_subset.x, atol=1e-6
        )
        np.testing.assert_allclose(
            output3.grid_subset.y, output.grid_subset.y, atol=1e-6
        )

    with pytest.raises(ValueError, match="proj4_str"):
        Netcdf(pm, workdir, pattern, regular_grid=True)
    with pytest.raises(ValueError, match="proj4_str"):
        Netcdf(pm, workdir, pattern, grid_indices=indices)
    with pytest.raises(ValueError, match="shape"):
        Netcdf(pm, workdir, pattern, domain_name="meps", grid_indices=indices[:5])
    with pytest.raises(ValueError, match="interp_res"):
        Netcdf(
            pm, workdir, pattern, domain_name="meps", interp_res=0.1, regular_grid=True
        )


if __name__ == "__main__":
    test_domain_name()
    test_deterministic()
    test_ensemble()
