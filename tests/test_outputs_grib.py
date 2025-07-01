import os
import tempfile

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from bris import projections
from bris.outputs.grib import Grib
from bris.predict_metadata import PredictMetadata


def get_meps_lats_lons():
    # Define CRS and transformers
    lcc_crs = CRS.from_proj4(projections.get_proj4_str("meps"))
    geo_crs = CRS.from_epsg(4326)
    to_geo = Transformer.from_crs(lcc_crs, geo_crs, always_xy=True)

    x0 = -1060084.00
    y0 = -1332517.875
    dx = 10_000
    dy = 10_000
    nx = 238
    ny = 268
    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy

    # Create meshgrid of projected coordinates
    x_grid, y_grid = np.meshgrid(x, y)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    return to_geo.transform(x_flat, y_flat)  # flat


def test_file_exists():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]

    lons, lats = get_meps_lats_lons()

    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [268, 238]  # NOTE: Hard-coded to MEPS domain for now

    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes
    times = times.astype("datetime64[s]")

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.grib")
        workdir = os.path.join(temp_dir, "test_gridded")

        output = Grib(pm, workdir, pattern, domain_name="meps")

        for member in range(num_members):
            output.add_forecast(times, member, pred)
        output.finalize()

        output_filename = os.path.join(temp_dir, "test_20230101T06Z.grib")
        assert os.path.exists(output_filename)


def test_grib_attrs():
    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lons, lats = get_meps_lats_lons()
    altitudes = np.array([100, 200])
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 1
    field_shape = [268, 238]  # NOTE: Hard-coded to MEPS domain for now

    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes
    times = times.astype("datetime64[s]")

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.grib")
        workdir = os.path.join(temp_dir, "test_gridded")

        output = Grib(pm, workdir, pattern, domain_name="meps")

        for member in range(num_members):
            output.add_forecast(times, member, pred)
        output.finalize()

        filters = [
            {
                "shortName": "2t",
                "typeOfLevel": "heightAboveGround",
                "level": 2,
                "units": "K",
            },
            {
                "cfVarName": "u",
                "typeOfLevel": "isobaricInhPa",
                "level": 600,
                "units": "m s**-1",
            },
            {
                "cfVarName": "u",
                "typeOfLevel": "isobaricInhPa",
                "level": 800,
                "units": "m s**-1",
            },
            {
                "cfVarName": "v",
                "typeOfLevel": "isobaricInhPa",
                "level": 500,
                "units": "m s**-1",
            },
            {
                "cfVarName": "u10",
                "typeOfLevel": "heightAboveGround",
                "level": 10,
                "units": "m s**-1",
            },
        ]

        for f in filters:
            with xr.open_dataset(
                os.path.join(temp_dir, "test_20230101T06Z.grib"),
                backend_kwargs={
                    "filter_by_keys": f,
                },
                decode_timedelta=True,
            ) as file:
                for var in file.data_vars:
                    assert file[var].attrs.get("GRIB_typeOfLevel") == f.get(
                        "typeOfLevel"
                    )
                    assert file[file[var].attrs.get("GRIB_typeOfLevel")] == f.get(
                        "level"
                    )
                    assert (
                        file[var].attrs.get("GRIB_numberOfPoints") == file[var][0].size
                    )
                    assert file[var].attrs.get("GRIB_units") == f.get("units")


if __name__ == "__main__":
    test_file_exists()
    test_grib_attrs()
