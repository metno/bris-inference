import os
import tempfile

import eccodes as ecc
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

    altitudes = np.zeros(len(lats))
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
    altitudes = np.zeros(len(lats))
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


def test_missing_values():
    """Points without a value are encoded as missing with a bitmap: a diagnostic variable has
    no values at lead time 0, and an ensemble member may never arrive"""
    variables = ["2t", "tp"]
    lons, lats = get_meps_lats_lons()
    lats, lons = lats[:12], lons[:12]
    leadtimes = np.arange(0, 3600 * 3, 3600)
    num_members = 2
    pm = PredictMetadata(
        variables, lats, lons, np.zeros(12), leadtimes, num_members, [3, 4]
    )

    rng = np.random.default_rng(0)
    pred = rng.random(pm.shape).astype(np.float32) + 1
    pred[0, :, 1] = np.nan  # tp at lead time 0
    pred[1, [2, 7], 0] = np.nan  # two points of 2t at lead time 1
    times = (1672552800 + leadtimes).astype("datetime64[s]")

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "test_%Y%m%dT%HZ.grib")
        output = Grib(pm, os.path.join(temp_dir, "work"), pattern, domain_name="meps")
        # Member 1 never arrives
        output.add_forecast(times, 0, pred)
        output.finalize()

        missing = {}
        values = {}
        with open(os.path.join(temp_dir, "test_20230101T06Z.grib"), "rb") as file:
            while (grib := ecc.codes_grib_new_from_file(file)) is not None:
                # Temperature is parameter category 0, precipitation category 1
                name = {0: "2t", 1: "tp"}[ecc.codes_get(grib, "parameterCategory")]
                key = (
                    name,
                    ecc.codes_get(grib, "forecastTime"),
                    ecc.codes_get(grib, "perturbationNumber"),
                )
                ecc.codes_set(grib, "missingValue", 1e30)
                values[key] = ecc.codes_get_values(grib)
                missing[key] = (
                    ecc.codes_get(grib, "bitmapPresent"),
                    ecc.codes_get(grib, "numberOfMissing"),
                )
                ecc.codes_release(grib)

    assert len(missing) == 2 * 3 * 2
    # Member 0: all of tp at lead time 0, two points of 2t at lead time 1, else complete
    assert missing[("tp", 0, 1)] == (1, 12)
    assert missing[("2t", 1, 1)] == (1, 2)
    assert np.array_equal(np.where(values[("2t", 1, 1)] == 1e30)[0], [2, 7])
    for key in [("2t", 0, 1), ("2t", 2, 1), ("tp", 1, 1), ("tp", 2, 1)]:
        assert missing[key] == (0, 0), key
    np.testing.assert_allclose(values[("2t", 2, 1)], pred[2, :, 0], rtol=1e-5)
    # tp is written in kg/m^2
    np.testing.assert_allclose(values[("tp", 2, 1)], pred[2, :, 1] * 1000, rtol=1e-5)
    # Member 1: nothing
    for (_, _, member), (bitmap, number) in missing.items():
        if member == 2:
            assert (bitmap, number) == (1, 12)


if __name__ == "__main__":
    test_file_exists()
    test_grib_attrs()
