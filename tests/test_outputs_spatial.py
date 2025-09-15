import os
import tempfile

import numpy as np
import xarray as xr

from bris.outputs.spatial import DCTPowerSpectrum, SHPowerSpectrum
from bris.predict_metadata import PredictMetadata


def test_SHPowerSpectrum():
    variables = ["z_500", "t_850", "msl"]
    lats = np.array([-85, 0, 85])
    lons = np.array([-175, 40, 175])
    altitudes = [20, 10, 30]
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 2
    field_shape = [3]
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "test_SHPowerSpectrum.nc")
        workdir = os.path.join(temp_dir, "test_SHPowerSpectrum")
        variable = "z_500"
        delta_degrees = 10

        out1 = SHPowerSpectrum(pm, workdir, filename, variable)
        out2 = SHPowerSpectrum(pm, workdir, filename, variable, delta_degrees)

        for member in range(num_members):
            out1.add_forecast(times, member, pred)
            out2.add_forecast(times, member, pred)

        # Only finalize the one with delta degrees because its easier to test on
        out2.finalize()

        assert os.path.exists(filename)

        with xr.open_dataset(filename) as file:
            var_name = "sh_power_spectrum_" + variable
            assert var_name in file.variables

            dimensions = ["time", "leadtime", "l", "ensemble_member"]
            for dim in dimensions:
                assert dim in file.coords, dim

            n_lats = int(np.floor((lats.max() - lats.min()) / delta_degrees))
            assert len(file["l"]) == n_lats - 1


def get_test_regular_latlons():
    lats = np.array(
        [
            51.633816,
            51.638386,
            51.642944,
            51.647488,
            51.655403,
            51.659977,
            51.664536,
            51.669086,
            51.67699,
            51.68157,
            51.68613,
            51.69068,
            51.69858,
            51.70316,
            51.707726,
            51.71228,
        ]
    )
    lons = np.array(
        [
            1.6049105,
            1.6396946,
            1.6744866,
            1.7092867,
            1.5975349,
            1.632337,
            1.667147,
            1.7019651,
            1.5901512,
            1.6249714,
            1.6597995,
            1.6946356,
            1.5827595,
            1.6175977,
            1.6524439,
            1.6872982,
        ]
    )
    return lats, lons, [4, 4]


def test_DCTPowerSpectrum():
    variables = ["z_500", "t_850", "msl"]
    lats, lons, field_shape = get_test_regular_latlons()
    altitudes = np.ones(len(lats))
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 2
    pm = PredictMetadata(
        variables, lats, lons, altitudes, leadtimes, num_members, field_shape
    )

    pred = np.random.rand(*pm.shape)
    frt = 1672552800
    times = frt + leadtimes

    with tempfile.TemporaryDictionairy() as temp_dir:
        filename = os.path.join(temp_dir, "test_DCTPowerSpectrum.nc")
        workdir = os.path.join(temp_dir, "test_DCTPowerSpectrum")
        variable = "msl"
        domain_name = "meps"
        n_bins = 5

        out1 = DCTPowerSpectrum(
            pm, workdir, filename, variable, domain_name=domain_name
        )
        out2 = DCTPowerSpectrum(
            pm, workdir, filename, variable, domain_name=domain_name, n_bins=n_bins
        )

        for member in range(num_members):
            out1.add_forecast(times, member, pred)
            out2.add_forecast(times, member, pred)

        # Only finalize the one with n_bins since its easier to test on
        out2.finalize()

        assert os.path.exists(filename)

        with xr.open_dataset(filename) as file:
            var_name = "power_spectrum_" + variable
            assert var_name in file.variables

            dimensions = ["time", "leadtime", "k", "ensemble_member"]
            for dim in dimensions:
                assert dim in file.coords, dim

            assert len(file["k"]) == n_bins


if __name__ == "__main__":
    test_SHPowerSpectrum()
    test_DCTPowerSpectrum()
