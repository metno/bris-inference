import numpy as np
import tempfile

from bris.outputs.spatial import SHPowerSpectrum


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

    pred = np.random.rand(pm.shape)
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
            for dim in dimension:
                assert dim in file.coords, dim

            n_lats = int(np.floor((lats.max() - lats.min()) / delta_degrees))
            assert len(file["l"]) == n_lats - 1


if __name__ == "__main__":
    test_SHPowerSpectrum()
