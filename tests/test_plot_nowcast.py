import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from bris.plot_nowcast import plot_nowcast


def test_plot_nowcast_creates_gif_and_png():
    time = np.array(
        [
            np.datetime64("2026-06-19T06:45:00"),
            np.datetime64("2026-06-19T06:50:00"),
            np.datetime64("2026-06-19T06:55:00"),
        ]
    )
    y = np.arange(3, dtype=np.float32)
    x = np.arange(4, dtype=np.float32)
    ensemble_member = np.array([0, 1], dtype=np.int32)
    data = np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4)

    ds = xr.Dataset(
        data_vars={
            "lwe_precipitation_rate": (
                ("time", "ensemble_member", "y", "x"),
                data,
                {"units": "mm/h"},
            )
        },
        coords={
            "time": time.astype("datetime64[s]").astype("int64"),
            "ensemble_member": ensemble_member,
            "y": y,
            "x": x,
            "latitude": (("y", "x"), np.array([[60.0, 60.0, 60.0, 60.0], [61.0, 61.0, 61.0, 61.0], [62.0, 62.0, 62.0, 62.0]], dtype=np.float32)),
            "longitude": (("y", "x"), np.array([[5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)),
        },
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "forecast.nc"
        output = Path(temp_dir) / "preview.gif"
        ds.to_netcdf(source)

        gif_path, png_path = plot_nowcast(str(source), str(output))

        assert gif_path == output
        assert png_path == output.with_suffix(".png")
        assert gif_path.exists()
        assert png_path.exists()
        assert gif_path.stat().st_size > 0
        assert png_path.stat().st_size > 0
