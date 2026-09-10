import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf

from bris import obs, sources


def _make_zarr(path, variables, lat, lon, dates, values):
    """Write a minimal anemoi-datasets-like zarr store."""
    z = zarr.open(str(path), mode="w")
    z.create_dataset(
        "data",
        data=values.astype(np.float32),
        chunks=(1, values.shape[1], 1, values.shape[3]),
    )
    z.create_dataset("latitudes", data=lat.astype(np.float64))
    z.create_dataset("longitudes", data=lon.astype(np.float64))
    z.create_dataset("dates", data=dates.astype("datetime64[s]"))
    z.attrs["variables"] = list(variables)
    return str(path)


@pytest.fixture
def grid():
    lat, lon = np.meshgrid(
        np.arange(-10, 10.01, 1.0), np.arange(0, 20.01, 1.0), indexing="ij"
    )
    return lat.ravel(), lon.ravel()


@pytest.fixture
def stores(tmp_path, grid):
    lat, lon = grid
    n = len(lat)
    dates = np.array(
        ["2023-01-01T00", "2023-01-01T06", "2023-01-01T12", "2023-01-01T18"],
        dtype="datetime64[s]",
    )
    idx = np.arange(n, dtype=np.float32)
    t = np.arange(len(dates), dtype=np.float32)[:, None]
    main_vars = ["2t", "10u", "10v", "tp", "z"]
    main = np.stack(
        [
            273.15 + 0.01 * idx + t,  # 2t [K]
            3.0 + 0 * idx + t,  # 10u
            4.0 + 0 * idx + 0 * t,  # 10v
            0.001 * (idx % 5) + 0 * t,  # tp [m]
            9.81 * 100.0 * (idx % 7) + 0 * t,  # z -> altitude 0..600 m
        ],
        axis=1,
    )[:, :, None, :]
    extra = (0.5 + 0 * idx + 0.1 * t)[:, None, None, :]  # tcc
    return {
        "main": _make_zarr(tmp_path / "main.zarr", main_vars, lat, lon, dates, main),
        "extra": _make_zarr(tmp_path / "extra.zarr", ["tcc"], lat, lon, dates, extra),
        "n": n,
        "dates": dates,
    }


def test_select_points_spacing(grid):
    lat, lon = grid
    global_index, location_id = obs.select_points(
        lat, lon, area=[5, 2, -5, 12], spacing=2.5
    )
    assert len(global_index) == 25  # 5 x 5 target nodes, all distinct nearest points
    assert len(np.unique(location_id)) == 25
    assert np.all(lat[global_index] <= 5) and np.all(lat[global_index] >= -5)
    assert np.all(lon[global_index] >= 2) and np.all(lon[global_index] <= 12)


def test_select_points_every(grid):
    lat, lon = grid
    global_index, location_id = obs.select_points(lat, lon, area=None, every=10)
    assert len(global_index) == len(lat) // 10 + (1 if len(lat) % 10 else 0)
    np.testing.assert_array_equal(global_index, location_id)  # no crop: same indexing


def test_select_points_all(grid):
    lat, lon = grid
    global_index, _ = obs.select_points(lat, lon, area=[1, 0, 0, 1])
    assert len(global_index) == 4


def test_select_points_errors(grid):
    lat, lon = grid
    with pytest.raises(ValueError):
        obs.select_points(lat, lon, area=[0, 0, 5, 5])  # N <= S
    with pytest.raises(ValueError):
        obs.select_points(lat, lon, spacing=1.0, every=2)


def _verif(filename, variable, units=None):
    entry = {"filename": str(filename), "variable": variable}
    if units:
        entry["units"] = units
    return {"verif": entry}


def test_parse_recipe():
    r = obs.parse_recipe("/a.zarr")
    assert r["paths"] == ["/a.zarr"] and r["area"] is None
    r = obs.parse_recipe(
        {
            "dataset": {"join": [{"dataset": "/a.zarr"}, "/b.zarr"]},
            "area": [5, 2, -5, 12],
            "every_loc": 10,
            "start": "2023-01-01",
        }
    )
    assert r["paths"] == ["/a.zarr", "/b.zarr"]
    assert (
        r["area"] == [5, 2, -5, 12] and r["every"] == 10 and r["start"] == "2023-01-01"
    )
    with pytest.raises(ValueError, match="unsupported keys"):
        obs.parse_recipe({"dataset": "/a.zarr", "select": ["2t"]})
    with pytest.raises(ValueError, match="conflicting"):
        obs.parse_recipe(
            {
                "dataset": {"dataset": "/a.zarr", "area": [1, 0, 0, 1]},
                "area": [2, 0, 0, 2],
            }
        )
    with pytest.raises(ValueError, match="either"):
        obs.parse_recipe({"dataset": "/a.zarr", "every_loc": 2, "spacing": 1.0})


def test_run_writes_verif_files(tmp_path, stores):
    out = tmp_path / "out"
    config = OmegaConf.create(
        {
            "start_date": "2023-01-01T06:00:00",
            "end_date": "2023-01-01T18:00:00",
            "workers": 1,
            "dataset": {
                "dataset": {
                    "join": [{"dataset": stores["main"]}, {"dataset": stores["extra"]}]
                },
                "area": [5, 2, -5, 12],
                "spacing": 2.5,
            },
            "outputs": [
                _verif(out / "t2m" / "analysis.nc", "2t", "degC"),
                _verif(out / "precip6h" / "analysis.nc", "tp", "mm"),
                _verif(out / "ws10m" / "analysis.nc", "ws"),
                _verif(out / "tcc" / "analysis.nc", "tcc"),
            ],
        }
    )
    written = obs.run(config)
    assert [p.name for p in written] == ["analysis.nc"] * 4
    assert {p.parent.name for p in written} == {"t2m", "precip6h", "ws10m", "tcc"}

    # Readable by the bris verif source, with the expected values and units
    src = sources.Verif(str(out / "t2m" / "analysis.nc"))
    locations = src.locations
    assert len(locations) == 25
    assert src.units == "degC"
    start = 1672552800  # 2023-01-01T06
    result = src.get("2t", start, start + 12 * 3600, 6 * 3600)
    data = result.get_data("2t", start)
    # location ids index the cropped grid; recover global indices from lat/lon to check values
    lat = np.array([loc.lat for loc in locations])
    lon = np.array([loc.lon for loc in locations])
    gi = (np.round(lat + 10) * 21 + np.round(lon)).astype(int)
    np.testing.assert_allclose(data, 273.15 + 0.01 * gi + 1.0 - 273.15, atol=1e-3)
    np.testing.assert_allclose(
        [loc.elev for loc in locations], 100.0 * (gi % 7), atol=1e-3
    )

    tp = sources.Verif(str(out / "precip6h" / "analysis.nc"))
    assert tp.units == "mm"
    np.testing.assert_allclose(
        tp.get("tp", start, start, 3600).get_data("tp", start),
        1.0 * (gi % 5),
        atol=1e-3,
    )

    ws = sources.Verif(str(out / "ws10m" / "analysis.nc"))
    assert ws.units == "m/s"
    np.testing.assert_allclose(
        ws.get("ws", start, start, 3600).get_data("ws", start),
        np.hypot(4.0, 4.0),
        atol=1e-3,
    )

    tcc = sources.Verif(str(out / "tcc" / "analysis.nc"))
    np.testing.assert_allclose(
        tcc.get("tcc", start, start, 3600).get_data("tcc", start), 0.6, atol=1e-3
    )
    assert len(tcc.file["time"]) == 3


def test_run_rejects_mismatched_grid(tmp_path, stores, grid):
    lat, lon = grid
    other = _make_zarr(
        tmp_path / "other.zarr",
        ["tcc"],
        lat + 1.0,
        lon,
        stores["dates"],
        np.zeros((len(stores["dates"]), 1, 1, len(lat)), np.float32),
    )
    config = OmegaConf.create(
        {
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-01-01T18:00:00",
            "dataset": {"join": [stores["main"], other]},
            "outputs": [_verif(tmp_path / "tcc.nc", "tcc")],
        }
    )
    with pytest.raises(ValueError, match="latitudes"):
        obs.run(config)


def test_main_cli(tmp_path, stores):
    cfg = tmp_path / "obs.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "dataset": {
                    "dataset": stores["main"],
                    "start": "2023-01-01T00:00:00",
                    "end": "2023-01-01T18:00:00",
                    "every_loc": 50,
                },
                "outputs": [_verif(tmp_path / "cli" / "t2m.nc", "2t", "degC")],
            }
        ),
        cfg,
    )
    assert obs.main(["--config", str(cfg)]) == 0
    src = sources.Verif(str(tmp_path / "cli" / "t2m.nc"))
    assert len(src.file["time"]) == 4  # period from the recipe's start/end
    assert len(src.locations) == stores["n"] // 50 + 1

    # top-level start_date/end_date apply when the recipe has none; -sd overrides them
    OmegaConf.save(
        OmegaConf.create(
            {
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-01-01T18:00:00",
                "dataset": {"dataset": stores["main"], "every_loc": 50},
                "outputs": [_verif(tmp_path / "cli2" / "t2m.nc", "2t", "degC")],
            }
        ),
        cfg,
    )
    assert obs.main(["--config", str(cfg), "-sd", "2023-01-01T12:00:00"]) == 0
    src = sources.Verif(str(tmp_path / "cli2" / "t2m.nc"))
    assert len(src.file["time"]) == 2
