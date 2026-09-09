"""Create Verif observation files from anemoi-datasets zarr stores.

    bris-obs --config obs.yaml

Extracts a set of grid points from an analysis dataset over a period and writes one
Verif file per variable (dimensions ``time`` and ``location``), which the ``verif``
observation source of the ``verif`` output can read. This makes it possible to verify
forecasts against analyses at a subset of points without reading the full dataset at
inference time.

Example config::

    start_date: 2023-01-01T00:00:00
    end_date: 2023-12-31T18:00:00
    output: /path/to/verification/{name}/analysis.nc   # {name} is replaced per output
    workers: 16                                        # parallel readers (1 = in-process)

    points:
      area: [40, -25, -40, 55]   # N, W, S, E (optional, whole grid if omitted)
      spacing: 2.5               # nearest grid point to each node of a regular lat/lon grid
      # every: 1000              # alternative: every n-th grid point (index stride)

    datasets:                    # zarr paths; the first defines grid, dates and altitudes (z)
      - /path/to/analysis.zarr
      - /path/to/analysis-extra-variables.zarr   # same grid and dates, e.g. cloud variables

    outputs:
      - {name: t2m, variable: 2t, units: degC}
      - {name: mslp, variable: msl, units: hPa}
      - {name: precip6h, variable: tp, units: mm}
      - {name: ws10m, variable: ws}             # derived from 10u/10v; anemoi units kept
      - {name: tcc, variable: tcc}

The zarr stores are read directly (one chunk decompression per timestep and chunk of
variables), which is orders of magnitude faster than going through
``anemoi.datasets.open_dataset`` with an ``area`` crop for point subsets. Only plain zarr
stores are supported for that reason, not open_dataset recipes.

Point selection: ``spacing`` picks, for each node of a regular latitude/longitude grid
over the area, the nearest dataset grid point (even coverage, every point is a real grid
point). ``every`` keeps every n-th point in storage order; on reduced Gaussian grids this
aliases with the row length and gives very uneven coverage, so ``spacing`` is preferred.
The ``location`` id is the index of the point in the (area-cropped) grid.
"""

import json
import logging
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from bris import units as bris_units
from bris.conventions import anemoi as anemoi_conventions
from bris.conventions import cf

LOGGER = logging.getLogger(__name__)

GRAVITY = 9.81  # same constant as the anemoidataset source uses for z -> altitude
DERIVED_VARIABLES = {"ws": ("10u", "10v")}

_STORES: dict = {}  # per-process zarr handles, keyed by path


def _open_zarr(path: str):
    import zarr

    return zarr.open(path, mode="r")


def _worker_init(paths: list[str]) -> None:
    global _STORES
    _STORES = {path: _open_zarr(path) for path in paths}


def _read_step(args) -> tuple[int, np.ndarray]:
    """Read one time index for all planned (dataset, variable) rows at the selected points.

    Args:
        args: (time index, plan, point index), where plan maps a zarr path to a list of
            (output row, variable index in that store) tuples.
    """
    t_index, plan, point_index = args
    n_rows = sum(len(rows) for rows in plan.values())
    out = np.full((n_rows, len(point_index)), np.nan, dtype=np.float32)
    for path, rows in plan.items():
        data = _STORES[path]["data"]
        # Variables are chunked in groups; decompress each chunk once for all rows in it
        vchunk = data.chunks[1]
        for group in sorted({v // vchunk for _, v in rows}):
            lo, hi = group * vchunk, min((group + 1) * vchunk, data.shape[1])
            block = data[t_index, lo:hi, 0, :]
            for row, v in rows:
                if lo <= v < hi:
                    out[row] = block[v - lo, point_index]
    return t_index, out


def _wrap_longitudes(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    return ((lon + 180.0) % 360.0) - 180.0


def _xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def select_points(
    lat: np.ndarray,
    lon: np.ndarray,
    area: list | tuple | None = None,
    spacing: float | None = None,
    every: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select observation points from a grid.

    Args:
        lat, lon: grid coordinates (degrees, any longitude convention)
        area: [N, W, S, E] bounding box in degrees (longitudes in -180..180), or None
        spacing: keep the nearest grid point to each node of a regular lat/lon grid with
            this spacing (degrees) over the area
        every: keep every n-th point of the (cropped) grid in storage order

    Returns:
        global_index: indices into the full grid
        location_id: indices into the cropped grid (used as Verif location ids)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon180 = _wrap_longitudes(lon)

    if area is not None:
        north, west, south, east = (float(x) for x in area)
        if north <= south:
            raise ValueError(f"area must be [N, W, S, E] with N > S, got {list(area)}")
        in_lat = (lat <= north) & (lat >= south)
        if west <= east:
            in_lon = (lon180 >= west) & (lon180 <= east)
        else:  # box crossing the date line
            in_lon = (lon180 >= west) | (lon180 <= east)
        cropped = np.flatnonzero(in_lat & in_lon)
    else:
        north, west, south, east = 90.0, -180.0, -90.0, 180.0
        cropped = np.arange(len(lat))

    if len(cropped) == 0:
        raise ValueError(f"no grid points inside area {list(area)}")

    if spacing is not None and every is not None:
        raise ValueError("points: give either 'spacing' or 'every', not both")

    if spacing is not None:
        from scipy.spatial import cKDTree

        spacing = float(spacing)
        if spacing <= 0:
            raise ValueError("points.spacing must be positive")
        target_lat = np.arange(south, north + 1e-9, spacing)
        if area is None:
            target_lon = np.arange(-180.0, 180.0, spacing)
        elif west <= east:
            target_lon = np.arange(west, east + 1e-9, spacing)
        else:
            target_lon = _wrap_longitudes(np.arange(west, east + 360.0 + 1e-9, spacing))
        tlat, tlon = np.meshgrid(target_lat, target_lon, indexing="ij")
        tree = cKDTree(_xyz(lat[cropped], lon180[cropped]))
        _, nearest = tree.query(_xyz(tlat.ravel(), tlon.ravel()))
        location_id = np.unique(nearest)
    elif every is not None:
        every = int(every)
        if every < 1:
            raise ValueError("points.every must be >= 1")
        if every > 1:
            LOGGER.warning(
                "points.every keeps every %d-th point in storage order; on reduced Gaussian "
                "grids this aliases with the row length and gives uneven coverage. "
                "Consider points.spacing instead.",
                every,
            )
        location_id = np.arange(0, len(cropped), every)
    else:
        location_id = np.arange(len(cropped))

    return cropped[location_id], location_id


def _unixtime(dates: np.ndarray) -> np.ndarray:
    dates = np.asarray(dates).astype("datetime64[s]")
    return (dates - np.datetime64("1970-01-01T00:00:00", "s")).astype(np.int64).astype(np.float64)


def run(config) -> list[Path]:
    """Create the observation files described by ``config``. Returns the written paths."""
    t0 = time.perf_counter()
    config = OmegaConf.to_container(config, resolve=True) if not isinstance(config, dict) else config

    # ---- datasets: plain zarr paths, the first one is the reference grid -------------------
    paths = []
    for entry in config["datasets"]:
        path = entry["dataset"] if isinstance(entry, dict) else entry
        if not isinstance(path, str):
            raise ValueError(
                "bris-obs reads zarr stores directly: 'datasets' entries must be paths, not open_dataset recipes"
            )
        paths.append(path)
    if not paths:
        raise ValueError("config needs at least one entry in 'datasets'")

    stores = {path: _open_zarr(path) for path in paths}
    names = {path: list(stores[path].attrs["variables"]) for path in paths}
    primary = stores[paths[0]]
    lat = primary["latitudes"][:]
    lon = primary["longitudes"][:]
    dates = primary["dates"][:].astype("datetime64[s]")
    for path in paths[1:]:
        for key, ref in (("latitudes", lat), ("longitudes", lon), ("dates", dates)):
            if not np.array_equal(stores[path][key][:].astype(ref.dtype), ref):
                raise ValueError(f"dataset {path}: '{key}' differs from the first dataset {paths[0]}")

    # ---- period ---------------------------------------------------------------------------
    start = np.datetime64(str(config["start_date"]), "s")
    end = np.datetime64(str(config["end_date"]), "s")
    t_indices = np.flatnonzero((dates >= start) & (dates <= end))
    if len(t_indices) == 0:
        raise ValueError(f"no dates in [{start}, {end}]; dataset covers {dates[0]} .. {dates[-1]}")
    LOGGER.info("%d timesteps: %s .. %s", len(t_indices), dates[t_indices[0]], dates[t_indices[-1]])

    # ---- points ---------------------------------------------------------------------------
    points_cfg = config.get("points") or {}
    global_index, location_id = select_points(
        lat, lon, points_cfg.get("area"), points_cfg.get("spacing"), points_cfg.get("every")
    )
    LOGGER.info("%d points selected (%s)", len(global_index), json.dumps(points_cfg))

    # ---- read plan: which (store, variable) rows to read ------------------------------------
    outputs = config["outputs"]
    if not outputs:
        raise ValueError("config needs at least one entry in 'outputs'")
    needed = []  # source variable names, in row order
    for out in outputs:
        for var in DERIVED_VARIABLES.get(out["variable"], (out["variable"],)):
            if var not in needed:
                needed.append(var)
    has_altitude = "z" in names[paths[0]]
    if has_altitude and "z" not in needed:
        needed.append("z")
    if not has_altitude:
        LOGGER.warning("Variable 'z' not in %s: altitudes set to 0", paths[0])

    plan: dict[str, list[tuple[int, int]]] = {}
    row_of = {}
    for row, var in enumerate(needed):
        for path in paths:
            if var in names[path]:
                plan.setdefault(path, []).append((row, names[path].index(var)))
                row_of[var] = row
                break
        else:
            raise ValueError(f"variable '{var}' not found in any dataset")

    # ---- read ----------------------------------------------------------------------------------
    jobs = [(int(t), plan, global_index) for t in t_indices]
    values = np.full((len(t_indices), len(needed), len(global_index)), np.nan, dtype=np.float32)
    position = {int(t): k for k, t in enumerate(t_indices)}
    workers = int(config.get("workers", 1))
    if workers > 1:
        with Pool(workers, initializer=_worker_init, initargs=(paths,)) as pool:
            results = pool.imap_unordered(_read_step, jobs, chunksize=4)
            for k, (t_index, out) in enumerate(results):
                values[position[t_index]] = out
                if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                    LOGGER.info("read %d/%d timesteps (%.0f s)", k + 1, len(jobs), time.perf_counter() - t0)
    else:
        _worker_init(paths)
        for k, job in enumerate(jobs):
            t_index, out = _read_step(job)
            values[position[t_index]] = out
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                LOGGER.info("read %d/%d timesteps (%.0f s)", k + 1, len(jobs), time.perf_counter() - t0)

    altitude = values[0, row_of["z"]] / GRAVITY if has_altitude else np.zeros(len(global_index), np.float32)
    unixtime = _unixtime(dates[t_indices])
    lon180 = _wrap_longitudes(lon[global_index])

    # ---- write one Verif file per output ---------------------------------------------------------
    import xarray as xr

    provenance = {
        "datasets": paths,
        "points": points_cfg,
        "start_date": str(dates[t_indices[0]]),
        "end_date": str(dates[t_indices[-1]]),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    pattern = config["output"]
    if "{name}" not in pattern and len(outputs) > 1:
        raise ValueError("'output' must contain '{name}' when there is more than one output")

    written = []
    for out in outputs:
        variable = out["variable"]
        name = out.get("name", variable)
        if variable in DERIVED_VARIABLES:
            u, v = (values[:, row_of[c]] for c in DERIVED_VARIABLES[variable])
            obs = np.sqrt(u**2 + v**2)
        else:
            obs = values[:, row_of[variable]]

        from_units = anemoi_conventions.get_units(variable)
        units = out.get("units", from_units)
        if units is not None and from_units is not None and units != from_units:
            obs, units = bris_units.convert(obs, from_units, units)
        elif units is not None and from_units is None:
            LOGGER.warning("%s: anemoi units unknown, writing values unchanged with units '%s'", variable, units)

        cfname = cf.get_metadata(variable)["cfname"]
        ds = xr.Dataset(
            {
                "obs": (("time", "location"), np.asarray(obs, dtype=np.float32)),
                "lat": (("location",), lat[global_index].astype(np.float32)),
                "lon": (("location",), lon180.astype(np.float32)),
                "altitude": (("location",), np.asarray(altitude, dtype=np.float32)),
            },
            coords={"time": unixtime, "location": location_id.astype(np.int32)},
            attrs={
                "long_name": cfname,
                "standard_name": cfname,
                "Conventions": "verif_1.0.0",
                "anemoi_variable": variable,
                "provenance": json.dumps(provenance, default=str),
                **({"units": units} if units is not None else {}),
            },
        )
        ds["time"].attrs["units"] = "seconds since 1970-01-01 00:00:00 +00:00"

        filename = Path(out.get("filename") or pattern.format(name=name))
        filename.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(filename, encoding={"obs": {"zlib": True, "complevel": 4}})
        LOGGER.info(
            "%-10s %s  range %.3g .. %.3g %s  nan=%d",
            name, filename, np.nanmin(obs), np.nanmax(obs), units or "", int(np.isnan(obs).sum()),
        )
        written.append(filename)

    LOGGER.info("done in %.0f s", time.perf_counter() - t0)
    return written


def parse_args(arg_list: list[str] | None) -> dict:
    parser = ArgumentParser(description="Create Verif observation files from anemoi-datasets zarr stores")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-sd", type=str, dest="start_date", required=False)
    parser.add_argument("-ed", type=str, dest="end_date", required=False)
    parser.add_argument("-w", type=int, dest="workers", required=False)
    args, _ = parser.parse_known_args(arg_list)
    return {k: v for k, v in args.__dict__.items() if v is not None}


def main(arg_list: list[str] | None = None) -> int:
    args = parse_args(arg_list)
    logging.basicConfig(
        level=logging.DEBUG if args.pop("debug", False) else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    config = OmegaConf.load(args.pop("config"))
    config = OmegaConf.merge(config, OmegaConf.create(args))
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
