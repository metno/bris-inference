#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from glob import glob
from importlib.resources import files
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import dask
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer

MEPS_VARIABLES = [
    "t2m",
    "tp",
    "lcc",
    "lsm",
    "mcc",
    "msl",
    "z",
    "u10",
    "v10",
    "cape",
    "cin",
    "d2m",
    "wind_gust_10m",
]
NETATMO_VARIABLES = ["ff", "fg", "dd", "pr", "ta", "rr", "rr_hourly", "uu"]


def _netatmo_station_template() -> xr.Dataset:
    station_file = files("bris").joinpath("schema/unique_lonlat_2020_2025.txt")
    locations = pd.read_csv(station_file, sep="\t")
    locations = locations[
        (locations.longitude < 40.73)
        & (locations.longitude > -8.08)
        & (locations.latitude < 73.06)
        & (locations.latitude > 53.14)
    ].copy()
    longitude = np.asarray(locations["longitude"].values, dtype=np.float64)
    latitude = np.asarray(locations["latitude"].values, dtype=np.float64)
    longitude = np.where(longitude == 0.0, 0.0, longitude)
    latitude = np.where(latitude == 0.0, 0.0, latitude)
    locations["id"] = np.char.add(
        np.char.mod("%.4f", longitude),
        np.char.add("_", np.char.mod("%.4f", latitude)),
    )
    return xr.Dataset.from_dataframe(locations.set_index("id")).set_coords(
        ["longitude", "latitude"]
    )


def expand_to_full_netatmo_grid(data: xr.Dataset) -> xr.Dataset:
    template = _netatmo_station_template()
    if "time" not in data.dims:
        raise ValueError("Netatmo dataset is missing time dimension.")
    spatial_dims = [dimension for dimension in data.dims if dimension != "time"]
    if not spatial_dims:
        raise ValueError("Netatmo dataset is missing a spatial dimension.")
    spatial_dim = spatial_dims[0]
    if "longitude" not in data or "latitude" not in data:
        raise ValueError("Netatmo dataset is missing longitude/latitude coordinates.")

    longitude = np.round(np.asarray(data["longitude"].values, dtype=np.float64), 4)
    latitude = np.round(np.asarray(data["latitude"].values, dtype=np.float64), 4)
    longitude = np.where(longitude == 0.0, 0.0, longitude)
    latitude = np.where(latitude == 0.0, 0.0, latitude)
    location_ids = np.char.add(
        np.char.mod("%.4f", longitude),
        np.char.add("_", np.char.mod("%.4f", latitude)),
    )
    data = data.assign_coords(id=(spatial_dim, location_ids))
    if spatial_dim != "id":
        data = data.swap_dims({spatial_dim: "id"})
    keep = ~pd.Index(data["id"].values).duplicated(keep="first")
    if not bool(keep.all()):
        data = data.isel(id=np.flatnonzero(keep))
    data = data.reindex(id=template["id"].values)
    return data.assign_coords(
        longitude=("id", template["longitude"].values),
        latitude=("id", template["latitude"].values),
    )


def _times(start: str, end: str, frequency: str) -> pd.DatetimeIndex:
    pandas_frequency = f"{frequency[:-1]}min" if frequency.endswith("m") else frequency
    values = pd.date_range(start, end, freq=pandas_frequency)
    if values.tz is not None:
        values = values.tz_convert(None)
    return values


def _normalise_time(dataset: xr.Dataset) -> xr.Dataset:
    if "time" not in dataset:
        raise ValueError("Retrieved dataset has no time coordinate.")
    if np.issubdtype(dataset["time"].dtype, np.integer):
        dataset["time"] = pd.to_datetime(dataset["time"].values, unit="s", utc=True).tz_convert(None)
    else:
        dataset["time"] = pd.to_datetime(dataset["time"].values).tz_localize(None)
    dataset["time"].attrs.update(standard_name="time", long_name="time", axis="T")
    return dataset.sortby("time")


def _require_exact_times(dataset: xr.Dataset, requested: pd.DatetimeIndex, label: str) -> xr.Dataset:
    available = pd.DatetimeIndex(pd.to_datetime(dataset["time"].values)).tz_localize(None)
    missing = requested.difference(available)
    if len(missing):
        raise ValueError(f"{label} is missing exact times: {', '.join(str(value) for value in missing)}")
    return dataset.sel(time=requested)


def _write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for variable in dataset.variables:
        dataset[variable].encoding.clear()
    with dask.config.set(scheduler="synchronous"):
        dataset.to_netcdf(path, engine="netcdf4")


def _write_recipe(
    path: Path,
    source_netcdf: Path,
    variables: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    cell_chunk: int,
) -> None:
    recipe = {
        "description": "Operational BRIS multisource nowcast input from Weathermart",
        "dates": {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frequency": frequency,
        },
        "input": {
            "xarray": {
                "url": str(source_netcdf),
                "param": variables,
            }
        },
        "build": {
            "group_by": "daily",
            "variable_naming": "param",
            "allow_nans": True,
        },
        "output": {
            "chunking": {
                "dates": max(1, len(_times(str(start), str(end), frequency))),
                "values": cell_chunk,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))


def prepare_radar(args: argparse.Namespace) -> None:
    from weathermart.retrievers.radar import NordicRadarRetriever, align_to_template
    from weathermart.utils import assign_latlon_coords

    requested = _times(args.start, args.end, "5min")
    retriever = NordicRadarRetriever()
    if args.endpoint == "prod":
        retriever.PROD_ROOT = str(args.root)
    else:
        retriever.ARCHIVE_ROOT = str(args.root)

    files: list[str] = []
    for timestamp in requested:
        patterns, _ = retriever._get_file_patterns(timestamp, args.endpoint)
        timestamp_text = timestamp.strftime("%Y%m%dT%H%M%SZ")
        for pattern in patterns:
            matches = glob(str(pattern))
            if args.endpoint == "prod":
                matches = [path for path in matches if timestamp_text in Path(path).name]
            files.extend(matches)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No raw Nordic radar files found in {args.root}.")

    dataset = retriever._open_files(files)
    dataset = _normalise_time(dataset)
    dataset = _require_exact_times(dataset, requested, "Raw Nordic radar")
    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        if "lat" in dataset and "lon" in dataset:
            dataset = dataset.set_coords(["lat", "lon"])
        else:
            projection = retriever.crs["laea" if any("laea" in path for path in files) else "lcc"]
            dataset = assign_latlon_coords(dataset, crs=projection)
    dataset = align_to_template(dataset)
    if "lwe_precipitation_rate" not in dataset:
        raise KeyError("Raw Nordic radar does not contain lwe_precipitation_rate.")
    dataset = dataset[["lwe_precipitation_rate"]]
    dataset["lwe_precipitation_rate"] = dataset["lwe_precipitation_rate"].where(
        dataset["lwe_precipitation_rate"] < 1.0e6
    )
    dataset.attrs.update(source="Weathermart NORDIC_RADAR", qc_filtered="false")
    _write_netcdf(dataset, args.netcdf)
    _write_recipe(
        args.recipe,
        args.netcdf,
        ["lwe_precipitation_rate"],
        requested[0],
        requested[-1],
        "5m",
        args.cell_chunk,
    )


def _select_meps_zarr_valid_time(root: Path, valid_time: pd.Timestamp) -> xr.Dataset:
    paths = []
    first_day = (valid_time - pd.Timedelta(hours=6)).normalize()
    for day in pd.date_range(first_day, valid_time.normalize(), freq="D"):
        path = root / day.strftime("%Y%m%d")
        if path.is_dir():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No nowcast_meps_fcst daily Zarr found in {root} for {valid_time}.")

    datasets = []
    target_y = None
    target_x = None
    for path in paths:
        daily = xr.open_zarr(path, consolidated=False)
        missing = sorted(set(MEPS_VARIABLES).difference(daily.data_vars))
        if missing:
            raise KeyError(f"{path} is missing MEPS variable(s): {missing}")
        keep = [*MEPS_VARIABLES, "latitude", "longitude"]
        daily = daily[[name for name in keep if name in daily]]
        if target_y is None:
            target_y = daily["y"]
            target_x = daily["x"]
        elif daily.sizes["y"] != target_y.size or daily.sizes["x"] != target_x.size:
            daily = daily.sel(y=target_y, x=target_x, method="nearest")
        else:
            daily = daily.assign_coords(y=target_y.data, x=target_x.data)
        datasets.append(daily)

    archive = xr.concat(
        datasets,
        dim="forecast_reference_time",
        coords="minimal",
        compat="override",
    ).sortby("forecast_reference_time")
    reference_times = pd.DatetimeIndex(pd.to_datetime(archive.forecast_reference_time.values))
    lead_times = pd.TimedeltaIndex(pd.to_timedelta(archive.lead_time.values))

    selected = {}
    for variable in MEPS_VARIABLES:
        values = archive[variable]
        require_positive_lead = variable == "tp"
        if require_positive_lead:
            values = values.diff("lead_time", label="upper").clip(min=0)
            variable_leads = pd.TimedeltaIndex(pd.to_timedelta(values.lead_time.values))
        else:
            variable_leads = lead_times
        candidates = []
        for reference_index, reference_time in enumerate(reference_times):
            lead = valid_time - reference_time
            if require_positive_lead and lead <= pd.Timedelta(0):
                continue
            matches = np.flatnonzero(variable_leads == lead)
            if matches.size:
                candidates.append((reference_time, reference_index, int(matches[0])))
        if not candidates:
            raise KeyError(f"No latest MEPS forecast candidate for {variable} at {valid_time}.")
        _, reference_index, lead_index = max(candidates, key=lambda item: item[0])
        field = values.isel(
            forecast_reference_time=reference_index,
            lead_time=lead_index,
            drop=True,
        )
        selected[variable] = xr.DataArray(
            field.data,
            dims=field.dims,
            coords={dim: field.coords[dim] for dim in field.dims},
            attrs=field.attrs,
        )

    output = xr.Dataset(selected).expand_dims(time=[valid_time.to_datetime64()])
    for coordinate_name in ("latitude", "longitude"):
        if coordinate_name in archive:
            coordinate = archive[coordinate_name]
            for dimension in tuple(coordinate.dims):
                if dimension not in {"y", "x"}:
                    coordinate = coordinate.isel({dimension: 0}, drop=True)
            output = output.assign_coords(
                {coordinate_name: (("y", "x"), coordinate.transpose("y", "x").data)}
            )
    output.attrs.update(
        source="Weathermart nowcast_meps_fcst daily Zarr",
        archive_root=str(root),
        selection="latest forecast cycle per valid time; tp is a deaccumulated one-hour interval",
    )
    return output


def prepare_meps(args: argparse.Namespace) -> None:
    from weathermart.retrievers.meps_netcdf import MEPSNetcdfRetriever

    valid_time = pd.Timestamp(args.valid_time).tz_localize(None).floor("h")
    requested = pd.date_range(valid_time - pd.Timedelta(hours=1), valid_time, freq="1h")
    if args.backend == "zarr":
        dataset = xr.concat(
            [_select_meps_zarr_valid_time(args.root, timestamp) for timestamp in requested],
            dim="time",
            coords="minimal",
            compat="override",
        )
    else:
        retriever = MEPSNetcdfRetriever(archive_root=args.root)
        dataset = retriever.retrieve(
            source="MEPS_FCST",
            variables=MEPS_VARIABLES,
            dates=list(requested),
            archive_root=args.root,
            lead_hours="0,1,2,3,4,5,6",
            continuous_hourly=True,
        )
    dataset = _normalise_time(dataset)
    dataset = _require_exact_times(dataset, requested, "MEPS")
    dataset.attrs.update(selection="latest forecast available for each requested valid time")
    retrieval_coordinates = {}
    for coordinate_name in ("forecast_reference_time", "lead_time"):
        if coordinate_name in dataset.coords and coordinate_name not in dataset.dims:
            retrieval_coordinates[coordinate_name] = str(dataset[coordinate_name].values[0])
            dataset = dataset.drop_vars(coordinate_name)
    dataset.attrs.update(
        {
            f"selected_{name}": value
            for name, value in retrieval_coordinates.items()
        }
    )
    _write_netcdf(dataset, args.netcdf)
    _write_recipe(
        args.recipe,
        args.netcdf,
        MEPS_VARIABLES,
        requested[0],
        requested[-1],
        "1h",
        args.cell_chunk,
    )


def prepare_netatmo(args: argparse.Namespace) -> None:
    requested = _times(args.start, args.end, "5min")
    datasets = []
    if args.backend in {"auto", "zarr"}:
        for day in sorted(set(requested.normalize())):
            path = args.root / day.strftime("%Y%m%d")
            if path.is_dir():
                datasets.append(xr.open_zarr(path, consolidated=False)[NETATMO_VARIABLES])
    if datasets:
        dataset = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time").sortby("time")
    elif args.backend in {"auto", "production"}:
        import weathermart.retrievers.netatmo as netatmo_module
        from weathermart.retrievers.netatmo import netatmo_to_xarray_parallel

        os.environ["NETATMO_ROOT"] = str(args.root)
        netatmo_module.ROOT = str(args.root)
        netatmo_module.CHUNK_SIZE = 1
        netatmo_module.MAX_WORKERS = min(len(requested), max(1, os.cpu_count() or 4))
        dataset = netatmo_to_xarray_parallel(requested.tz_localize("UTC"), NETATMO_VARIABLES)
        if dataset is None or not dataset.data_vars:
            raise FileNotFoundError(f"Weathermart returned no production Netatmo data from {args.root}.")
        dataset = expand_to_full_netatmo_grid(dataset)
    else:
        raise FileNotFoundError(f"No Weathermart Netatmo daily Zarr found in {args.root}.")
    dataset = _normalise_time(dataset)
    dataset = _require_exact_times(dataset, requested, "Netatmo")
    dataset = expand_to_full_netatmo_grid(dataset)
    dataset.attrs.update(source="Weathermart NETATMO daily Zarr")
    _write_netcdf(dataset, args.netcdf)
    _write_recipe(
        args.recipe,
        args.netcdf,
        NETATMO_VARIABLES,
        requested[0],
        requested[-1],
        "5m",
        args.cell_chunk,
    )


def prepare_lightning(args: argparse.Namespace) -> None:
    from weathermart.retrievers.frost import (
        FrostRetriever,
        _centers_to_edges,
        _open_template_grid,
        _parse_lightning_ualf,
        _wkt_polygon_from_bounds,
    )

    requested = _times(args.start, args.end, "5min")
    template, x_name, y_name, x, y, template_crs = _open_template_grid(
        args.template,
        args.template_crs,
    )
    longitude = template["longitude"] if "longitude" in template else template["lon"]
    latitude = template["latitude"] if "latitude" in template else template["lat"]
    geometry = _wkt_polygon_from_bounds(
        (float(np.nanmin(longitude.values)), float(np.nanmax(longitude.values))),
        (float(np.nanmin(latitude.values)), float(np.nanmax(latitude.values))),
    )
    client_id, client_secret = FrostRetriever._load_credentials(str(args.credentials))
    transformer = Transformer.from_crs("EPSG:4326", template_crs, always_xy=True)
    x_edges = _centers_to_edges(x)
    y_edges = _centers_to_edges(y)
    counts = np.zeros((len(requested), len(y), len(x)), dtype=np.float32)

    for day in sorted(set(requested.normalize())):
        day_start = day.tz_localize("UTC")
        day_stop = day_start + pd.Timedelta(days=1)
        response = FrostRetriever.request_from_frost(
            endpoint="lightning",
            client_id=client_id,
            client_secret=client_secret,
            args={
                "referencetime": f"{day_start:%Y-%m-%dT%H:%M:%S}/{day_stop:%Y-%m-%dT%H:%M:%S}",
                "geometry": geometry,
            },
            fmt="ualf",
        )
        observations = _parse_lightning_ualf(response.text)
        if observations.empty:
            continue
        buckets = observations["time"].dt.floor("5min")
        time_indices = requested.get_indexer(buckets)
        projected_x, projected_y = transformer.transform(
            observations["lon"].to_numpy(dtype=np.float64),
            observations["lat"].to_numpy(dtype=np.float64),
        )
        x_indices = np.digitize(projected_x, x_edges) - 1
        y_indices = np.digitize(projected_y, y_edges) - 1
        valid = (
            (time_indices >= 0)
            & (x_indices >= 0)
            & (x_indices < len(x))
            & (y_indices >= 0)
            & (y_indices < len(y))
        )
        np.add.at(
            counts,
            (time_indices[valid], y_indices[valid], x_indices[valid]),
            1.0,
        )

    dataset = xr.Dataset(
        {"thunder_count": (("time", y_name, x_name), counts)},
        coords={
            "time": requested,
            y_name: y,
            x_name: x,
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "source": "Weathermart LIGHTNING via Frost",
            "temporal_resolution": "5min",
            "crs": template_crs,
        },
    )
    dataset["thunder_count"].attrs.update(
        long_name="Observed lightning count in five-minute interval",
        units="1",
    )
    _write_netcdf(dataset, args.netcdf)
    _write_recipe(
        args.recipe,
        args.netcdf,
        ["thunder_count"],
        requested[0],
        requested[-1],
        "5m",
        args.cell_chunk,
    )


def _add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--netcdf", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--cell-chunk", type=int, default=100000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare BRIS multisource nowcast inputs with Weathermart.")
    commands = parser.add_subparsers(required=True)

    radar = commands.add_parser("radar")
    radar.add_argument("--start", required=True)
    radar.add_argument("--end", required=True)
    radar.add_argument("--root", required=True, type=Path)
    radar.add_argument("--endpoint", choices=("prod", "archive"), default="prod")
    _add_common_output_args(radar)
    radar.set_defaults(func=prepare_radar)

    meps = commands.add_parser("meps")
    meps.add_argument("--valid-time", required=True)
    meps.add_argument("--root", required=True, type=Path)
    meps.add_argument("--backend", choices=("zarr", "netcdf"), default="zarr")
    _add_common_output_args(meps)
    meps.set_defaults(func=prepare_meps)

    netatmo = commands.add_parser("netatmo")
    netatmo.add_argument("--start", required=True)
    netatmo.add_argument("--end", required=True)
    netatmo.add_argument("--root", required=True, type=Path)
    netatmo.add_argument("--backend", choices=("auto", "zarr", "production"), default="auto")
    _add_common_output_args(netatmo)
    netatmo.set_defaults(func=prepare_netatmo)

    lightning = commands.add_parser("lightning")
    lightning.add_argument("--start", required=True)
    lightning.add_argument("--end", required=True)
    lightning.add_argument("--template", required=True, type=Path)
    lightning.add_argument("--template-crs", required=True)
    lightning.add_argument("--credentials", required=True, type=Path)
    _add_common_output_args(lightning)
    lightning.set_defaults(func=prepare_lightning)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
