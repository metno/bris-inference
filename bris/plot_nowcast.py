from __future__ import annotations

import argparse
import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_stdlib_inspect_path = Path(sysconfig.get_paths()["stdlib"]) / "inspect.py"
_stdlib_inspect_spec = importlib.util.spec_from_file_location("inspect", _stdlib_inspect_path)
if _stdlib_inspect_spec is None or _stdlib_inspect_spec.loader is None:
    raise ImportError(f"Unable to load stdlib inspect from {_stdlib_inspect_path}")
_stdlib_inspect = importlib.util.module_from_spec(_stdlib_inspect_spec)
_stdlib_inspect_spec.loader.exec_module(_stdlib_inspect)
sys.modules["inspect"] = _stdlib_inspect

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.tri import Triangulation

NORWAY_TZ = ZoneInfo("Europe/Oslo")
NORWAY_MAP = ccrs.Stereographic(central_longitude=18.5, central_latitude=63.5)
GEO_MAX_POINTS = 500_000
ENSEMBLE_PROBABILITY_THRESHOLD = 0.1

PRECIP_COLORS = [
    "#ffffff",
    "#04e9e7",
    "#019ff4",
    "#0300f4",
    "#02fd02",
    "#01c501",
    "#008e00",
    "#fdf802",
    "#e5bc00",
    "#fd9500",
    "#fd0000",
    "#d40000",
    "#bc0000",
    "#f800fd",
]
PRECIP_LEVELS = [0, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 100]
PRECIP_TICK_LABELS = ["0", "0.05", "0.1", "0.25", "0.5", "1", "1.5", "2", "3", "4", "5", "6", "7"]

# Discrete bins for wet-member count / probability panel.
WET_MEMBER_ZERO_COLOR = (1.0, 1.0, 1.0, 1.0)
WET_MEMBER_TAB10 = plt.get_cmap("tab10")


def _resolve_input(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        nc_files = sorted(candidate.glob("nordic_radar_*.nc"))
        if not nc_files:
            nc_files = sorted(candidate.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in {candidate}")
        return nc_files[-1]
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _pick_variable(ds: xr.Dataset, requested: str | None) -> str:
    if requested:
        if requested not in ds.data_vars:
            raise KeyError(
                f"Variable {requested!r} not found. Available: {', '.join(ds.data_vars)}"
            )
        return requested

    preferred = [
        "lwe_precipitation_rate",
        "precipitation_amount",
        "tp",
    ]
    for name in preferred:
        if name in ds.data_vars:
            return name

    for name in ds.data_vars:
        if name not in {"forecast_reference_time", "projection"}:
            return name

    raise ValueError("No plottable data variables found")


def _to_datetimes(time_values: np.ndarray) -> pd.DatetimeIndex:
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values, utc=True)
    return pd.to_datetime(values, unit="s", utc=True)


def _format_local_time(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(NORWAY_TZ).strftime("%Y-%m-%d %H:%M %Z")


def _prepare_dataarray(ds: xr.Dataset, variable: str, ensemble_mode: str) -> xr.DataArray:
    da = ds[variable]
    if "ensemble_member" in da.dims:
        if ensemble_mode == "mean":
            da = da.mean("ensemble_member")
        elif ensemble_mode == "median":
            da = da.median("ensemble_member")
        else:
            da = da.isel(ensemble_member=int(ensemble_mode))

    if "time" not in da.dims:
        raise ValueError(f"Variable {variable!r} does not have a time dimension")
    return da


def _frame_limits_many(dataarrays: list[xr.DataArray], samples: int = 5) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for da in dataarrays:
        sample = da.isel(time=slice(0, min(samples, da.sizes["time"]))).load()
        arr = np.asarray(sample.values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return 0.0, 1.0
    sample_values = np.concatenate(values)
    vmin = float(np.nanpercentile(sample_values, 1))
    vmax = float(np.nanpercentile(sample_values, 99))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(sample_values))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(sample_values))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _frame_limits(da: xr.DataArray, samples: int = 5) -> tuple[float, float]:
    sample = da.isel(time=slice(0, min(samples, da.sizes["time"]))).load()
    values = np.asarray(sample.values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(values, 1))
    vmax = float(np.nanpercentile(values, 99))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(values))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _is_precip_variable(name: str) -> bool:
    needle = name.lower()
    return "precip" in needle or needle in {"tp", "rr", "rain", "lwe_precipitation_rate"}


def _coerce_frame(da: xr.DataArray, time_index: int) -> np.ndarray:
    frame = da.isel(time=time_index).values
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame after selection, got shape {frame.shape}")
    return frame


def _ensemble_member_count(da: xr.DataArray) -> int:
    return int(da.sizes.get("ensemble_member", 1))


def _coerce_wet_member_count_frame(
    da: xr.DataArray, time_index: int, threshold: float
) -> np.ndarray:
    frame = da.isel(time=time_index)
    values = np.asarray(frame.values, dtype=np.float32)
    if "ensemble_member" in frame.dims:
        ensemble_axis = frame.get_axis_num("ensemble_member")
        values = np.moveaxis(values, ensemble_axis, 0)
        exceed = np.where(np.isfinite(values), values > threshold, False)
        count = exceed.sum(axis=0, dtype=np.int16)
    else:
        count = np.where(np.isfinite(values), values > threshold, False).astype(np.int16)
    if count.ndim != 2:
        raise ValueError(f"Expected a 2D wet-member-count frame, got shape {count.shape}")
    return count


def _wet_member_count_metadata(
    da: xr.DataArray,
) -> tuple[np.ndarray, ListedColormap, BoundaryNorm, list[int], list[str]]:
    n_members = max(1, _ensemble_member_count(da))
    levels = np.arange(-0.5, n_members + 1.5, 1.0)
    tab10_samples = [WET_MEMBER_TAB10(i % WET_MEMBER_TAB10.N) for i in range(n_members)]
    colors: list[Any] = [WET_MEMBER_ZERO_COLOR, *tab10_samples]
    cmap = ListedColormap(colors, name=f"wet_member_count_{n_members}")
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    ticks = list(range(1, n_members + 1))
    ticklabels = [f"{k}/{n_members}" for k in ticks]
    return levels, cmap, norm, ticks, ticklabels


def _downsample_indices(size: int, max_points: int) -> np.ndarray:
    if size <= max_points:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, max_points, dtype=np.int64)


def _infer_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"Expected exactly 2 spatial dimensions, found {spatial_dims!r} in {da.dims!r}"
        )
    return spatial_dims[0], spatial_dims[1]


def _resolve_geo_coords(da: xr.DataArray) -> tuple[np.ndarray | None, np.ndarray | None]:
    lon = None
    lat = None
    for lon_name in ("longitude", "lon"):
        if lon_name in da.coords:
            lon = np.asarray(da.coords[lon_name].values)
            break
    for lat_name in ("latitude", "lat"):
        if lat_name in da.coords:
            lat = np.asarray(da.coords[lat_name].values)
            break
    return lon, lat


def _project_lonlat(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = NORWAY_MAP.transform_points(ccrs.PlateCarree(), lon, lat)
    return points[..., 0].astype(np.float32, copy=False), points[..., 1].astype(np.float32, copy=False)


def _add_geo_features(ax: plt.Axes, *, white_background: bool = False) -> None:
    if white_background:
        ax.set_facecolor("white")
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)
        coastline_color = "#c7d2d7"
        border_color = "#d7e0e3"
    else:
        ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#efefef", zorder=0)
        coastline_color = "#b7c3c7"
        border_color = "#c8d2d5"

    ax.coastlines(resolution="50m", linewidth=0.45, color=coastline_color, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.35, edgecolor=border_color, zorder=2)


def _build_geo_triangulation(lon: np.ndarray, lat: np.ndarray) -> Triangulation:
    x, y = _project_lonlat(lon, lat)
    return Triangulation(x.ravel(), y.ravel())


def _triangulation_extent(triangulation: Triangulation) -> tuple[float, float, float, float]:
    x = np.asarray(triangulation.x, dtype=np.float64)
    y = np.asarray(triangulation.y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError("Unable to infer map extent from projected coordinates")
    return (
        float(np.nanmin(x[finite])),
        float(np.nanmax(x[finite])),
        float(np.nanmin(y[finite])),
        float(np.nanmax(y[finite])),
    )


def _render_frame(
    ax: plt.Axes,
    da: xr.DataArray,
    time_index: int,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str | ListedColormap,
    *,
    norm: BoundaryNorm | None = None,
    geo_triangulation: Triangulation | None = None,
    geo_extent: tuple[float, float, float, float] | None = None,
    geo_point_indices: np.ndarray | None = None,
) -> plt.Artist:
    frame = _coerce_frame(da, time_index)
    ydim, xdim = _infer_spatial_dims(da)

    lon, lat = _resolve_geo_coords(da)
    if geo_triangulation is not None and lon is not None and lat is not None and lon.shape == frame.shape and lat.shape == frame.shape:
        color_kwargs: dict[str, Any] = {"cmap": cmap, "norm": norm, "shading": "gouraud", "rasterized": True}
        if norm is None:
            color_kwargs.update({"vmin": vmin, "vmax": vmax})
        values = frame.ravel()
        if geo_point_indices is not None:
            values = values[geo_point_indices]
        image = ax.tripcolor(geo_triangulation, values, **color_kwargs)
        if geo_extent is not None:
            ax.set_xlim(geo_extent[0], geo_extent[1])
            ax.set_ylim(geo_extent[2], geo_extent[3])
        _add_geo_features(ax)
        ax.set_aspect("equal", adjustable="box")
    else:
        xcoord = da.coords.get(xdim)
        ycoord = da.coords.get(ydim)
        image_kwargs: dict[str, Any] = {"cmap": cmap, "norm": norm, "aspect": "auto"}
        if norm is None:
            image_kwargs.update({"vmin": vmin, "vmax": vmax})
        if xcoord is not None and ycoord is not None and xcoord.ndim == 1 and ycoord.ndim == 1:
            extent = (
                float(np.nanmin(xcoord.values)),
                float(np.nanmax(xcoord.values)),
                float(np.nanmin(ycoord.values)),
                float(np.nanmax(ycoord.values)),
            )
            image = ax.imshow(frame, origin="lower", extent=extent, **image_kwargs)
            ax.set_xlabel(xdim)
            ax.set_ylabel(ydim)
        else:
            image = ax.imshow(frame, origin="lower", **image_kwargs)
            ax.set_xlabel(xdim)
            ax.set_ylabel(ydim)

    ax.set_title(title)
    return image


def _render_wet_member_count_frame(
    ax: plt.Axes,
    da: xr.DataArray,
    time_index: int,
    title: str,
    *,
    threshold: float,
    levels: np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    geo_triangulation: Triangulation | None = None,
    geo_extent: tuple[float, float, float, float] | None = None,
    geo_point_indices: np.ndarray | None = None,
) -> plt.Artist:
    frame_da = da.isel(time=time_index)
    frame = _coerce_wet_member_count_frame(da, time_index, threshold).astype(np.float32)
    spatial_dims = [dim for dim in frame_da.dims if dim != "ensemble_member"]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"Expected exactly 2 spatial dimensions for wet-member-count frame, "
            f"found {spatial_dims!r} in {frame_da.dims!r}"
        )
    ydim, xdim = spatial_dims

    lon, lat = _resolve_geo_coords(da)
    if geo_triangulation is not None and lon is not None and lat is not None and lon.shape == frame.shape and lat.shape == frame.shape:
        values = frame.ravel()[geo_point_indices] if geo_point_indices is not None else frame.ravel()
        image = ax.tricontourf(
            geo_triangulation,
            values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="neither",
            antialiased=True,
        )
        if geo_extent is not None:
            ax.set_xlim(geo_extent[0], geo_extent[1])
            ax.set_ylim(geo_extent[2], geo_extent[3])
        _add_geo_features(ax, white_background=True)
        ax.set_aspect("equal", adjustable="box")
    else:
        xcoord = da.coords.get(xdim)
        ycoord = da.coords.get(ydim)
        if xcoord is not None and ycoord is not None and xcoord.ndim == 1 and ycoord.ndim == 1:
            image = ax.contourf(
                np.asarray(xcoord.values),
                np.asarray(ycoord.values),
                np.ma.masked_less_equal(frame, 0.0),
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="neither",
                antialiased=True,
            )
            ax.set_xlabel(xdim)
            ax.set_ylabel(ydim)
        else:
            image = ax.contourf(
                np.ma.masked_less_equal(frame, 0.0),
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="neither",
                antialiased=True,
            )
            ax.set_xlabel(xdim)
            ax.set_ylabel(ydim)

    ax.set_title(title)
    return image


def _clear_contour_set(contour_set: Any) -> None:
    remove = getattr(contour_set, "remove", None)
    if callable(remove):
        remove()
        return
    collections = getattr(contour_set, "collections", None)
    if collections is not None:
        for collection in list(collections):
            collection.remove()


def _update_artist(artist: plt.Artist, frame: np.ndarray, *, flattened: bool = False) -> None:
    if hasattr(artist, "set_array"):
        artist.set_array(np.asarray(frame).ravel() if flattened else np.asarray(frame))
        return
    artist.set_data(frame)


def _add_fixed_horizontal_colorbar(
    fig: plt.Figure,
    mappable: Any,
    axes: list[plt.Axes] | np.ndarray,
    *,
    ticks: list[float] | None,
    label: str,
    geo_mode: bool,
) -> Any:
    axes_arr = np.atleast_1d(axes)
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes_arr]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    bottom = min(box.y0 for box in boxes)
    span = right - left
    if geo_mode:
        width = min(0.34, span * 0.46)
        height = 0.014
        cbar_bottom = max(0.030, bottom - 0.040)
    else:
        width = span * 0.62
        height = 0.018
        cbar_bottom = max(0.040, bottom - 0.050)
    cbar_left = left + 0.5 * (span - width)
    cax = fig.add_axes([cbar_left, cbar_bottom, width, height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    if label:
        cbar.set_label(label, fontsize=16, labelpad=1)
    cbar.ax.tick_params(labelsize=16, pad=1, length=2)
    return cbar


def _add_fixed_vertical_colorbar(
    fig: plt.Figure,
    mappable: Any,
    ax: plt.Axes,
    *,
    ticks: list[float] | list[int] | None,
    ticklabels: list[str] | None,
    label: str,
) -> Any:
    fig.canvas.draw()
    box = ax.get_position()
    cax = fig.add_axes([
        max(0.004, box.x0 - 0.038),
        box.y0,
        0.008,
        box.height,
    ])
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical", ticks=ticks)
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=16, pad=1, length=2)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels, fontsize=16)
    return cbar


def _save_ensemble_rain_timeseries(
    *,
    ds: xr.Dataset,
    var_name: str,
    output_path: Path,
    title: str,
) -> Path:
    da = ds[var_name]
    if "time" not in da.dims:
        raise ValueError(f"Variable {var_name!r} does not have a time dimension")

    times = _to_datetimes(ds["time"].values)
    has_ensemble = "ensemble_member" in da.dims

    if has_ensemble:
        spatial_dims = [dim for dim in da.dims if dim not in {"time", "ensemble_member"}]
    else:
        spatial_dims = [dim for dim in da.dims if dim != "time"]
    if not spatial_dims:
        raise ValueError(f"Variable {var_name!r} has no spatial dimensions to average")

    if has_ensemble:
        member_ts = da.mean(dim=spatial_dims, skipna=True)
        member_ts = member_ts.transpose("time", "ensemble_member", ...)
        mean_ts = member_ts.mean(dim="ensemble_member", skipna=True)
        member_values = np.asarray(member_ts.values, dtype=np.float32)
        if member_values.shape[0] != len(times):
            raise ValueError("Timeseries length does not match time coordinate")
    else:
        mean_ts = da.mean(dim=spatial_dims, skipna=True)
        member_values = None

    mean_values = np.asarray(mean_ts.values, dtype=np.float32)
    if mean_values.shape[0] != len(times):
        raise ValueError("Timeseries length does not match time coordinate")

    fig, ax = plt.subplots(figsize=(10.5, 3.6), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.86)
    if has_ensemble and member_values is not None:
        n_members = int(member_values.shape[1])
        colors = plt.get_cmap("tab20")(np.linspace(0.05, 0.95, max(n_members, 2)))
        for member_index in range(n_members):
            ax.plot(
                times,
                member_values[:, member_index],
                color=colors[member_index % len(colors)],
                linewidth=1.0,
                alpha=0.7,
                label=f"member {member_index}",
            )
        ax.plot(times, mean_values, color="black", linewidth=2.1, label="ensemble mean")
        ax.legend(loc="upper left", ncol=2, fontsize=16, frameon=False)
    else:
        ax.plot(times, mean_values, color="#08519c", linewidth=1.8)
        ax.fill_between(times, 0.0, mean_values, color="#9ecae1", alpha=0.35, linewidth=0)
    ax.axhline(0.0, color="#7f7f7f", linewidth=0.7, alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("rain rate (mm/h)")
    ax.set_xlabel("time")
    ax.tick_params(labelsize=16)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=0, ha="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_nowcast(
    input_path: str,
    output_path: str | None = None,
    variable: str | None = None,
    ensemble: str = "mean",
    fps: int = 5,
    cmap: str = "turbo",
    dpi: int = 140,
) -> tuple[Path, Path]:
    source = _resolve_input(input_path)
    output = Path(output_path) if output_path else source.with_suffix(".gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    timeseries_path = output.with_name("timeseries.png")

    with xr.open_dataset(source, decode_times=False) as ds:
        var_name = _pick_variable(ds, variable)
        compare_mode = ensemble == "compare"
        da = _prepare_dataarray(ds, var_name, ensemble if not compare_mode else "0")
        if da.sizes["time"] == 0:
            raise ValueError(f"No time steps found in {source}")

        compare_da = None
        wet_levels = None
        wet_cmap = None
        wet_norm = None
        wet_ticks = None
        wet_ticklabels = None
        if compare_mode:
            compare_da = _prepare_dataarray(ds, var_name, "mean")
            if compare_da.sizes["time"] != da.sizes["time"]:
                raise ValueError("Comparison views must have the same number of time steps")
            wet_levels, wet_cmap, wet_norm, wet_ticks, wet_ticklabels = _wet_member_count_metadata(ds[var_name])

        times = _to_datetimes(ds["time"].values)
        if len(times) != da.sizes["time"]:
            raise ValueError("Time coordinate length does not match data")

        _save_ensemble_rain_timeseries(
            ds=ds,
            var_name=var_name,
            output_path=timeseries_path,
            title=f"{source.name} ensemble rain meteogram",
        )

        if compare_da is not None:
            vmin, vmax = _frame_limits_many([da, compare_da])
        else:
            vmin, vmax = _frame_limits(da)
        units = da.attrs.get("units", "")
        use_precip_palette = _is_precip_variable(var_name)
        if use_precip_palette:
            cmap_obj: str | ListedColormap = ListedColormap(PRECIP_COLORS, name="precip")
            norm = BoundaryNorm(PRECIP_LEVELS, cmap_obj.N, clip=True)
            vmin, vmax = PRECIP_LEVELS[0], PRECIP_LEVELS[-1]
        else:
            cmap_obj = cmap
            norm = None

        lon, lat = _resolve_geo_coords(da)
        frame0 = _coerce_frame(da, 0)
        geo_mode = lon is not None and lat is not None and lon.shape == frame0.shape and lat.shape == frame0.shape
        geo_point_indices = None
        if geo_mode:
            finite = np.isfinite(lon) & np.isfinite(lat)
            finite_indices = np.flatnonzero(finite)
            geo_point_indices = finite_indices[_downsample_indices(int(np.count_nonzero(finite)), GEO_MAX_POINTS)]
            geo_triangulation = _build_geo_triangulation(lon.ravel()[geo_point_indices], lat.ravel()[geo_point_indices])
        else:
            geo_triangulation = None
        geo_extent = _triangulation_extent(geo_triangulation) if geo_mode and geo_triangulation is not None else None

        if compare_mode:
            if geo_mode:
                fig, axes = plt.subplots(
                    ncols=3,
                    figsize=(16.2, 5.2),
                    subplot_kw={"projection": NORWAY_MAP},
                    constrained_layout=False,
                )
                fig.subplots_adjust(left=0.070, right=0.992, top=0.905, bottom=0.060, wspace=0.000)
            else:
                fig, axes = plt.subplots(ncols=3, figsize=(16.2, 5.2), constrained_layout=False)
                fig.subplots_adjust(left=0.070, right=0.992, top=0.905, bottom=0.065, wspace=0.000)
            axes = np.atleast_1d(axes)
            panel_defs = [
                ("ensemble 0", da, "precip"),
                ("ensemble mean", compare_da if compare_da is not None else da, "precip"),
                (f"wet members > {ENSEMBLE_PROBABILITY_THRESHOLD:g} mm/h", ds[var_name], "wet_count"),
            ]
        else:
            if geo_mode:
                fig, ax = plt.subplots(
                    figsize=(10, 7.6),
                    subplot_kw={"projection": NORWAY_MAP},
                    constrained_layout=False,
                )
                fig.subplots_adjust(left=0.035, right=0.970, top=0.925, bottom=0.095)
            else:
                fig, ax = plt.subplots(figsize=(10, 7.6), constrained_layout=False)
                fig.subplots_adjust(left=0.080, right=0.965, top=0.925, bottom=0.095)
            axes = np.atleast_1d(ax)
            panel_defs = [(var_name, da, "precip")]

        images: list[plt.Artist] = []
        for ax, (panel_label, panel_da, panel_kind) in zip(axes, panel_defs, strict=True):
            if panel_kind == "wet_count":
                if wet_levels is None or wet_cmap is None or wet_norm is None:
                    raise RuntimeError("Wet-member metadata was not initialised")
                image = _render_wet_member_count_frame(
                    ax,
                    panel_da,
                    0,
                    f"{panel_label}  {_format_local_time(times[0])}",
                    threshold=ENSEMBLE_PROBABILITY_THRESHOLD,
                    levels=wet_levels,
                    cmap=wet_cmap,
                    norm=wet_norm,
                    geo_triangulation=geo_triangulation,
                    geo_extent=geo_extent,
                    geo_point_indices=geo_point_indices,
                )
            else:
                image = _render_frame(
                    ax,
                    panel_da,
                    0,
                    f"{panel_label}  {_format_local_time(times[0])}",
                    vmin,
                    vmax,
                    cmap_obj,
                    norm=norm,
                    geo_triangulation=geo_triangulation,
                    geo_extent=geo_extent,
                    geo_point_indices=geo_point_indices,
                )
            images.append(image)

        if compare_mode:
            _add_fixed_vertical_colorbar(
                fig,
                images[0],
                axes[0],
                ticks=PRECIP_LEVELS[:-1] if use_precip_palette else None,
                ticklabels=PRECIP_TICK_LABELS if use_precip_palette else None,
                label=str(units),
            )
            if wet_ticks is None or wet_ticklabels is None:
                raise RuntimeError("Wet-member colourbar metadata was not initialised")
            _add_fixed_vertical_colorbar(
                fig,
                images[2],
                axes[2],
                ticks=wet_ticks,
                ticklabels=wet_ticklabels,
                label=f"Wet members > {ENSEMBLE_PROBABILITY_THRESHOLD:g} mm/h",
            )
        else:
            _add_fixed_vertical_colorbar(
                fig,
                images[0],
                axes[0],
                ticks=PRECIP_LEVELS[:-1] if use_precip_palette else None,
                ticklabels=PRECIP_TICK_LABELS if use_precip_palette else None,
                label=str(units),
            )

        def update(frame_index: int) -> tuple[plt.Artist, ...]:
            artists: list[plt.Artist] = []
            for idx, (ax, image, (panel_label, panel_da, panel_kind)) in enumerate(
                zip(axes, images, panel_defs, strict=True)
            ):
                if panel_kind == "wet_count":
                    _clear_contour_set(image)
                    if wet_levels is None or wet_cmap is None or wet_norm is None:
                        raise RuntimeError("Wet-member metadata was not initialised")
                    new_image = _render_wet_member_count_frame(
                        ax,
                        panel_da,
                        frame_index,
                        f"{panel_label}  {_format_local_time(times[frame_index])}",
                        threshold=ENSEMBLE_PROBABILITY_THRESHOLD,
                        levels=wet_levels,
                        cmap=wet_cmap,
                        norm=wet_norm,
                        geo_triangulation=geo_triangulation,
                        geo_extent=geo_extent,
                        geo_point_indices=geo_point_indices,
                    )
                    images[idx] = new_image
                    artists.append(new_image)
                else:
                    frame = _coerce_frame(panel_da, frame_index)
                    if geo_mode and geo_point_indices is not None:
                        image.set_array(frame.ravel()[geo_point_indices])
                    else:
                        _update_artist(image, frame, flattened=geo_mode)
                    artists.append(image)
                    ax.set_title(f"{panel_label}  {_format_local_time(times[frame_index])}")
            return tuple(artists)

        animation = FuncAnimation(
            fig,
            update,
            frames=da.sizes["time"],
            interval=max(1, 1000 // max(1, fps)),
        )

        fig.savefig(png_path, dpi=dpi)
        animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)

    return output, png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a BRIS nowcast NetCDF as a GIF")
    parser.add_argument("input", help="Forecast NetCDF file or a directory containing one")
    parser.add_argument(
        "-o",
        "--output",
        help="Output GIF path. Defaults to <input>.gif",
    )
    parser.add_argument(
        "-v",
        "--variable",
        help="Variable to plot. Defaults to the first precipitation-like variable.",
    )
    parser.add_argument(
        "--ensemble",
        default="mean",
        help="Ensemble member to plot, 'mean'/'median' to aggregate, or 'compare' for member 0, mean, and wet-member-count panels.",
    )
    parser.add_argument("--fps", type=int, default=5, help="GIF frame rate")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap")
    parser.add_argument("--dpi", type=int, default=140, help="Image DPI")
    args = parser.parse_args(argv)

    output, png_path = plot_nowcast(
        args.input,
        args.output,
        args.variable,
        args.ensemble,
        args.fps,
        args.cmap,
        args.dpi,
    )
    print(output)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
