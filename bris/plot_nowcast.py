#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

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

NORWAY_MAP = ccrs.Stereographic(central_longitude=18.5, central_latitude=63.5)
GEO_MAX_POINTS = 500_000

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
PRECIP_TICK_LABELS = [
    "0",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "1",
    "1.5",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
]


def add_geo_features(axis: plt.Axes, *, white_background: bool = False) -> None:
    """Add the geographic background shared by BRIS nowcast plots."""
    if white_background:
        axis.set_facecolor("white")
        axis.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white", zorder=0)
        axis.add_feature(cfeature.LAND.with_scale("110m"), facecolor="white", zorder=0)
        coastline_color = "#c7d2d7"
        border_color = "#d7e0e3"
    else:
        axis.add_feature(
            cfeature.OCEAN.with_scale("110m"), facecolor="#dbe9f4", zorder=0
        )
        axis.add_feature(
            cfeature.LAND.with_scale("110m"), facecolor="#efefef", zorder=0
        )
        coastline_color = "#b7c3c7"
        border_color = "#c8d2d5"
    axis.coastlines(resolution="50m", linewidth=0.45, color=coastline_color, zorder=2)
    axis.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linestyle=":",
        linewidth=0.35,
        edgecolor=border_color,
        zorder=2,
    )


def add_geo_gridlines(axis: plt.Axes) -> None:
    """Add unobtrusive labelled latitude/longitude gridlines."""
    gridlines = axis.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="#c7d2d7",
        alpha=0.65,
        linestyle="-",
        zorder=0.5,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9}
    gridlines.ylabel_style = {"size": 9}


def geographic_triangulation(
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> tuple[Triangulation, np.ndarray, tuple[float, float, float, float]]:
    """Project finite lon/lat points onto the standard nowcast map."""
    finite_indices = np.flatnonzero(np.isfinite(longitude) & np.isfinite(latitude))
    if finite_indices.size == 0:
        raise ValueError("No finite latitude/longitude coordinates are available for plotting.")
    if finite_indices.size > GEO_MAX_POINTS:
        selection = np.linspace(
            0,
            finite_indices.size - 1,
            GEO_MAX_POINTS,
            dtype=np.int64,
        )
        point_indices = finite_indices[selection]
    else:
        point_indices = finite_indices
    points = NORWAY_MAP.transform_points(
        ccrs.PlateCarree(),
        longitude.ravel()[point_indices],
        latitude.ravel()[point_indices],
    )
    triangulation = Triangulation(points[:, 0], points[:, 1])
    extent = (
        float(np.nanmin(points[:, 0])),
        float(np.nanmax(points[:, 0])),
        float(np.nanmin(points[:, 1])),
        float(np.nanmax(points[:, 1])),
    )
    return triangulation, point_indices, extent


def render_triangulated_field(
    axis: plt.Axes,
    values: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    cmap: Any,
    norm: Any,
    white_background: bool = False,
    shading: str = "gouraud",
) -> Any:
    """Render a georeferenced field using the shared nowcast styling."""
    triangulation, point_indices, extent = geographic_triangulation(
        longitude,
        latitude,
    )
    sampled_values = np.ma.masked_invalid(values.ravel()[point_indices])
    image = axis.tripcolor(
        triangulation,
        sampled_values,
        cmap=cmap,
        norm=norm,
        shading=shading,
        rasterized=True,
        zorder=1,
    )
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    add_geo_features(axis, white_background=white_background)
    axis.set_aspect("equal", adjustable="box")
    return image


def add_fixed_vertical_colorbar(
    figure: plt.Figure,
    mappable: Any,
    axis: plt.Axes,
    *,
    ticks: list[float],
    ticklabels: list[str],
    label: str,
    side: str = "left",
) -> Any:
    """Add a fixed-width vertical colorbar beside a geographic panel."""
    figure.canvas.draw()
    box = axis.get_position()
    if side == "left":
        colorbar_bounds = [max(0.004, box.x0 - 0.038), box.y0, 0.008, box.height]
    elif side == "right":
        colorbar_bounds = [min(0.982, box.x1 + 0.018), box.y0, 0.012, box.height]
    else:
        raise ValueError(f"Unsupported colorbar side: {side}")
    colorbar_axis = figure.add_axes(colorbar_bounds)
    colorbar = figure.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=ticks,
    )
    colorbar.set_label(label, fontsize=16)
    colorbar.ax.tick_params(labelsize=16, pad=1, length=2)
    colorbar.ax.yaxis.set_ticks_position(side)
    colorbar.ax.yaxis.set_label_position(side)
    colorbar.ax.set_yticklabels(ticklabels, fontsize=16)
    return colorbar


def _single_datetime(dataset: xr.Dataset, name: str) -> np.datetime64:
    if name not in dataset:
        raise ValueError(f"{name} is missing from precipitation output.")
    values = np.asarray(dataset[name].values).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must contain exactly one value.")
    return np.datetime64(pd.Timestamp(values[0]).to_datetime64(), "ns")


def _coordinate_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    name = next((candidate for candidate in candidates if candidate in dataset), None)
    if name is None:
        raise ValueError(f"Missing coordinate; expected one of {candidates}.")
    return name


def select_diverse_members(
    input_path: Path,
    lead_minutes: list[int],
    spatial_stride: int,
    member_count: int = 4,
) -> list[int]:
    with xr.open_dataset(input_path) as dataset:
        variable_name = "lwe_precipitation_rate"
        if variable_name not in dataset:
            raise ValueError(f"{variable_name} is missing from precipitation output.")
        field = dataset[variable_name]
        member_values = np.asarray(dataset["ensemble_member"].values, dtype=int)
        if sorted(member_values.tolist()) != list(range(10)):
            raise ValueError(
                f"Expected ensemble members 0 through 9, found {member_values.tolist()}."
            )
        if member_count > member_values.size:
            raise ValueError("Cannot select more members than are available.")

        reference_time = _single_datetime(dataset, "forecast_reference_time")
        times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
        time_indices = []
        for lead in lead_minutes:
            valid_time = reference_time + np.timedelta64(lead, "m")
            matches = np.flatnonzero(times == valid_time)
            if matches.size != 1:
                raise ValueError(f"Expected exactly one field at lead +{lead} minutes.")
            time_indices.append(int(matches[0]))

        spatial_dims = tuple(
            dimension
            for dimension in field.dims
            if dimension not in {"time", "ensemble_member"}
        )
        stride_selection = {
            dimension: slice(None, None, spatial_stride) for dimension in spatial_dims
        }
        sampled = np.asarray(
            field.isel(time=time_indices, **stride_selection)
            .transpose("ensemble_member", "time", *spatial_dims)
            .values,
            dtype=np.float32,
        )

    common_support = np.isfinite(sampled).all(axis=0)
    wet_support = common_support & (np.max(sampled, axis=0) > 0.05)
    feature_support = wet_support if np.count_nonzero(wet_support) >= 100 else common_support
    if not np.any(feature_support):
        raise ValueError("No common finite precipitation support is available for member selection.")

    feature_vectors = np.log1p(np.clip(sampled[:, feature_support], 0.0, None))
    distances = np.zeros((member_values.size, member_values.size), dtype=np.float64)
    for first, second in combinations(range(member_values.size), 2):
        difference = feature_vectors[first] - feature_vectors[second]
        distance = float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))
        distances[first, second] = distance
        distances[second, first] = distance

    best_indices: tuple[int, ...] | None = None
    best_score = -np.inf
    for candidate in combinations(range(member_values.size), member_count):
        score = sum(
            distances[first, second]
            for first, second in combinations(candidate, 2)
        )
        if score > best_score:
            best_score = score
            best_indices = candidate
    if best_indices is None:
        raise RuntimeError("Failed to select diverse precipitation members.")
    return sorted(int(member_values[index]) for index in best_indices)


def plot_member_evolution(
    input_path: Path,
    output_path: Path,
    members: list[int],
    lead_minutes: list[int],
    spatial_stride: int,
) -> None:
    with xr.open_dataset(input_path) as dataset:
        variable_name = "lwe_precipitation_rate"
        if variable_name not in dataset:
            raise ValueError(f"{variable_name} is missing from precipitation output.")
        field = dataset[variable_name]
        if "ensemble_member" not in field.dims:
            raise ValueError("Precipitation output has no ensemble_member dimension.")
        if "time" not in field.dims:
            raise ValueError("Precipitation output has no time dimension.")

        member_values = np.asarray(dataset["ensemble_member"].values, dtype=int)
        if sorted(member_values.tolist()) != list(range(10)):
            raise ValueError(
                f"Expected ensemble members 0 through 9, found {member_values.tolist()}."
            )
        missing_members = sorted(set(members).difference(member_values.tolist()))
        if missing_members:
            raise ValueError(f"Requested ensemble members are missing: {missing_members}.")

        reference_time = _single_datetime(dataset, "forecast_reference_time")
        times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
        selected_time_indices: list[int] = []
        selected_times: list[np.datetime64] = []
        for lead in lead_minutes:
            valid_time = reference_time + np.timedelta64(lead, "m")
            matches = np.flatnonzero(times == valid_time)
            if matches.size != 1:
                raise ValueError(f"Expected exactly one field at lead +{lead} minutes.")
            selected_time_indices.append(int(matches[0]))
            selected_times.append(valid_time)

        latitude_name = _coordinate_name(dataset, ("latitude", "lat"))
        longitude_name = _coordinate_name(dataset, ("longitude", "lon"))
        latitude = np.asarray(dataset[latitude_name].values)[
            ::spatial_stride, ::spatial_stride
        ]
        longitude = np.asarray(dataset[longitude_name].values)[
            ::spatial_stride, ::spatial_stride
        ]
        finite_coordinates = np.isfinite(latitude) & np.isfinite(longitude)
        point_indices = np.flatnonzero(finite_coordinates.ravel())
        if point_indices.size == 0:
            raise ValueError("No finite latitude/longitude coordinates are available.")
        points = NORWAY_MAP.transform_points(
            ccrs.PlateCarree(),
            longitude.ravel()[point_indices],
            latitude.ravel()[point_indices],
        )
        triangulation = Triangulation(points[:, 0], points[:, 1])
        extent = (
            float(np.nanmin(points[:, 0])),
            float(np.nanmax(points[:, 0])),
            float(np.nanmin(points[:, 1])),
            float(np.nanmax(points[:, 1])),
        )
        units = str(field.attrs.get("units", "mm/h"))

        cmap = ListedColormap(PRECIP_COLORS, name="precip")
        norm = BoundaryNorm(PRECIP_LEVELS, cmap.N, clip=True)
        figure, axes = plt.subplots(
            nrows=len(members),
            ncols=len(lead_minutes),
            figsize=(24, 18.5),
            subplot_kw={"projection": NORWAY_MAP},
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.045,
            right=0.995,
            top=0.94,
            bottom=0.045,
            wspace=0.005,
            hspace=0.0,
        )

        image = None
        for row, member in enumerate(members):
            for column, (lead, time_index, valid_time) in enumerate(
                zip(lead_minutes, selected_time_indices, selected_times, strict=True)
            ):
                axis = axes[row, column]
                values = np.asarray(
                    field.sel(ensemble_member=member).isel(time=time_index).values,
                    dtype=np.float64,
                )[::spatial_stride, ::spatial_stride]
                sampled_values = np.ma.masked_invalid(values.ravel()[point_indices])
                image = axis.tripcolor(
                    triangulation,
                    sampled_values,
                    cmap=cmap,
                    norm=norm,
                    shading="gouraud",
                    rasterized=True,
                    zorder=1,
                )
                axis.set_xlim(extent[0], extent[1])
                axis.set_ylim(extent[2], extent[3])
                add_geo_features(axis)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(
                        f"+{lead} min\n{pd.Timestamp(valid_time):%H:%M} UTC",
                        fontsize=19,
                        pad=5,
                    )
                if column == 0:
                    axis.text(
                        -0.065,
                        0.5,
                        f"Member {member}",
                        transform=axis.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=21,
                    )

        if image is None:
            raise RuntimeError("No precipitation panels were rendered.")
        figure.suptitle(
            f"{pd.Timestamp(reference_time):%Y-%m-%d %H:%M} UTC",
            fontsize=28,
            y=0.995,
        )
        colorbar = figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            ticks=PRECIP_LEVELS[:-1],
            fraction=0.018,
            pad=0.012,
            aspect=90,
        )
        colorbar.set_label(f"Precipitation rate ({units})", fontsize=20)
        colorbar.ax.set_xticklabels(PRECIP_TICK_LABELS, fontsize=16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, facecolor="white")
        plt.close(figure)


def animate_members_and_median(
    input_path: Path,
    output_path: Path,
    members: list[int],
    spatial_stride: int,
    fps: int,
) -> None:
    with xr.open_dataset(input_path) as dataset:
        variable_name = "lwe_precipitation_rate"
        if variable_name not in dataset:
            raise ValueError(f"{variable_name} is missing from precipitation output.")
        field = dataset[variable_name]
        member_values = np.asarray(dataset["ensemble_member"].values, dtype=int)
        if sorted(member_values.tolist()) != list(range(10)):
            raise ValueError(
                f"Expected ensemble members 0 through 9, found {member_values.tolist()}."
            )
        missing_members = sorted(set(members).difference(member_values.tolist()))
        if missing_members:
            raise ValueError(f"Requested ensemble members are missing: {missing_members}.")

        reference_time = _single_datetime(dataset, "forecast_reference_time")
        times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
        latitude_name = _coordinate_name(dataset, ("latitude", "lat"))
        longitude_name = _coordinate_name(dataset, ("longitude", "lon"))
        spatial_dims = dataset[latitude_name].dims
        stride_selection = {
            dimension: slice(None, None, spatial_stride) for dimension in spatial_dims
        }
        latitude = np.asarray(dataset[latitude_name].isel(stride_selection).values)
        longitude = np.asarray(dataset[longitude_name].isel(stride_selection).values)
        finite_coordinates = np.isfinite(latitude) & np.isfinite(longitude)
        point_indices = np.flatnonzero(finite_coordinates.ravel())
        points = NORWAY_MAP.transform_points(
            ccrs.PlateCarree(),
            longitude.ravel()[point_indices],
            latitude.ravel()[point_indices],
        )
        triangulation = Triangulation(points[:, 0], points[:, 1])
        extent = (
            float(np.nanmin(points[:, 0])),
            float(np.nanmax(points[:, 0])),
            float(np.nanmin(points[:, 1])),
            float(np.nanmax(points[:, 1])),
        )
        sampled = np.asarray(
            field.isel(stride_selection)
            .transpose("time", "ensemble_member", *spatial_dims)
            .values,
            dtype=np.float32,
        )
        fixed_valid_support = np.isfinite(sampled).all(axis=(0, 1)).ravel()[
            point_indices
        ]
        triangulation.set_mask(
            np.any(~fixed_valid_support[triangulation.triangles], axis=1)
        )
        ensemble_median = np.ma.median(
            np.ma.masked_invalid(sampled), axis=1
        ).filled(np.nan)
        member_indices = {
            member: int(np.flatnonzero(member_values == member)[0])
            for member in members
        }
        panel_fields = [
            sampled[:, member_indices[members[0]]],
            sampled[:, member_indices[members[1]]],
            ensemble_median,
            sampled[:, member_indices[members[2]]],
            sampled[:, member_indices[members[3]]],
        ]
        panel_titles = [
            f"Member {members[0]}",
            f"Member {members[1]}",
            "ENSEMBLE MEDIAN",
            f"Member {members[2]}",
            f"Member {members[3]}",
        ]
        units = str(field.attrs.get("units", "mm/h"))

    cmap = ListedColormap(PRECIP_COLORS, name="precip")
    norm = BoundaryNorm(PRECIP_LEVELS, cmap.N, clip=True)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(20, 6.1),
        subplot_kw={"projection": NORWAY_MAP},
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.995,
        top=0.84,
        bottom=0.17,
        wspace=0.01,
    )
    images = []
    for index, (axis, values, title) in enumerate(
        zip(axes[0], panel_fields, panel_titles, strict=True)
    ):
        image = axis.tripcolor(
            triangulation,
            np.ma.masked_invalid(values[0].ravel()[point_indices]),
            cmap=cmap,
            norm=norm,
            shading="gouraud",
            rasterized=True,
            zorder=1,
        )
        images.append(image)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        add_geo_features(axis)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        if index == 2:
            axis.set_title(
                title,
                fontsize=21,
                color="#171000",
                pad=8,
                bbox={
                    "facecolor": "#ffca28",
                    "edgecolor": "#7a5200",
                    "linewidth": 2.5,
                    "boxstyle": "round,pad=0.3",
                },
            )
            axis.spines["geo"].set_edgecolor("#ffb300")
            axis.spines["geo"].set_linewidth(5.0)
        else:
            axis.set_title(title, fontsize=19, pad=8)

    time_title = figure.suptitle("", fontsize=25, y=0.975)
    colorbar = figure.colorbar(
        images[0],
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        ticks=PRECIP_LEVELS[:-1],
        fraction=0.04,
        pad=0.045,
        aspect=80,
    )
    colorbar.set_label(f"Precipitation rate ({units})", fontsize=17)
    colorbar.ax.set_xticklabels(PRECIP_TICK_LABELS, fontsize=13)

    def update(time_index: int) -> tuple[plt.Artist, ...]:
        for image, values in zip(images, panel_fields, strict=True):
            image.set_array(
                np.ma.masked_invalid(values[time_index].ravel()[point_indices])
            )
        valid_time = times[time_index]
        lead_minutes = int(
            (valid_time - reference_time) / np.timedelta64(1, "m")
        )
        lead_label = "analysis" if lead_minutes == 0 else f"+{lead_minutes} min"
        time_title.set_text(
            f"{pd.Timestamp(valid_time):%Y-%m-%d %H:%M} UTC · {lead_label}"
        )
        return (*images, time_title)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=max(1, 1000 // max(1, fps)),
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(figure)


def plot_nowcast(
    input_path: Path,
    output_path: Path,
    *,
    members: list[int] | None = None,
    lead_minutes: list[int] | None = None,
    spatial_stride: int = 8,
    animation_path: Path | None = None,
    fps: int = 4,
) -> tuple[Path, Path | None]:
    """Create the standard member-evolution plot and optional GIF preview."""
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be positive.")
    selected_leads = lead_minutes or [5, 10, 20, 60, 90, 120]
    if len(selected_leads) != 6:
        raise ValueError("Exactly six lead times are required.")
    selected_members = members or select_diverse_members(
        input_path,
        selected_leads,
        spatial_stride,
    )
    if len(selected_members) != 4:
        raise ValueError("Exactly four ensemble members are required.")

    plot_member_evolution(
        input_path,
        output_path,
        selected_members,
        selected_leads,
        spatial_stride,
    )
    if animation_path is not None:
        animate_members_and_median(
            input_path,
            animation_path,
            selected_members,
            spatial_stride,
            fps,
        )
    return output_path, animation_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot selected BRIS precipitation members through +120 minutes."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--members",
        type=int,
        nargs=4,
        help="Four explicit member IDs; omitted to select the most spatially diverse set.",
    )
    parser.add_argument(
        "--lead-minutes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 60, 90, 120],
    )
    parser.add_argument("--spatial-stride", type=int, default=8)
    parser.add_argument("--animation", type=Path)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args(argv)
    if args.spatial_stride < 1:
        parser.error("--spatial-stride must be positive.")
    if len(args.lead_minutes) != 6:
        parser.error("Exactly six lead times are required.")
    members = args.members or select_diverse_members(
        args.input,
        args.lead_minutes,
        args.spatial_stride,
    )
    print(f"Selected diverse precipitation members: {members}")
    plot_nowcast(
        args.input,
        args.output,
        members=members,
        lead_minutes=args.lead_minutes,
        spatial_stride=args.spatial_stride,
        animation_path=args.animation,
        fps=args.fps,
    )
    print(args.output)
    if args.animation is not None:
        print(args.animation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
