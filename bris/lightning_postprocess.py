#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import CRS, Transformer

from bris.plot_nowcast import (
    NORWAY_MAP,
    add_geo_gridlines,
    render_triangulated_field,
)

DECODER_LOGIT_SCALE = 0.009828979149460793
BALANCED_PRIOR = 2.7748810500779243e-05
OPERATING_THRESHOLD = 0.87
GRID_SPACING_METRES = 8000.0
GRID_SHAPE = (350, 312)
EXPECTED_COVERED_CELLS = 98_720
ACTIVITY_LEAD_MINUTES = np.array([5, 10, 15, 20, 25, 30], dtype=np.int64)
EXPECTED_MEMBERS = np.arange(10, dtype=np.int64)
LAEA_CRS = CRS.from_proj4(
    "+proj=laea +lon_0=18.5 +lat_0=63.5 +datum=WGS84 +units=m +no_defs"
)


def packaged_grid_definition() -> Path:
    return Path(str(files("bris").joinpath("schema/lightning_8km_grid.json")))


def _datetime64_ns(value: object) -> np.datetime64:
    return np.datetime64(pd.Timestamp(value).to_datetime64(), "ns")


def _coordinate_names(dataset: xr.Dataset) -> tuple[str, str]:
    latitude_name = next(
        (name for name in ("latitude", "lat") if name in dataset), None
    )
    longitude_name = next(
        (name for name in ("longitude", "lon") if name in dataset), None
    )
    if latitude_name is None or longitude_name is None:
        raise ValueError("Lightning logits must contain latitude and longitude coordinates.")
    return latitude_name, longitude_name


def _coordinate_fingerprint(latitude: np.ndarray, longitude: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(latitude, dtype="<f8").tobytes())
    digest.update(np.asarray(longitude, dtype="<f8").tobytes())
    return digest.hexdigest()


def _load_grid_definition(
    latitude: np.ndarray,
    longitude: np.ndarray,
    grid_definition_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    fingerprint = _coordinate_fingerprint(latitude, longitude)
    forward = Transformer.from_crs("EPSG:4326", LAEA_CRS, always_xy=True)
    projected_x, projected_y = forward.transform(longitude, latitude)
    finite = np.isfinite(projected_x) & np.isfinite(projected_y)
    if not finite.any():
        raise ValueError("The lightning grid contains no finite projected coordinates.")

    computed_origin_x = float(
        np.floor(np.nanmin(projected_x[finite]) / GRID_SPACING_METRES)
        * GRID_SPACING_METRES
    )
    computed_origin_y = float(
        np.floor(np.nanmin(projected_y[finite]) / GRID_SPACING_METRES)
        * GRID_SPACING_METRES
    )

    if grid_definition_path.exists():
        definition = json.loads(grid_definition_path.read_text())
        if definition.get("coordinate_sha256") != fingerprint:
            raise ValueError(
                "Lightning-grid coordinates changed; refusing to recompute the operational mapping."
            )
        origin_x = float(definition["origin_x"])
        origin_y = float(definition["origin_y"])
        shape = tuple(int(value) for value in definition["shape"])
        if shape != GRID_SHAPE:
            raise ValueError(f"Stored lightning grid shape is {shape}, expected {GRID_SHAPE}.")
    else:
        origin_x = computed_origin_x
        origin_y = computed_origin_y
        shape = GRID_SHAPE

    cell_x = np.full(np.shape(projected_x), -1, dtype=np.int64)
    cell_y = np.full(np.shape(projected_y), -1, dtype=np.int64)
    cell_x[finite] = np.floor(
        (projected_x[finite] - origin_x) / GRID_SPACING_METRES
    ).astype(np.int64)
    cell_y[finite] = np.floor(
        (projected_y[finite] - origin_y) / GRID_SPACING_METRES
    ).astype(np.int64)
    in_grid = (
        finite
        & (cell_x >= 0)
        & (cell_x < shape[1])
        & (cell_y >= 0)
        & (cell_y < shape[0])
    )
    if not in_grid.any():
        raise ValueError("No native lightning-grid points map to the fixed 8-km grid.")
    flat_cell_index = cell_y * shape[1] + cell_x
    covered_cells = np.unique(flat_cell_index[in_grid]).size
    if covered_cells != EXPECTED_COVERED_CELLS:
        raise ValueError(
            f"Fixed 8-km mapping covers {covered_cells:,} cells, "
            f"expected {EXPECTED_COVERED_CELLS:,}."
        )

    inferred_shape = (
        int(np.nanmax(cell_y[finite])) + 1,
        int(np.nanmax(cell_x[finite])) + 1,
    )
    if not grid_definition_path.exists():
        if inferred_shape != GRID_SHAPE:
            raise ValueError(
                f"New lightning grid maps to {inferred_shape}, expected verified shape {GRID_SHAPE}."
            )
        grid_definition_path.parent.mkdir(parents=True, exist_ok=True)
        grid_definition_path.write_text(
            json.dumps(
                {
                    "coordinate_sha256": fingerprint,
                    "crs": LAEA_CRS.to_proj4(),
                    "grid_spacing_metres": GRID_SPACING_METRES,
                    "origin_x": origin_x,
                    "origin_y": origin_y,
                    "shape": list(shape),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    elif (origin_x, origin_y) != (computed_origin_x, computed_origin_y):
        raise ValueError("Stored 8-km grid origin does not match the fixed source grid.")

    return flat_cell_index.ravel(), in_grid.ravel(), finite.ravel(), origin_x, origin_y


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / DECODER_LOGIT_SCALE
    probability = np.empty_like(scaled, dtype=np.float64)
    nonnegative = scaled >= 0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-scaled[nonnegative]))
    exp_scaled = np.exp(scaled[~nonnegative])
    probability[~nonnegative] = exp_scaled / (1.0 + exp_scaled)
    probability[~np.isfinite(values)] = np.nan
    return probability


def _average_to_fixed_grid(
    frame_probability: np.ndarray,
    flat_cell_index: np.ndarray,
    in_grid: np.ndarray,
) -> np.ndarray:
    output_size = GRID_SHAPE[0] * GRID_SHAPE[1]
    output = np.full(
        (*frame_probability.shape[:-1], output_size), np.nan, dtype=np.float64
    )
    for index in np.ndindex(frame_probability.shape[:-1]):
        values = frame_probability[index]
        valid = in_grid & np.isfinite(values)
        sums = np.bincount(
            flat_cell_index[valid], weights=values[valid], minlength=output_size
        )
        counts = np.bincount(flat_cell_index[valid], minlength=output_size)
        np.divide(sums, counts, out=output[index], where=counts > 0)
    return output.reshape(*frame_probability.shape[:-1], *GRID_SHAPE)


def _forecast_reference_time(dataset: xr.Dataset) -> np.datetime64:
    if "forecast_reference_time" not in dataset:
        raise ValueError("forecast_reference_time is missing from lightning logits.")
    values = np.asarray(dataset["forecast_reference_time"].values).reshape(-1)
    if values.size != 1:
        raise ValueError("forecast_reference_time must contain exactly one value.")
    return _datetime64_ns(values[0])


def postprocess_lightning(
    input_path: Path,
    output_path: Path,
    grid_definition_path: Path,
) -> xr.Dataset:
    with xr.open_dataset(input_path) as dataset:
        if "thunder_count" not in dataset:
            raise ValueError("thunder_count is missing from lightning logits.")
        if "time" not in dataset.coords:
            raise ValueError("time is missing from lightning logits.")
        if "ensemble_member" not in dataset.coords:
            raise ValueError("ensemble_member is missing from lightning logits.")

        members = np.asarray(dataset["ensemble_member"].values, dtype=np.int64)
        if not np.array_equal(members, EXPECTED_MEMBERS):
            raise ValueError(
                f"ensemble_member must be exactly 0..9, received {members.tolist()}."
            )

        forecast_reference_time = _forecast_reference_time(dataset)
        requested_times = forecast_reference_time + ACTIVITY_LEAD_MINUTES.astype(
            "timedelta64[m]"
        )
        available_times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
        selected_indices = []
        for requested_time in requested_times:
            matches = np.flatnonzero(available_times == requested_time)
            if matches.size != 1:
                raise ValueError(
                    "Required lightning valid times must be exactly +5,+10,+15,+20,+25,+30 "
                    f"minutes; missing or duplicated {requested_time}."
                )
            selected_indices.append(int(matches[0]))

        latitude_name, longitude_name = _coordinate_names(dataset)
        latitude = np.asarray(dataset[latitude_name].values, dtype=np.float64)
        longitude = np.asarray(dataset[longitude_name].values, dtype=np.float64)
        spatial_dims = dataset[latitude_name].dims
        if dataset[longitude_name].dims != spatial_dims:
            raise ValueError("Latitude and longitude dimensions differ.")

        ordered = dataset["thunder_count"].isel(time=selected_indices).transpose(
            "time", "ensemble_member", *spatial_dims
        )
        decoder_values = np.asarray(ordered.values)
    if not np.isfinite(decoder_values).any():
        raise ValueError("thunder_count has no finite values on the selected model domain.")
    flattened_decoder_values = decoder_values.reshape(6, 10, -1)
    valid_model_domain = np.isfinite(flattened_decoder_values).any(axis=(0, 1))
    if not np.isfinite(flattened_decoder_values[:, :, valid_model_domain]).all():
        raise ValueError(
            "thunder_count must be finite for all six leads and ten members on the valid model domain."
        )

    flat_cell_index, in_grid, _, origin_x, origin_y = _load_grid_definition(
        latitude,
        longitude,
        grid_definition_path,
    )
    frame_probability = stable_sigmoid(flattened_decoder_values)
    frame_probability_8km = _average_to_fixed_grid(
        frame_probability,
        flat_cell_index,
        in_grid,
    )
    member_activity_probability = 1.0 - np.prod(
        1.0 - frame_probability_8km, axis=0
    )
    raw_activity_probability = np.mean(member_activity_probability, axis=0)
    numerator = BALANCED_PRIOR * raw_activity_probability
    denominator = numerator + (1.0 - BALANCED_PRIOR) * (
        1.0 - raw_activity_probability
    )
    corrected_probability = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    finite_probability = np.isfinite(corrected_probability)
    if not finite_probability.any():
        raise ValueError("corrected_probability has no finite values on the model domain.")
    if int(finite_probability.sum()) != EXPECTED_COVERED_CELLS:
        raise ValueError(
            f"corrected_probability is finite on {int(finite_probability.sum()):,} cells, "
            f"expected {EXPECTED_COVERED_CELLS:,}."
        )
    if np.nanmin(corrected_probability) < 0 or np.nanmax(corrected_probability) > 1:
        raise ValueError("corrected_probability is outside [0, 1].")

    x = origin_x + (np.arange(GRID_SHAPE[1]) + 0.5) * GRID_SPACING_METRES
    y = origin_y + (np.arange(GRID_SHAPE[0]) + 0.5) * GRID_SPACING_METRES
    grid_x, grid_y = np.meshgrid(x, y)
    inverse = Transformer.from_crs(LAEA_CRS, "EPSG:4326", always_xy=True)
    grid_longitude, grid_latitude = inverse.transform(grid_x, grid_y)
    lightning_mask = finite_probability & (
        corrected_probability >= OPERATING_THRESHOLD
    )
    display_category = np.zeros(GRID_SHAPE, dtype=np.uint8)
    display_category[
        finite_probability & (corrected_probability >= OPERATING_THRESHOLD)
    ] = 1
    display_category[finite_probability & (corrected_probability >= 0.95)] = 2
    display_category[finite_probability & (corrected_probability >= 0.99)] = 3

    output = xr.Dataset(
        data_vars={
            "corrected_probability": (
                ("y", "x"),
                corrected_probability.astype(np.float32),
                {
                    "long_name": "Probability of any lightning during forecast lead +5 to +30 minutes",
                    "units": "1",
                },
            ),
            "raw_activity_probability": (
                ("y", "x"),
                raw_activity_probability.astype(np.float32),
                {"units": "1"},
            ),
            "lightning_mask": (
                ("y", "x"),
                lightning_mask.astype(np.uint8),
                {
                    "long_name": "Operational lightning mask",
                    "threshold": OPERATING_THRESHOLD,
                },
            ),
            "display_category": (
                ("y", "x"),
                display_category,
                {
                    "flag_values": np.array([0, 1, 2, 3], dtype=np.uint8),
                    "flag_meanings": "below_threshold somewhat_likely likely very_likely",
                },
            ),
            "latitude": (("y", "x"), grid_latitude.astype(np.float32)),
            "longitude": (("y", "x"), grid_longitude.astype(np.float32)),
        },
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "forecast_reference_time": forecast_reference_time,
            "activity_window_start": forecast_reference_time
            + np.timedelta64(5, "m"),
            "activity_window_end": forecast_reference_time
            + np.timedelta64(30, "m"),
        },
        attrs={
            "title": "BRIS calibrated 30-minute lightning probability",
            "source": str(input_path),
            "decoder_logit_scale": DECODER_LOGIT_SCALE,
            "balanced_prior": BALANCED_PRIOR,
            "operating_threshold": OPERATING_THRESHOLD,
            "activity_lead_minutes": "5,10,15,20,25,30",
            "activity_window_aggregation": "noisy-OR per member, then mean over 10 members",
            "spatial_aggregation": "finite-value arithmetic mean of frame probabilities on fixed 8-km LAEA cells",
            "projection": LAEA_CRS.to_proj4(),
            "grid_definition": str(grid_definition_path),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(
        output_path,
        encoding={
            "corrected_probability": {"zlib": True, "complevel": 4},
            "raw_activity_probability": {"zlib": True, "complevel": 4},
        },
    )
    return output


def plot_probability(dataset: xr.Dataset, output_path: Path) -> None:
    threshold = float(dataset.attrs.get("operating_threshold", OPERATING_THRESHOLD))
    if threshold != OPERATING_THRESHOLD:
        raise ValueError(
            f"Lightning plot threshold is {threshold}, expected fixed threshold {OPERATING_THRESHOLD}."
        )
    probability = dataset["corrected_probability"].where(
        dataset["corrected_probability"] >= threshold
    )
    levels = [threshold, 0.95, 0.99, 1.0000001]
    category_midpoints = np.array(
        [
            (levels[0] + levels[1]) / 2.0,
            (levels[1] + levels[2]) / 2.0,
            (levels[2] + 1.0) / 2.0,
        ]
    )
    color_positions = (category_midpoints - threshold) / (1.0 - threshold)
    cmap = ListedColormap(plt.get_cmap("viridis")(color_positions))
    norm = BoundaryNorm(levels, cmap.N)
    fig, axis = plt.subplots(
        figsize=(10, 7.6),
        subplot_kw={"projection": NORWAY_MAP},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.030, right=0.775, top=0.925, bottom=0.095)
    image = render_triangulated_field(
        axis,
        np.asarray(probability.values),
        np.asarray(dataset["longitude"].values),
        np.asarray(dataset["latitude"].values),
        cmap=cmap,
        norm=norm,
        white_background=True,
    )
    add_geo_gridlines(axis)
    reference_time = pd.Timestamp(dataset["forecast_reference_time"].values)
    fig.suptitle(
        f"Lightning probability  {reference_time:%Y-%m-%d %H:%M} UTC  "
        "(any lightning during +5 to +30 min)",
        y=0.975,
    )
    colorbar_axis = fig.add_axes([0.810, 0.205, 0.024, 0.590])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=[0.91, 0.97, 0.995],
    )
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    colorbar.ax.set_yticklabels(
        ["somewhat likely", "likely", "very likely"],
        fontsize=11,
    )
    colorbar.ax.tick_params(pad=5, length=3)
    colorbar.set_label(
        f"Corrected probability (displayed from {threshold:.2f})",
        fontsize=11,
        labelpad=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BRIS lightning decoder output to calibrated +5-to-+30-minute probability."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--grid-definition",
        type=Path,
        default=packaged_grid_definition(),
        help="Fixed-grid JSON packaged with bris-inference.",
    )
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = postprocess_lightning(args.input, args.output, args.grid_definition)
    if args.plot is not None:
        plot_probability(output, args.plot)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
