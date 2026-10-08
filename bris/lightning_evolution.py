#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer

from bris.plot_nowcast import NORWAY_MAP, render_triangulated_field

from .lightning_postprocess import (
    BALANCED_PRIOR,
    DECODER_LOGIT_SCALE,
    EXPECTED_COVERED_CELLS,
    EXPECTED_MEMBERS,
    GRID_SHAPE,
    GRID_SPACING_METRES,
    LAEA_CRS,
    OPERATING_THRESHOLD,
    _average_to_fixed_grid,
    _coordinate_names,
    _forecast_reference_time,
    _load_grid_definition,
    packaged_grid_definition,
    stable_sigmoid,
)

WINDOW_LEADS = (
    np.arange(5, 31, 5, dtype=np.int64),
    np.arange(15, 41, 5, dtype=np.int64),
    np.arange(30, 61, 5, dtype=np.int64),
    np.arange(45, 111, 5, dtype=np.int64),
)
DISPLAY_THRESHOLD = OPERATING_THRESHOLD
DISPLAY_LEVELS = [DISPLAY_THRESHOLD, 0.95, 0.99, 1.0000001]


def _correct_probability(raw_probability: np.ndarray) -> np.ndarray:
    numerator = BALANCED_PRIOR * raw_probability
    denominator = numerator + (1.0 - BALANCED_PRIOR) * (1.0 - raw_probability)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def create_lightning_evolution(
    input_path: Path,
    output_path: Path,
    grid_definition_path: Path,
) -> xr.Dataset:
    with xr.open_dataset(input_path) as dataset:
        if "thunder_count" not in dataset:
            raise ValueError("thunder_count is missing from lightning logits.")
        members = np.asarray(dataset["ensemble_member"].values, dtype=np.int64)
        if not np.array_equal(members, EXPECTED_MEMBERS):
            raise ValueError(
                f"ensemble_member must be exactly 0..9, received {members.tolist()}."
            )

        reference_time = _forecast_reference_time(dataset)
        available_times = np.asarray(dataset["time"].values, dtype="datetime64[ns]")
        selected_indices: list[list[int]] = []
        for window in WINDOW_LEADS:
            window_indices = []
            for lead in window:
                requested_time = reference_time + np.timedelta64(int(lead), "m")
                matches = np.flatnonzero(available_times == requested_time)
                if matches.size != 1:
                    raise ValueError(
                        f"Expected exactly one lightning field at lead +{lead} minutes."
                    )
                window_indices.append(int(matches[0]))
            selected_indices.append(window_indices)

        latitude_name, longitude_name = _coordinate_names(dataset)
        latitude = np.asarray(dataset[latitude_name].values, dtype=np.float64)
        longitude = np.asarray(dataset[longitude_name].values, dtype=np.float64)
        spatial_dims = dataset[latitude_name].dims
        if dataset[longitude_name].dims != spatial_dims:
            raise ValueError("Latitude and longitude dimensions differ.")

        flat_cell_index, in_grid, _, origin_x, origin_y = _load_grid_definition(
            latitude,
            longitude,
            grid_definition_path,
        )
        corrected_windows = []
        raw_windows = []
        for window, indices in zip(WINDOW_LEADS, selected_indices, strict=True):
            ordered = dataset["thunder_count"].isel(time=indices).transpose(
                "time", "ensemble_member", *spatial_dims
            )
            decoder_values = np.asarray(ordered.values).reshape(len(window), 10, -1)
            valid_model_domain = np.isfinite(decoder_values).any(axis=(0, 1))
            if not valid_model_domain.any():
                raise ValueError(
                    f"thunder_count has no valid domain for leads {window.tolist()}."
                )
            if not np.isfinite(decoder_values[:, :, valid_model_domain]).all():
                raise ValueError(
                    "thunder_count must be finite for every selected lead and ten members "
                    f"for window {window.tolist()}."
                )
            frame_probability = stable_sigmoid(decoder_values)
            frame_probability_8km = _average_to_fixed_grid(
                frame_probability,
                flat_cell_index,
                in_grid,
            )
            member_activity_probability = 1.0 - np.prod(
                1.0 - frame_probability_8km, axis=0
            )
            raw_probability = np.mean(member_activity_probability, axis=0)
            corrected_probability = _correct_probability(raw_probability)
            finite = np.isfinite(corrected_probability)
            if int(finite.sum()) != EXPECTED_COVERED_CELLS:
                raise ValueError(
                    f"Window {window.tolist()} covers {int(finite.sum()):,} cells, "
                    f"expected {EXPECTED_COVERED_CELLS:,}."
                )
            if np.any(
                (corrected_probability[finite] < 0.0)
                | (corrected_probability[finite] > 1.0)
            ):
                raise ValueError(
                    f"Corrected probability is outside [0, 1] for {window.tolist()}."
                )
            raw_windows.append(raw_probability.astype(np.float32))
            corrected_windows.append(corrected_probability.astype(np.float32))

    x = origin_x + (np.arange(GRID_SHAPE[1]) + 0.5) * GRID_SPACING_METRES
    y = origin_y + (np.arange(GRID_SHAPE[0]) + 0.5) * GRID_SPACING_METRES
    grid_x, grid_y = np.meshgrid(x, y)
    inverse = Transformer.from_crs(LAEA_CRS, "EPSG:4326", always_xy=True)
    grid_longitude, grid_latitude = inverse.transform(grid_x, grid_y)
    starts = reference_time + np.array(
        [window[0] for window in WINDOW_LEADS], dtype="timedelta64[m]"
    )
    ends = reference_time + np.array(
        [window[-1] for window in WINDOW_LEADS], dtype="timedelta64[m]"
    )

    output = xr.Dataset(
        data_vars={
            "corrected_probability": (
                ("activity_window", "y", "x"),
                np.stack(corrected_windows),
                {
                    "long_name": "Probability of any lightning in each selected activity window",
                    "units": "1",
                },
            ),
            "raw_activity_probability": (
                ("activity_window", "y", "x"),
                np.stack(raw_windows),
                {"units": "1"},
            ),
            "latitude": (("y", "x"), grid_latitude.astype(np.float32)),
            "longitude": (("y", "x"), grid_longitude.astype(np.float32)),
        },
        coords={
            "activity_window": np.arange(len(WINDOW_LEADS), dtype=np.int8),
            "activity_window_start": ("activity_window", starts),
            "activity_window_end": ("activity_window", ends),
            "x": ("x", x),
            "y": ("y", y),
            "forecast_reference_time": reference_time,
        },
        attrs={
            "title": "BRIS lightning probability evolution for selected activity windows",
            "source": str(input_path),
            "decoder_logit_scale": DECODER_LOGIT_SCALE,
            "balanced_prior": BALANCED_PRIOR,
            "window_frame_leads_minutes": ";".join(
                ",".join(str(int(lead)) for lead in window) for window in WINDOW_LEADS
            ),
            "activity_window_aggregation": "noisy-OR per member, then mean over 10 members",
            "spatial_aggregation": "finite-value arithmetic mean of frame probabilities on fixed 8-km LAEA cells",
            "projection": LAEA_CRS.to_proj4(),
            "grid_definition": str(grid_definition_path),
            "operating_threshold_first_window_only": OPERATING_THRESHOLD,
            "threshold_note": "0.87 is validated only for the first +5-to-+30-minute window and is not applied to this evolution field",
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


def plot_lightning_evolution(dataset: xr.Dataset, output_path: Path) -> None:
    probability = np.asarray(dataset["corrected_probability"].values)
    longitude = np.asarray(dataset["longitude"].values)
    latitude = np.asarray(dataset["latitude"].values)
    reference_time = pd.Timestamp(dataset["forecast_reference_time"].values)

    cmap = ListedColormap(["#440154", "#21918C", "#FDE725"])
    norm = BoundaryNorm(DISPLAY_LEVELS, cmap.N)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=probability.shape[0],
        figsize=(24, 7.0),
        subplot_kw={"projection": NORWAY_MAP},
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.0,
        right=0.935,
        top=0.92,
        bottom=0.0,
        wspace=0.0,
    )

    image = None
    for index, axis in enumerate(axes[0]):
        image = render_triangulated_field(
            axis,
            np.where(
                probability[index] >= DISPLAY_THRESHOLD,
                probability[index],
                np.nan,
            ),
            longitude,
            latitude,
            cmap=cmap,
            norm=norm,
            white_background=True,
            shading="flat",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        start_lead = int(WINDOW_LEADS[index][0])
        end_lead = int(WINDOW_LEADS[index][-1])
        axis.text(
            0.5,
            0.975,
            f"+{start_lead} to +{end_lead} min",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=25,
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1},
        )

    if image is None:
        raise RuntimeError("No lightning panels were rendered.")
    figure.canvas.draw()
    panel_bottom = min(axis.get_position().y0 for axis in axes[0])
    panel_top = max(axis.get_position().y1 for axis in axes[0])
    figure.text(
        0.467,
        min(0.985, panel_top + 0.025),
        f"{reference_time:%Y-%m-%d %H:%M} UTC",
        ha="center",
        va="bottom",
        fontsize=25,
    )
    colorbar = figure.colorbar(
        image,
        cax=figure.add_axes([0.945, panel_bottom, 0.008, panel_top - panel_bottom]),
        orientation="vertical",
        ticks=[0.91, 0.97, 0.995],
    )
    colorbar.ax.set_yticklabels(["somewhat likely", "likely", "very likely"])
    colorbar.ax.tick_params(labelsize=18)
    colorbar.set_label(
        f"corrected probability ≥ {DISPLAY_THRESHOLD:.2f}",
        fontsize=18,
        labelpad=2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=160,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and plot selected BRIS lightning-probability windows."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("plot", type=Path)
    parser.add_argument("--grid-definition", type=Path, default=packaged_grid_definition())
    args = parser.parse_args()
    output = create_lightning_evolution(
        args.input,
        args.output,
        args.grid_definition,
    )
    plot_lightning_evolution(output, args.plot)
    print(args.output)
    print(args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
