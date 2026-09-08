#!/usr/bin/env python3
"""Plot FSS against lead time from a bris-inference NetCDF output file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="FSS NetCDF output file")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: <input stem>_leadtime.png)",
    )
    parser.add_argument(
        "--variable",
        help="FSS variable name, for example fss_tp; required when the file has multiple FSS variables",
    )
    parser.add_argument(
        "--max-leadtime",
        type=float,
        default=60,
        help="Largest lead time in hours to plot (default: 60)",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure")
    return parser.parse_args()


def get_fss_variable(dataset: xr.Dataset, variable: str | None) -> str:
    if variable is not None:
        if variable not in dataset.data_vars:
            raise ValueError(f"{variable!r} is not a variable in {dataset.encoding['source']}")
        return variable

    candidates = [name for name in dataset.data_vars if name.startswith("fss_")]
    if len(candidates) != 1:
        raise ValueError(
            "Specify --variable when the file does not contain exactly one "
            f"FSS variable; found {candidates}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_leadtime.png")

    with xr.open_dataset(args.input) as dataset:
        variable = get_fss_variable(dataset, args.variable)
        scores = dataset[variable]
        mean_dimensions = [
            dimension
            for dimension in ("time", "ensemble_member")
            if dimension in scores.dims
        ]
        scores = scores.mean(dim=mean_dimensions, skipna=True)
        scores = scores.where(scores["leadtime"] <= args.max_leadtime, drop=True)
        scores.load()

        if not np.isfinite(scores.values).any():
            raise ValueError(
                f"{variable} has no finite FSS values. This usually means that "
                "observations were unavailable for the requested valid times or "
                "that neither field exceeded the selected thresholds."
            )

        thresholds = scores["threshold"].values
        sizes = scores["neighbourhood_size"].values
        units = scores["threshold"].attrs.get("units", "")
        figure, axes = plt.subplots(
            len(thresholds),
            1,
            figsize=(9, max(3.5, 2.8 * len(thresholds))),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for axis, threshold in zip(axes, thresholds):
            threshold_scores = scores.sel(threshold=threshold)
            for size in sizes:
                values = threshold_scores.sel(neighbourhood_size=size)
                axis.plot(
                    values["leadtime"],
                    values,
                    marker="o",
                    markersize=3,
                    label=f"{size} grid cells",
                )
            axis.set_title(f"Threshold: {threshold:g} {units}".rstrip())
            axis.set_ylabel("Mean FSS")
            axis.set_ylim(0, 1.02)
            axis.grid(True, alpha=0.3)

        axes[0].legend(title="Neighbourhood", ncols=min(3, len(sizes)))
        axes[-1].set_xlabel("Lead time (hours)")
        figure.suptitle(
            f"{variable}: mean across {', '.join(mean_dimensions) or 'stored cases'}"
        )
        figure.savefig(output, dpi=180)

    print(f"Wrote {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
