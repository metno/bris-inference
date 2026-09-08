"""Fractions skill score output for regular gridded forecasts."""

import datetime
from math import gcd
from functools import reduce

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from bris import utils
from bris.conventions import cf
from bris.outputs import Output
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata
from bris.sources import Source


class FractionsSkillScore(Output):
    """Calculate the fractions skill score (FSS) against gridded observations.

    The observation source must provide one observation for every forecast grid
    point. ``neighbourhood_sizes`` are square window widths in grid cells, which
    is appropriate for the regular MEPS grid.
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
        obs_source: Source,
        thresholds: list[float],
        neighbourhood_sizes: list[int],
        units: str | None = None,
        remove_intermediate: bool = True,
    ) -> None:
        extra_variables = (
            ["ws"]
            if variable == "ws" and variable not in predict_metadata.variables
            else []
        )
        super().__init__(predict_metadata, extra_variables)
        if not self.pm.is_gridded:
            raise ValueError("FractionsSkillScore only supports gridded data")
        if variable not in self.pm.variables:
            raise ValueError(f"Variable {variable!r} is not available")
        if not thresholds:
            raise ValueError("At least one threshold must be provided")
        if not neighbourhood_sizes or any(size < 1 for size in neighbourhood_sizes):
            raise ValueError("neighbourhood_sizes must contain positive integers")

        self.variable = variable
        self.obs_source = obs_source
        self.thresholds = np.asarray(thresholds, dtype=np.float32)
        self.neighbourhood_sizes = np.asarray(neighbourhood_sizes, dtype=np.int32)
        self.units = units
        self.filename = filename
        self.remove_intermediate = remove_intermediate
        self.intermediate = Intermediate(self.pm, workdir)
        self.observation_indices = self._get_observation_indices()

    def _get_observation_indices(self) -> np.ndarray:
        """Map source locations to forecast locations, requiring a full grid."""
        locations = self.obs_source.locations
        if len(locations) != self.pm.num_points:
            raise ValueError(
                "FSS observations must cover every forecast grid point "
                f"({len(locations)} observations for {self.pm.num_points} points)"
            )

        observation_points = np.array([(loc.lat, loc.lon) for loc in locations])
        forecast_points = np.column_stack((self.pm.lats, self.pm.lons))
        distances, indices = cKDTree(observation_points).query(forecast_points)
        if np.any(distances > 1e-4) or len(np.unique(indices)) != len(indices):
            raise ValueError(
                "FSS observations must be on the same grid as the forecast"
            )
        return indices

    def _add_forecast(
        self, times: list, ensemble_member: int, pred: np.ndarray
    ) -> None:
        self.intermediate._add_forecast(times, ensemble_member, pred)

    @staticmethod
    def _fraction(field: np.ndarray, size: int) -> np.ndarray:
        """Return neighbourhood event fractions, preserving missing values."""
        valid = np.isfinite(field)
        counts = uniform_filter(valid.astype(np.float32), size=size, mode="constant")
        fractions = uniform_filter(
            np.where(valid, field, 0.0).astype(np.float32), size=size, mode="constant"
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, fractions / counts, np.nan)

    @classmethod
    def calculate_fss(
        cls, forecast: np.ndarray, observation: np.ndarray, threshold: float, size: int
    ) -> float:
        """Calculate FSS for one forecast/observation field pair."""
        forecast_fraction = cls._fraction(
            np.where(np.isfinite(forecast), forecast >= threshold, np.nan), size
        )
        observation_fraction = cls._fraction(
            np.where(np.isfinite(observation), observation >= threshold, np.nan), size
        )
        valid = np.isfinite(forecast_fraction) & np.isfinite(observation_fraction)
        if not np.any(valid):
            return np.nan

        numerator = np.sum((forecast_fraction[valid] - observation_fraction[valid]) ** 2)
        denominator = np.sum(
            forecast_fraction[valid] ** 2 + observation_fraction[valid] ** 2
        )
        return np.nan if denominator == 0 else float(1 - numerator / denominator)

    def _observation_frequency(self, frts: list[np.datetime64]) -> int:
        valid_times = sorted(
            {
                int(utils.datetime_to_unixtime(frt) + leadtime)
                for frt in frts
                for leadtime in self.pm.leadtimes
            }
        )
        differences = np.diff(valid_times)
        differences = differences[differences > 0]
        if len(differences):
            return reduce(gcd, differences.tolist())
        positive_leadtimes = self.pm.leadtimes[self.pm.leadtimes > 0]
        return int(positive_leadtimes.min()) if len(positive_leadtimes) else 3600

    def finalize(self) -> None:
        frts = self.intermediate.get_forecast_reference_times()
        if not frts:
            return

        valid_times = np.array(
            [
                utils.datetime_to_unixtime(frt) + leadtime
                for frt in frts
                for leadtime in self.pm.leadtimes
            ]
        )
        observation_frequency = self._observation_frequency(frts)
        observation_start = int(valid_times.min())
        observation_end = int(valid_times.max())
        observations = self.obs_source.get(
            self.variable,
            observation_start,
            (
                observation_end
                if observation_end > observation_start
                else observation_end + observation_frequency
            ),
            observation_frequency,
        )
        data = np.full(
            (
                len(frts),
                self.pm.num_leadtimes,
                len(self.thresholds),
                len(self.neighbourhood_sizes),
                self.pm.num_members,
            ),
            np.nan,
            dtype=np.float32,
        )
        variable_index = self.pm.variables.index(self.variable)
        ny, nx = self.pm.field_shape

        for frt_index, frt in enumerate(frts):
            forecast = self.intermediate.get_forecast(frt)
            for leadtime_index, leadtime in enumerate(self.pm.leadtimes):
                valid_time = int(utils.datetime_to_unixtime(frt) + leadtime)
                observation = observations.get_data(self.variable, valid_time)
                if observation is None:
                    continue
                observation = observation[self.observation_indices].reshape(ny, nx)
                for member in range(self.pm.num_members):
                    field = forecast[leadtime_index, :, variable_index, member].reshape(ny, nx)
                    for threshold_index, threshold in enumerate(self.thresholds):
                        for size_index, size in enumerate(self.neighbourhood_sizes):
                            data[frt_index, leadtime_index, threshold_index, size_index, member] = (
                                self.calculate_fss(field, observation, threshold, int(size))
                            )

        coordinates = {
            "time": (
                ["time"],
                utils.datetime_to_unixtime(frts).astype(np.double),
                cf.get_attributes("time"),
            ),
            "leadtime": (
                ["leadtime"],
                self.pm.leadtimes.astype(np.float32) / 3600,
                {"units": "hour"},
            ),
            "threshold": (["threshold"], self.thresholds, {"units": self.units or "1"}),
            "neighbourhood_size": (
                ["neighbourhood_size"],
                self.neighbourhood_sizes,
                {"units": "grid_cells"},
            ),
            "ensemble_member": (["ensemble_member"], np.arange(self.pm.num_members)),
        }
        dataset = xr.Dataset(coords=coordinates)
        dimensions = [
            "time",
            "leadtime",
            "threshold",
            "neighbourhood_size",
            "ensemble_member",
        ]
        if self.pm.num_members == 1:
            data = data[..., 0]
            dimensions.pop()
        dataset[f"fss_{self.variable}"] = (dimensions, data)
        datestr = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S +00:00")
        dataset.attrs["history"] = f"{datestr} Created by bris-inference"
        dataset.attrs["Conventions"] = "CF-1.6"

        utils.create_directory(self.filename)
        dataset.to_netcdf(self.filename, mode="w", engine="netcdf4")
        if self.remove_intermediate:
            self.intermediate.cleanup()