import time as pytime

import numpy as np

import bris.units
from bris import utils
from bris.outputs import Output
from bris.outputs.intermediate import IntermediateEnsembleAccumulator
from bris.outputs.netcdf import Netcdf
from bris.predict_metadata import PredictMetadata


class EnsembleStatistics(Output):
    """Writes statistics across the ensemble (mean, standard deviation, min, max) to NetCDF,
    without ever storing the individual ensemble members.

    For each batch (forecast reference time), every member contributes its prediction, its
    squared prediction, and its min/max. These are reduced across the members that run in
    parallel by the writer, and accumulated on disk (see IntermediateEnsembleAccumulator) across
    members that run in sequence. On finalize, the statistics are computed from the accumulated
    quantities and written using the same conventions as the netcdf output.

    Compared to the netcdf output with an ensemble, the intermediate storage is independent of
    the number of members: at most 4 arrays (each the size of one member) per forecast reference
    time, and finalize never needs to hold the full ensemble in memory.

    Example config:
        - ensemble_statistics:
            filename_pattern: /path/ens_{statistic}_%Y%m%dT%HZ.nc
            variables: [2t, 10u, 10v]
            extra_variables: [ws]
            statistics: [mean, std]
            domain_name: meps
    """

    reduce_across_members = True
    ensemble_reductions = {"sum": "sum", "sum_sq": "sum", "min": "min", "max": "max"}
    valid_statistics = ("mean", "std", "min", "max")
    statistic_token = "{statistic}"

    # CF cell_methods for each statistic
    cell_methods = {
        "mean": "realization: mean",
        "std": "realization: standard_deviation",
        "min": "realization: minimum",
        "max": "realization: maximum",
    }

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename_pattern: str,
        variables: list | None = None,
        extra_variables: list | None = None,
        statistics: list | None = None,
        ddof: int = 0,
        interp_res=None,
        latrange=None,
        lonrange=None,
        proj4_str=None,
        domain_name=None,
        mask_file: str = "",
        mask_field=None,
        global_attributes=None,
        remove_intermediate: bool = True,
        compression: bool = False,
    ):
        """
        Args:
            filename_pattern: Save statistics to this filename after time tokens are expanded.
                Must contain the token {statistic} (replaced by e.g. "mean") when more than one
                statistic is requested.
            variables: If None, compute statistics for all variables
            extra_variables: Derive these extra variables (e.g. ws) before computing statistics
            statistics: Which of "mean", "std", "min", "max" to write. Default: mean and std
            ddof: Delta degrees of freedom used for std (0: population, 1: sample)
            interp_res, latrange, lonrange, proj4_str, domain_name, mask_file, mask_field,
                global_attributes, compression: See the netcdf output
        """
        super().__init__(predict_metadata, extra_variables)

        if statistics is None:
            statistics = ["mean", "std"]
        statistics = list(statistics)
        if len(statistics) == 0:
            raise ValueError("At least one statistic must be requested")
        for statistic in statistics:
            if statistic not in self.valid_statistics:
                raise ValueError(
                    f"Invalid statistic '{statistic}'. Must be one of {self.valid_statistics}"
                )
        if len(statistics) > 1 and self.statistic_token not in filename_pattern:
            raise ValueError(
                f"filename_pattern must contain {self.statistic_token} when writing more than one statistic"
            )
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")

        self.statistics = statistics
        self.ddof = ddof
        self.filename_pattern = filename_pattern
        self.remove_intermediate = remove_intermediate

        # Only accumulate the variables that will be written
        if variables is None:
            self.extract_variables = list(self.pm.variables)
        else:
            self.extract_variables = list(variables)
            for name in self.extra_variables:
                if name not in self.extract_variables:
                    self.extract_variables += [name]
        for variable in self.extract_variables:
            if variable not in self.pm.variables:
                raise ValueError(
                    f"Variable '{variable}' is not available in the predictions ({self.pm.variables})"
                )
        self.variable_indices = [
            self.pm.variables.index(v) for v in self.extract_variables
        ]

        self.intermediate = IntermediateEnsembleAccumulator(
            workdir, self.ensemble_reductions
        )

        # The netcdf writers see the statistics as a deterministic (single member) forecast
        # containing only the extracted variables
        writer_pm = PredictMetadata(
            self.extract_variables,
            self.pm.lats,
            self.pm.lons,
            self.pm.altitudes,
            self.pm.leadtimes,
            1,
            self.pm.field_shape,
        )
        self.writers = {}
        for statistic in self.statistics:
            pattern = filename_pattern.replace(self.statistic_token, statistic)
            self.writers[statistic] = _StatisticNetcdf(
                statistic,
                writer_pm,
                workdir,
                pattern,
                variables=self.extract_variables,
                interp_res=interp_res,
                latrange=latrange,
                lonrange=lonrange,
                proj4_str=proj4_str,
                domain_name=domain_name,
                mask_file=mask_file,
                mask_field=mask_field,
                global_attributes=global_attributes,
                compression=compression,
            )

    def _member_contribution(
        self, times: list, ensemble_member: int, pred: np.ndarray
    ) -> dict[str, np.ndarray]:
        # Accumulate in double precision, since sum_sq - sum^2/n loses precision in float32 for
        # variables with a large mean (e.g. temperature in K)
        curr = pred[..., self.variable_indices].astype(np.float64)
        return {
            "sum": curr,
            "sum_sq": curr**2,
            "min": curr,
            "max": curr,
        }

    def _add_forecast(
        self, times: list, ensemble_member: int, pred: np.ndarray
    ) -> None:
        """Fallback for callers that do not reduce across members (e.g. when members run in
        sequence on a single process): accumulate this member directly"""
        contributions = self._member_contribution(times, ensemble_member, pred)
        self.add_reduced_forecast(times, contributions, 1)

    def add_reduced_forecast(
        self, times: list, contributions: dict[str, np.ndarray], num_members: int
    ) -> None:
        t0 = pytime.perf_counter()
        self.intermediate.accumulate(times[0], contributions, num_members)
        utils.LOGGER.debug(
            f"EnsembleStatistics.add_reduced_forecast for {times[0]} ({num_members} members) in {pytime.perf_counter() - t0:.1f}s"
        )

    def compute_statistics(
        self, accumulated: dict[str, np.ndarray], num_members: int
    ) -> dict[str, np.ndarray]:
        """Computes the requested statistics from the accumulated quantities

        Args:
            accumulated: dict with "sum", "sum_sq", "min", "max", each with dimensions
                (leadtime, location, variable)
            num_members: Number of members that have been accumulated

        Returns:
            dict statistic -> array with dimensions (leadtime, location, variable)
        """
        assert num_members > 0
        ret = {}
        mean = accumulated["sum"] / num_members
        for statistic in self.statistics:
            if statistic == "mean":
                ret[statistic] = mean
            elif statistic == "std":
                if num_members - self.ddof <= 0:
                    ret[statistic] = np.full(mean.shape, np.nan)
                else:
                    # Sum of squared deviations from the mean
                    ss = accumulated["sum_sq"] - num_members * mean**2
                    ss = np.maximum(ss, 0)  # Guard against round-off
                    ret[statistic] = np.sqrt(ss / (num_members - self.ddof))
            elif statistic in ["min", "max"]:
                ret[statistic] = accumulated[statistic]
            else:
                raise ValueError(f"Unknown statistic '{statistic}'")
        return ret

    def finalize(self) -> None:
        t0 = pytime.perf_counter()
        frts = self.intermediate.get_forecast_reference_times()
        if len(frts) == 0:
            utils.LOGGER.warning(
                "EnsembleStatistics.finalize: No forecasts have been accumulated"
            )

        for frt in frts:
            num_members = self.intermediate.get_num_members(frt)
            if num_members != self.pm.num_members:
                utils.LOGGER.warning(
                    f"EnsembleStatistics: {num_members} members accumulated for {frt}, expected {self.pm.num_members}"
                )
            accumulated = self.intermediate.get(frt)
            statistics = self.compute_statistics(accumulated, num_members)

            frt_ut = int(utils.datetime_to_unixtime(frt))
            lead_times = [frt + lt for lt in self.pm.leadtimes]
            for statistic, values in statistics.items():
                writer = self.writers[statistic]
                filename = writer.get_filename(frt_ut)
                # Netcdf.write expects (leadtime, location, variable, member)
                writer.write(filename, lead_times, values.astype(np.float32)[..., None])

        if self.remove_intermediate:
            self.intermediate.cleanup()
        utils.LOGGER.debug(
            f"EnsembleStatistics.finalize: {pytime.perf_counter() - t0:.1f}s"
        )


class _StatisticNetcdf(Netcdf):
    """Netcdf writer for a single ensemble statistic. Standard deviations are a difference
    quantity, so only the scale part of unit conversions applies to them (e.g. K -> C must not
    subtract 273.15)."""

    def __init__(self, statistic: str, *args, **kwargs):
        self.statistic = statistic
        super().__init__(*args, **kwargs)

    def _convert_units(
        self, ar: np.ndarray, from_units: str, to_units: str
    ) -> np.ndarray:
        if self.statistic == "std":
            zero, one = bris.units.convert(np.array([0.0, 1.0]), from_units, to_units)[
                0
            ]
            return ar * (one - zero)
        return super()._convert_units(ar, from_units, to_units)

    def _extra_variable_attrs(self) -> dict:
        return {"cell_methods": EnsembleStatistics.cell_methods[self.statistic]}
