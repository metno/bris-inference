import cfunits
import gridpp
import numpy as np
import scipy.interpolate
from scipy.spatial import Delaunay
import xarray as xr
from bris import utils
from bris.conventions import anemoi as anemoi_conventions
from bris.conventions import cf
from bris.outputs import Output
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata


class Verif(Output):
    """This class writes verification files in Verif format. See github.com/WFRT/verif."""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable=None,
        obs_sources=list,
        units=None,
        thresholds=[],
        quantile_levels=[],
        consensus_method="control",
        elev_gradient=None,
    ):
        """
        Args:
            units: Units to put in Verif file. Should be the same as the observations
            elev_gradient: Apply this elevation gradient when downscaling from grid to point (e.g.
                -0.0065 for temperature)
        """
        extra_variables = list()
        if variable not in predict_metadata.variables:
            extra_variables += [variable]
        super().__init__(predict_metadata, extra_variables)

        self.filename = filename
        self.fcst = dict()
        self.variable = variable
        self.units = units
        self.thresholds = thresholds
        self.quantile_levels = quantile_levels
        self.consensus_method = consensus_method
        self.elev_gradient = elev_gradient

        for level in self.quantile_levels:
            assert level >= 0 and level <= 1, f"level={level} must be between 0 and 1"

        if self._is_gridded_input:
            self.igrid = gridpp.Grid(
                self.pm.grid_lats, self.pm.grid_lons, self.pm.grid_altitudes
            )

        self.ipoints = gridpp.Points(self.pm.lats, self.pm.lons, self.pm.altitudes)
        self.ipoints_tuple = np.column_stack((self.pm.lats, self.pm.lons))
        self.ialtitudes = self.pm.altitudes

        obs_lats = list()
        obs_lons = list()
        obs_altitudes = list()
        obs_ids = list()

        self.obs_sources = obs_sources

        for obs_source in obs_sources:
            obs_lats += [loc.lat for loc in obs_source.locations]
            obs_lons += [loc.lon for loc in obs_source.locations]
            obs_altitudes += [loc.elev for loc in obs_source.locations]
            obs_ids += [loc.id for loc in obs_source.locations]

        self.obs_lats = np.array(obs_lats, np.float32)
        self.obs_lons = np.array(obs_lons, np.float32)
        self.obs_altitudes = np.array(obs_altitudes, np.float32)
        self.obs_ids = np.array(obs_ids, np.int32)
        self.opoints = gridpp.Points(self.obs_lats, self.obs_lons, self.obs_altitudes)
        self.opoints_tuple = np.column_stack((self.obs_lats, self.obs_lons))

        
        if not self._is_gridded_input:
            self.triangulation = Delaunay(self.ipoints_tuple)
        else:
            self.triangulation = None
        
        # The intermediate will only store the final output locations
        intermediate_pm = PredictMetadata(
            [variable],
            self.obs_lats,
            self.obs_lons,
            self.obs_altitudes,
            predict_metadata.leadtimes,
            predict_metadata.num_members,
        )
        self.intermediate = Intermediate(intermediate_pm, workdir)

    def _add_forecast(self, times: list, ensemble_member: int, pred: np.array):
        """Add forecasts to this object. Will be written when .write() is called

        Args:
            times: List of np.datetime64 objects
            pred: 3D array of forecasts with dimensions (time, points, variables)
        """
        print("### Adding forecast", times[0])

        Iv = self.pm.variables.index(self.variable)
        if self._is_gridded_input:
            pred = self.reshape_pred(pred)
            pred = pred[..., Iv]  # Extract single variable
            interpolated_pred = gridpp.bilinear(self.igrid, self.opoints, pred)

            if self.elev_gradient is not None:
                interpolated_altitudes = gridpp.bilinear(
                    self.igrid, self.opoints, self.igrid.get_elevs()
                )
                daltitude = self.opoints.get_elevs() - interpolated_altitudes
                interpolated_pred += self.elev_gradient * daltitude
            interpolated_pred = interpolated_pred[
                :, :, None
            ]  # Add in variable dimension
        else:
            pred = pred[..., [Iv]]

            altitude_correction = None
            if self.elev_gradient is not None:
                interpolator = scipy.interpolate.LinearNDInterpolator(
                    self.triangulation, self.ialtitudes
                )
                interpolated_altitudes = interpolator(self.opoints_tuple)
                altitude_correction = self.opoints.get_elevs() - interpolated_altitudes

            num_leadtimes = pred.shape[0]
            num_points = self.opoints.size()

            interpolated_pred = np.nan * np.zeros(
                [num_leadtimes, num_points, 1], np.float32
            )
            for lt in range(num_leadtimes):
                interpolator = scipy.interpolate.LinearNDInterpolator(
                    self.triangulation, pred[lt, :, 0]
                )
                interpolated_pred[lt, :, 0] = interpolator(self.opoints_tuple)
                if altitude_correction is not None:
                    interpolated_pred[lt, :, 0] += (
                        self.elev_gradient * altitude_correction
                    )

            # Much faster, but not a linear interpolator
            # interpolated_pred = gridpp.nearest(self.ipoints, self.opoints, pred[..., 0])
            # interpolated_pred = interpolated_pred[:, :, None]

        anemoi_units = anemoi_conventions.get_units(self.variable)

        if self.units is None:
            # Update the units so they can be written out
            self.units = anemoi_units
        elif anemoi_units is not None and self.units != anemoi_units:
            to_units = cfunits.Units(self.units)
            from_units = cfunits.Units(anemoi_units)
            cfunits.Units.conform(interpolated_pred, from_units, to_units, inplace=True)

        self.intermediate.add_forecast(times, ensemble_member, interpolated_pred)

    @property
    def _is_gridded_input(self):
        return self.pm.is_gridded

    @property
    def _num_locations(self):
        return self.opoints.size()

    @property
    def num_members(self):
        return self.intermediate.num_members

    def finalize(self):
        """Write forecasts and observations to file"""

        coords = dict()
        coords["time"] = (["time"], [], cf.get_attributes("time"))
        coords["leadtime"] = (
            ["leadtime"],
            self.intermediate.pm.leadtimes.astype(np.float32),
            {"units": "hour"},
        )
        coords["location"] = (["location"], self.obs_ids)
        coords["lat"] = (
            ["location"],
            self.obs_lats,
            cf.get_attributes("latitude"),
        )
        coords["lon"] = (
            ["location"],
            self.obs_lons,
            cf.get_attributes("longitude"),
        )
        coords["altitude"] = (
            ["location"],
            self.obs_altitudes,
            cf.get_attributes("surface_altitude"),
        )
        """
        coords["ensemble_member"] = (
                ["ensemble_member"],
                self.ensemble_members,
                cf.get_attributes("ensemble_member"),
        )
        """
        if self.num_members > 1:
            if len(self.thresholds) > 0:
                coords["threshold"] = (
                    ["threshold"],
                    self.thresholds,
                )
            if len(self.quantile_levels) > 0:
                coords["quantile"] = (
                    ["quantile"],
                    self.quantile_levels,
                )
        self.ds = xr.Dataset(coords=coords)

        frts = self.intermediate.get_forecast_reference_times()
        self.ds["time"] = utils.datetime_to_unixtime(frts).astype(np.double)

        # Load forecasts
        fcst = np.nan * np.zeros(
            [
                len(frts),
                self.intermediate.pm.num_leadtimes,
                self.intermediate.pm.num_points,
            ],
            np.float32,
        )
        for i, frt in enumerate(frts):
            curr = self.intermediate.get_forecast(frt)[..., 0, :]
            fcst[i, ...] = self.compute_consensus(curr)

        self.ds["fcst"] = (["time", "leadtime", "location"], fcst)

        # Load threshold forecasts
        if len(self.thresholds) > 0 and self.num_members > 1:
            cdf = np.nan * np.zeros(
                [
                    len(frts),
                    self.intermediate.pm.num_leadtimes,
                    self.intermediate.pm.num_points,
                    len(self.thresholds),
                ],
                np.float32,
            )
            for i, frt in enumerate(frts):
                curr = self.intermediate.get_forecast(frt)[..., 0, :]
                for t, threshold in enumerate(self.thresholds):
                    cdf[i, ..., t] = self.compute_threshold_prob(curr, threshold)

            self.ds["cdf"] = (["time", "leadtime", "location", "threshold"], cdf)

        # Load quantile forecasts
        if len(self.quantile_levels) > 0 and self.num_members > 1:
            x = np.nan * np.zeros(
                [
                    len(frts),
                    self.intermediate.pm.num_leadtimes,
                    self.intermediate.pm.num_points,
                    len(self.quantile_levels),
                ],
                np.float32,
            )
            for i, frt in enumerate(frts):
                curr = self.intermediate.get_forecast(frt)[
                    :, :, 0, :
                ]  # Remove variable dimension
                for t, quantile_level in enumerate(self.quantile_levels):
                    x[i, ..., t] = self.compute_quantile(curr, quantile_level)

            self.ds["x"] = (["time", "leadtime", "location", "quantile"], x)

        # Find which valid times we need observations for
        frts_ut = utils.datetime_to_unixtime(frts)
        a, b = np.meshgrid(frts_ut, np.array(self.intermediate.pm.leadtimes))
        valid_times = a + b
        valid_times = valid_times.transpose()
        if len(valid_times) == 0:
            print("### No valid times")
            return

        # valid_times = np.sort(np.unique(valid_times.flatten()))
        unique_valid_times = np.sort(np.unique(valid_times.flatten()))

        start_time = int(np.min(unique_valid_times))
        end_time = int(np.max(unique_valid_times))

        if start_time == end_time:
            # Any number will do
            frequency = 3600
        else:
            frequency = int(np.min(np.diff(unique_valid_times)))

        # Fill in retrieved observations into our obs array.
        obs = np.nan * np.zeros(
            [
                len(frts),
                self.intermediate.pm.num_leadtimes,
                self.intermediate.pm.num_points,
            ],
            np.float32,
        )
        count = 0
        for obs_source in self.obs_sources:
            curr = obs_source.get(self.variable, start_time, end_time, frequency)
            from_units = (
                cfunits.Units(obs_source.units)
                if obs_source.units is not None
                else None
            )
            to_units = cfunits.Units(self.units) if self.units is not None else None
            for t, valid_time in enumerate(unique_valid_times):
                Itimes, Ileadtimes = np.where(valid_times == valid_time)
                data = curr.get_data(self.variable, valid_time)
                if data is not None:
                    if None not in [obs_source.units, self.units]:
                        cfunits.Units.conform(data, from_units, to_units, inplace=True)

                    Iout = range(count, len(obs_source.locations) + count)
                    for i in range(len(Itimes)):
                        # Copy observation into all times/leadtimes that matches this valid time
                        obs[Itimes[i], Ileadtimes[i], Iout] = data
            count += len(obs_source.locations)

        self.ds["obs"] = (["time", "leadtime", "location"], obs)

        self.ds.attrs["units"] = self.units
        self.ds.attrs["verif_version"] = "1.0.0"
        self.ds.attrs["standard_name"] = cf.get_metadata(self.variable)["cfname"]

        utils.create_directory(self.filename)
        self.ds.to_netcdf(self.filename, mode="w", engine="netcdf4")

    def compute_consensus(self, pred):
        assert len(pred.shape) == 3, pred.shape

        if self.consensus_method == "control":
            return pred[..., 0]
        elif self.consensus_method == "mean":
            return np.mean(pred, axis=-1)
        else:
            raise NotImplementedError(
                f"Unknown consensus method {self.consensus_method}"
            )

    def compute_quantile(self, ar, level, fair=True):
        """Extracts a quantile from an array

        Args:
            ar: N-D numpy array, where last dimension is ensemble
            level: a number between 0 and 1
            fair: Adjust for sampling error

        Returns:
            (N-1)-D numpy array with quantiles
        """
        assert level >= 0 and level <= 1, f"level={level} must be between 0 and 1"

        if fair:
            # What quantile level do we assign the lowest member?
            # For 10 members we want 0.05, 0.15, ..., 0.95
            num_members = ar.shape[-1]
            lower = 0.5 * 1 / num_members
            upper = 1 - lower
            percentile = (level - lower) / (upper - lower) * 100
            percentile = max(min(percentile, 100), 0)
        else:
            percentile = level

        q = np.percentile(ar, percentile, axis=-1)
        return q

    def compute_threshold_prob(self, ar, threshold, fair=True):
        """Compute probability less than a threshold for an ensemble
        Args:
            ar: N-D numpy array, where last dimensions is ensmelbe
            threshold: Threshold to compute fraction of members that are less than this
            fair: Adjust for sampling error

        Returns:
            (N-1)-D numpy array of probabilities
        """
        p = np.mean(ar <= threshold, axis=-1)
        if fair:
            num_members = ar.shape[-1]
            lower = 0.5 * 1 / num_members
            upper = 1 - lower

            p *= (upper - lower) + lower
            p[p > 1] = 1
            p[p < 0] = 0
        return p
