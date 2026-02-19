import gridpp
import numpy as np
import scipy.interpolate
import xarray as xr
from scipy.spatial import Delaunay, cKDTree

import bris.units
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
        variable: str = None,
        variable_type: str = None,
        obs_sources: list = None,
        units: str = None,
        thresholds: list = None,
        quantile_levels: list = None,
        consensus_method: str = "control",
        elev_gradient: float = None,
        max_distance: float = None,
        fair_threshold: bool = True,
        fair_quantile: bool = True,
        fair_crps: bool = True,
        remove_intermediate: bool = True,
        compression: bool = False,
    ) -> None:
        """
        Args:
            units: Units to put in Verif file. Should be the same as the observations
            elev_gradient: Apply this elevation gradient when downscaling from grid to point (e.g.
                -0.0065 for temperature)
            variable_type: 'logit' or 'threshold_probability'
            consensus_method: Whether to use the control member ('control') or ensemble mean
            ('mean') as the forecast prediction
            max_distance: Maximum distance between input and output points
            fair_quantile: Whether or not quantile is adjusted for sampling error
            fair_threshold: Whether or not threshold is adjusted for sampling error
            fair_crps: Whether or not to adjust crps for ensemble size
            compression: If True, write compressed output files
        """
        if quantile_levels is None:
            quantile_levels = []
        for level in quantile_levels:
            assert 0 <= level <= 1, f"level={level} must be between 0 and 1"

        extra_variables = []
        if variable not in predict_metadata.variables:
            extra_variables += [variable]

        super().__init__(predict_metadata, extra_variables)

        self.filename = filename
        self.fcst = {}
        self.variable = variable
        self.variable_type = variable_type
        self.obs_sources = obs_sources
        self.units = units
        self.thresholds = thresholds if thresholds is not None else []
        self.quantile_levels = quantile_levels
        self.consensus_method = consensus_method
        self.elev_gradient = elev_gradient
        self.max_distance = max_distance
        self.fair_quantile = fair_quantile
        self.fair_threshold = fair_threshold
        self.fair_crps = fair_crps

        if self.pm.altitudes is None and elev_gradient is not None:
            raise ValueError(
                "Cannot do elevation gradient since input field does not have altitude"
            )

        if self._is_gridded_input:
            if self.pm.altitudes is not None:
                self.igrid = gridpp.Grid(
                    self.pm.grid_lats, self.pm.grid_lons, self.pm.grid_altitudes
                )
            else:
                self.igrid = gridpp.Grid(self.pm.grid_lats, self.pm.grid_lons)

        self.ipoints_array = np.column_stack((self.pm.lats, self.pm.lons))
        self.ialtitudes = self.pm.altitudes

        self.ipoints, self.opoints, self.obs_ids = self.get_points(
            self.pm, obs_sources, self.max_distance
        )
        self.opoints_array = np.column_stack(
            (self.opoints.get_lats(), self.opoints.get_lons())
        )
        _tree = cKDTree(self.ipoints_array)
        _, _indices = _tree.query(self.opoints_array, distance_upper_bound=1e-4)
        _valid_matches = _indices < len(self.ipoints_array)
        _matching_indices = _indices[_valid_matches]

        self.interpolate = True
        if len(_matching_indices) == len(self.opoints_array):
            self.verif_indices = _matching_indices
            self.interpolate = False

        self.triangulation = self.ipoints_array
        if (
            not self._is_gridded_input
            and self.ipoints_array.shape[0] > 3
            and self.interpolate
        ):
            # This speeds up interpolation from irregular points to observation points
            # but Delaunay needs enough points for this to work
            self.triangulation = Delaunay(self.ipoints_array)

        # The intermediate will only store the final output locations
        intermediate_pm = PredictMetadata(
            [variable],
            self.opoints.get_lats(),
            self.opoints.get_lons(),
            self.opoints.get_elevs(),
            predict_metadata.leadtimes,
            predict_metadata.num_members,
        )
        self.intermediate = Intermediate(intermediate_pm, workdir)
        self.remove_intermediate = remove_intermediate
        self.compression = compression

    def _add_forecast(self, times: list, ensemble_member: int, pred: np.array) -> None:
        """Add forecasts to this object. Will be written when .write() is called

        Args:
            times: List of np.datetime64 objects
            pred: 3D array of forecasts with dimensions (time, points, variables)
        """

        Iv = self.pm.variables.index(self.variable)
        if not self.interpolate:
            interpolated_pred = pred[:, self.verif_indices, Iv][:, :, np.newaxis]
        else:
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
                    interpolated_altitudes = interpolator(self.opoints_array)
                    altitude_correction = (
                        self.opoints.get_elevs() - interpolated_altitudes
                    )

                num_leadtimes = pred.shape[0]
                num_points = self.opoints.size()

                interpolated_pred = np.nan * np.zeros(
                    [num_leadtimes, num_points, 1], np.float32
                )
                for lt in range(num_leadtimes):
                    interpolator = scipy.interpolate.LinearNDInterpolator(
                        self.triangulation, pred[lt, :, 0]
                    )
                    interpolated_pred[lt, :, 0] = interpolator(self.opoints_array)
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
            to_units = self.units
            from_units = anemoi_units
            bris.units.convert(interpolated_pred, from_units, to_units, inplace=True)

        self.intermediate.add_forecast(times, ensemble_member, interpolated_pred)

    @property
    def _is_gridded_input(self) -> bool:
        return self.pm.is_gridded

    @property
    def _num_locations(self) -> int:
        return self.opoints.size()

    @property
    def num_members(self) -> int:
        return self.pm.num_members

    @staticmethod
    def create_nan_array(shape, dtype=np.float32) -> np.ndarray:
        """Create numpy array of NaNs to be overwritten"""
        return np.full(shape, np.nan, dtype)

    def finalize(self) -> None:
        """Write forecasts and observations to file"""

        frts = self.intermediate.get_forecast_reference_times()
        frts_unix = utils.datetime_to_unixtime(frts).astype(np.double)

        coords = {}
        coords["time"] = (["time"], frts_unix, cf.get_attributes("time"))
        coords["leadtime"] = (
            ["leadtime"],
            self.intermediate.pm.leadtimes.astype(np.float32) / 3600,
            {"units": "hour"},
        )
        assert len(self.obs_ids) == len(self.opoints.get_lats()), (
            len(self.obs_ids),
            len(self.opoints.get_lats()),
        )
        coords["location"] = (["location"], self.obs_ids)
        coords["lat"] = (
            ["location"],
            self.opoints.get_lats(),
            cf.get_attributes("latitude"),
        )
        coords["lon"] = (
            ["location"],
            self.opoints.get_lons(),
            cf.get_attributes("longitude"),
        )
        coords["altitude"] = (
            ["location"],
            self.opoints.get_elevs(),
            cf.get_attributes("surface_altitude"),
        )
        if self.num_members > 1:
            coords["ensemble_member"] = (
                ["ensemble_member"],
                np.arange(self.num_members),
            )
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
        if self.variable_type == "logit":
            # Add threshold variable
            coords["threshold"] = (["threshold"], self.thresholds)

        self.ds = xr.Dataset(coords=coords)

        # Load forecasts
        fcst_shape = (
            len(frts),
            self.intermediate.pm.num_leadtimes,
            self.intermediate.pm.num_points,
        )
        fcst = self.create_nan_array(fcst_shape)
        ens_mean = self.create_nan_array(fcst_shape)
        ens_var = self.create_nan_array(fcst_shape)
        ens = self.create_nan_array(fcst_shape + (self.num_members,))
        for i, frt in enumerate(frts):
            curr = self.intermediate.get_forecast(frt)[..., 0, :]
            fcst[i, ...] = self.compute_consensus(curr)
            if self.num_members > 1:
                ens[i, ...] = curr
                ens_mean[i, ...] = curr.mean(axis=-1)
                ens_var[i, ...] = curr.var(axis=-1)

        if self.variable_type in ["logit", "threshold_probability"]:
            cdf = np.copy(fcst)
            # Apply sigmoid activation function to logits
            if self.variable_type == "logit":
                cdf = 1 / (1 + np.exp(-cdf))

            # Add more axes
            cdf = np.expand_dims(cdf, axis=-1)
            cdf = np.tile(cdf, (1, 1, 1, len(self.thresholds)))
            self.ds["cdf"] = (["time", "leadtime", "location", "threshold"], 1 - cdf)

        else:
            self.ds["fcst"] = (["time", "leadtime", "location"], fcst)

            if self.num_members > 1:
                self.ds["ensemble"] = (
                    ["time", "leadtime", "location", "ensemble_member"],
                    ens,
                )
                self.ds["ensemble_mean"] = (["time", "leadtime", "location"], ens_mean)
                self.ds["ensemble_variance"] = (["time", "leadtime", "location"], ens_var)
            # Load threshold forecasts
            if len(self.thresholds) > 0 and self.num_members > 1:
                cdf = self.create_nan_array(fcst_shape + (len(self.thresholds),))
                for i, frt in enumerate(frts):
                    curr = self.intermediate.get_forecast(frt)[..., 0, :]
                    for t, threshold in enumerate(self.thresholds):
                        cdf[i, ..., t] = self.compute_threshold_prob(
                            curr, threshold, self.fair_threshold
                        )

                        self.ds["cdf"] = (
                            ["time", "leadtime", "location", "threshold"],
                            cdf,
                        )

            # Load quantile forecasts
            if len(self.quantile_levels) > 0 and self.num_members > 1:
                x = self.create_nan_array(fcst_shape + (len(self.quantile_levels),))
                for i, frt in enumerate(frts):
                    curr = self.intermediate.get_forecast(frt)[
                        :, :, 0, :
                    ]  # Remove variable dimension
                    for t, quantile_level in enumerate(self.quantile_levels):
                        x[i, ..., t] = self.compute_quantile(
                            curr, quantile_level, self.fair_quantile
                        )

                        self.ds["x"] = (["time", "leadtime", "location", "quantile"], x)

        # Find which valid times we need observations for
        frts_ut = utils.datetime_to_unixtime(frts)
        a, b = np.meshgrid(frts_ut, np.array(self.intermediate.pm.leadtimes))
        valid_times = a + b
        valid_times = valid_times.transpose()
        if len(valid_times) == 0:
            utils.LOGGER.warning("Could not finalize verif, no valid times")
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
        obs = self.create_nan_array(fcst_shape)
        count = 0
        for obs_source in self.obs_sources:
            curr = obs_source.get(self.variable, start_time, end_time, frequency)
            from_units = obs_source.units
            to_units = self.units
            for _t, valid_time in enumerate(unique_valid_times):
                Itimes, Ileadtimes = np.where(valid_times == valid_time)
                data = curr.get_data(self.variable, valid_time)
                if data is not None:
                    if None not in [obs_source.units, self.units]:
                        bris.units.convert(data, from_units, to_units, inplace=True)

                    Iout = range(count, len(obs_source.locations) + count)
                    for i in range(len(Itimes)):
                        # Copy observation into all times/leadtimes that matches this valid time
                        obs[Itimes[i], Ileadtimes[i], Iout] = data
            count += len(obs_source.locations)

        self.ds["obs"] = (["time", "leadtime", "location"], obs)

        if self.num_members > 1:
            crps = self.compute_crps(ens, obs, self.fair_crps)
            self.ds["ensemble_crps"] = (["time", "leadtime", "location"], crps)

        self.ds.attrs["units"] = self.units
        self.ds.attrs["verif_version"] = "1.0.0"
        self.ds.attrs["standard_name"] = cf.get_metadata(self.variable)["cfname"]

        utils.create_directory(self.filename)

        data_variables = [
            "obs",
            "fcst",
            "ensemble",
            "ensemble_mean",
            "ensemble_variance",
            "cdf",
            "x",
            "ensemble_crps",
            "pit",
        ]
        if self.compression:
            nc_encoding = {v: {"zlib": True} for v in data_variables if v in self.ds}
        else:
            nc_encoding = dict()

        self.ds.to_netcdf(
            self.filename,
            mode="w",
            engine="netcdf4",
            unlimited_dims=["time"],
            encoding=nc_encoding,
        )

        if self.remove_intermediate:
            self.intermediate.cleanup()

    def compute_consensus(self, pred) -> np.ndarray:
        assert len(pred.shape) == 3, pred.shape

        if self.consensus_method == "control":
            return pred[..., 0]
        if self.consensus_method == "mean":
            return np.mean(pred, axis=-1)
        raise NotImplementedError(f"Unknown consensus method {self.consensus_method}")

    def compute_quantile(self, ar, level, fair=True) -> np.ndarray:
        """Extracts a quantile from an array

        Args:
            ar: N-D numpy array, where last dimension is ensemble
            level: a number between 0 and 1
            fair: Adjust for sampling error

        Returns:
            (N-1)-D numpy array with quantiles
        """
        assert 0 <= level <= 1, f"level={level} must be between 0 and 1"

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

    def compute_threshold_prob(self, ar, threshold, fair=True) -> np.ndarray:
        """Compute probability less than a threshold for an ensemble
        Args:
            ar: N-D numpy array, where last dimensions is ensemble
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

    def compute_crps(self, preds, targets, fair=True) -> np.ndarray:
        """Continuous Ranked Probability Score (CRPS).

        Args:
            preds: numpy.ndarray
                Predictions, shape (time, leadtime, location, ens_size)
            targets: numpy.ndarray
                Targets, shape (time, leadtime, location)
            fair: bool
                Defaults to true

        Returns:
            crps: numpy.ndarray
                Shape (time, leadtime, location)
        """

        coef = (
            -1.0 / (self.num_members * (self.num_members - 1))
            if fair
            else -1.0 / (self.num_members**2)
        )

        mae = np.mean(np.abs(targets[..., None] - preds), axis=-1)

        # var = np.abs(preds[..., None] - preds[..., None, :])
        var = np.zeros(preds.shape[:-1])
        for i in range(self.num_members):  # loop version to reduce memory usage
            var += np.sum(np.abs(preds[..., i, None] - preds[..., i + 1 :]), axis=-1)
        var *= coef
        return mae + var

    @staticmethod
    def get_points(
        predict_metadata, obs_sources, max_distance=None
    ) -> tuple[gridpp.Points, gridpp.Points, np.ndarray | list]:
        """Returns point objects for input and output, filtering out output points that are too
        far outside the input"""
        obs_lats = []
        obs_lons = []
        obs_altitudes = []
        obs_ids = []
        for obs_source in obs_sources:
            obs_lats += [loc.lat for loc in obs_source.locations]
            obs_lons += [loc.lon for loc in obs_source.locations]
            obs_altitudes += [loc.elev for loc in obs_source.locations]
            obs_ids += [loc.id for loc in obs_source.locations]

        if predict_metadata.altitudes is not None:
            ipoints = gridpp.Points(
                predict_metadata.lats, predict_metadata.lons, predict_metadata.altitudes
            )
        else:
            ipoints = gridpp.Points(predict_metadata.lats, predict_metadata.lons)
        opoints = gridpp.Points(
            np.array(obs_lats), np.array(obs_lons), np.array(obs_altitudes)
        )

        if max_distance is not None:
            dist = gridpp.distance(ipoints, opoints)
            ipoint = np.where(dist < max_distance)[0]
            opoints = opoints.subset(ipoint)

            obs_ids = np.array(obs_ids)[ipoint]

        assert opoints.size() == len(obs_ids)

        return ipoints, opoints, obs_ids
