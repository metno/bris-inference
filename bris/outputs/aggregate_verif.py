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


class AggregateVerif(Verif):
    """Writes verification files aggregated over locations in Verif format. See github.com/WFRT/verif."""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str = None,
        obs_sources: list = None,
        units: str = None,
        elev_gradient: float = None,
        max_distance: float = None,
        remove_intermediate: bool = True,
        compression: bool = False,
    ) -> None:
        extra_variables = []
        if variable not in predict_metadata.variables:
            extra_variables += [variable]

        super().__init__(predict_metadata, extra_variables)

        self.filename = filename
        self.fcst = {}
        self.variable = variable
        self.obs_sources = obs_sources
        self.units = units
        self.elev_gradient = elev_gradient
        self.max_distance = max_distance

        self.metrics = ["rmse", "crps", "obs", "fcst", "bias"]
        self.locations = ["nhem", "shem"]


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

        self.matching_locations = False
        if len(_matching_indices) == len(self.opoints_array):
            self.verif_indices = _matching_indices
            self.matching_locations = True

        self.triangulation = self.ipoints_array
        if (
            not self._is_gridded_input
            and self.ipoints_array.shape[0] > 3
            and not self.matching_locations
        ):
            # This speeds up interpolation from irregular points to observation points
            # but Delaunay needs enough points for this to work
            self.triangulation = Delaunay(self.ipoints_array)

        # The intermediate will only store the final output locations
        intermediate_pm = PredictMetadata(
            self.metrics,
            np.zeros(len(self.locations), np.float32),
            np.zeros(len(self.locations), np.float32),
            np.zeros(len(self.locations), np.float32),
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

        interpolated_pred = self.interpolate(pred)
        obs = self.get_obs(times)

        shape = [pred.shape[0], len(self.locations), len(self.metrics)]
        scores = np.nan * np.zeros(shape, np.float32)

        for l, location in enumerate(self.locations):
            if location == "nhem":
                point_indices = np.where(self.opoints.get_lats() >= 0)[0]
            elif location == "shem":
                # TODO: What to do about equator points
                point_indices = np.where(self.opoints.get_lats() <= 0)[0]
            else:
                raise NotImplementedError()

            if len(point_indices) == 0:
                print(f"Skipping, {location}")
                continue

            selected_pred = interpolated_pred[:, point_indices]
            selected_obs = obs[:, point_indices]

            for m, metric in enumerate(self.metrics):
                if metric == "fcst":
                    curr_scores = np.mean(selected_pred, axis=1)
                elif metric == "obs":
                    curr_scores = np.mean(selected_obs, axis=1)
                elif metric == "rmse":
                    curr_scores = np.sqrt(np.mean((selected_obs - selected_pred)**2, axis=1))

                anemoi_units = anemoi_conventions.get_units(self.variable)
                if self.units is None:
                    # Update the units so they can be written out
                    # Should be done in constructor
                    self.units = anemoi_units
                elif anemoi_units is not None and self.units != anemoi_units:
                    # TODO: Units depend on the metric
                    to_units = self.units
                    from_units = anemoi_units
                    bris.units.convert(curr_scores, from_units, to_units, inplace=True)
                scores[:, l, m] = curr_scores

        self.intermediate.add_forecast(times, ensemble_member, scores)

    def interpolate(self, pred):
        """Returns 2D array (leadtime, point)"""
        Iv = self.pm.variables.index(self.variable)
        if self.matching_locations:
            interpolated_pred = pred[:, self.verif_indices, Iv]
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

                interpolated_pred = np.nan * np.zeros([num_leadtimes, num_points], np.float32)
                for lt in range(num_leadtimes):
                    interpolator = scipy.interpolate.LinearNDInterpolator(
                        self.triangulation, pred[lt, :, 0]
                    )
                    interpolated_pred[lt, :] = interpolator(self.opoints_array)
                    if altitude_correction is not None:
                        interpolated_pred[lt, :] += (
                            self.elev_gradient * altitude_correction
                        )
        return interpolated_pred

    def get_obs(self, unixtimes):
        """Returns a 2D array (time, location)"""

        for t, unixtime in enumerate(unixtimes):
            obs = np.zeros([0], np.float32)
            for obs_source in self.obs_sources:
                curr = obs_source.get(self.variable, unixtime, unixtime, 3600)
                curr_obs = curr.get_data(self.variable, unixtime)

                from_units = obs_source.units
                to_units = self.units
                if curr_obs is not None:
                    if None not in [obs_source.units, self.units]:
                        bris.units.convert(curr_obs, from_units, to_units, inplace=True)
                obs = np.append(obs, curr_obs)

            if t == 0:
                allobs = np.zeros([len(unixtimes), len(obs)], np.float32)
            allobs[t, :] = obs
        return allobs

    @property
    def _is_gridded_input(self) -> bool:
        return self.pm.is_gridded

    @property
    def _num_locations(self) -> int:
        return len(self.locations)

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

        leadtimes_seconds = self.intermediate.pm.leadtimes.astype(np.float32)

        coords = {}
        coords["time"] = (["time"], frts_unix, cf.get_attributes("time"))
        coords["leadtime"] = (
            ["leadtime"],
            leadtimes_seconds / 3600,
            {"units": "hour"},
        )
        assert len(self.obs_ids) == len(self.opoints.get_lats()), (
            len(self.obs_ids),
            len(self.opoints.get_lats()),
        )
        coords["location"] = (["location"], self.locations)

        self.ds = xr.Dataset(coords=coords)

        scores_shape = (
            len(frts),
            self.intermediate.pm.num_leadtimes,
            self._num_locations,
            len(self.metrics),
        )
        scores = self.create_nan_array(scores_shape)
        for i, frt in enumerate(frts):
            # last dimension (member) is a singleton
            scores[i, ...] = self.intermediate.get_forecast(frt)[..., 0]

        for m, metric in enumerate(self.metrics):
            self.ds[metric] = (["time", "leadtime", "location"], scores[..., m])

        self.ds.attrs["units"] = self.units
        self.ds.attrs["verif_version"] = "1.0.0"
        self.ds.attrs["standard_name"] = cf.get_metadata(self.variable)["cfname"]

        utils.create_directory(self.filename)

        data_variables = [
            "obs",
            "fcst",
            "rmse",
            "crps",
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
