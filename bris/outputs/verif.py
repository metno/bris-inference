import gridpp
import numpy as np
import xarray as xr


from bris.output import Output
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata
from bris.utils import create_directory
from bris.conventions import cf


class Verif(Output):
    """This class writes verification files in Verif format. See github.com/WFRT/verif."""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
        units: str,
        obs_sources=list(),
        elev_gradient=None,
    ):
        """
        Args:
            units: Units to put in Verif file. Should be the same as the observations
            elev_gradient: Apply this elevation gradient when downscaling from grid to point (e.g.
                -0.0065 for temperature)
        """
        super().__init__(predict_metadata)

        self.filename = filename
        self.fcst = dict()
        self.variable = variable
        self.units = units
        self.elev_gradient = elev_gradient

        self.igrid = gridpp.Grid(
            self.pm.grid_lats, self.pm.grid_lons, self.pm.grid_elevs
        )

        self.ipoints = gridpp.Points(self.pm.lats, self.pm.lons, self.pm.elevs)

        obs_lats = list()
        obs_lons = list()
        obs_elevs = list()
        obs_ids = list()

        self.obs_sources = obs_sources

        for obs_source in obs_sources:
            obs_lats += [loc.lat for loc in obs_source.locations]
            obs_lons += [loc.lon for loc in obs_source.locations]
            obs_elevs += [loc.elev for loc in obs_source.locations]
            obs_ids += [loc.id for loc in obs_source.locations]

        self.obs_lats = np.array(obs_lats, np.float32)
        self.obs_lons = np.array(obs_lons, np.float32)
        self.obs_elevs = np.array(obs_elevs, np.float32)
        self.obs_ids = np.array(obs_ids, np.int32)
        self.opoints = gridpp.Points(self.obs_lats, self.obs_lons, self.obs_elevs)

        # The intermediate will only store the final output locations
        intermediate_pm = PredictMetadata(
            [variable],
            self.obs_lats,
            self.obs_lons,
            predict_metadata.leadtimes,
            predict_metadata.num_members,
        )
        self.intermediate = Intermediate(intermediate_pm, workdir)

    def _add_forecast(
        self, forecast_reference_time: int, ensemble_member: int, pred: np.array
    ):
        """Add forecasts to this object. Will be written when .write() is called

        Args:
            forecast_reference_time: Unix time of the forecast initialization [s]
            pred: 3D array of forecasts with dimensions (time, points, variables)
        """

        if self._is_gridded_input:
            pred = self.reshape_pred(pred)
            pred = pred[..., 0]  # Extract single variable
            interpolated_pred = gridpp.bilinear(self.igrid, self.opoints, pred)

            if self.elev_gradient is not None:
                interpolated_elevs = gridpp.bilinear(
                    self.igrid, self.opoints, self.igrid.get_elevs()
                )
                delev = self.opoints.get_elevs() - interpolated_elevs
                interpolated_pred += self.elev_gradient * delev
            interpolated_pred = interpolated_pred[
                :, :, None
            ]  # Add in variable dimension
        else:
            pred = pred[..., 0]
            interpolated_pred = gridpp.simple_gradient(self.ipoints, self.opoints, pred)

            if self.elev_gradient is not None:
                interpolated_elevs = gridpp.nearest(
                    self.ipoints, self.opoints, self.ipoints.get_elevs()
                )
                delev = self.opoints.get_elevs() - interpolated_elevs
                interpolated_pred += self.elev_gradient * delev
            interpolated_pred = interpolated_pred[:, :, None]

        self.intermediate.add_forecast(
            forecast_reference_time, ensemble_member, interpolated_pred
        )

    @property
    def _is_gridded_input(self):
        return self.pm.field_shape is not None

    @property
    def _num_locations(self):
        return self.opoints.size()

    def finalize(self):
        """Write forecasts and observations to file"""

        coords = dict()
        coords["time"] = (["time"], [], cf.get_attributes("time"))
        coords["leadtime"] = (
            ["leadtime"],
            self.pm.leadtimes.astype(np.float32),
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
            self.obs_elevs,
            cf.get_attributes("surface_altitude"),
        )
        self.ds = xr.Dataset(coords=coords)

        frts = self.intermediate.get_forecast_reference_times().astype(np.double)
        self.ds["time"] = frts

        # Load forecasts
        fcst = np.nan * np.zeros(
            [
                len(frts),
                len(self.intermediate.pm.leadtimes),
                self.intermediate.pm.num_points,
            ],
            np.float32,
        )
        for i, frt in enumerate(frts):
            fcst[i, ...] = self.intermediate.get_forecast(frt, 0)[..., 0]

        self.ds["fcst"] = (["time", "leadtime", "location"], fcst)

        # Find which valid times we need observations for
        a, b = np.meshgrid(frts, np.array(self.intermediate.pm.leadtimes) * 3600)
        valid_times = a + b
        valid_times = valid_times.transpose()
        if len(valid_times) == 0:
            print(frts, self.pm.leadtimes)
            raise Exception("No valid times")

        # valid_times = np.sort(np.unique(valid_times.flatten()))
        unique_valid_times = np.sort(np.unique(valid_times.flatten()))

        start_time = int(np.min(unique_valid_times))
        end_time = int(np.max(unique_valid_times))
        frequency = int(np.min(np.diff(unique_valid_times)))

        # Fill in retrieved observations into our obs array.
        obs = np.nan * np.zeros(
            [
                len(frts),
                len(self.intermediate.pm.leadtimes),
                self.intermediate.pm.num_points,
            ],
            np.float32,
        )
        count = 0
        for obs_source in self.obs_sources:
            curr = obs_source.get(self.variable, start_time, end_time, frequency)
            for t, valid_time in enumerate(unique_valid_times):
                I = np.where(valid_times == valid_time)
                data = curr.get_data(self.variable, valid_time)
                if data is not None:
                    Iout = range(count, len(obs_source.locations) + count)
                    for i in range(len(I[0])):
                        # Copy observation into all times/leadtimes that matches this valid time
                        obs[I[0][i], I[1][i], Iout] = data
            count += len(obs_source.locations)

        self.ds["obs"] = (["time", "leadtime", "location"], obs)

        self.ds.attrs["units"] = self.units
        self.ds.attrs["verif_version"] = "1.0.0"
        self.ds.attrs["standard_name"] = cf.get_metadata(self.variable)["cfname"]

        create_directory(self.filename)
        self.ds.to_netcdf(self.filename, mode="w", engine="netcdf4")
