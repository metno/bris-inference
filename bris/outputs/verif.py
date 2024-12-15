from ..output import Output
from .intermediate import Intermediate
from ..predict_metadata import PredictMetadata


class Verif(Output):
    """This class writes verification files in Verif format. See github.com/WFRT/verif."""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
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
        self.points = None
        self.frost_variable_name = frost_variable_name
        self._frost_client_id = frost_client_id
        self.grid = gridpp.Grid(grid_lats, grid_lons, grid_elevs)
        self.units = units
        self.elev_gradient = elev_gradient

        metadata = get_station_metadata(self.frost_client_id, wmo=True, country="Norge")

        self.station_ids = [id for id in metadata]
        obs_lats = [metadata[id]["lat"] for id in self.station_ids]
        obs_lons = [metadata[id]["lon"] for id in self.station_ids]
        obs_elevs = [metadata[id]["elev"] for id in self.station_ids]
        # Frost uses SN18700, whereas in Verif we want just 18700
        self.obs_ids = [int(id.replace("SN", "")) for id in metadata]
        self.points = gridpp.Points(obs_lats, obs_lons, obs_elevs)

        coords = dict()
        coords["time"] = (
            ["time"],
            [],
            {
                "units": "seconds since 1970-01-01 00:00:00 +00:00",
                "var.standard_name": "forecast_reference_time",
            },
        )
        coords["leadtime"] = (["leadtime"], self.pm.leadtimes, {"units": "hour"})
        coords["location"] = (["location"], self.obs_ids)
        coords["lat"] = (
            ["location"],
            obs_lats,
            {"units": "degree_north", "standard_name": "latitude"},
        )
        coords["lon"] = (
            ["location"],
            obs_lons,
            {"units": "degree_east", "standard_name": "longitude"},
        )
        coords["altitude"] = (
            ["location"],
            obs_elevs,
            {"units": "m", "standard_name": "surface_altitude"},
        )
        self.ds = xr.Dataset(coords=coords)

        # The intermediate will only store the final output locations
        intermediate_om = PredictMetadata([variable], obs_lats, obs_lons, predict_metadata.leadtimes, predict_metadata.num_members)
        self.intermediate = Intermediate(intermediate_pm)

    def _add_forecast(self, forecast_reference_time: int, ensemble_member: int, pred: np.array):
        """Add forecasts to this object. Will be written when .write() is called

        Args:
            forecast_reference_time: Unix time of the forecast initialization [s]
            pred: 3D array of forecasts with dimensions (time, y, x)
        """
        # 1. Interpolate to points
        # 2. Store to intermediate file
        num_leadtimes, num_locations, num_variables = pred.shape
        shape = [num_leadtimes, self._num_locations, 1]
        interpolated_pred = np.nan * np.zeros(shape, np.float32)

        """
        if self.elev_gradient is None:
            self.fcst[forecast_reference_time] = gridpp.bilinear(
                self.grid, self.points, pred
            )
        else:
            # print("ELEVATION CORRECTION", np.mean(self.grid.get_elevs()), np.mean(self.points.get_elevs()))
            self.fcst[forecast_reference_time] = gridpp.simple_gradient(
                self.grid, self.points, pred, self.elev_gradient, gridpp.Bilinear
            )
        """

        self.intermediate.add(forecast_reference_time, ensemble_member, interpolated_pred)

    @property
    def _num_locations(self):
        return len(self.obs_lats)

    def finalize(self):
        """Write forecasts and observations to file"""

        create_directory(self.filename)

        # Add forecasts
        frts = list(self.fcst.keys())
        frts.sort()
        self.ds["time"] = frts
        fcst = np.nan * np.zeros(
            [len(frts), len(self.ds.leadtime), len(self.ds.location)], np.float32
        )
        for i, frt in enumerate(frts):
            fcst[i, ...] = self.fcst[frt]

        self.ds["fcst"] = (["time", "leadtime", "location"], fcst)

        if not self.fetch_path:
            # Find which valid times we need observations for
            a, b = np.meshgrid(self.ds.time, self.ds.leadtime * 3600)
            valid_times = a + b
            valid_times = valid_times.transpose()
            if len(valid_times) == 0:
                print(self.ds.time, self.ds.leadtime)
                raise Exception("No valid times")
            start_time = np.min(valid_times)
            end_time = np.max(valid_times)

            # Load the observations. Note we might not get the same locations and times we requested, so
            # we have to do a matching.
            print(f"Loading {self.frost_variable_name} observations from frost.met.no")
            obs_times, obs_locations, obs_values = get(
                start_time,
                end_time,
                self.frost_variable_name,
                self.frost_client_id,
                station_ids=self.station_ids,
                time_resolutions=["PT1H"],
                debug=False,
            )
            obs_ids = [loc.id for loc in obs_locations]
            Iin, Iout = get_common_indices(obs_ids, self.obs_ids)

            # Fill in retrieved observations into our obs array.
            obs = np.nan * np.zeros(
                [len(frts), len(self.ds.leadtime), len(self.ds.location)], np.float32
            )
            for t, obs_time in enumerate(obs_times):
                I = np.where(valid_times == obs_time)
                for i in range(len(I[0])):
                    # Copy observation into all times/leadtimes that matches this valid time
                    obs[I[0][i], I[1][i], Iout] = obs_values[t, Iin]

            self.ds["obs"] = (["time", "leadtime", "location"], obs)

        else:
            fvar_to_param = {
                "air_temperature": "t2m",
                "wind_speed": "ws10m",
                "air_pressure_at_sea_level": "mslp",
                "sum(precipitation_amount PT6H)": "precip6h",
            }
            fetch_path = self.fetch_path.split("PARAM")
            ref_ds = f"{fetch_path[0]}{fvar_to_param[self.frost_variable_name]}{fetch_path[1]}"
            rds = xr.open_dataset(ref_ds)
            self.ds["obs"] = rds["obs"]

        self.ds.attrs["units"] = self.units

        self.ds.to_netcdf(self.filename, mode="w", engine="netcdf4")
