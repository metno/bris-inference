from functools import cached_property
import numpy as np


from bris.source import Source


class Frost(Source):
    """This class writes verification files in Verif format. See github.com/WFRT/verif."""

    def __init__(
        self,
        filename: str,
        frost_variable_name: str,
        leadtimes: list,
        grid_lats: np.array,
        grid_lons: np.array,
        grid_elevs: np.array,
        units: str,
        elev_gradient=None,
        frost_client_id=None,
        fetch_path=None,
    ):
        """
        Args:
            frost_variable_name: Name of observation variable in frost.met.no to use for verification
            leadtimes: List of leadtimes in hours to write
            grid_lats: 1D or 2D array of latitudes of the forecast grid
            grid_lons: 1D or 2D array of longitudes of the forecast grid
            grid_elevs: 2D array of altitudes of the forecast grid
            units: Units to put in Verif file. Should be the same as the observations
            elev_gradient: Apply this elevation gradient when downscaling from grid to point (e.g.
                -0.0065 for temperature)
        """
        self.filename = filename
        self.fcst = dict()
        self.points = None
        self.frost_variable_name = frost_variable_name
        self._frost_client_id = frost_client_id
        self.leadtimes = leadtimes
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
        coords["leadtime"] = (["leadtime"], leadtimes, {"units": "hour"})
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

        self.fetch_path = fetch_path  # if None or False fetches observations from frost, else gets precached observations from file at path.

    @cached_property
    def frost_client_id(self):
        """Returns the frost_client_id if provided under construction, or from ~/.frostid"""
        if self._frost_client_id is None:
            if os.path.exists("~/.frostrc"):
                with open("~/.frostrc", "r") as file:
                    frost_credentials = json.loads(file)
                if "frost_client_id" in frost_credentials:
                    return frost_credentials["frost_client_id"]
                else:
                    raise RuntimeError("Could not find frost_client_id in ~/.frostrc")
            else:
                raise RuntimeError(
                    "frost_client_id is not provided and ~/.frostrc does not exist"
                )
        else:
            return self._frost_client_id

    def add_forecast(self, forecast_reference_time: float, field: np.array):
        """Add forecasts to this object. Will be written when .write() is called

        Args:
            forecast_reference_time: Unix time of the forecast initialization [s]
            field: 3D array of forecasts with dimensions (time, y, x)
        """
        # print(field.shape, len(self.leadtimes))
        assert field.shape[0] == len(self.leadtimes)
        assert len(field.shape) == 3
        assert field.shape[1] == self.grid.size()[0]
        assert field.shape[2] == self.grid.size()[1]

        if self.elev_gradient is None:
            self.fcst[forecast_reference_time] = gridpp.bilinear(
                self.grid, self.points, field
            )
        else:
            # print("ELEVATION CORRECTION", np.mean(self.grid.get_elevs()), np.mean(self.points.get_elevs()))
            self.fcst[forecast_reference_time] = gridpp.simple_gradient(
                self.grid, self.points, field, self.elev_gradient, gridpp.Bilinear
            )

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
