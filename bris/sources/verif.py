from functools import cached_property
from typing import Optional

import numpy as np
import xarray as xr

from bris.observations import Location, Observations
from bris.sources import Source


class Verif(Source):
    """Loads observations from a Verif file (https:://github.com/WFRT/Verif)

    Fetches observations across times and leadtimes to maximize the number of observations available
    for a given time.

    This also parses Verif files that do not have a leadtime dimension, relying only on the time
    dimension.
    """

    def __init__(self, filename: str, obs_variable: str = "obs"):
        # Don't convert the unixtime to datetime objects
        self.file = xr.open_dataset(filename, decode_times=False)
        self.obs_variable = obs_variable

        self.has_leadtime = (
            "leadtime" in self.file.variables and "leadtime" in self.file.dims
        )

    @cached_property
    def locations(self):
        num_locations = len(self.file["location"])
        _locations = []
        if "latitude" in self.file:
            lats = self.file["latitude"].values
        else:
            lats = self.file["lat"].values
        if "longitude" in self.file:
            lons = self.file["longitude"].values
        else:
            lons = self.file["lon"].values
        for i in range(num_locations):
            location = Location(
                lats[i],
                lons[i],
                self.file["altitude"].values[i],
                self.file["location"].values[i],
            )
            _locations += [location]
        return _locations

    def get(self, variable, start_time, end_time, frequency):
        assert frequency > 0
        assert end_time >= start_time

        requested_times = np.arange(start_time, end_time + 1, frequency)
        num_requested_times = len(requested_times)

        raw_obs = self.file[self.obs_variable].values

        data = np.nan * np.zeros([num_requested_times, len(self.locations)], np.float32)
        for t, requested_time in enumerate(requested_times):
            if self.has_leadtime:
                i, j = np.where(self._all_times == requested_time)
                if len(i) > 0:
                    data[t, :] = raw_obs[i[0], j[0], :]
            else:
                i = np.where(self._all_times == requested_time)[0]
                if len(i) > 0:
                    data[t, :] = raw_obs[i[0], :]

        observations = Observations(self.locations, requested_times, {variable: data})
        return observations

    @cached_property
    def _all_times(self) -> np.ndarray:
        if self.has_leadtime:
            a, b = np.meshgrid(
                self.file["leadtime"].values * 3600, self.file["time"].values
            )
            return (a + b)[:]  # (time, leadtime)
        return self.file["time"].values[:, None]

    @cached_property
    def _times(self):
        return np.sort(np.unique(self._all_times))

    @cached_property
    def units(self) -> Optional[str]:
        if hasattr(self.file, "units"):
            return self.file.units
        return None
