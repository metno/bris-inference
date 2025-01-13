from functools import cached_property

import numpy as np
import xarray as xr

from bris.observations import Location, Observations
from bris.sources import Source


class Verif(Source):
    """Loads observations from a Verif file (https:://github.com/WFRT/Verif)

    Fetches observations across times and leadtimes to maximize the number of observations available
    for a given time.
    """

    def __init__(self, filename: str):
        self.file = xr.open_dataset(filename)

    @cached_property
    def locations(self):
        num_locations = len(self.file["location"])
        _locations = list()
        for i in range(num_locations):
            location = Location(
                self.file["lat"].values[i],
                self.file["lon"].values[i],
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

        raw_obs = self.file["obs"].values

        data = np.nan * np.zeros([num_requested_times, len(self.locations)], np.float32)
        for t, requested_time in enumerate(requested_times):
            i, j = np.where(self._all_times == requested_time)
            if len(i) > 0:
                data[t, :] = raw_obs[i[0], j[0], :]

        observations = Observations(self.locations, requested_times, {variable: data})
        return observations

    @cached_property
    def _all_times(self):
        a, b = np.meshgrid(
            self.file["leadtime"].values * 3600, self.file["time"].values
        )
        return (a + b)[:]  # (time, leadtime)

    @cached_property
    def _times(self):
        return np.sort(np.unique(self._all_times))

    @cached_property
    def units(self):
        if hasattr(self.file, "units"):
            return self.file.units
        else:
            return None
