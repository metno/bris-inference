import xarray as xr
import numpy as np


from bris.source import Source
from bris.observations import Observations
from bris.observations import Location


class VerifNetcdf(Source):
    """Loads observations from a Verif file (https:://github.com/WFRT/Verif)

    Fetches observations across times and leadtimes to maximize the number of observations available
    for a given time.
    """

    def __init__(self, filename: str):
        self.file = xr.open_dataset(filename)

    def get(self, variable, start_time, end_time, frequency):
        num_locations = len(self.file["location"])
        locations = list()
        for i in range(num_locations):
            location = Location(
                self.file["lat"],
                self.file["lon"],
                self.file["altitude"],
                self.file["location"],
            )
            locations += [location]

        a, b = np.meshgrid(
            self.file["leadtime"].values * 3600, self.file["time"].values
        )
        all_times = a + b  # (time, leadtime)
        times = np.sort(np.unique(all_times[:]))
        num_times = len(times)

        raw_obs = self.file["obs"].values
        data = np.zeros([num_times, num_locations], np.float32)
        for t in range(num_times):
            i, j = np.where(all_times == times[t])
            if len(i) > 0:
                data[t, :] = raw_obs[i[0], j[0], :]

        observations = Observations(locations, times, {variable: data})
        return observations
