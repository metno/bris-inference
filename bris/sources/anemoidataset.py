from functools import cached_property

import numpy as np

from anemoi.datasets import open_dataset

from bris.observations import Location, Observations
from bris.utils import datetime_to_unixtime
from bris.sources import Source

class AnemoiDataset(Source):
    """Loads truth from an anemoi datasets zarr dataset.

    Creates a file that can be read by verif
    """

    def __init__(self, dataset_dict: dict):
        """
        Args:
            dataset_dict: open_dataset recipie, dictionairy.
        """
        
        self.dataset = open_dataset(dataset_dict)
        print(self.dataset.dates)

    @cached_property
    def locations(self):
        num_locations = len(self.dataset.longitudes)
        _latitudes = self.dataset.latitudes
        _longitudes = self.dataset.longitudes
        _longitudes[_longitudes < -180] += 360
        _longitudes[_longitudes > 180] -= 360 

        if "z" in self.dataset.variables:
            _altitudes = self.dataset[0, self.dataset.name_to_index["z"], 0,:] / 9.81
        else:
            print("Warning: Could not find field 'z' in dataset, setting altitude to 0 everywhere.")
            _altitudes = np.zeros_like(_latitudes)
        _locations = list()
        for i in range(num_locations):
            location = Location(
                _latitudes[i],
                _longitudes[i],
                _altitudes[i],
                i                
            )
            _locations += [location]
        return _locations
    
    def get(self, variable, start_time, end_time, frequency):
        assert frequency > 0
        assert end_time > start_time

        requested_times = np.arange(start_time, end_time + 1, frequency)
        num_requested_times = len(requested_times)
        
        var_index = self.dataset.name_to_index[variable]

        data = np.nan * np.zeros([num_requested_times, len(self.locations)], np.float32)
        for t, requested_time in enumerate(requested_times):
            i = np.where(self._all_times == requested_time)[0]
            if len(i) > 0:
                data[t, :] = self.dataset[int(i[0]), var_index, 0, :]

        observations = Observations(self.locations, requested_times, {variable: data})
        return observations
    
    @cached_property
    def _all_times(self):
        return datetime_to_unixtime(self.dataset.dates)
    
    @cached_property
    def units(self):
        return None

    
