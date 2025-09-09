from functools import cached_property
import re

import numpy as np
from anemoi.datasets import open_dataset

from bris.conventions.anemoi import get_units as get_anemoi_units
from bris.observations import Location, Observations
from bris.sources import Source
from bris.utils import datetime_to_unixtime


class AnemoiDataset(Source):
    """Loads data from an anemoi datasets zarr dataset.

    Used to create a verif file that can be used to evaluate forecasts against analysis.
    """

    def __init__(self, dataset_dict: dict, variable: str, every_loc: int = 1):
        """
        Args:
            dataset_dict: open_dataset recipe, dictionary.
            variable: variable to fetch from dataset. Can also specify how to 
                      derive from other variables using standard Pythonic math
                      expressions and specifying variables in brackets (e.g. [10u])
            every: include every this number of locations in the verif file
        """

        self.dataset = open_dataset(dataset_dict)

        self.variable = variable
        self.derive = False
        if "[" in variable:
            self.derive = True
            variables = re.findall(r'\[([^\[\]]+)\]', variable)
            self.derive_template = re.sub(r'\[[^\[\]]+\]', 'np.array({})', variable)
            self.variable_index = []
            for v in variables:
                v_idx = self.dataset.name_to_index[v]
                self.variable_index.append(v_idx)
        else:
            self.variable_index = self.dataset.name_to_index[variable]
        self.every_loc = every_loc

    @cached_property
    def locations(self):
        num_locations = len(self.dataset.longitudes)
        _latitudes = self.dataset.latitudes
        _longitudes = self.dataset.longitudes
        _longitudes[_longitudes < -180] += 360
        _longitudes[_longitudes > 180] -= 360

        if "z" in self.dataset.variables:
            _altitudes = self.dataset[0, self.dataset.name_to_index["z"], 0, :] / 9.81
        else:
            print(
                "Warning: Could not find field 'z' in dataset, setting altitude to 0 everywhere."
            )
            _altitudes = np.zeros_like(_latitudes)
        _locations = []
        for i in range(0, num_locations, self.every_loc):
            location = Location(_latitudes[i], _longitudes[i], _altitudes[i], i)
            _locations += [location]
        return _locations

    def get(self, variable, start_time, end_time, frequency):
        assert frequency > 0
        assert end_time > start_time

        requested_times = np.arange(start_time, end_time + 1, frequency)
        num_requested_times = len(requested_times)

        data = np.nan * np.zeros([num_requested_times, len(self.locations)], np.float32)
        for t, requested_time in enumerate(requested_times):
            i = np.where(self._all_times == requested_time)[0]
            if len(i) > 0:
                j = int(i[0])
                if j in self.dataset.missing:
                    print(
                        f"Date {self.dataset.dates[int(i[0])]} missing from verif dataset"
                    )
                else:
                    if self.derive:
                        all_data = []
                        for v_idx in self.variable_index:
                            all_data.append(self.dataset[j, v_idx, 0, :: self.every_loc].tolist())
                        data[t, :] = eval(self.derive_template.format(*all_data))
                    else:
                        data[t, :] = self.dataset[
                            j, self.variable_index, 0, :: self.every_loc
                        ]

        observations = Observations(self.locations, requested_times, {variable: data})
        return observations

    @cached_property
    def _all_times(self):
        return datetime_to_unixtime(self.dataset.dates)

    @cached_property
    def units(self):
        return get_anemoi_units(self.variable)
