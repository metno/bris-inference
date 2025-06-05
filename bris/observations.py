import numpy as np


class Observations:
    def __init__(self, locations, times, data):
        """
        data: dict with key variable_name and value: 2D numpy array with dimensions (time, location)
        """
        for _k, v in data.items():
            assert len(times) == v.shape[0]
            assert len(locations) == v.shape[1]

        self.locations = locations
        self.times = times
        self.data = data

    @property
    def variables(self):
        return self.data.keys()

    def get_data(self, variable, unixtime):
        # print(self.times, unixtime)
        indices = np.where(self.times == unixtime)[0]
        if len(indices) == 0:
            return None
        index = indices[0]
        return self.data[variable][index, ...]

    def __str__(self):
        string = "Observations:\n"
        string += f"   num locations: {len(self.locations)}\n"
        string += f"   num times: {len(self.times)}\n"
        string += f"   num variables: {len(self.variables)}"
        return string


class Location:
    def __init__(self, lat, lon, elev=None, location_id=None):
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.id = location_id

        if self.lon < -180:
            self.lon += 360
        if self.lon > 180:
            self.lon -= 360
