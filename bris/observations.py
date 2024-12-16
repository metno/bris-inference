import numpy as np


class Observations:
    def __init__(self, locations, times, data):
        """
        data: dict with key variable_name and value: 2D numpy array with dimensions (time, location)
        """
        for k, v in data.items():
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
        else:
            index = indices[0]
            return self.data[variable][index, ...]

    def __str__(self):
        string = "Observations:\n"
        string += "   num locations: %s\n" % len(self.locations)
        string += "   num times: %s\n" % len(self.times)
        string += "   num variables: %s" % len(self.variables)
        return string


class Location:
    def __init__(self, lat, lon, elev=None, id=None):
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.id = id
