class PredictMetadata:
    """This class stores metadata about each dimension of a batch"""
    def __init__(self, variables, lats, lons, leadtimes):
        assert lats.shape == lons.shape

        self.variables = variables
        self.lats = lats
        self.lons = lons
        self.leadtimes = leadtimes
