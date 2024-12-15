import numpy as np


class PredictMetadata:
    """This class stores metadata about each dimension of a batch"""
    def __init__(self, variables, lats, lons, leadtimes, num_members, field_shape=None):
        assert lats.shape == lons.shape

        assert np.prod(field_shape) == len(lats)

        self.variables = variables
        self.lats = lats
        self.lons = lons
        self.leadtimes = leadtimes
        self.num_members = num_members
        self.field_shape = field_shape

    @property
    def shape(self):
        """The shape that you can expect predict_step to provide. Note that ensemble is done in
        data parallel, so not included in this shape.
        """
        # This is goverend by predict_step in forecaster.py
        return [len(self.leadtimes), len(self.lats), len(self.variables)]
