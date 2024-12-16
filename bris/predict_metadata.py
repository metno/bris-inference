import numpy as np


class PredictMetadata:
    """This class stores metadata about each dimension of a batch"""
    def __init__(self, variables, lats, lons, leadtimes, num_members, field_shape=None):
        assert lats.shape == lons.shape, (lats.shape, lons.shape)

        if field_shape is not None:
            assert np.prod(field_shape) == len(lats)

        self.variables = variables
        self.lats = np.array(lats)
        self.lons = np.array(lons)
        # TODO
        self.elevs = np.zeros(lats.shape)
        self.leadtimes = np.array(leadtimes)
        self.num_members = num_members
        self.field_shape = field_shape

    @property
    def num_points(self):
        return len(self.lats)

    @property
    def shape(self):
        """The shape that you can expect predict_step to provide. Note that ensemble is done in
        data parallel, so not included in this shape.
        """
        # This is goverend by predict_step in forecaster.py
        return [len(self.leadtimes), len(self.lats), len(self.variables)]

    @property
    def is_gridded(self):
        return self.field_shape is not None

    @property
    def grid_lats(self):
        assert self.is_gridded

        return np.reshape(self.lats, self.field_shape)

    @property
    def grid_lons(self):
        assert self.is_gridded

        return np.reshape(self.lons, self.field_shape)

    @property
    def grid_elevs(self):
        assert self.is_gridded

        return np.zeros(self.field_shape, np.float32)
