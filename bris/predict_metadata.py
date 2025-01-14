import numpy as np

from bris import utils


class PredictMetadata:
    """This class stores metadata about each dimension of a batch"""

    def __init__(
        self, variables, lats, lons, altitudes, num_leadtimes, num_members, field_shape=None
    ):
        assert utils.is_number(num_leadtimes)
        if field_shape is not None:
            assert np.prod(field_shape) == len(lats), (field_shape, len(lats))

        self.variables = variables

        # Ensure lons are on the interval -180, 180
        self.lats = np.array(lats)
        self.lons = np.array(lons)

        assert self.lats.shape == self.lons.shape, (self.lats.shape, self.lons.shape)

        self.lons[self.lons < -180] += 360
        self.lons[self.lons > 180] -= 360

        self.altitudes = altitudes
        self.num_leadtimes = num_leadtimes
        self.num_members = num_members
        self.field_shape = field_shape

    @property
    def num_points(self):
        return len(self.lats)

    @property
    def num_variables(self):
        return len(self.variables)

    @property
    def shape(self):
        """The shape that you can expect predict_step to provide. Note that ensemble is done in
        data parallel, so not included in this shape.
        """
        # This is goverend by predict_step in forecaster.py
        return [self.num_leadtimes, len(self.lats), len(self.variables)]

    @property
    def is_gridded(self):
        return self.field_shape is not None and len(self.field_shape) == 2

    @property
    def grid_lats(self):
        assert self.is_gridded

        return np.reshape(self.lats, self.field_shape)

    @property
    def grid_lons(self):
        assert self.is_gridded

        return np.reshape(self.lons, self.field_shape)

    @property
    def grid_altitudes(self):
        assert self.is_gridded

        return np.reshape(self.altitudes, self.field_shape)
