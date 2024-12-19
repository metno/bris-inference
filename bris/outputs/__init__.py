import numpy as np

from bris import sources

from bris.predict_metadata import PredictMetadata


def instantiate(name: str, predict_metadata: PredictMetadata, workdir: str, init_args):
    """Creates an object of type name with config

    Args:
        predict_metadata: Contains metadata about the bathc the output will recive
        init_args: Arguments to pass to Output constructor
    """
    if name == "verif":
        # Parse obs sources
        obs_sources = list()
        for s in init_args["obs_sources"]:
            for name, opts in s.items():
                obs_sources += [sources.instantiate(name, opts)]
        init_args["obs_sources"] = obs_sources
        return Verif(predict_metadata, workdir, **init_args)

    elif name == "netcdf":
        return Netcdf(predict_metadata, workdir, **init_args)

    else:
        raise ValueError(f"Invalid output: {name}")

def get_required_variables(name, init_args):
    """What variables does this output require? Return None if it will process all variables
    provided
    """

    if name == "netcdf":
        if "variables" in init_args:
            return init_args["variables"]
        else:
            return None

    elif name == "verif":
        return [init_args["variable"]]

    else:
        raise ValueError(f"Invalid output: {name}")

def expand_tokens(string, variable):
    return string.replace("%V", variable)


class Output:
    """This class writes output for a specific part of the domain"""

    def __init__(self, predict_metadata: PredictMetadata):
        """Creates an object of type name with config

        Args:
            predict_metadata: Contains metadata about the batch the output will recieve
        """

        self.pm = predict_metadata

    def add_forecast(
        self, forecast_reference_time: int, ensemble_member: int, pred: np.array
    ):
        """
        Args:
            timestamp: Seconds since 1970
            pred: 2D array (location, variable)
        """
        assert pred.shape[0] == len(self.pm.leadtimes)
        assert pred.shape[1] == len(self.pm.lats)
        assert pred.shape[2] == len(self.pm.variables)
        assert ensemble_member >= 0
        assert ensemble_member < self.pm.num_members

        self._add_forecast(forecast_reference_time, ensemble_member, pred)

    def _add_forecast(
        self, forecast_reference_time: int, ensemble_member: int, pred: np.array
    ):
        raise NotImplementedError()

    def finalize(self):
        """Finalizes the output. Subclasses can override this if necessary."""
        pass

    def reshape_pred(self, pred):
        """Reshape predictor matrix from points to 2D based on field_shape"""
        assert self.pm.field_shape is not None

        T, L, V = pred.shape
        shape = [T, self.pm.field_shape[0], self.pm.field_shape[1], V]
        pred = np.reshape(pred, shape)
        return pred

    def flatten_pred(self, pred):
        """Reshape predictor matrix on 2D points back to the original 1D set of points"""
        assert len(pred.shape) == 4
        T, Y, X, V = pred.shape

        shape = [T, self.pm.field_shape[0] * self.pm.field_shape[1], V]
        pred = np.reshape(pred, shape)
        return pred


from .intermediate import Intermediate
from .netcdf import Netcdf
from .verif import Verif
