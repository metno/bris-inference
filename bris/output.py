import outputs
from .predict_metadata import PredictMetadata


def instantiate(name, predict_metadata: PredictMetadata, init_args):
    """Creates an object of type name with config

    Args:
        predict_metadata: Contains metadata about the bathc the output will recive
        init_args: Arguments to pass to Output constructor
    """
    if name == "verif":
        filename = expand_tokens(init_args["filename"], init_args["variable"])
        return outputs.Verif(filename, init_args["frost_variable_name"])
    elif name == "netcdf":
        return outputs.Netcdf(init_args["filename"])


def expand_tokens(string, variable):
    return string.replace("%V", variable)


class Output:
    """This class writes output for a specific part of the domain"""

    def __init__(self, predict_metadata: PredictMetadata):
        """Creates an object of type name with config

        Args:
            predict_metadata: Contains metadata about the bathc the output will recive
        """

        self.pm = predict_metadata

    def add_forecast(self, forecast_reference_time: int, pred: np.array):
        """
        Args:
            timestamp: Seconds since 1970
            pred: 2D array (location, variable)
        """
        assert array.shape[0] == len(self.pm.lats)
        assert array.shape[1] == len(self.pm.variables)

        self._add(forecast_reference_time, pred)

    @abstractmethod
    def _add_forecast(self, forecast_reference_time: int, pred: np.array):
        raise NotImplementedError()

    def flush(self):
        """Flushes data to disk. Subclasses can override this if necessary."""
        pass

    def finalize(self):
        """Finalizes the output. Subclasses can override this if necessary."""
        pass
