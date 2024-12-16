from bris.predict_metadata import PredictMetadata


def instantiate(name, predict_metadata: PredictMetadata, workdir: str, init_args):
    """Creates an object of type name with config

    Args:
        predict_metadata: Contains metadata about the bathc the output will recive
        init_args: Arguments to pass to Output constructor
    """
    if name == "frost":
        return sources.Frost(init_args["frost_variable_name"])
    elif name == "verif_netcdf":
        return sources.VerifNetcdf(init_args["filename"])

def expand_tokens(string, variable):
    return string.replace("%V", variable)


class Source:
    """Abstract base class that retrieves observations"""

    def __init__(self):
        pass

    def get(self, variable, start_time, end_time, frequency):
        """Extracts data for a given variable for a time period"""
        raise NotImplementedError()

    @property
    def locations(self):
        """Returns a list of the available locations"""
        raise NotImplementedError
