from bris import sources


def instantiate(name: str, init_args: dict):
    """Creates an object of type name with config

    Args:
        predict_metadata: Contains metadata about the bathc the output will recive
        init_args: Arguments to pass to Output constructor
    """
    if name == "frost":
        return sources.frost.Frost(init_args["frost_variable_name"])
    elif name == "verif":
        return sources.verif.Verif(init_args["filename"])
    else:
        raise ValueError(f"Invalid source: {name}")


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

from .frost import Frost
from .verif import Verif
