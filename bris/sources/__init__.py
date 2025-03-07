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
    elif name == "anemoidataset":
        return sources.anemoidataset.AnemoiDataset(
            init_args["dataset"], init_args["variable"]
        )
    else:
        raise ValueError(f"Invalid source: {name}")


class Source:
    """Abstract base class that retrieves observations"""

    def __init__(self):
        pass

    def get(self, variable: str, start_time: int, end_time: int, frequency: int):
        """Extracts data for a given variable for a time period

        Args:
            variable: Name of variable to retrieve
            start_time: Starting date to retrieve observation from
            start_time: End date to retrieve observation to
            frequency: Frequency in seconds to retrieve observations for
        """
        raise NotImplementedError()

    @property
    def locations(self) -> list:
        """Returns a list of bris.observations.Location. Subclasses must override this."""
        raise NotImplementedError()

    @property
    def units(self) -> str:
        raise NotImplementedError()


from .frost import Frost
from .verif import Verif
from .anemoidataset import AnemoiDataset
