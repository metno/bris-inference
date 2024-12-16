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
