class Source:
    """Abstract base class that retrieves observations"""

    def __init__(self):
        pass

    def get(self, variable, start_time, end_time, frequency):
        pass
