class GriddedNetcdf(bris.output.Output):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def flush(self):
        pass

    def finalize(self):
        pass
