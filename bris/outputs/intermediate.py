import numpy as np
import glob
import os


from ..output import Output
from bris import utils


class Intermediate(Output):
    """This output saves data into an intermediate format, that can be used by other outputs to
    cache data. It saves one forecast run in each file (i.e. a separate file for each
    forecast_reference_time and ensemble_member
    """
    def __init__(self, predict_metadata, workdir):
        self.pm = predict_metadata
        self.workdir = workdir

    def _add_forecast(self, forecast_reference_time, ensemble_member, pred):
        filename = self.get_filename(forecast_reference_time, ensemble_member)
        utils.create_directory(filename)

        np.save(filename, pred)

    def get_filename(self, forecast_reference_time, ensemble_member):
        return f"{self.workdir}/{forecast_reference_time:d}_{ensemble_member:d}.npy"

    def get_forecast_reference_times(self):
        """Returns all forecast reference times that have been saved"""
        filenames = self.get_filenames()
        frts = list()
        for filename in filenames:
            frt, _ = filename.split('/')[-1].split('_')
            frts += [int(frt)]

        frts = list(set(frts))
        frts.sort()

        return frts

    def get_forecast(self, forecast_reference_time, ensemble_member):
        assert isinstance(forecast_reference_time, int)
        assert isinstance(ensemble_member, int)

        filename = self.get_filename(forecast_reference_time, ensemble_member)
        if os.path.exists(filename):
            pred = np.load(filename)
        else:
            pred = None

        return pred

    def get_filenames(self):
        return glob.glob(f"{self.workdir}/*_*.npy")

    def finalize(self):
        # clean up files
        for filename in self.get_filenames():
            # delete file
            pass
