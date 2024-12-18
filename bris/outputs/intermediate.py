import glob
import os

import numpy as np
from bris import utils
from bris.output import Output


class Intermediate(Output):
    """This output saves data into an intermediate format, that can be used by other outputs to
    cache data. It saves one forecast run in each file (i.e. a separate file for each
    forecast_reference_time and ensemble_member
    """

    def __init__(self, predict_metadata, workdir):
        super().__init__(predict_metadata)
        self.pm = predict_metadata
        self.workdir = workdir

    def _add_forecast(self, forecast_reference_time, ensemble_member, pred):
        filename = self.get_filename(forecast_reference_time, ensemble_member)
        utils.create_directory(filename)

        np.save(filename, pred)

    def get_filename(self, forecast_reference_time, ensemble_member):
        return f"{self.workdir}/{forecast_reference_time:.0f}_{ensemble_member:.0f}.npy"

    def get_forecast_reference_times(self):
        """Returns all forecast reference times that have been saved"""
        filenames = self.get_filenames()
        frts = list()
        for filename in filenames:
            frt, _ = filename.split("/")[-1].split("_")
            frts += [int(frt)]

        frts = list(set(frts))
        frts.sort()

        return np.array(frts, np.int32)

    def get_forecast(self, forecast_reference_time, ensemble_member=None):
        """Fetches forecasts from stored numpy files

        Args:
            forecast_reference_time: Unixtime of forecast initialization [seconds]
            ensemble_member: If an integer, retrieve this member number otherwise retrieve the full
                ensemble

        Returns:
            np.array: 3D (leadtime, points, variables) if member is selected
                      4D otherwise (leadtime, points, variables, members)
        """
        assert utils.is_number(forecast_reference_time)

        if ensemble_member is None:
            shape = [
                self.pm.num_leadtimes,
                self.pm.num_points,
                self.pm.num_variables,
                self.pm.num_members,
            ]
            pred = np.nan * np.zeros(shape)
            for e in range(self.pm.num_members):
                filename = self.get_filename(forecast_reference_time, e)
                if os.path.exists(filename):
                    pred[..., e] = np.load(filename)
        else:
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
