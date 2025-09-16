import glob
import os
import time
from typing import Optional

import numpy as np

from bris import utils
from bris.outputs import Output
from bris.predict_metadata import PredictMetadata


class Intermediate(Output):
    """This output saves data into an intermediate format, that can be used by other outputs to
    cache data. It saves one forecast run in each file (i.e. a separate file for each
    forecast_reference_time and ensemble_member
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        extra_variables: Optional[list] = None,
    ) -> None:
        super().__init__(predict_metadata, extra_variables)
        self.workdir = workdir

    def _add_forecast(self, times, ensemble_member, pred) -> None:
        t0 = time.perf_counter()
        filename = self.get_filename(times[0], ensemble_member)
        utils.create_directory(filename)

        np.save(filename, pred)
        utils.LOGGER.debug(
            f"Intermediate._add_forecast for {filename} in {time.perf_counter() - t0:.1f}s"
        )

    def get_filename(self, forecast_reference_time, ensemble_member) -> str:
        frt_ut = utils.datetime_to_unixtime(forecast_reference_time)
        return f"{self.workdir}/{frt_ut:.0f}_{ensemble_member:.0f}.npy"

    def get_forecast_reference_times(self) -> list[np.datetime64]:
        """Returns all forecast reference times that have been saved"""
        filenames = self.get_filenames()
        frts = []
        for filename in filenames:
            frt_ut, _ = filename.split("/")[-1].split("_")
            frt = utils.unixtime_to_datetime(int(frt_ut))
            frts += [frt]

        frts = list(set(frts))
        frts.sort()

        return frts

    def get_forecast(
        self, forecast_reference_time, ensemble_member=None
    ) -> np.ndarray | None:
        """Fetches forecasts from stored numpy files

        Args:
            forecast_reference_time: Unixtime of forecast initialization [seconds]
            ensemble_member: If an integer, retrieve this member number otherwise retrieve the full
                ensemble

        Returns:
            np.array: 3D (leadtime, points, variables) if member is selected
                      4D otherwise (leadtime, points, variables, members)
        """

        t0 = time.perf_counter()
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
                utils.LOGGER.debug(
                    f"Intermediate.get_forecast for {filename} in {time.perf_counter() - t0:.1f}s"
                )
        else:
            assert isinstance(ensemble_member, int)

            filename = self.get_filename(forecast_reference_time, ensemble_member)
            pred = np.load(filename) if os.path.exists(filename) else None
            utils.LOGGER.debug(
                f"Intermediate.get_forecast for {filename} in {time.perf_counter() - t0:.1f}s"
            )
        return pred

    @property
    def num_members(self) -> int:
        filenames = self.get_filenames()

        max_member = 0
        for filename in filenames:
            _, member = filename.split("/")[-1].split(".npy")[0].split("_")
            max_member = max(int(member), max_member)
        return max_member + 1

    def get_filenames(self) -> list[str]:
        return glob.glob(f"{self.workdir}/*_*.npy")

    def cleanup(self) -> None:
        """Removes up all intermediate files and removes the workdir. Called in finalize of the main output."""

        t0 = time.perf_counter()
        for _filename in self.get_filenames():
            try:
                os.remove(_filename)
            except OSError as e:
                utils.LOGGER.error(f"Error during cleanup of {_filename}: {e}")

        try:
            os.rmdir(self.workdir)
        except OSError as e:
            utils.LOGGER.error(f"Error removing workdir {self.workdir}: {e}")
        utils.LOGGER.debug(f"Intermediate.cleanup in {time.perf_counter() - t0:.1f}s")

    async def finalize(self):
        pass
