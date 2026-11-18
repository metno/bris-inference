import glob
import os
import re
import time

import numpy as np
import zarr

from bris import utils, zarrstore
from bris.outputs import Output
from bris.predict_metadata import PredictMetadata


class Intermediate(Output):
    """This output saves data into an intermediate format, that can be used by other outputs to
    cache data. It saves one forecast run in each Zarr store (i.e. a separate store for each
    forecast_reference_time), with one chunk per ensemble member, so that the members can be
    written by different processes at the same time (see bris.zarrstore).
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        extra_variables: list | None = None,
    ) -> None:
        super().__init__(predict_metadata, extra_variables)
        self.workdir = workdir

    @property
    def member_shape(self) -> tuple:
        """Shape of what is saved for one ensemble member"""
        return tuple(self.pm.shape)

    def _add_forecast(self, times, ensemble_member, pred) -> None:
        t0 = time.perf_counter()
        filename = self.get_filename(times[0])
        shape = self.member_shape + (self.pm.num_members,)

        def build(tmp: str) -> None:
            zarr.open_group(tmp, mode="w").create_dataset(
                "data",
                shape=shape,
                chunks=self.member_shape + (1,),
                dtype="f4",
                fill_value=np.float32(np.nan),
                compressor=None,
            )

        zarrstore.ensure(filename, build, fingerprint=str(shape))
        zarr.open_group(filename, mode="r+")["data"][..., ensemble_member] = pred
        utils.LOGGER.debug(
            f"Intermediate._add_forecast for {filename} in {time.perf_counter() - t0:.1f}s"
        )

    def get_filename(self, forecast_reference_time: str) -> str:
        frt_ut = utils.datetime_to_unixtime(forecast_reference_time)
        return f"{self.workdir}/{frt_ut:.0f}.zarr"

    def get_forecast_reference_times(self) -> list[np.datetime64]:
        """Returns all forecast reference times that have been saved"""
        frts = [
            utils.unixtime_to_datetime(int(os.path.basename(filename).split(".")[0]))
            for filename in self.get_filenames()
        ]
        frts.sort()

        return frts

    def get_forecast(
        self,
        forecast_reference_time: str,
        ensemble_member: int | None = None,
    ) -> np.ndarray | None:
        """Fetches forecasts from the stores

        Args:
            forecast_reference_time: Unixtime of forecast initialization [seconds]
            ensemble_member: If an integer, retrieve this member number otherwise retrieve the full
                ensemble

        Returns:
            np.array: 3D (leadtime, points, variables) if member is selected, or None if that
                          member has not been saved
                      4D otherwise (leadtime, points, variables, members), with NaN for the
                          members that have not been saved
        """

        t0 = time.perf_counter()
        filename = self.get_filename(forecast_reference_time)
        data = None
        if os.path.exists(filename):
            data = zarr.open_group(filename, mode="r")["data"]

        if ensemble_member is None:
            if data is None:
                shape = self.member_shape + (self.pm.num_members,)
                pred = np.full(shape, np.nan, dtype=np.float32)
            else:
                pred = data[...]
        else:
            assert isinstance(ensemble_member, int)
            pred = None
            if data is not None and ensemble_member in self._saved_members(filename):
                pred = data[..., ensemble_member]
        utils.LOGGER.debug(
            f"Intermediate.get_forecast for {filename} in {time.perf_counter() - t0:.1f}s"
        )
        return pred

    def _saved_members(self, filename: str) -> set[int]:
        """Which ensemble members have been saved in this store?"""
        members = set()
        for chunk in os.listdir(os.path.join(filename, "data")):
            if re.fullmatch(r"\d+(\.\d+)*", chunk):
                members.add(int(chunk.split(".")[-1]))
        return members

    @property
    def num_members(self) -> int:
        """One more than the highest ensemble member that has been saved"""
        members = set()
        for filename in self.get_filenames():
            members |= self._saved_members(filename)
        return max(members, default=-1) + 1

    def get_filenames(self) -> list[str]:
        return [
            filename
            for filename in glob.glob(f"{self.workdir}/*.zarr")
            if re.fullmatch(r"-?\d+\.zarr", os.path.basename(filename))
        ]

    def cleanup(self) -> None:
        """Removes up all intermediate files and removes the workdir. Called in finalize of the main output."""
        t0 = time.perf_counter()
        for _filename in self.get_filenames():
            zarrstore.remove(_filename)

        try:
            os.rmdir(self.workdir)
        except OSError as e:
            utils.LOGGER.warning(f"Error removing workdir {self.workdir}: {e}")
        utils.LOGGER.debug(f"Intermediate.cleanup in {time.perf_counter() - t0:.1f}s")

    def finalize(self):
        pass


class IntermediateSpatial(Intermediate):
    """Intermediate output for spatial metrics, inheriting from Intermediate."""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        metric_shape: tuple,
        extra_variables: list | None = None,
    ) -> None:
        super().__init__(predict_metadata, workdir, extra_variables)
        self.metric_shape = metric_shape

    @property
    def member_shape(self) -> tuple:
        return (self.pm.num_leadtimes,) + tuple(self.metric_shape)
