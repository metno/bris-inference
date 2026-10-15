import glob
import os
import time

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
        extra_variables: list | None = None,
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

    def get_filename(self, forecast_reference_time: str, ensemble_member: int) -> str:
        frt_ut = utils.datetime_to_unixtime(forecast_reference_time)
        return f"{self.workdir}/{frt_ut:.0f}_{ensemble_member:.0f}.npy"

    def get_forecast_reference_times(self) -> list[np.datetime64]:
        """Returns all forecast reference times that have been saved"""
        filenames = self.get_filenames()
        frts: list[np.datetime64] = []
        for filename in filenames:
            frt_ut, _ = filename.split("/")[-1].split("_")
            frt = utils.unixtime_to_datetime(int(frt_ut))
            frts += [frt]

        frts = list(set(frts))
        frts.sort()

        return frts

    def get_forecast(
        self, forecast_reference_time: str, ensemble_member: int | None = None
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
            pred = np.nan * np.zeros(shape, dtype=np.float32)
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
                utils.LOGGER.warning(f"Error during cleanup of {_filename}: {e}")

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

    def get_forecast(self, forecast_reference_time, ensemble_member=None):
        if ensemble_member is None:
            shape = (
                (self.pm.num_leadtimes,) + self.metric_shape + (self.pm.num_members,)
            )
            pred = np.nan * np.zeros(shape, dtype=np.float32)
            for e in range(self.pm.num_members):
                filename = self.get_filename(forecast_reference_time, e)
                if os.path.exists(filename):
                    pred[..., e] = np.load(filename)
        else:
            assert isinstance(ensemble_member, int)

            filename = self.get_filename(forecast_reference_time, ensemble_member)
            pred = np.load(filename) if os.path.exists(filename) else None

        return pred


class IntermediateEnsembleAccumulator:
    """Stores quantities that have been reduced across ensemble members (e.g. ensemble sums),
    instead of one forecast per member. One directory is used per forecast reference time, holding
    one .npy file per accumulated quantity plus a member count.

    Contributions for the same forecast reference time can be added several times (e.g. when
    members run in sequence, or when several groups of members are reduced separately). They are
    combined with the reduction operation registered for each quantity ("sum", "min" or "max").
    All accumulation happens through read-modify-write of the files, so calls for the same
    forecast reference time must come from a single process and must not overlap in time. The
    writer guarantees this, since only the ensemble-root rank receives reduced contributions and
    it waits for the previous batch to be stored before starting on the next.
    """

    valid_reductions = ("sum", "min", "max")
    count_name = "num_members"

    def __init__(self, workdir: str, reductions: dict[str, str]) -> None:
        """
        Args:
            workdir: Directory to store files in
            reductions: name -> reduction operation for each quantity to accumulate
        """
        for name, op in reductions.items():
            if op not in self.valid_reductions:
                raise ValueError(
                    f"Invalid reduction '{op}' for '{name}'. Must be one of {self.valid_reductions}"
                )
            if name == self.count_name or not name.isidentifier():
                raise ValueError(f"Invalid accumulator name '{name}'")
        self.workdir = workdir
        self.reductions = dict(reductions)

    def accumulate(
        self,
        forecast_reference_time,
        contributions: dict[str, np.ndarray],
        num_members: int,
    ) -> None:
        """Combines contributions (already reduced across num_members members) with what has
        been stored for this forecast reference time"""
        t0 = time.perf_counter()
        assert set(contributions.keys()) == set(self.reductions.keys()), (
            sorted(contributions.keys()),
            sorted(self.reductions.keys()),
        )
        assert num_members > 0

        directory = self.get_directory(forecast_reference_time)
        os.makedirs(directory, exist_ok=True)

        for name, op in self.reductions.items():
            filename = self.get_filename(forecast_reference_time, name)
            new = np.asarray(contributions[name])
            if os.path.exists(filename):
                prev = np.load(filename)
                assert prev.shape == new.shape, (name, prev.shape, new.shape)
                new = self.combine(prev, new, op)
            self._atomic_save(filename, new)

        count = self.get_num_members(forecast_reference_time) + num_members
        self._atomic_save(
            self.get_filename(forecast_reference_time, self.count_name),
            np.array(count, dtype=np.int64),
        )
        utils.LOGGER.debug(
            f"IntermediateEnsembleAccumulator.accumulate {num_members} members into {directory} in {time.perf_counter() - t0:.1f}s"
        )

    @staticmethod
    def combine(prev: np.ndarray, new: np.ndarray, op: str) -> np.ndarray:
        if op == "sum":
            return prev + new
        if op == "min":
            return np.minimum(prev, new)
        if op == "max":
            return np.maximum(prev, new)
        raise ValueError(f"Unknown reduction '{op}'")

    @staticmethod
    def _atomic_save(filename: str, array: np.ndarray) -> None:
        """Write to a temporary file and rename, so a crash cannot leave a partial file"""
        tmp_filename = filename + ".tmp.npy"
        np.save(tmp_filename, array)
        os.replace(tmp_filename, filename)

    def get(self, forecast_reference_time) -> dict[str, np.ndarray]:
        """Returns the accumulated quantities for this forecast reference time"""
        ret = {}
        for name in self.reductions:
            filename = self.get_filename(forecast_reference_time, name)
            ret[name] = np.load(filename)
        return ret

    def get_num_members(self, forecast_reference_time) -> int:
        """How many members have been accumulated for this forecast reference time?"""
        filename = self.get_filename(forecast_reference_time, self.count_name)
        if not os.path.exists(filename):
            return 0
        return int(np.load(filename))

    def get_directory(self, forecast_reference_time) -> str:
        frt_ut = utils.datetime_to_unixtime(forecast_reference_time)
        return f"{self.workdir}/{frt_ut:.0f}"

    def get_filename(self, forecast_reference_time, name: str) -> str:
        return f"{self.get_directory(forecast_reference_time)}/{name}.npy"

    def get_forecast_reference_times(self) -> list[np.datetime64]:
        """Returns all forecast reference times that have been accumulated"""
        frts = []
        for directory in glob.glob(f"{self.workdir}/*"):
            basename = os.path.basename(directory)
            if os.path.isdir(directory) and basename.isdigit():
                frts += [utils.unixtime_to_datetime(int(basename))]
        frts.sort()
        return frts

    def cleanup(self) -> None:
        """Removes all intermediate files and the workdir"""
        t0 = time.perf_counter()
        for frt in self.get_forecast_reference_times():
            directory = self.get_directory(frt)
            for filename in glob.glob(f"{directory}/*.npy"):
                try:
                    os.remove(filename)
                except OSError as e:
                    utils.LOGGER.warning(f"Error during cleanup of {filename}: {e}")
            try:
                os.rmdir(directory)
            except OSError as e:
                utils.LOGGER.warning(f"Error removing directory {directory}: {e}")
        try:
            os.rmdir(self.workdir)
        except OSError as e:
            utils.LOGGER.warning(f"Error removing workdir {self.workdir}: {e}")
        utils.LOGGER.debug(
            f"IntermediateEnsembleAccumulator.cleanup in {time.perf_counter() - t0:.1f}s"
        )
