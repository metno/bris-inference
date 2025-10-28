import multiprocessing
import time
from collections.abc import Sequence
from multiprocessing.sharedctypes import Synchronized

import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.trainer.trainer import Trainer

from .utils import LOGGER


class CustomWriter(BasePredictionWriter):
    """This class is used in a callback to the trainer to write data to output."""

    def __init__(
        self,
        outputs: list[dict],
        current_batch_no: Synchronized,
        process_list: list[multiprocessing.Process] | None,
        write_interval: str = "batch",
        max_processes: int = 2,
    ) -> None:
        """
        Args:
            outputs (dict): Dict of domain-name to dict, where dict has "start", "end", and
                "outputs", where "outputs" is a list of Output objects that the writer will call.

            process_list (list): reference to empty list to add new process objects to, so the
                caller can keep track of background writer processs spawned by this function. Caller
                must run .join() on each process in list to wait for them to finish.

            write_interval (str): Only "batch" is supported.
        """
        super().__init__(write_interval)

        self.outputs = outputs
        self.process_list = process_list
        self.current_batch_no = current_batch_no
        self.max_processes = max_processes

    def write_on_batch_end(
        self,
        trainer: Trainer,  # Not used
        pl_module: LightningModule,  # Not used
        prediction,
        batch_indices: Sequence[int] | None,  # Not used
        batch,  # Not used
        batch_idx: int,
        dataloader_idx: int,  # Not used
    ) -> None:
        """
        Args:
            prediction: This comes from predict_step in forecaster
        """

        LOGGER.debug(f"CustomWriter self.process_list contains {self.process_list}")

        times = prediction["times"]
        ensemble_member = prediction["ensemble_member"]

        # TODO: Why is this here, don't we want all data-parallel processes to write to disk?
        if prediction["group_rank"] == 0:  # related to model parallel?
            for output_dict in self.outputs:
                pred = prediction["pred"][output_dict["decoder_index"]]
                assert pred.shape[0] == 1, "Batchsize (per dataparallel) should be 1"
                pred = np.squeeze(pred, axis=0)
                pred = pred[
                    ...,
                    output_dict["start_gridpoint"] : output_dict["end_gridpoint"],
                    :,
                ]

                for output in output_dict["outputs"]:
                    if self.process_list is None:  # Disable background processes
                        output.add_forecast(times, ensemble_member, pred)
                        LOGGER.debug(
                            f"CustomWriter starting add_forecast for member <{ensemble_member}>, times {times}."
                        )
                    else:
                        # If on new batch, wait for previous to complete writing
                        if self.current_batch_no.value != batch_idx:
                            while len(self.process_list) > 0:
                                t0 = time.perf_counter()
                                p = self.process_list.pop()
                                p.join()
                                LOGGER.debug(
                                    f"CustomWriter waited {time.perf_counter() - t0:.1f}s for {p} to complete previous batch_idx ({self.current_batch_no.value})."
                                )
                        # If too many processes, wait for previous to complete writing
                        if len(self.process_list) > self.max_processes:
                            while len(self.process_list) > 0:
                                t0 = time.perf_counter()
                                p = self.process_list.pop()
                                p.join()
                                LOGGER.debug(
                                    f"CustomWriter waited {time.perf_counter() - t0:.1f}s for {p} to complete before creating new processes."
                                )
                        with self.current_batch_no.get_lock():
                            self.current_batch_no.value = batch_idx

                        process = multiprocessing.Process(
                            target=output.add_forecast,
                            args=(times, ensemble_member, pred),
                        )
                        self.process_list.append(process)
                        process.start()
                        LOGGER.debug(
                            f"CustomWriter starting {process.name} of add_forecast for member <{ensemble_member}>, times {times} for writing, batch_idx {batch_idx}."
                        )
