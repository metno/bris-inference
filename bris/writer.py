import os
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

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
        process_list: list[Future] | None,
        write_interval: str = "batch",
        max_processes: int = os.cpu_count(),
    ) -> None:
        """
        Args:
            outputs (dict): Dict of domain-name to dict, where dict has "start", "end", and
                "outputs", where "outputs" is a list of Output objects that the writer will call.

            process_list (list): reference to empty list to add new process objects to, so the
                caller can keep track of background writer processs spawned by this function. Caller
                must run .join() on each process in list to wait for them to finish.

            write_interval (str): Only "batch" is supported.

            max_processes (int): Max background writing processes. Don't set <1.
        """
        super().__init__(write_interval)

        self.outputs = outputs
        self.process_list = process_list
        if max_processes < 1:
            max_processes = 1
        self.pool = ThreadPoolExecutor(max_workers=max_processes)
        LOGGER.debug(f"CustomWriter max_processes set to {max_processes}")

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

        # Wait for processes from the previous batch to finish
        LOGGER.debug(f"CustomWriter process_list contains {self.process_list}")
        while len(self.process_list) > 0:
            LOGGER.debug(
                "CustomWriter waiting for previous process to complete before writing new data."
            )
            process = self.process_list.pop()
            process.result()
            LOGGER.debug("CustomWriter previous process completed.")

        times = prediction["times"]
        ensemble_member = prediction["ensemble_member"]
        if prediction["group_rank"] == 0:
            for output_dict in self.outputs:
                pred = prediction["pred"][output_dict["decoder_name"]]
                assert pred.shape[0] == 1, "Batchsize (per dataparallel) should be 1"
                pred = np.squeeze(pred, axis=0)
                pred = pred[
                    ...,
                    output_dict["start_gridpoint"] : output_dict["end_gridpoint"],
                    :,
                ]

                for output in output_dict["outputs"]:
                    if self.process_list is not None:
                        self.process_list.append(
                            self.pool.submit(
                                output.add_forecast, times, ensemble_member, pred
                            )
                        )
                        LOGGER.debug(
                            f"CustomWriter starting async add_forecast for member <{ensemble_member}>, times {times} for writing, batch_idx {batch_idx}."
                        )
                    else:
                        output.add_forecast(times, ensemble_member, pred)
                        LOGGER.debug(
                            f"CustomWriter added forecast for member <{ensemble_member}>, times {times} for writing, batch_idx {batch_idx}."
                        )

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # Not used
    ) -> None:
        """Called when prediction ends."""
        LOGGER.debug(
            "CustomWriter on_predict_end called, waiting for all processes to finish."
        )
        # Wait for all processes to finish
        if self.process_list is not None:
            while len(self.process_list) > 0:
                process = self.process_list.pop()
                process.result()
        LOGGER.debug("CustomWriter all processes finished.")
        self.pool.shutdown(wait=True)
