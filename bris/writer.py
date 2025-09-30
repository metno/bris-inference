import multiprocessing
from collections.abc import Sequence

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
        process_list: list[multiprocessing.Process] | None,
        write_interval: str = "batch",
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

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction,
        batch_indices: Sequence[int] | None,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Args:
            prediction: This comes from predict_step in forecaster
        """

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
                        process = multiprocessing.Process(
                            target=output.add_forecast,
                            args=(times, ensemble_member, pred),
                        )
                        self.process_list.append(process)
                        process.start()
                        LOGGER.debug(
                            f"CustomWriter starting background process {process.name} of add_forecast for member <{ensemble_member}>, times {times}."
                        )
