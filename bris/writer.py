import threading

import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter

from .utils import LOGGER


class CustomWriter(BasePredictionWriter):
    """This class is used in a callback to the trainer to write data to output."""

    def __init__(
        self, outputs: dict, thread_list: list, write_interval: str = "batch"
    ) -> None:
        """
        Args:
            outputs (dict): Dict of domain-name to dict, where dict has "start", "end", and
                "outputs", where "outputs" is a list of Output objects that the writer will call.

            thread_list (list): reference to empty list to add new Thread() objects to, so the
                caller can keep track of background writer threads spawned by this function. Caller
                must run .join() on each thread in list to wait for them to finish.

            write_interval (str): Only "batch" is supported.
        """
        super().__init__(write_interval)

        self.outputs = outputs
        self.thread_list = thread_list

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
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
                    if self.thread_list is None:  # Disable threading
                        output.add_forecast(times, ensemble_member, pred)
                        LOGGER.debug(
                            f"CustomWriter starting add_forecast for member <{ensemble_member}>, "
                            f"{output.filename_pattern}"
                        )
                    else:
                        thread = threading.Thread(
                            target=output.add_forecast,
                            args=(times, ensemble_member, pred),
                        )
                        self.thread_list.append(thread)
                        thread.start()
                        LOGGER.debug(
                            f"CustomWriter starting background thread {thread.name} of add_forecast for member <{ensemble_member}>, "
                            f"{output.filename_pattern}."
                        )
