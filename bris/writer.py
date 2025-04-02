import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    """This class is used in a callback to the trainer to write data into output"""

    def __init__(self, outputs: dict, write_interval):
        """
        Args:
            outputs (dict): Dict of domain-name to dict, where dict has "start", "end", and
                "outputs", where "outputs" is a list of Output objects that the writer will call
        """
        super().__init__(write_interval)

        self.outputs = outputs

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
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
                    output.add_forecast(
                        times, ensemble_member, pred
                    )  # change timestamp to times
