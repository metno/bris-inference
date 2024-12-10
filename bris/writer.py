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
        # TODO: What are the different elements of the prediction list?
        # This comes from predict_step in forecaster...

        timestamp = prediction[3][0].split(":")[0]

        if prediction[1] == 0:  # if on first model parallel gpu of data parallel group
            # pred = prediction[0]
            # time_index = int(self.avail_samples//self.data_par_num * prediction[2] + batch_idx)
            # pred = pred[0] #remove batch dimension for now since we will always use data parallel for larger batches.
            for grid_name, grid_config in self.outputs.items():
                # Filter grid points
                if "start" in grid_config:
                    start_index = grid_config["start"]
                    end_index = grid_config["end"]
                    pred = prediction[0][0, :, start_index:end_index]
                else:
                    pred = prediction[0][0, :, :]

                # This should be done by the output class
                # pred = self.I.to_grid(pred, grid)
                # pred = self.I.map_to_dict(pred, self.I.inf_conf.select.vars)
                # pred["wind_speed_10m"] = np.sqrt(
                #     pred["x_wind_10m"] ** 2 + pred["y_wind_10m"] ** 2
                # )

                for output in grid_config["outputs"]:
                    output.add(timestamp, pred)
