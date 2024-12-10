import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.forecaster import GraphForecaster
from omegaconf import DictConfig


class BrisForecaster(GraphForecaster):
    def __init__(
        self,
        *,
        config: DictConfig,
        statistics: dict,
        data_indices: IndexCollection,
        graph_data: dict,
        metadata: dict,
        select_indices=None,
        select_indices_truth=None,
        grid_points_range=None,
    ) -> None:
        super().__init__(
            config=config,
            statistics=statistics,
            data_indices=data_indices,
            graph_data=graph_data,
            metadata=metadata,
        )
        self.select_indices = select_indices
        self.select_indices_truth = select_indices_truth
        self.grid_points_range = grid_points_range

    def advance_input_predict(
        self, x: torch.Tensor, y_pred: torch.Tensor, forcing: torch.Tensor
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[
            ..., self.data_indices.model.output.prognostic
        ]

        # get forcing constants:
        x[:, -1, :, :, self.data_indices.model.input.forcing] = forcing

        return x

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, int, int]:
        with torch.no_grad():
            batch, forcing, time_stamp = batch
            y_preds = np.zeros(
                (
                    batch.shape[0],
                    self.rollout + 1,
                    self.grid_points_range[1] - self.grid_points_range[0],
                    len(self.select_indices),
                ),
                dtype=np.float32,
            )
            y_preds[0] = (
                batch[
                    :,
                    self.multi_step - 1,
                    0,
                    self.grid_points_range[0] : self.grid_points_range[1],
                    self.select_indices_truth,
                ]
                .cpu()
                .numpy()
            )  # insert truth at lead time 0

            batch = self.model.pre_processors(batch)
            # x = batch[:, 0 : self.multi_step, :, :, self.data_indices.data.input.full]
            x = batch[..., self.data_indices.data.input.full]
            # print("batch shape", batch.shape)

            for rollout_step in range(self.rollout):
                # LOGGER.info("rollout step %i", rollout_step)
                y_pred = self(x)
                x = self.advance_input_predict(x, y_pred, forcing[:, rollout_step])

                # print("ypred shape", y_pred.shape)
                y_preds[:, rollout_step + 1] = (
                    self.model.post_processors(y_pred, in_place=False)[
                        :,
                        0,
                        self.grid_points_range[0] : self.grid_points_range[1],
                        self.select_indices,
                    ]
                    .cpu()
                    .numpy()
                )

            return [
                y_preds,
                self.model_comm_group_rank,
                self.model_comm_group_id,
                time_stamp,
            ]  # return the model parallel rank so that only one process per model writes to file.
