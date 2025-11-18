import torch

from .brispredictor import BrisPredictor
from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..utils import LOGGER

class MultiDomainPredictor(BrisPredictor):
    def __init__(
        self,
        *args,
        checkpoints: dict[str, Checkpoint],
        datamodule: DataModule,
        checkpoints_config: dict,
        required_variables: dict,
        release_cache: bool = False,
        fcstep_const: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            checkpoints=checkpoints,
            datamodule=datamodule,
            checkpoints_config=checkpoints_config,
            required_variables=required_variables,
            release_cache=release_cache,
            fcstep_const=fcstep_const,
            **kwargs,
        )

        LOGGER.info("Multi domain predictor is initialized")
        self.graph_label = checkpoints_config["forecaster"].get("graph_label", None)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.graph_label:
            return self.model(
                x, 
                model_comm_group=self.model_comm_group,
                graph_label=self.graph_label, 
                **kwargs
                )
        else:
            kwargs["graph_label"] = None
            return super().forward(x, **kwargs)

    @torch.inference_mode
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        predictions = super().predict_step(
            batch=batch,
            batch_idx=batch_idx
        )
        return predictions


