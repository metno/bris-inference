import torch
from functools import cached_property
import pytorch_lightning as pl


from .checkpoint import Checkpoint
from .writer import CustomWriter


class Inference:
    def __init__(
        self,
        model,
        datamodule,
        callbacks,
        num_gpus_per_node=1,
        num_nodes=1,
        precision=None,
        device=None,
        deterministic=True,
    ):

        self.checkpoint = checkpoint
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.deterministic = deterministic

        self._device = device

    def run(self):
        self.trainer.predict(self.model, datamodule=self.datamodule, return_predictions=False)

    @cached_property
    def trainer(self) -> pl.Trainer:
        trainer = pl.Trainer(
            accelerator=self.device,
            deterministic=self.deterministic,
            detect_anomaly=self.config.diagnostics.debug.anomaly_detection,
            strategy=DDPGroupStrategy(
                self.config.hardware.num_gpus_per_model,
                static_graph=not self.config.training.accum_grad_batches > 1,
            ),
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            precision=self.config.training.precision,
            inference_mode=True,
            use_distributed_sampler=False,
            callbacks=self.callbacks,
        )
        return trainer

    @property
    def device(self) -> str:
        if self._device is None:
            if torch.cuda.is_available() and torch.backends.cuda.is_built():
                return "cuda"
            else:
                return "cpu"
        else:
            return self._device
