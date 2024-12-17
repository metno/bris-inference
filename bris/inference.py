from typing import Any, Optional

import torch
from functools import cached_property
import pytorch_lightning as pl

from anemoi.training.distributed.strategy import DDPGroupStrategy
from anemoi.utils.config import DotDict

from .checkpoint import Checkpoint
from .writer import CustomWriter
from .data.datamodule import DataModule


class Inference:
    def __init__(
        self,
        config: DotDict,
        model: pl.LightningModule,
        callbacks: Any,
        datamodule: DataModule,
        checkpoint: Checkpoint,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:

        self.config = config
        self.model = model
        self.checkpoint = checkpoint
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.deterministic = self.config.deterministic
        self.precision = precision
        self._device = device

    @property
    def device(self) -> str:
        if self._device is None:
            if torch.cuda.is_available() and torch.backends.cuda.is_built():
                return "cuda"
            else:
                return "cpu"
        else:
            return self._device

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
            precision=self.config.precision if not self.precision else self.precision,
            inference_mode=True,
            use_distributed_sampler=False,
            callbacks=self.callbacks,
        )
        return trainer

    def run(self):
        self.trainer.predict(
            self.model, datamodule=self.datamodule, return_predictions=False
        )
