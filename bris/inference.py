import logging

from functools import cached_property
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from anemoi.utils.config import DotDict

from .checkpoint import Checkpoint
from .data.datamodule import DataModule
from .utils import check_anemoi_dataset_version
from .writer import CustomWriter

LOGGER = logging.getLogger(__name__)


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
                LOGGER.info(f"Specified device not set. Found GPU")
                return "cuda"
            else:
                LOGGER.info(f"Specified device not set. Could not find gpu, using CPU")
                return "cpu"
        else:
            LOGGER.info(f"Using specified device: {self._device}")
            return self._device

    @cached_property
    def strategy(self):
        if check_anemoi_dataset_version(self.checkpoint._metadata):
            LOGGER.info("Anemoi training package found, using its strategy")
            from anemoi.training.distributed.strategy import DDPGroupStrategy

            return DDPGroupStrategy(
                self.config.hardware.num_gpus_per_model,
                self.config.dataloader.get(
                    "read_group_size", self.config.hardware.num_gpus_per_model
                ),
                static_graph=not self.checkpoint.config.training.accum_grad_batches > 1,
            )
        else:
            LOGGER.info(
                "Anemoi training package not found! Using aifs-mono legacy strategy"
            )
            from bris.data.legacy.distributed.strategy import DDPGroupStrategy

            return DDPGroupStrategy(
                self.config.hardware.num_gpus_per_model,
                static_graph=not self.checkpoint.config.training.accum_grad_batches > 1,
            )

    @cached_property
    def trainer(self) -> pl.Trainer:
        trainer = pl.Trainer(
            accelerator=self.device,
            deterministic=self.deterministic,
            detect_anomaly=False,
            strategy=self.strategy,
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            precision="16-mixed",
            inference_mode=True,
            use_distributed_sampler=False,
            callbacks=self.callbacks,
        )
        return trainer

    def run(self):
        self.trainer.predict(
            self.model, datamodule=self.datamodule, return_predictions=False
        )
