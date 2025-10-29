import logging
import time
from functools import cached_property
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from anemoi.utils.config import DotDict

from bris.ddp_strategy import DDPGroupStrategy
from bris.utils import LOGGER

from .data.datamodule import DataModule


class Inference:
    def __init__(
        self,
        config: DotDict,
        model: pl.LightningModule,
        callbacks: Any,
        datamodule: DataModule,
        num_gpus_per_ensemble,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.precision = precision
        self._device = device
        self.num_gpus_per_ensemble = num_gpus_per_ensemble

        torch.set_float32_matmul_precision("high")

    @property
    def device(self) -> str:
        if self._device is None:
            if torch.cuda.is_available() and torch.backends.cuda.is_built():
                LOGGER.info("Specified device not set. Found GPU")
                return "cuda"

            LOGGER.warning("Specified device not set. Could not find gpu, using CPU")
            return "cpu"

        LOGGER.info("Using specified device: %s", self._device)
        return self._device

    @cached_property
    def strategy(self):
        return DDPGroupStrategy(
            num_gpus_per_model=self.config.hardware.num_gpus_per_model,
            num_gpus_per_ensemble=self.num_gpus_per_ensemble,
            read_group_size=1,
            static_graph=False,
        )

    @cached_property
    def trainer(self) -> pl.Trainer:
        trainer = pl.Trainer(
            logger=False,
            accelerator=self.device,
            deterministic=False,
            detect_anomaly=False,
            strategy=self.strategy,
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            precision="bf16",
            inference_mode=True,
            use_distributed_sampler=False,
            callbacks=self.callbacks,
        )
        return trainer

    def run(self):
        t0 = time.perf_counter()
        LOGGER.debug("Bris/Inference/run Predicting")
        self.trainer.predict(
            self.model, datamodule=self.datamodule, return_predictions=False
        )
        LOGGER.debug(f"bris/Inference.run: {time.perf_counter() - t0:.1f}s")
