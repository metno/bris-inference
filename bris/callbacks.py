import json
import logging
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pytorch_lightning as pl
import torch
import torchinfo
from anemoi.utils.config import DotDict
from omegaconf import DictConfig
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

LOGGER = logging.getLogger(__name__)


class NewModelCheckpoint(ModelCheckpoint):
    """A checkpoint callback that saves the model after every validation epoch."""

    def __init__(self, ckpt_metadata, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ckpt_metadata = ckpt_metadata
        self._model_metadata = None
        self._tracker_metadata = None
        self._tracker_name = None

    def _torch_drop_down(self, trainer: pl.Trainer) -> torch.nn.Module:
        # Get the model from the DataParallel wrapper, for single and multi-gpu cases
        assert hasattr(
            trainer, "model"
        ), "Trainer has no attribute 'model'! Is the Pytorch Lightning version correct?"
        return (
            trainer.model.module.model
            if hasattr(trainer.model, "module")
            else trainer.model.model
        )

    @rank_zero_only
    def model_metadata(self, model):
        if self._model_metadata is not None:
            return self._model_metadata

        self._model_metadata = {
            "model": model.__class__.__name__,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "summary": repr(
                torchinfo.summary(
                    model,
                    depth=50,
                    verbose=0,
                    row_settings=["var_names"],
                ),
            ),
        }

        return self._model_metadata

    def _save_checkpoint(
        self, trainer: pl.Trainer, lightning_checkpoint_filepath: str
    ) -> None:
        if trainer.is_global_zero:
            model = self._torch_drop_down(trainer)
            # We want a different uuid each time we save the model
            # so we can tell them apart in the catalogue (i.e. different epochs)
            checkpoint_uuid = self.ckpt_metadata.uuid
            trainer.lightning_module._hparams["metadata"]["uuid"] = checkpoint_uuid

            trainer.lightning_module._hparams["metadata"]["model"] = (
                self.model_metadata(model)
            )
            trainer.lightning_module._hparams["metadata"][
                "tracker"
            ] = self.ckpt_metadata.tracker

            trainer.lightning_module._hparams["metadata"]["training"] = {
                "current_epoch": self.ckpt_metadata.training.current_epoch,
                "global_step": self.ckpt_metadata.training.global_step,
                "elapsed_time": self.ckpt_metadata.training.elapsed_time,
            }

            Path(lightning_checkpoint_filepath).parent.mkdir(
                parents=True, exist_ok=True
            )

            save_config = model.config
            model.config = None

            save_metadata = model.metadata
            model.metadata = None

            metadata = {**save_metadata}

            inference_checkpoint_filepath = Path(
                lightning_checkpoint_filepath
            ).parent / Path(
                "inference-" + str(Path(lightning_checkpoint_filepath).name),
            )

            torch.save(model, inference_checkpoint_filepath)

            with ZipFile(inference_checkpoint_filepath, "a") as zipf:
                base = Path(inference_checkpoint_filepath).stem
                zipf.writestr(
                    f"{base}/ai-models.json",
                    json.dumps(metadata),
                )

            model.config = save_config
            model.metadata = save_metadata

            self._last_global_step_saved = trainer.global_step

        trainer.strategy.barrier()

        # saving checkpoint used for pytorch-lightning based training
        trainer.save_checkpoint(lightning_checkpoint_filepath, weights_only=False)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = lightning_checkpoint_filepath

        # notify loggers
        if trainer.is_global_zero:
            from weakref import proxy

            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def get_callbacks(
    *, filename: str, ckpt_metadata: DotDict, config: Optional[DictConfig] = None
) -> list:
    """Setup callbacks for PyTorch Lightning trainer.

    Parameters
    ----------
    config : DictConfig
        Job configuration

    Returns
    -------
    List
        A list of PyTorch Lightning callbacks
    """
    checkpoint_settings = {
        "dirpath": filename,
        "verbose": False,
        # save weights, optimizer states, LR-schedule states, hyperparameters etc.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html#contents-of-a-checkpoint
        "save_weights_only": False,
    }

    trainer_callbacks = []

    trainer_callbacks.extend(
        # save_top_k: the save_top_k flag can either save the best or the last k checkpoints
        # depending on the monitor flag on ModelCheckpoint.
        # See https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html for reference
        [
            NewModelCheckpoint(
                ckpt_metadata=ckpt_metadata,
                filename=filename,
                **checkpoint_settings,
            ),
        ],
    )

    # trainer_callbacks.append(ParentUUIDCallback(ckpt_metadata.config))
    # trainer_callbacks.append(ConfigDumper(ckpt_metadata.config))

    return trainer_callbacks
