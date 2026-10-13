import logging
import math
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from ..checkpoint import Checkpoint
from ..utils import check_anemoi_training

LOGGER = logging.getLogger(__name__)


class BasePredictor(pl.LightningModule):
    """
    An abstract class for implementing custom predictors.

    Holds the communication groups (set by the DDP strategy) and what every predictor shares:
    the ensemble member bookkeeping and the assembly of the prediction dict that the writer
    consumes. Subclasses implement predict_step.

    Methods
    -------

    __init__

    set_model_comm_group, set_ens_comm_group, set_reader_groups

    fcstep: The forecast step passed to the model.

    prediction: Build the return value of predict_step.

    predict_step (abstract)
    """

    def __init__(
        self,
        *args: Any,
        checkpoints: dict[str, Checkpoint],
        hardware_config: dict,
        num_members_in_parallel: int,
        release_cache: bool = False,
        fcstep_const: bool = False,
        **kwargs: Any,
    ):
        """
        Init model_comm* variables for distributed training.

        Args:
            checkpoints {"forecaster": checkpoint_object}
            hardware_config {"num_gpus_per_model": int, "num_gpus_per_node": int, "num_nodes": int}
            num_members_in_parallel: int
                Number of ensemble members in parallel. Used to track ensemble_id of each model when running
                ensemble members in sequence.
            release_cache
                Release cache (torch.cuda.empty_cache()) after each prediction step. This is useful for large
                models, but may slow down the prediction.
            fcstep_const
                For inference on non-rollout trained ensemble models. Keep fcstep constant at 0 as in training
                if True.
        """

        super().__init__(*args, **kwargs)
        self.num_members_in_parallel = num_members_in_parallel
        self.release_cache = release_cache
        self.fcstep_const = fcstep_const
        self.batch_info = {}

        if check_anemoi_training(checkpoints["forecaster"].metadata):
            self.legacy = False

            self.model_comm_group = None
            self.model_comm_group_id = 0
            self.model_comm_group_rank = 0
            self.model_comm_num_groups = 1

            self.ens_comm_group = None
            self.ens_comm_group_id = 0
            self.ens_comm_group_rank = 0
            self.ens_comm_num_groups = 1

        else:
            self.legacy = True

            self.model_comm_group = None
            self.model_comm_group_id = (
                int(os.environ.get("SLURM_PROCID", "0"))
                // hardware_config["num_gpus_per_model"]
            )
            self.model_comm_group_rank = (
                int(os.environ.get("SLURM_PROCID", "0"))
                % hardware_config["num_gpus_per_model"]
            )
            self.model_comm_num_groups = math.ceil(
                hardware_config["num_gpus_per_node"]
                * hardware_config["num_nodes"]
                / hardware_config["num_gpus_per_model"],
            )

            self.ens_comm_group = None
            try:
                self.ens_comm_num_groups = int(hardware_config["members_in_parallel"])
            except KeyError:
                self.ens_comm_num_groups = 1
            self.ens_comm_group_id = (
                int(os.environ.get("SLURM_PROCID", "0")) // self.ens_comm_num_groups
            )
            self.ens_comm_group_rank = (
                int(os.environ.get("SLURM_PROCID", "0")) % self.ens_comm_num_groups
            )

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        self.model_comm_group = model_comm_group
        if not self.legacy:
            self.model_comm_group_id = model_comm_group_id
            self.model_comm_group_rank = model_comm_group_rank
            self.model_comm_num_groups = model_comm_num_groups
            self.model_comm_group_size = model_comm_group_size

    def set_ens_comm_group(
        self,
        ens_comm_group: ProcessGroup,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        member_id: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        if not self.legacy:
            self.ens_comm_group_id = ens_comm_group_id
            self.ens_comm_group_rank = ens_comm_group_rank
            self.ens_comm_num_groups = ens_comm_num_groups
            self.ens_comm_group_size = ens_comm_group_size
            self.member_id = member_id

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def fcstep(self, forecast_step: int) -> int:
        """The forecast step passed to the model: 0 for all steps if fcstep_const."""
        return 0 if self.fcstep_const else forecast_step

    def allgather_batch(self, batch: tuple) -> tuple:
        """
        Allgather the batch-shards across the reader group.
        """
        return batch  # Not implemented properly, https://github.com/metno/bris-inference/issues/123

    def update_batch_info(self, time: np.datetime64) -> None:
        """Count the forecasts ending at ``time`` seen so far, to number the ensemble members."""
        self.batch_info[time] = self.batch_info.get(time, 0) + 1

    def prediction(
        self, pred: dict[str, torch.Tensor], times: list[np.datetime64]
    ) -> dict:
        """
        The return value of predict_step, as the writer expects it.

        Args:
            pred (dict): Dataset name -> predictions, shape (batch, leadtimes, points, variables).
            times (list): Valid time of each lead time.
        """
        end_time = times[-1]
        self.update_batch_info(end_time)
        return {
            "pred": {ds: y.to(torch.float32).numpy() for ds, y in pred.items()},
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": self.member_id
            + self.num_members_in_parallel * (self.batch_info[end_time] - 1),
        }

    @abstractmethod
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        pass
