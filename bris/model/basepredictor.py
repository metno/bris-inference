import logging
import math
import os
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Union

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

    Methods
    -------

    __init__

    set_model_comm_group

    set_reader_groups

    set_static_forcings (abstract)

    forward (abstract)

    advance_input_predict (abstract)

    predict_step (abstract)
    """

    def __init__(
        self,
        *args: Any,
        checkpoints: dict[str, Checkpoint],
        hardware_config: dict,
        num_members_in_parallel: int,
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
        """

        super().__init__(*args, **kwargs)
        self.num_members_in_parallel = num_members_in_parallel

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

    @abstractmethod
    def set_static_forcings(
        self,
        datareader: Iterable,
    ) -> None:
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        pass

    @abstractmethod
    def advance_input_predict(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        y_pred: Union[torch.Tensor, list[torch.Tensor]],
        time: np.datetime64,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        pass
