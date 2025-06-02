import logging
import random
from collections.abc import Iterator
from functools import cached_property
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset, get_worker_info

from bris.data.grid_indices import BaseGridIndices
from bris.utils import get_base_seed, get_usable_indices

LOGGER = logging.getLogger(__name__)

class ZipDataset(NativeGridDataset):
    def __init__(
        self,
        data_reader,
        grid_indices,
        rollout=1,
        multistep=1,
        timeincrement=1,
        label="generic",
    ):
        self.label = label
        self.data = data_reader

        self.rollout = rollout
        self.timeincrement = timeincrement
        self.grid_indices = grid_indices

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        assert all(
            dset_shape[self.ensemble_dim] == self.data.shape[0][self.ensemble_dim]
            for dset_shape in self.data.shape
        ), "Ensemble size must match for all datasets"
        self.ensemble_size = self.data.shape[0][self.ensemble_dim]

    def __iter__(self) -> Iterator[tuple[tuple[torch.Tensor], str]]:
        shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement
            x = self.data[start : end : self.timeincrement]
            batch = []
            for j, data in enumerate(x):
                grid_shard_indices = self.grid_indices[j].get_shard_indices(
                    self.reader_group_rank
                )
                batch.append(
                    torch.from_numpy(
                        rearrange(
                            data[..., grid_shard_indices],
                            "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
                        )
                    )
                )

            self.ensemble_dim = 1

            yield (tuple(batch), str(self.data.dates[i]))
