import logging
from collections.abc import Iterator
from typing import Callable

import torch
from einops import rearrange

from bris.data.dataset import NativeGridDataset

LOGGER = logging.getLogger(__name__)


class ZipDataset(NativeGridDataset):
    def __init__(
        self,
        data_reader: Callable,
        grid_indices,
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        label: str = "generic",
        num_members_in_sequence: int = 1,
    ) -> None:
        super().__init__(
            data_reader,
            grid_indices,
            rollout,
            multistep,
            timeincrement,
            label,
            init_ensemble_size=False,
            num_members_in_sequence=num_members_in_sequence,
        )

        self.grid_indices = grid_indices
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
