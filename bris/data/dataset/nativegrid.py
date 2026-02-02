import logging
import random
from collections.abc import Iterator
from functools import cached_property
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from bris.data.grid_indices import BaseGridIndices
from bris.utils import LOGGER, get_base_seed, get_usable_indices


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids.
    Note that ZipDataset inherits from this.

    Methods
    -------

        __init__

        valid_date_indices

        set_comm_group_info

        per_worker_init

        __iter__
    """

    def __init__(
        self,
        data_readers: dict[str, Callable],
        grid_indices: dict[str, type[BaseGridIndices]],
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        label: str = "generic",
        init_ensemble_size: bool = True,
        num_members_in_sequence: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Args:
        data_reader : dict[str, Callable]
            dict containing the dataset name and a user function that opens and 
            returns the zarr array data

        grid_indices : dict[str, type[BaseGridIndices]]
            dict containing the dataset name and indices of the grid to keep. Defaults to None, which keeps all spatial indices

        rollout : int, optional
            length of rollout window, default 1

        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, default 1

        timeincrement : int, optional
            time increment between samples, default 1

        label : str, optional
            label for the dataset, by default "generic"

        init_ensemble_size: bool, default True
            In sub-classes this must be set to false and done in the sub-class instead. See ZipDataset for example.

        num_members_in_sequence : int, default 1
            Number of ensemble members in the sequence. This is used to repeat the indices
            for each member in the sequence.
        """
        self.label = label
        self.data = data_readers
        self.dataset_names = list(data_readers.keys())

        self.rollout = rollout
        self.timeincrement = timeincrement
        self.grid_indices = grid_indices
        self.num_members_in_sequence = num_members_in_sequence

        # Lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # Lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        # Additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2

        if init_ensemble_size:
            self.ensemble_size = self.data[self.dataset_names[0]].shape[self.ensemble_dim]
            for dataset_name in self.dataset_names:
                assert self.data[dataset_name].shape[self.ensemble_dim] == self.ensemble_size, (
                    f"Ensemble size mismatch for dataset {dataset_name}: "
                    f"{self.data[dataset_name].shape[self.ensemble_dim]} != {self.ensemble_size}"
                )
        

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return common valid date indices for all datasets.

        A date t is valid if we can sample the sequence
            (t - multistep + 1, ..., t + rollout)
        without missing data (if time_increment is 1).

        If there are no missing dates, total number of valid ICs is
        dataset length minus rollout minus additional multistep inputs
        (if time_increment is 1).
        """
        dataset_length = len(self.data[self.dataset_names[0]])
        for dataset_name in self.dataset_names[1:]:
            assert len(self.data[dataset_name]) == dataset_length, (
                f"Dataset length mismatch for dataset {dataset_name}: "
                f"{len(self.data[dataset_name])} != {dataset_length}"
            )
        valid_indices = {}
        for dataset_name in self.dataset_names:
            valid_indices[dataset_name] = get_usable_indices(
                self.data[dataset_name].missing,
                dataset_length,
                self.rollout,
                self.multi_step,
                self.timeincrement,
            )
        
        common_valid_indices = set(valid_indices[self.dataset_names[0]])
        for dataset_name in self.dataset_names[1:]:
            common_valid_indices &= set(valid_indices[dataset_name])
        common_valid_indices = np.array(sorted(common_valid_indices), dtype=np.uint32)

        return common_valid_indices

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.ens_comm_num_groups

        assert shard_size > 0, (
            f"Number of samples per data parallel worker is {shard_size}. "
            f"Check your config file and ensure that the number of samples is greater than or equal to than the number of data parallel workers. "
            f"num_data_parallel = num_nodes * num_gpus_per_node / num_gpus_per_model"
        )
        if len(self.valid_date_indices) % self.ens_comm_num_groups != 0:
            LOGGER.warning(
                f"Dataloader has {len(self.valid_date_indices)} samples, which is not divisible by "
                f"{self.ens_comm_num_groups} data parallel workers. This will lead to "
                f"{len(self.valid_date_indices) % self.ens_comm_num_groups} unprocessed samples.",
                "num_data_parallel = num_nodes * num_gpus_per_node / num_gpus_per_model",
            )
        shard_start = self.ens_comm_group_id * shard_size
        shard_end = (self.ens_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.tile(
            np.arange(low, high, dtype=np.uint32), self.num_members_in_sequence
        )

        base_seed = get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], str]]:
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi Datasets from zarr files. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """

        shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            batch = {}
            for dataset_name in self.dataset_names:
                                
                grid_shard_indices = self.grid_indices[dataset_name].get_shard_indices(
                    self.reader_group_rank
                )
                x = self.data[dataset_name][start : end : self.timeincrement, :, :, :]
                x = x[..., grid_shard_indices]  # select the grid shard
                x = rearrange(
                    x,
                    "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
                )
                batch[dataset_name] = torch.from_numpy(x)
            
            self.ensemble_dim = 1

            yield (batch, str(self.data[self.dataset_names[0]].dates[i]))
