"""
    This submodule includes legacy NativeGridDataset used in 
    aifs-mono.

"""

import os
import random
from functools import cached_property
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset, get_worker_info

from bris.data.legacy.utils import get_base_seed, get_code_logger, get_usable_indices

LOGGER = get_code_logger(__name__)


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    BASE_SEED = 42

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        model_comm_group_rank: int = 0,
        model_comm_group_id: int = 0,
        model_comm_num_groups: int = 1,
        spatial_index: Optional[list[int]] = None,
        shuffle: bool = True,
        label: str = "generic",
        logging: str = "INFO",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        timeincrement : int, optional
            time increment in the dataset for each model step, by default 1
        model_comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        model_comm_group_id: int, optional
            device group ID, default 0
        model_comm_num_groups : int, optional
            total number of device groups, by default 1
        spatial_index : list[int], optional
            indices of the spatial indices to keep. Defaults to None, which keeps all spatial indices.
        shuffle : bool, optional
            Shuffle batches, by default True

        Raises
        ------
        RuntimeError
            Multistep value cannot be negative.
        """
        LOGGER.setLevel(logging)
        self.label = label

        self.data = data_reader

        self.rollout = rollout
        self.timeincrement = timeincrement

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # DDP-relevant info
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_id = model_comm_group_id
        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: Optional[np.ndarray] = None
        self.shuffle = shuffle
        self.spatial_index = spatial_index

        # Data dimensions
        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def valid_dates(self) -> np.ndarray:
        """Return valid dates."""
        return get_usable_indices(
            self.data.missing,
            len(self.data),
            self.rollout,
            self.multi_step,
            self.timeincrement,
        )

    @cached_property
    def build_einops_dim_order(self) -> str:
        ensemble = "ensemble " if self.ensemble_size > 1 else ""
        self.ensemble_dim = 1
        return f"dates variables {ensemble}gridpoints -> dates {ensemble}gridpoints variables"

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

        # Total number of valid ICs is given by usable dates:
        len_corrected = len(self.valid_dates)

        # Divide this equally across shards (one shard per group!)
        shard_size = len_corrected // self.model_comm_num_groups
        shard_start = (
            self.model_comm_group_id * shard_size
        )  # + (self.multi_step - 1) * self.timeincrement
        shard_end = (
            self.model_comm_group_id + 1
        ) * shard_size  # min((self.model_comm_group_id + 1) * shard_size, len(self.data) - self.rollout * self.timeincrement)

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)

        LOGGER.debug(
            "Worker %d (pid %d, global_rank %d, model comm group %d) has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        chunk_index_ideal = np.arange(low, high, dtype=np.uint32)
        self.chunk_index_range = self.valid_dates[chunk_index_ideal]

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        # should we check lightning env variable "PL_SEED_WORKERS" here?
        # but we alwyas want to seed these anyways ...

        base_seed = get_base_seed()

        seed = (
            base_seed * (self.model_comm_group_id + 1) - worker_id
        )  # note that test, validation etc. datasets get same seed
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.debug(
            "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, group_rank %d, base_seed %d) using seed %d, sanity rnd %f",
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            base_seed,
            seed,
            sanity_rnd,
        )

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by Anemoi datasets from zarr files. This iterator
        yields chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.chunk_index_range, size=self.n_samples_per_worker, replace=False
            )
        else:
            shuffled_chunk_indices = self.chunk_index_range

        LOGGER.debug(
            "Worker pid %d, label %s, worker id %d, global_rank %d, model comm group %d, group_rank %d using indices[0:10]: %s",
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            x = self.data[start : end : self.timeincrement]
            if self.spatial_index is not None:
                x = x[..., self.spatial_index]
            x = rearrange(
                x,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )
            self.ensemble_dim = 1

            yield torch.from_numpy(x), str(self.data.dates[i])

    def __getitem__(self, t_idx):
        """
        Expects a list of time indices and returns the corresponding data.
        """
        x = self.data[t_idx[0] : t_idx[-1] + 1]
        # if self.ensemble_size == 1:
        #    x = x[:, :, 0, :]
        # x = rearrange(x, self.build_einops_dim_order)
        if self.spatial_index is not None:
            x = x[..., self.spatial_index]
        x = rearrange(
            x,
            "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
        )
        self.ensemble_dim = 1

        return torch.from_numpy(x[np.newaxis, ...])  # add batch dimension.

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Rollout: {self.rollout}
            Multistep: {self.multi_step}
            Timeincrement: {self.timeincrement}
        """


class EnsNativeGridDataset(NativeGridDataset):
    """Iterable ensemble dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        comm_group_rank: int = 0,
        comm_group_id: int = 0,
        comm_num_groups: int = 1,
        spatial_index: Optional[list[int]] = None,
        shuffle: bool = True,
        label: str = "generic",
        logging: str = "INFO",
        ens_members_per_device: int = 1,
        num_gpus_per_ens: int = 1,
        num_gpus_per_model: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the zarr array data
        rollout : int, optional
            length of rollout window, by default 12
        multistep : int, optional
            collate (t-1, ... t - multistep) into the input state vector, by default 1
        ens_members_per_device: int, optional
            number of ensemble members input for each GPU device, by default 1
        comm_group_rank : int, optional
            process rank in the torch.distributed group (important when running on multiple GPUs), by default 0
        comm_group_id: int, optional
            device group ID, default 0
        comm_num_groups : int, optional
            total number of device groups, by default 1
        spatial_index : list[int], optional
            indices of the spatial indices to keep. Defaults to None, which keeps all spatial indices.
        shuffle : bool, optional
            Shuffle batches, by default True

        Raises
        ------
        RuntimeError
            Multistep value cannot be negative.
        """
        super().__init__(
            data_reader=data_reader,
            rollout=rollout,
            multistep=multistep,
            timeincrement=timeincrement,
            model_comm_group_rank=comm_group_rank,
            model_comm_group_id=comm_group_id,
            model_comm_num_groups=comm_num_groups,
            spatial_index=spatial_index,
            shuffle=shuffle,
            label=label,
        )

        LOGGER.setLevel(logging)

        self._seed: Optional[int] = None
        self._worker_id: Optional[int] = None

        self.comm_group_id = comm_group_id
        self.comm_group_rank = comm_group_rank

        # Lazy init
        self.ens_members_per_device = ens_members_per_device
        self.num_gpus_per_ens = num_gpus_per_ens
        self.num_gpus_per_model = num_gpus_per_model

    @property
    def num_eda_members(self) -> int:
        """Return number of EDA members."""
        return self.data.shape[2] - 1

    @property
    def eda_flag(self) -> bool:
        """Return whether EDA is enabled."""
        return self.data.shape[2] > 1

    def sample_eda_members(self, num_eda_members: int = 9) -> np.ndarray:
        """Subselect EDA ensemble members assigned to the current device."""
        tot_ens = (
            self.ens_members_per_device
            * self.num_gpus_per_ens
            // self.num_gpus_per_model
        )

        assert (
            tot_ens <= num_eda_members
        ), f"Can't generate an ensemble of size {tot_ens} from {num_eda_members} EDA perturbations"

        eda_member_gen_idx = self.rng.choice(
            range(num_eda_members), size=tot_ens, replace=False
        )
        offset = 1  # index=0 analysis, index=1 EDA recentred
        eda_member_gen_idx += offset

        effective_rank = self.comm_group_rank // self.num_gpus_per_model
        eda_member_idx = np.sort(
            eda_member_gen_idx[
                effective_rank
                * self.ens_members_per_device : self.ens_members_per_device
                * (1 + effective_rank)
            ],
        )

        LOGGER.debug(
            "GPU with global rank %s, Worker id %s, comm_group_id %s, comm_group_rank %s will receive EDA member(s) %s",
            self.global_rank,
            self._worker_id,
            self.comm_group_id,
            self.comm_group_rank,
            eda_member_gen_idx,
        )

        return eda_member_gen_idx, eda_member_idx

    def __iter__(self):
        """Return an iterator over the dataset.

        The datasets are retrieved by ECML Tools from zarr files. This iterator yields
        chunked batches for DDP and sharded training.
        """
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.chunk_index_range, size=self.n_samples_per_worker, replace=False
            )
        else:
            shuffled_chunk_indices = self.chunk_index_range

        for i in shuffled_chunk_indices:
            # start and end time indices, for analysis and EDA
            start = i - (self.multi_step - 1) * self.timeincrement
            end_an = i + (self.rollout + 1) * self.timeincrement
            end_eda = i + self.timeincrement

            if self.eda_flag:
                eda_member_gen_idx, eda_member_idx = self.sample_eda_members(
                    self.num_eda_members
                )
            else:
                eda_member_gen_idx = None
                eda_member_idx = None

            """
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            x = self.data[start : end : self.timeincrement]
            if self.spatial_index is not None:
                x = x[..., self.spatial_index]
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            """
            self.ensemble_dim = 1
            x_an = self.data[start : end_an : self.timeincrement]  # , :, 0:1, ...]
            # x_an = rearrange(torch.from_numpy(x_an), "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
            x_an = rearrange(
                x_an,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )

            x_pert: Optional[torch.Tensor] = None
            if self.eda_flag:

                x_pert = self.data[start : end_eda : self.timeincrement, ...]
                x_pert = x_pert[:, :, eda_member_idx, ...]
                sample = (
                    x_an,
                    rearrange(
                        torch.from_numpy(x_pert),
                        "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
                    ),
                )
            else:
                sample = (x_an,)

            # Handle debug logging
            self._log_debug_info(
                self._worker_id,
                os.getpid(),
                self.global_rank,
                self.comm_group_id,
                self.comm_group_rank,
                start,
                end_an,
                self.eda_flag,
                eda_member_gen_idx,
                eda_member_idx,
                analysis_shape=list(sample[0].shape),
                perturbation_shape=list(sample[1].shape) if len(sample) > 1 else "n/a",
            )

            yield sample, str(self.data.dates[i])


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process.

    Calls WeatherBenchDataset.per_worker_init() on each dataset object.

    Parameters
    ----------
    worker_id : int
        Worker ID

    Raises
    ------
    RuntimeError
        If worker_info is None
    """
    worker_info = get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = (
        worker_info.dataset
    )  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
