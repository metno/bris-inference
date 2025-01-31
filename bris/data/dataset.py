from typing import Any

import torch
from numpy import datetime64
from torch.utils.data import IterableDataset
from einops import rearrange


class Dataset(IterableDataset):
    def __init__(
        self,
        dataCls: Any,
    ):
        """
        Wrapper for a given anemoi.training.data.dataset class
        to include timestamp in the iterator.
        """

        super().__init__()
        if hasattr(dataCls, "data"):
            self.data = dataCls.data
        else:
            raise RuntimeError("dataCls does not have attribute data")
        self.dataCls = dataCls

    def per_worker_init(self, n_workers, worker_id):
        """
        Delegate per_worker_init to the underlying dataset.
        Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        if hasattr(self.dataCls, "per_worker_init"):
            self.dataCls.per_worker_init(n_workers=n_workers, worker_id=worker_id)
        else:
            raise RuntimeError(
                "Warning: Underlying dataset does not implement 'per_worker_init'."
            )
           
    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        if hasattr(self.dataCls, "set_comm_group_info"):
            self.dataCls.set_comm_group_info(
                global_rank=global_rank,
                model_comm_group_id=model_comm_group_id,
                model_comm_group_rank=model_comm_group_rank,
                model_comm_num_groups=model_comm_num_groups,
                reader_group_rank=reader_group_rank,
                reader_group_size=reader_group_size,
            )
        else:
            raise RuntimeError(
                "Warning: Underlying dataset does not implement 'set_comm_group_info'."
            )


    def __iter__(
        self,
    ) -> tuple[torch.Tensor, datetime64] | tuple[tuple[torch.Tensor], datetime64]:

        for idx, x in enumerate(iter(self.dataCls)):
            yield (x, str(self.data.dates[self.dataCls.chunk_index_range[idx] + self.dataCls.multi_step - 1]))
