from typing import Any

import torch
from numpy import datetime64
from torch.utils.data import IterableDataset, get_worker_info


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
        if hasattr(self.data, "per_worker_init"):
            self.data.per_worker_init(n_workers=n_workers, worker_id=worker_id)
        else:
            raise RuntimeError(
                "Warning: Underlying dataset does not implement 'per_worker_init'."
            )

    def __iter__(
        self,
    ) -> tuple[torch.Tensor, datetime64] | tuple[tuple[torch.Tensor], datetime64]:

        for idx, x in enumerate(iter(self.data)):
            yield (x, str(self.data.dates[idx]))
