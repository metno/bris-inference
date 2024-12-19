import logging
import os
from functools import cached_property
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, errors
from torch.utils.data import DataLoader, get_worker_info
from torch_geometric.data import HeteroData

from bris.checkpoint import Checkpoint
from bris.data.dataset import Dataset
from bris.utils import check_anemoi_dataset_version, check_anemoi_training

LOGGER = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DotDict = None,
        checkpoint_object: Checkpoint = None,
    ) -> None:
        """
        DataModule instance and DataSets.

        It reads the spatial indices from the graph object. These spatial indices
        correspond to the order (and data points) that will be read from each data
        source. If not specified, the dataset will be read full in the order that it is
        stored.
        """
        super().__init__()

        assert isinstance(
            config, DictConfig
        ), f"Expecting config to be DotDict object, but got {type(config)}"

        self.config = config
        self.graph = checkpoint_object.graph
        self.ckptObj = checkpoint_object
        self.timestep = config.timestep
        self.frequency = config.frequency
        self.legacy = not check_anemoi_training(metadata=self.ckptObj._metadata)

    def predict_dataloader(self) -> DataLoader:
        """
        Creates a dataloader for prediction

        args:
            None
        return:

        """
        return self._get_dataloader(self.ds_predict)

    def _get_dataloader(self, ds):
        """
        Creates torch dataloader object for
        ds. Batch_size, num_workers, prefetch_factor
        and pin_memory can be adjusted in the config
        under dataloader.

        args:
            ds: anemoi.datasets.data.open_dataset object

        return:
            torch dataloader initialized on anemoi dataset object
        """
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size,
            # number of worker processes
            num_workers=self.config.dataloader.num_workers,
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=self.config.dataloader.get("pin_memory", True),
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )

    @cached_property
    def ds_predict(self) -> Any:
        """
        creates predict input instance

        args:
            None
        return:
            Anemoi dataset open_dataset object
        """
        return self._get_dataset(self.data_reader)

    def _get_dataset(
        self,
        data_reader,
    ):
        """
        Instantiates a given dataset class
        from anemoi.training.data.dataset.
        This assumes that the python path for
        the class is defined, and anemoi-training
        for a given branch is installed with pip
        in order to access the class. This
        method returns an instantiated instance of
        a given data class. This supports
        data distributed parallel (DDP) and model
        sharding.

        args:
            data_reader: anemoi open_dataset object

        return:
            an dataset class object
        """
        if self.legacy:
            # TODO: fix imports and pip packages for legacy version
            LOGGER.info(
                """Did not find anemoi.training version in checkpoint metadata, assuming 
                        the model was trained with aifs-mono and using legacy functionality"""
            )
            LOGGER.warning("Ensemble legacy mode has yet to be implemented!")
            from .legacy.dataset import EnsNativeGridDataset, NativeGridDataset
            from .legacy.utils import _legacy_slurm_proc_id

            model_comm_group_rank, model_comm_group_id, model_comm_num_groups = (
                _legacy_slurm_proc_id(self.config)
            )

            spatial_mask = {}
            for mesh_name, mesh in self.graph.items():
                if (
                    isinstance(mesh_name, str)
                    and mesh_name
                    != self.ckptObj._metadata.config.graphs.hidden_mesh.name
                ):
                    spatial_mask[mesh_name] = mesh.get("dataset_idx", None)
            spatial_index = spatial_mask[
                self.ckptObj._metadata.config.graphs.encoders[0]["src_mesh"]
            ]

            dataCls = NativeGridDataset(
                data_reader=data_reader,
                rollout=0,  # we dont perform rollout during inference
                multistep=self.ckptObj.multistep,
                timeincrement=self.timeincrement,
                model_comm_group_rank=model_comm_group_rank,
                model_comm_group_id=model_comm_group_id,
                model_comm_num_groups=model_comm_num_groups,
                spatial_index=spatial_index,
                shuffle=False,
                label="predict",
            )
            return dataCls
        else:
            dataCls = instantiate(
                config=self.config.dataloader.datamodule,
                data_reader=data_reader,
                rollout=0,  # we dont perform rollout during inference
                multistep=self.ckptObj.multistep,
                timeincrement=self.timeincrement,
                shuffle=False,
                label="predict",
            )

        return Dataset(dataCls)

    @cached_property
    def data_reader(self):
        """
        Creates an anemoi open_dataset object for
        a given dataset (or set of datasets). If the path
        of the dataset(s) is given as command line args,
        trailing '/' is removed and paths are added to
        dataset key. The config.dataset is highly adjustable
        and see: https://anemoi-datasets.readthedocs.io/en/latest/
        on how to open your dataset in various ways.

        args:
            None
        return:
            An anemoi open_dataset object
        """
        from anemoi.datasets import open_dataset
        return open_dataset(self.config.dataloader.predict)

    @cached_property
    def timeincrement(self) -> int:
        """Determine the step size relative to the data frequency."""
        try:
            frequency = frequency_to_seconds(self.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.frequency}"
            raise ValueError(msg) from e

        try:
            timestep = frequency_to_seconds(self.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.timestep}"
            raise ValueError(msg) from e

        assert timestep % frequency == 0, (
            f"Timestep ({self.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.frequency} == {frequency})."
        )

        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    @cached_property
    def grids(self) -> tuple:
        """
        Retrieves a tuple of flatten grid shape(s).
        """
        if isinstance(self.data_reader.grids[0], int):
            return (self.data_reader.grids,)
        else: 
            return self.data_reader.grids

    @cached_property
    def latitudes(self) -> tuple:
        """
        Retrieves latitude from data_reader method
        """
        if isinstance(self.data_reader.latitudes, np.ndarray):
            return (self.data_reader.latitudes,)
        else:
            return self.data_reader.latitudes

    @cached_property
    def longitudes(self) -> tuple:
        """
        Retrieves longitude from data_reader method
        """
        if isinstance(self.data_reader.longitudes, np.ndarray):
            return (self.data_reader.longitudes,)
        else:
            return self.data_reader.longitudes

    @cached_property
    def field_shape(self) -> tuple:
        """
        Retrieves the field shape(s) for different
        type of grids. For XY-regular grid the field shape
        is on (x,y) format. For non-regular grids e.g gaussian
        grid the field shape is the flatten shape of the array.
        For example o96 -> (40320,)

        """
        if hasattr(self.data_reader, "datasets"):
            field_shape = ()
            for dataset in self.data_reader.datasets:
                if hasattr(dataset, "datasets"):
                    field_shape_dataset = ()
                    for sub_dataset in dataset.datasets:
                        field_shape_dataset += (sub_dataset.field_shape,)
                else:
                    field_shape_dataset = dataset.field_shape

                field_shape += (field_shape_dataset,)
        else:
            field_shape = ((self.data_reader.field_shape,),) 
        return field_shape #probably have to fix this for cutout


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
        # LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = (
        worker_info.dataset
    )  # the copy of the dataset held by this worker process.
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
