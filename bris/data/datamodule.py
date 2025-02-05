import logging
from functools import cached_property
from typing import Any

import numpy as np
import pytorch_lightning as pl
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, get_worker_info
import anemoi.datasets.data.subset
import anemoi.datasets.data.select

from bris.checkpoint import Checkpoint
from bris.data.dataset import Dataset
from bris.utils import check_anemoi_training, recursive_list_to_tuple

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
        and pin_memory have sane defaults, but can be adjusted in the config
        under dataloader.

        args:
            ds: anemoi.datasets.data.open_dataset object

        return:
            torch dataloader initialized on anemoi dataset object
        """
        return DataLoader(
            ds,
            batch_size=1,
            # number of worker processes
            num_workers=self.config.dataloader.get("num_workers", 1),
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=self.config.dataloader.get("pin_memory", True),
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=self.config.dataloader.get("prefetch_factor", 2),
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
            LOGGER.warning("WARNING! Ensemble legacy mode has yet to be implemented!")
            from .legacy.dataset import NativeGridDataset
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
            try:
                dataCls = instantiate(
                    config=self.config.dataloader.datamodule,
                    data_reader=data_reader,
                    rollout=0,  # we dont perform rollout during inference
                    multistep=self.ckptObj.multistep,
                    timeincrement=self.timeincrement,
                    grid_indices=self.grid_indices,
                    shuffle=False,
                    label="predict",
                )
            except:
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

    @property
    def name_to_index(self):
        """
        Returns a tuple of dictionaries, where each dict is:
            variable_name -> index
        """
        return self.ckptObj.name_to_index

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
        base_loader = OmegaConf.to_container(
            self.config.dataset, 
            resolve=True
            )
        return open_dataset(base_loader)

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
    def grid_indices(self):
        if check_anemoi_training(self.ckptObj.metadata):
            try:
                from anemoi.training.data.grid_indices import BaseGridIndices, FullGrid
            except ImportError:
                print("Warning! Could not import BaseGridIndices and FullGrid. Continuing without this module")

        grid_indices = FullGrid(
            nodes_name="data",
            reader_group_size=1
            )
        grid_indices.setup(self.graph)
        return grid_indices
    
    @cached_property
    def grids(self) -> tuple:
        """
        Retrieves a tuple of flatten grid shape(s).
        """
        if isinstance(self.data_reader.grids[0], (int, np.int32,np.int64)):
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
    def altitudes(self) -> tuple:
        """
        Retrives altitudes from geopotential height in the datasets
        """
        name_to_index = self.data_reader.name_to_index
        if isinstance(name_to_index, tuple):
            altitudes = ()
            for i, n2i in enumerate(name_to_index):
                if 'z' in n2i.keys():
                    altitudes += (self.data_reader[0][i][n2i['z'],0,:] / 9.81 ,)
                else:
                    altitudes += (None,)
        else:
            if 'z' in name_to_index.keys():
                altitudes = (self.data_reader[0][name_to_index['z'],0,:] / 9.81 ,)
            else:
                altitudes = (None,)

        return altitudes


    @cached_property
    def field_shape(self) -> tuple:

        field_shape = [None]*len(self.grids)
        for decoder_index, grids in enumerate(self.grids):
            field_shape[decoder_index] = [None]*len(grids)
            for dataset_index, grid in enumerate(grids):
                _field_shape = self._get_field_shape(decoder_index, dataset_index)
                if np.prod(_field_shape) == grid:
                    field_shape[decoder_index][dataset_index] = list(_field_shape)
                else:
                    field_shape[decoder_index][dataset_index] = [grid,]
        return recursive_list_to_tuple(field_shape)

    def _get_field_shape(self, decoder_index, dataset_index):
        data_reader = self.data_reader
        while isinstance(data_reader, (anemoi.datasets.data.subset.Subset, anemoi.datasets.data.select.Select)):
            data_reader = data_reader.dataset

        if hasattr(data_reader, "datasets"):
            dataset = data_reader.datasets[decoder_index]
            while isinstance(dataset, (anemoi.datasets.data.subset.Subset, anemoi.datasets.data.select.Select)):
                dataset = dataset.dataset
            
            if hasattr(dataset, "datasets"):
                return dataset.datasets[dataset_index].field_shape
            else:
                return dataset.field_shape
        else:
            assert (decoder_index == 0 and dataset_index == 0)
            return data_reader.field_shape

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
