import logging
from functools import cached_property
from typing import Any

import anemoi.datasets.data.select
import anemoi.datasets.data.subset
import numpy as np
import pytorch_lightning as pl
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from bris.checkpoint import Checkpoint
from bris.data.dataset import worker_init_func
from bris.data.grid_indices import BaseGridIndices, FullGrid
from bris.utils import recursive_list_to_tuple

LOGGER = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DotDict,
        checkpoint_object: Checkpoint,
        timestep: int,
        frequency: int,
        num_members_in_sequence: int = 1,
    ) -> None:
        """
        DataModule instance and DataSets.

        It reads the spatial indices from the graph object. These spatial indices
        correspond to the order (and data points) that will be read from each data
        source. If not specified, the dataset will be read full in the order that it is
        stored.
        """
        super().__init__()

        assert isinstance(config, DictConfig), (
            f"Expecting config to be DotDict object, but got {type(config)}"
        )

        self.config = config
        self.graph = checkpoint_object.graph
        self.checkpoint_object = checkpoint_object
        self.timestep = timestep
        self.frequency = frequency
        self.num_members_in_sequence = num_members_in_sequence
        self.dataset_names = list(self.data_readers.keys())

    def predict_dataloader(self) -> DataLoader:
        """
        Creates a dataloader for prediction

        args:
            None
        return:

        """
        return DataLoader(
            self.ds_predict,
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
        return self._get_dataset(self.data_readers)

    def _get_dataset(
        self,
        data_readers,
    ) -> IterableDataset:
        ds = instantiate(
            config=self.config.dataloader.datamodule,
            data_readers=data_readers,
            rollout=0,
            multistep=self.checkpoint_object.multistep,
            timeincrement=self.timeincrement,
            grid_indices=self.grid_indices,
            label="predict",
            num_members_in_sequence=self.num_members_in_sequence,
        )

        return ds

    @cached_property
    def data_readers(self):
        """
        Creates a dictionairy of open_dataset objects for
        a given dataset (or set of datasets). 
        The config.dataset is highly adjustable
        and see: https://anemoi-datasets.readthedocs.io/en/latest/
        on how to open your dataset in various ways.

        args:
            None
        return:
            A dictionary of dataset names and anemoi open_dataset objects
        """
        ds_cfg = OmegaConf.to_container(self.config.datasets, resolve=True)
        data_readers = {}
        for dataset_name, dataset_recipe in ds_cfg.items():
            data_readers[dataset_name] = open_dataset(dataset_recipe)
        
        return data_readers

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

    @property
    def name_to_index(self):
        """
        Returns a dictionary of dataset names and their name_to_index mapping.
        """

        return {
            ds_name: self.data_readers[ds_name].name_to_index for ds_name in self.dataset_names
        }

    @cached_property
    def grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = 1
        graph_cfg = self.checkpoint_object.config.graph

        grid_indices = {}
        for ds_name in self.dataset_names:
            gi = FullGrid(nodes_name=ds_name, reader_group_size=reader_group_size)
            gi.setup(self.graph)
            grid_indices[ds_name] = gi
        
        return grid_indices


    @cached_property
    def grids(self) -> tuple:
        """
        Retrieves a tuple of flatten grid shape(s).
        """
        return {
            ds_name: self.data_readers[ds_name].grids for ds_name in self.dataset_names
        }

    @cached_property
    def latitudes(self) -> tuple:
        """
        Retrieves latitude from data_reader method
        """
        return {
            ds_name: self.data_readers[ds_name].latitudes for ds_name in self.dataset_names
        }

    @cached_property
    def longitudes(self) -> tuple:
        """
        Retrieves longitude from data_reader method
        """
        return {
            ds_name: self.data_readers[ds_name].longitudes for ds_name in self.dataset_names
        }

    @cached_property
    def altitudes(self) -> tuple:
        """
        Retrives altitudes from geopotential height in the datasets
        """
        altitudes = {}
        for ds_name in self.dataset_names:
            name_to_index = self.data_readers[ds_name].name_to_index
            if "z" in name_to_index:
                altitudes[ds_name] = self.data_readers[ds_name][0][name_to_index["z"], 0, :] / 9.81
            else:
                altitudes[ds_name] = None

        return altitudes

    @cached_property
    def field_shape(self) -> tuple:
        """
        Retrieve field_shape of the datasets
        """
        field_shape = {}
        for decoder_name, grids in self.grids.items():
            field_shape[decoder_name] = [None] * len(grids)
            for dataset_index, grid in enumerate(grids):
                _field_shape = self._get_field_shape(decoder_name, dataset_index)
                if np.prod(_field_shape) == grid:
                    field_shape[decoder_name][dataset_index] = _field_shape
                else:
                    field_shape[decoder_name][dataset_index] = (grid,)
        
        return field_shape

    def _get_field_shape(self, decoder_name, dataset_index):
        data_reader = self.data_readers[decoder_name]

        if hasattr(data_reader, "datasets"):
            dataset = data_reader.datasets[decoder_index]
            while isinstance(
                dataset,
                (
                    anemoi.datasets.data.subset.Subset,
                    anemoi.datasets.data.select.Select,
                ),
            ):
                dataset = dataset.dataset

            if hasattr(dataset, "datasets"):
                return dataset.datasets[dataset_index].field_shape
        
            return dataset.field_shape
        
        return data_reader.field_shape