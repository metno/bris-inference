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
        return self._get_dataset(self.data_reader)

    def _get_dataset(
        self,
        data_reader,
    ) -> IterableDataset:
        ds = instantiate(
            config=self.config.dataloader.datamodule,
            data_reader=data_reader,
            rollout=0,
            multistep=self.checkpoint_object.multistep,
            timeincrement=self.timeincrement,
            grid_indices=self.grid_indices,
            label="predict",
            num_members_in_sequence=self.num_members_in_sequence,
        )

        return ds

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
        base_loader = OmegaConf.to_container(self.config.dataset, resolve=True)
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

    @property
    def name_to_index(self):
        """
        Returns a tuple of dictionaries, where each dict is:
            variable_name -> index
        """
        if isinstance(self.data_reader.name_to_index, dict):
            return (self.data_reader.name_to_index,)
        return self.data_reader.name_to_index

    @cached_property
    def grid_indices(self) -> type[BaseGridIndices]:
        # TODO: This currently only supports fullgrid for multi-encoder/decoder
        reader_group_size = 1  # Generalize this later
        graph_cfg = self.checkpoint_object.config.graph

        # Multi_encoder/decoder
        if "input_nodes" in graph_cfg:
            grid_indices = []
            for dset in graph_cfg.input_nodes.values():
                gi = FullGrid(nodes_name=dset, reader_group_size=reader_group_size)
                gi.setup(self.graph)
                grid_indices.append(gi)
        else:
            if hasattr(self.config.dataloader, "grid_indices"):
                grid_indices = instantiate(
                    self.config.dataloader.grid_indices,
                    reader_group_size=reader_group_size,
                )
                grid_indices.setup(self.graph)
                LOGGER.info("Using grid indices from dataloader config")
            else:
                try:
                    # anemoi-core
                    grid_indices = FullGrid(
                        nodes_name="data", reader_group_size=reader_group_size
                    )
                    grid_indices.setup(self.graph)
                except:
                    # legacy
                    grid_indices = FullGrid(
                        nodes_name="grid", reader_group_size=reader_group_size
                    )
                    grid_indices.setup(self.graph)
                LOGGER.info(
                    "grid_indices not found in dataloader config, defaulting to FullGrid"
                )
            grid_indices = [grid_indices]

        return grid_indices

    @cached_property
    def grids(self) -> tuple:
        """
        Retrieves a tuple of flatten grid shape(s).
        """
        if isinstance(self.data_reader.grids[0], (int, np.int32, np.int64)):
            return (self.data_reader.grids,)
        return self.data_reader.grids

    @cached_property
    def latitudes(self) -> tuple:
        """
        Retrieves latitude from data_reader method
        """
        if isinstance(self.data_reader.latitudes, np.ndarray):
            return (self.data_reader.latitudes,)
        return self.data_reader.latitudes

    @cached_property
    def longitudes(self) -> tuple:
        """
        Retrieves longitude from data_reader method
        """
        if isinstance(self.data_reader.longitudes, np.ndarray):
            return (self.data_reader.longitudes,)
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
                if "z" in n2i:
                    altitudes += (self.data_reader[0][i][n2i["z"], 0, :] / 9.81,)
                else:
                    altitudes += (None,)
        else:
            if "z" in name_to_index:
                altitudes = (self.data_reader[0][name_to_index["z"], 0, :] / 9.81,)
            else:
                altitudes = (None,)

        return altitudes

    @cached_property
    def field_shape(self) -> tuple:
        """
        Retrieve field_shape of the datasets
        """
        field_shape = [None] * len(self.grids)
        for decoder_index, grids in enumerate(self.grids):
            field_shape[decoder_index] = [None] * len(grids)
            for dataset_index, grid in enumerate(grids):
                _field_shape = self._get_field_shape(decoder_index, dataset_index)
                if np.prod(_field_shape) == grid:
                    field_shape[decoder_index][dataset_index] = list(_field_shape)
                else:
                    field_shape[decoder_index][dataset_index] = [
                        grid,
                    ]
        return recursive_list_to_tuple(field_shape)

    def _get_field_shape(self, decoder_index, dataset_index):
        data_reader = self.data_reader
        while isinstance(
            data_reader,
            (anemoi.datasets.data.subset.Subset, anemoi.datasets.data.select.Select),
        ):
            data_reader = data_reader.dataset

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
        assert decoder_index == 0 and dataset_index == 0
        return data_reader.field_shape
