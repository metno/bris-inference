import os
import logging

from omegaconf import OmegaConf, errors
from functools import cached_property
from typing import Optional, Any

import pytorch_lightning as pl

from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import get_worker_info
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData


from bris.checkpoint import Checkpoint
from bris.data.dataset import Dataset
from bris.utils import check_anemoi_dataset_version
from bris.utils import check_anemoi_training


LOGGER = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: DotDict = None,
        checkpoint_object: Checkpoint = None,
        paths: Optional[list[str]] = None,
        **kwargs: Any,
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

        self.global_rank = int(os.environ.get("SLURM_PROCID", "0"))  # global rank
        self.model_comm_group_id = (
            self.global_rank // self.config.run_options.num_gpus_per_model
        )  # id of the model communication group the rank is participating in
        self.model_comm_group_rank = (
            self.global_rank % self.config.run_options.num_gpus_per_model
        )  # rank within one model communication group
        total_gpus = (
            self.config.run_options.num_gpus_per_node
            * self.config.run_options.num_nodes
        )
        assert (
            total_gpus
        ) % self.config.run_options.num_gpus_per_model == 0, f"GPUs per model {self.config.run_options.num_gpus_per_model} does not divide total GPUs {total_gpus}"
        self.model_comm_num_groups = (
            self.config.run_options.num_gpus_per_node
            * self.config.run_options.num_nodes
            // self.config.run_options.num_gpus_per_model
        )  # number of model communication groups

        LOGGER.debug(
            "Rank %d model communication group number %d, with local model communication group rank %d",
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
        )

        if not kwargs.get("frequency", None) and not kwargs.get("timestep", None):
            try:
                self.timestep = self.config.timestep
                self.frequency = self.config.frequency
            except errors.ConfigAttributeError as e:
                raise ValueError(f"Missing either timestep, frequency or both") from e
        else:
            self.timestep = kwargs.get("timestep")
            self.frequency = kwargs.get("frequency")

        # assert isinstance(graph, HeteroData), f"Expecting graph to be torch geometric HeteroData object"

        if paths:
            # check if args paths exist
            for p in paths:
                assert os.path.exists(
                    p
                ), f"The given input data path does not exist. Got {p}"
            self.paths = paths
        #self.legacy = check_anemoi_dataset_version(metadata=self.ckptObj._metadata)
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
            LOGGER.info("""Did not find anemoi.training version in checkpoint metadata, assuming 
                        the model was trained with aifs-mono and using legacy functionality""")

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

            dataCls = instantiate(
                config=self.config.datamodule,
                data_reader=data_reader,
                rollout=0,  # we dont perform rollout during inference
                multistep=self.ckptObj.multistep,
                timeincrement=self.timeincrement,
                model_comm_group_rank=self.model_comm_group_rank,
                model_comm_group_id=self.model_comm_group_id,
                model_comm_num_groups=self.model_comm_num_groups,
                spatial_index=spatial_index,
                shuffle=False,
                label="predict",
            )
            LOGGER.info(
                f"Obtained data class for {dataCls.__name__}. Proceeding to wrap data class"
            )
        else:
            dataCls = instantiate(
                config=self.config.datamodule,
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

        if hasattr(self, "paths") and hasattr(self.config.datasets, "cutout"):
            assert len(self.config.datasets.cutout) == len(
                self.paths
            ), f"len(cutout) != len(paths)"
            # if paths is given with command line args
            # we want to replace the existing path in config with this
            # TODO :  make this more generic, what if cutout or open_dataset
            # struct is not given??

            for d, p in zip(self.config.datasets.cutout, self.paths):
                d["dataset"] = p if not p.endswith("/") else p.rstrip("/")
            return open_dataset(self.config.dataset)
        return open_dataset(self.config.dataset)

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
    def grids(self):
        """Returns a diction of grids and their grid point ranges"""
        return {"global": {"start": 1, "end": 2}, "meps": {"start": 3, "end": 4}}


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
