import os 

from omegaconf import OmegaConf 
from functools import cached_property
from typing import Optional, Any

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
from anemoi.utils.config import DotDict 
from torch_geometric.data import HeteroData
from omegaconf import DictConfig
from hydra.utils import instantiate
from .checkpoint import Checkpoint
import pytorch_lightning as pl
from torch.utils.data import get_worker_info
from torch.utils.data import DataLoader

class DataModule(AnemoiDatasetsDataModule):
    def __init__(self, graph, config):
        """DataModule instance and DataSets.

        It reads the spatial indices from the graph object. These spatial indices
        correspond to the order (and data points) that will be read from each data
        source. If not specified, the dataset will be read full in the order that it is
        stored.
        """
        self.graph = graph
        self.config = config

        spatial_mask = {}
        for mesh_name, mesh in self.graph.items():
            if (
                isinstance(mesh_name, str)
                and mesh_name != self.config.graphs.hidden_mesh.name
            ):
                spatial_mask[mesh_name] = mesh.get("dataset_idx", None)

        # Note: Only 1 dataset supported for now. If more datasets are specified, the first one will be used.
        super.__init__(
            self.config,
            spatial_index=spatial_mask[self.config.graphs.encoders[0]["src_mesh"]],
            predict_datareader=self.data_reader,
        )
        self.config.data.num_features = len(datamodule.ds_train.data.variables)

    @property
    def grids(self):
        """Returns a diction of grids and their grid point ranges"""
        return {"global": {"start": 1, "end": 2}, "meps": {"start": 3, "end": 4}}

class DataModuleTest(pl.LightningDataModule):
    def __init__(
            self, 
            #graph: HeteroData = None, 
            frequency: int = 6,
            config: DotDict = None,
            ckptObj: Checkpoint = None,
            paths: Optional[list[str]] = None,
            ) -> None:
        """
        DataModule instance and DataSets.

        It reads the spatial indices from the graph object. These spatial indices
        correspond to the order (and data points) that will be read from each data
        source. If not specified, the dataset will be read full in the order that it is
        stored.
        """
        # TODO: make this cleaner
        self.rollout = frequency // int(ckptObj._metadata.config.data.frequency[0])

        #assert isinstance(graph, HeteroData), f"Expecting graph to be torch geometric HeteroData object"
        assert isinstance(config, DictConfig), f"Expecting config to be DotDict object, but got {type(config)}"
        self.config = config
        if paths:
            # check if args paths exist
            for p in paths:
                assert os.path.exists(p), f"The given input data path does not exist. Got {p}"
            self.paths = paths

        #self.data_reader
        #self.__setup()
        # Get the DataLoader
        predict_loader = self.predict_dataloader()

        # Create an iterator from the DataLoader
        iterator = iter(predict_loader)

        # Get the first batch
        first_batch = next(iterator)

        # Print the first batch
        print(first_batch)    
    def ds_predict(self) -> Any:
        return self._get_dataset(
            self.data_reader,
            self.rollout
        ) 
    
    def predict_dataloader(self):
        return self._get_dataloader(self.data_reader,"predict")
    def _get_dataloader(self, ds, stage):
         return DataLoader(
            ds,
            batch_size=1,#self.config.dataloader.batch_size,
            # number of worker processes
            num_workers=1,#self.config.dataloader.num_workers[stage],
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=False,#self.config.dataloader.get("pin_memory", True),
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches
            prefetch_factor=2,#self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )
    def _get_dataset(
            self,
            data_reader,
            rollout
            ):
        # just for testing (not finalized)
        data = instantiate(
            self.config.datamodule,
            data_reader=data_reader,
            rollout=rollout,
            multistep=self.config.training.multistep_input,
            timeincrement=1,#self.timeincrement,
            model_comm_group_rank=0,#self.model_comm_group_rank,
            model_comm_group_id=0,#self.model_comm_group_id,
            model_comm_num_groups=0,#self.model_comm_num_groups,
            shuffle=False,
            label="test",
        )
    @cached_property
    def data_reader(self):
        from anemoi.datasets import open_dataset
        if hasattr(self, "paths") and hasattr(self.config.datasets, "cutout"):
            assert len(self.config.datasets.cutout) == len(self.paths), f"len(cutout) != len(paths)"
            # if paths is given with command line args
            # we want to replace the existing path in config with this
            # TODO :  make this more generic, what if cutout or open_dataset
            # struct is not given??

            for d, p in zip(self.config.datasets.cutout, self.paths):
                d["dataset"] = p if not p.endswith("/") else p.rstrip("/")
            return open_dataset(self.config.datasets)
        return open_dataset(self.config.datasets)
    

    def __setup(self):
        print(self.ckptObj.config.keys())
        print(self.ckptObj.config.dataloader)

        exit()
        metadata_config = self.__dotdict_to_dict(self.ckptObj.config)

        exit()
        omegaconf_dictconfig = OmegaConf.create(metadata_config)
        datamodule_setup = self.config.datamodule
        datamodule_setup["config"] = omegaconf_dictconfig

    def __dotdict_to_dict(self, d):
        if isinstance(d, DotDict):
            return {k: self.__dotdict_to_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.__dotdict_to_dict(i) for i in d]
        else:
            return d
        
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
        #LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
    print(dataset_obj)
    dataset_obj.per_worker_init(
        n_workers=worker_info.num_workers,
        worker_id=worker_id,
    )
if __name__ == "__main__":
    DataModuleTest()
