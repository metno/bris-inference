import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional, Any, List, Dict
from hydra import compose, initialize, initialize_config_dir

import hydra
import torch
import numpy as np
import zarr
import json

#import gridpp
#from yrlib_utils import *
import os

def create_directory(filename):
    """Creates all sub directories necessary to be able to write filename"""
    dir = os.path.dirname(filename)
    if dir != "":
        os.makedirs(dir, exist_ok=True)

from omegaconf import DictConfig
from omegaconf import OmegaConf

from aifs.data.datamodule import AnemoiDatasetsDataModule
from aifs.utils.logger import get_code_logger
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.interface import AnemoiModelInterface
from aifs.distributed.strategy import DDPGroupStrategy
from aifs.train.forecaster import GraphForecaster
from aifs.utils.jsonify import map_config_to_primitives
from anemoi.utils.provenance import gather_provenance_info
from aifs.utils.seeding import get_base_seed
from aifs.graphs.build import EncoderProcessorDecoderGraph

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

import scipy.interpolate

#from skimage.metrics import structural_similarity as ssim

from anemoi.datasets.data import open_dataset
import glob

LOGGER = get_code_logger(__name__)

class GraphPredictor(GraphForecaster):
    def __init__(
        self,
        *,
        config: DictConfig,
        statistics: dict,
        data_indices: IndexCollection,
        graph_data: dict,
        metadata: dict,
        select_indices = None,
        select_indices_truth = None,
        grid_points_range = None,
    ) -> None:
        super().__init__(config = config, statistics = statistics, data_indices = data_indices, graph_data = graph_data, metadata = metadata)
        self.select_indices = select_indices
        self.select_indices_truth = select_indices_truth
        self.grid_points_range = grid_points_range

    def advance_input_predict(self, x: torch.Tensor, y_pred: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        x = x.roll(-1, dims = 1)

        #Get prognostic variables
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[..., self.data_indices.model.output.prognostic]

        #get forcing constants:
        x[:, -1, :, :, self.data_indices.model.input.forcing] = forcing

        return x

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> tuple[torch.Tensor, int, int]:
        with torch.no_grad():
            batch, forcing, time_stamp = batch
            y_preds = np.zeros((batch.shape[0], self.rollout + 1, self.grid_points_range[1] - self.grid_points_range[0], len(self.select_indices)), dtype=np.float32)
            y_preds[0] = batch[:, self.multi_step - 1, 0, self.grid_points_range[0]:self.grid_points_range[1], self.select_indices_truth].cpu().numpy() #insert truth at lead time 0

            batch = self.model.pre_processors(batch)
            #x = batch[:, 0 : self.multi_step, :, :, self.data_indices.data.input.full]
            x = batch[..., self.data_indices.data.input.full]
            #print("batch shape", batch.shape)

            for rollout_step in range(self.rollout):
                #LOGGER.info("rollout step %i", rollout_step)
                y_pred = self(x)
                x = self.advance_input_predict(x, y_pred, forcing[:, rollout_step])

                #print("ypred shape", y_pred.shape)
                y_preds[:,rollout_step+1] = self.model.post_processors(y_pred, in_place = False)[:, 0, self.grid_points_range[0]:self.grid_points_range[1], self.select_indices].cpu().numpy()

            return [y_preds, self.model_comm_group_rank, self.model_comm_group_id, time_stamp] #return the model parallel rank so that only one process per model writes to file.

class CustomWriter(BasePredictionWriter):
    def __init__(self, inference, write_interval):
        super().__init__(write_interval)
        self.I = inference
        self.data_par_num = self.I.config.hardware.num_gpus_per_node/self.I.config.hardware.num_gpus_per_model * self.I.config.hardware.num_nodes
        #self.avail_samples = 623 #TODO: extract this from start and end date in inference config.
        self.vgrids = self.I.verif_grids
        self.grid_ranges = {grid: self.I.dset_properties[grid]['grid_points_range'] for grid in self.vgrids}
        self.hours_fc = str(self.I.inf_conf.dates.lead_times * self.I.inf_conf.dates.frequency)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if prediction[1] == 0: #if on first model parallel gpu of data parallel group
            #pred = prediction[0]
            #time_index = int(self.avail_samples//self.data_par_num * prediction[2] + batch_idx)
            #pred = pred[0] #remove batch dimension for now since we will always use data parallel for larger batches.
            for grid in self.grid_ranges.keys():
                if len(self.vgrids) == 2: #TODO: add support for e.g. 2 out of 3 grids being gotten from prediction and to be saved.
                    pred = prediction[0][0, :, self.grid_ranges[grid][0]:self.grid_ranges[grid][1]]
                elif len(self.vgrids) == 1:
                    pred = prediction[0][0]

                pred = self.I.to_grid(pred, grid)
                pred = self.I.map_to_dict(pred, self.I.inf_conf.select.vars)
                pred['wind_speed_10m'] = np.sqrt(pred['x_wind_10m']**2 + pred['y_wind_10m']**2)
                #pred['time'] = prediction[3][0] #remove batch dimension
                time_hours = prediction[3][0].split(":")[0]
                np.save(f"{self.I.inf_conf.paths.store}/predictions/{grid}_{self.hours_fc}hfc_{time_hours}.npy", pred)

class Inference:
    """Loads a checkpoint and runs inference on a dataset."""

    def __init__(self): #, inference_config: DictConfig):
        # Allow for lower internal precision of float32 matrix multiplications.
        # This can increase performance (and TensorCore usage, where available).
        torch.set_float32_matmul_precision("high")
        # Resolve the config to avoid shenanigans with lazy loading
        with initialize(version_base=None, config_path="aifs/config"):
            self.inf_conf = compose(config_name="inference")
        #OmegaConf.resolve(inference_config)
        #self.inf_conf = inference_config

        try:
            with initialize_config_dir(version_base=None, config_dir=self.inf_conf.paths.main):
                self.config = compose(config_name="config")
            print(f"Loaded config from experiments/{self.inf_conf.experiment.label}.")
        except:
            print("WARNING: Experiments folder did not contain a config file, using current repo config instead. Make sure all relevant parameters are equal to the ones used in training")
            with initialize(version_base=None, config_path="aifs/config"):
                self.config = compose(config_name="o96_1024c")

        self.update_configs()

        self.verif_grids = self.inf_conf.files.write.dsets
        #if 'meps' in self.dsets.keys() and 'meps' in self.inf_conf.files.write.dsets:
        #    self.verif_grid = 'meps'
        #else:
        #    self.verif_grid = 'era5'

        self.dataset = self.datamodule._get_dataset(self.data_reader, shuffle=False, rollout=1)
        #TODO: modify the chunk index range on self.dataset by subselecting dates before launching the predict step.
        self.altitude = self.data_reader[0].squeeze().swapaxes(0,1)[..., self.data_reader.name_to_index['z']]/9.81
        if self.inf_conf.select.selection:
            select_vars = self.inf_conf.select.vars
        else:
            select_vars = list(self.indices['name_to_index'].keys())

        create_directory(self.inf_conf.paths.store)
        create_directory(self.inf_conf.paths.store + "/predictions/")

    def update_configs(self) -> None:
        """
        Updates main config with information from the inference config.
        """
        self.config.training.rollout.start = self.inf_conf.dates.lead_times
        self.config.training.rollout.max = self.inf_conf.dates.lead_times
        self.config.training.rollout.epoch_increment = 0

        self.config.hardware.num_gpus_per_node = self.inf_conf.hardware.num_gpus_per_node
        self.config.hardware.num_gpus_per_model = self.inf_conf.hardware.num_gpus_per_model
        self.config.hardware.num_nodes = self.inf_conf.hardware.num_nodes

        self.config.diagnostics.log.mlflow.enabled = False
        self.config.dataloader.prefetch_factor = 1

        self.config.hardware.paths.graph = self.inf_conf.paths.home+ "/graphs/"
        #print("is in:", "1979-2022" in self.config.hardware.files.dataset)
        if "lustrep4" in self.inf_conf.paths.home: #lumi
            #print("dataset", self.config.hardware.files.dataset)
            if "1979-2022" in self.config.hardware.files.dataset: #if trained on 40 year data, we need to use data up until 2023 for inference.
                if "o96" in self.config.hardware.files.dataset:
                    self.config.hardware.files.dataset = "ERA5/aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6.zarr"
                elif "n320" in self.config.hardware.files.dataset:
                    self.config.hardware.files.dataset = "ERA5/aifs-od-an-oper-0001-mars-n320-2019-2023-6h-v6.zarr"
            
            if "2024" in self.inf_conf.dates.end:
                self.config.hardware.files.dataset = "ERA5/aifs-od-an-oper-0001-mars-n320-2023-2024-6h-v2.zarr" #aifs-od-an-oper-0001-mars-n320-2019-2023-6h-v6.zarr"
                self.config.dataloader.training.drop = ['cp']
                self.config.dataloader.training.dataset.cutout[1] = Path(self.config.hardware.paths.data, self.config.hardware.files.dataset)

        else: #ppi or other with similar dataset folders
            self.config.hardware.paths.data = self.inf_conf.paths.home + "/datasets/"
            self.config.dataloader.training.statistics = Path(self.config.hardware.paths.data, self.config.dataloader.training.statistics.split("/")[-1])
            if "2024" in self.inf_conf.dates.end:
                self.config.hardware.files.dataset = "aifs-od-an-oper-0001-mars-n320-2023-2024-6h-v2.zarr"
            else:
                self.config.hardware.files.dataset = self.config.hardware.files.dataset.split("/")[-1]
            if "dataset_lam" in self.config.hardware.files:
                self.config.hardware.files.dataset_lam = self.config.hardware.files.dataset_lam.split("/")[-1]
                self.config.dataloader.training.dataset.cutout[0] = Path(self.config.hardware.paths.data, self.config.hardware.files.dataset_lam)
                self.config.dataloader.training.dataset.cutout[1] = Path(self.config.hardware.paths.data, self.config.hardware.files.dataset)
            else:
                self.config.dataloader.training.dataset = Path(self.config.hardware.paths.data, self.config.hardware.files.dataset)
            

    @cached_property
    def dsets(self) -> dict:
        """
        Returns a dictionary of the full zarr dataset paths
        and information about their grid types
        """
        #TODO: Make this work with stretched grid again
        dsets = {}
        if 'dataset_lam' in self.config.hardware.files.keys():
            dsets['meps'] = {'path': self.config.hardware.paths.data + "/" + self.config.hardware.files.dataset_lam, 'grid': 'xy_regular'}
        dsets['era5'] = {'path': self.config.hardware.paths.data + "/" + self.config.hardware.files.dataset,'grid': 'latlon'}
        """
        base_path = self.config.hardware.paths.data
        for dlabel, dpath in self.config.hardware.files.datasets.items():
            dsets[dlabel] = {}
            dsets[dlabel]['path'] = base_path + dpath
            if dlabel == "era5":
                dsets[dlabel]['grid'] = "latlon"
            else:
                dsets[dlabel]['grid'] = "xy_regular"
        """
                
        return dsets

    @cached_property
    def datamodule(self) -> AnemoiDatasetsDataModule:
        """DataModule instance and DataSets.

        It reads the spatial indices from the graph object. These spatial indices
        correspond to the order (and data points) that will be read from each data
        source. If not specified, the dataset will be read full in the order that it is
        stored.
        """
        spatial_mask = {}
        for mesh_name, mesh in self.graph.items():
            if isinstance(mesh_name, str) and mesh_name != self.config.graphs.hidden_mesh.name:
                spatial_mask[mesh_name] = mesh.get("dataset_idx", None)

        # Note: Only 1 dataset supported for now. If more datasets are specified, the first one will be used.
        datamodule = AnemoiDatasetsDataModule(
            self.config,
            spatial_index=spatial_mask[self.config.graphs.encoders[0]["src_mesh"]],
            predict_datareader = self.data_reader,
        )
        self.config.data.num_features = len(datamodule.ds_train.data.variables)
        return datamodule

    @cached_property
    def graph(self) -> dict:
        graph_data_filename = Path(self.config.hardware.paths.graph, self.config.hardware.files.graph) #Path(self.config.hardware.paths.graph, self.config.hardware.files.graph)
        if graph_data_filename.exists(): #and not self.config.graphs.clobber:
            return torch.load(graph_data_filename)
        else:
            #return EncoderProcessorDecoderGraph(self.config).generate()
            assert False, "graph not found at path" + self.config.hardware.paths.graph + self.config.hardware.files.graph

    @cached_property
    def initial_seed(self) -> int:
        """Initial seed for the RNG.

        This sets the same initial seed for all ranks. Ranks are re-seeded in the
        strategy to account for model communication groups.
        """
        initial_seed = get_base_seed()
        rnd_seed = pl.seed_everything(initial_seed, workers=True)
        np_rng = np.random.default_rng(rnd_seed)
        (torch.rand(1), np_rng.random())
        LOGGER.debug(
            "Initial seed: Rank %d, initial seed %d, running with random seed: %d",
            int(os.environ.get("SLURM_PROCID", "0")),
            initial_seed,
            rnd_seed,
        )
        return initial_seed

    @cached_property
    def metadata(self) -> dict:
        """Metadata and provenance information."""
        return map_config_to_primitives(
            {
                "version": "1.0",
                "config": self.config,
                "seed": self.initial_seed,
                "run_id": "does_not_apply_here",
                "dataset": self.datamodule.metadata,
                "data_indices": self.datamodule.data_indices,
                "provenance_training": gather_provenance_info(),
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
            },
        )
    
    @cached_property
    def model(self) -> GraphPredictor:
        """Provide the model instance."""
        select_vars = self.inf_conf.select.vars
        select_indices = np.array([self.indices['name_to_index'][var] for var in select_vars])
        select_indices_truth = np.array([self.data_reader.name_to_index[var] for var in select_vars])

        kwargs = {
            "statistics": self.datamodule.statistics,
            "data_indices": self.datamodule.data_indices,
            "graph_data": self.graph,
            "metadata": self.metadata,
            "config": self.config,
            "select_indices": select_indices,
            "select_indices_truth": select_indices_truth,
            "grid_points_range": self.dset_properties[self.verif_grids[0]]['grid_points_range'] if len(self.verif_grids) == 1 else (0, sum(self.data_reader.grids)),
        }
        #LOGGER.info("Restoring only model weights from %s", self.last_checkpoint)

        #return GraphForecaster.load_from_checkpoint(self.last_checkpoint, **kwargs)
        model = GraphPredictor(**kwargs)
        ckpt = torch.load(self.last_checkpoint, 'cpu')
        for name, param in model.named_parameters():
            param.data = ckpt['state_dict'][name].data
        return model

    @cached_property
    def last_checkpoint(self) -> Optional[str]:
        checkpoint_files = glob.glob(self.inf_conf.paths.ckpt + f'*{self.inf_conf.experiment.epoch}*')

        if len(checkpoint_files) == 0:
            raise FileNotFoundError(f"No checkpoint file found for the specified identifier {self.inf_conf.experiment.epoch} at path {self.inf_conf.paths.ckpt}.")
        elif len(checkpoint_files) == 1:
            checkpoint_file = checkpoint_files[0]
        else:
            LOGGER.warning(
                "Multiple checkpoint files found for the specified identifier %s. Using the latest one.",
                self.inf_conf.experiment.epoch,
            )
            # Sort the filtered files based on their modification time (latest first)
            sorted_files = sorted(checkpoint_files, key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoint_file = sorted_files[0]

        return checkpoint_file

    @cached_property
    def trainer(self) -> pl.Trainer:
        """Provide the trainer instance."""
        return pl.Trainer(
            accelerator=self.device,
            deterministic=self.config.training.deterministic,
            detect_anomaly=self.config.diagnostics.debug.anomaly_detection,
            strategy=DDPGroupStrategy(
                self.config.hardware.num_gpus_per_model,
                static_graph=not self.config.training.accum_grad_batches > 1,
            ),
            devices=self.config.hardware.num_gpus_per_node,
            num_nodes=self.config.hardware.num_nodes,
            precision=self.config.training.precision,
            inference_mode = True,
            use_distributed_sampler=False,
            callbacks = [CustomWriter(self, write_interval = 'batch')],
        )

    @cached_property
    def data_reader(self):
        lead_times = int(self.inf_conf.dates.lead_times)
        frequency = int(self.inf_conf.dates.frequency)
        starting_date = np.datetime64(self.inf_conf.dates.start)
        end_date = np.datetime64(self.inf_conf.dates.end)

        base_loader = self.config.dataloader.training
        base_loader["start"] = str(starting_date)[:-3] #str(starting_date - np.timedelta64(frequency, 'h')) #[:-3] #the first prediction needs multistep dates, so we need to go back multistep-1 frequency.
        
        base_loader["end"] = str(end_date)[:-3] #str(end_date + np.timedelta64((lead_times+1)*frequency, 'h')) #[:-3] #the last prediction needs forcing parameters from lead_time frequency ahead in time.
        #[:-3] is done because ecml tools currently does not support hour precision in the dates, so we need to remove the last 3 characters.
        base_loader['frequency'] = f"{frequency}h"

        return open_dataset(OmegaConf.to_container(base_loader, resolve=True))
        #return self.datamodule._open_dataset(base_loader)

    @cached_property
    def device(self)-> str:
        if torch.cuda.is_available() and torch.backends.cuda.is_built():
            return "cuda"
        else:
            return "cpu"
    
    @cached_property
    def steps(self) -> int:
        """
        Returns the number of previous timesteps the model needs
        """
        return self.config.training.multistep_input - 1

    @cached_property
    def indices(self) -> dict:
        data_indices = IndexCollection(self.config, self.data_reader.name_to_index)
        #NOTE: what does this depend on in the config? Do we have to change the config outside of the inference config file to get the right indices?

        input_indices = data_indices.data.input.full

        diagnostic_indices = data_indices.model.output.diagnostic.to(self.device)
        forcing_indices = data_indices.model.input.forcing.to(self.device)

        not_forcing = torch.arange(len(input_indices)).to(self.device)
        not_forcing = not_forcing[~torch.isin(not_forcing, forcing_indices)]

        not_diagnostic = torch.arange(len(data_indices.model.output.full)).to(self.device)
        not_diagnostic = not_diagnostic[~torch.isin(not_diagnostic, diagnostic_indices)]

        return {
            'name_to_index': data_indices.model.output.name_to_index,
#            'data_nti': self.data_reader.name_to_index,
            'input': input_indices,
            'diagnostic': diagnostic_indices,
            'forcing': forcing_indices,
            'not_forcing': not_forcing,
            'not_diagnostic': not_diagnostic,
            'data_forcing': data_indices.data.input.forcing.to(self.device),
            'output': data_indices.data.output.full#.to(self.device)
        }
    
    @cached_property
    def dset_properties(self) -> dict:
        """
        Returns needed properties of the zarr archives for storing data.
        When we start using the newest version of ecml tools, we can get the properties directly from the data_reader instead.
        """
        properties = {}
        grids = self.data_reader.grids
        for i,dlabel in enumerate(self.dsets.keys()):
            dpath = self.dsets[dlabel]['path']
            dgrid = self.dsets[dlabel]['grid']
            
            print("Dataset path:", dpath)
            z = zarr.convenience.open(dpath, "r")
            lat = z.latitudes[:].astype(np.float32)
            lon = z.longitudes[:].astype(np.float32)
            lon[lon > 180] -= 360
            
            """
            if dlabel == 'era5' and 'cutout' in self.config.dataloader.dataset.keys():
                pass
                #print("doing cutout masking")
                #print(self.data_reader.grids)
                #drm = self.data_reader.mask
                #lat = lat[drm]
                #lon = lon[drm]
                #local_grid_points = len(lat)
                #NOTE: Currently unimplemented because .mask object does not exist anymore
                #no need to reimplement right now
            """

            properties[dlabel] = {
                "grid_points": grids[i],
                "grid_points_range": (sum(grids[:i]), sum(grids[:(i+1)])),
                "latitudes": lat,
                "longitudes": lon,
            }
            if dgrid == 'xy_regular':
                properties[dlabel]["y"] = z.y[:]
                properties[dlabel]["x"] = z.x[:]
                properties[dlabel]["ydim"] = len(properties[dlabel]["y"])
                properties[dlabel]["xdim"] = len(properties[dlabel]["x"])
                properties[dlabel]['latitudes'] = properties[dlabel]['latitudes'].reshape((properties[dlabel]["ydim"], properties[dlabel]["xdim"]))
                properties[dlabel]['longitudes'] = properties[dlabel]['longitudes'].reshape((properties[dlabel]["ydim"], properties[dlabel]["xdim"]))
                properties[dlabel]['altitude'] = self.altitude[properties[dlabel]["grid_points_range"][0]:properties[dlabel]["grid_points_range"][1]].reshape((properties[dlabel]["ydim"], properties[dlabel]["xdim"]))

            if dlabel == 'meps':
                properties[dlabel]['mapping'] = z.attrs['era_to_meps_mapping']

        return properties

    @cached_property
    def mapping(self) -> dict:
        """
        Maps the variable names from the era5 dataset to the meps dataset.
        """
        if 'meps' in self.dset_properties.keys():
            mapping = self.dset_properties['meps']['mapping']
            mapping['tp'] = 'precipitation_amount_acc6h'
            mapping['2d'] = 'dew_point_temperature_2m'
            return mapping
        return {
            '2t': 'air_temperature_2m',
            '10u': 'x_wind_10m',
            '10v': 'y_wind_10m',
            'msl': 'air_pressure_at_sea_level',
            'tp': 'precipitation_amount_acc6h',
            '2d': 'dew_point_temperature_2m',
        }

    @cached_property
    def create_truth(self) -> bool:
        if self.inf_conf.files.write.truth:
            return True
        
        for ver_type, write in self.inf_conf.files.write.verification.items():
            if write:
                return True
        
        return False

    def to_grid(self, data: np.array, dlabel: str) -> np.array:
        """
        Takes data with 1d grid points and reshapes it to 2d grid points.
        How to reshape depends on the nature of the grid, given by dlabel.
        interpolates if era?
        data is in shape (time, grid_points, variables)
        """
        if self.dsets[dlabel]['grid'] == 'xy_regular':
            return data.reshape((data.shape[0], self.dset_properties[dlabel]["ydim"], self.dset_properties[dlabel]["xdim"], data.shape[-1]))

        elif self.dsets[dlabel]['grid'] == 'latlon' and 'cutout' not in self.config.dataloader.dataset.keys():
            #IN the future, might want to just map to meps grid instead, but for that we would need the meps dataset loaded in era only checkpoints.
            era_lon_gridded_axis = np.arange(-20, 55, 0.25)
            era_lat_gridded_axis = np.arange(50, 75, 0.25)
            Y = len(era_lat_gridded_axis)
            X = len(era_lon_gridded_axis)
            era_lat_gridded, era_lon_gridded = np.meshgrid(era_lat_gridded_axis, era_lon_gridded_axis)
            era_lon_gridded = era_lon_gridded.transpose()
            era_lat_gridded = era_lat_gridded.transpose()

            # Extract forecasts at ERA points
            ypred_era = data
            T, L, P = ypred_era.shape

            # Interpolate irregular ERA grid to regular lat/lon grid
            era_lat, era_lon = self.dset_properties[dlabel]["latitudes"], self.dset_properties[dlabel]["longitudes"]
            icoords = np.zeros([len(era_lon), 2], np.float32)
            icoords[:, 0] = era_lon
            icoords[:, 1] = era_lat
            ocoords = np.zeros([Y*X, 2], np.float32)
            ocoords[:, 0] = era_lon_gridded.flatten()
            ocoords[:, 1] = era_lat_gridded.flatten()

            self.era_lat_gridded_axis = era_lat_gridded_axis
            self.era_lon_gridded_axis = era_lon_gridded_axis

            # This is somewhat slow, should be faster with gridpp.nearest()
            interpolated_era = np.zeros([T, Y, X, P], np.float32)
            for t in range(T):
                #print(f"Interpolating timestep {t}")
                for p in range(P):
                    interpolator = scipy.interpolate.NearestNDInterpolator(icoords, ypred_era[t, :, p])
                    q = interpolator(ocoords)
                    interpolated_era[t, :, :, p] = np.reshape(q, [Y, X])

            altitude = self.altitude[self.dset_properties[dlabel]["grid_points_range"][0]:self.dset_properties[dlabel]["grid_points_range"][1]]
            #print(altitude.shape, ypred_era.shape)
            self.dset_properties[dlabel]["altitude"] = np.reshape(scipy.interpolate.NearestNDInterpolator(icoords, altitude)(ocoords), [Y, X])
            self.dset_properties[dlabel]["latitudes_regular"] = era_lat_gridded
            self.dset_properties[dlabel]["longitudes_regular"] = era_lon_gridded

            return interpolated_era

        return data
    
    def map_to_dict(self, data: np.array, select_vars: list) -> dict:
        """
        Maps the data to a dictionary with the variable names as keys.
        variable names are mapped from era to meps names.
        These mapping keys can be found in the v3 of the zarr archive, but until we fully switch to using that one, they remain hardcoded here:
        """
        #pressure level support to be added later
        mapping = self.mapping
        plevels = {}
        for i,sv in enumerate(select_vars):
            try:
                pl = int(sv.split("_")[-1])
                #print(pl)
            except:
                pl = None

            if sv in mapping:
                sv = mapping[sv]
            
            if pl is not None:
                sv = sv[:-len(str(pl))] + "pl"
                if sv not in plevels.keys():
                    plevels[sv] = {}
                plevels[sv][pl] = i
            else:
                plevels[sv] = i

        return {var: data[..., i][:,np.newaxis,...] if type(i) == int else np.einsum('ijkl->iljk',data[..., np.array(list(i.values()))[np.argsort(list(i.keys()))]]) for var, i in plevels.items()}
        #Returns a dictionary, each key containing data of shape (time, pressure level, grid_y, grid_x). pressure is 1-dimensional if no pressure levels are present.

    def predict(self) -> None:
        trainer = self.trainer
        #self.datamodule.rollout = self.inf_conf.dates.lead_times
        trainer.predict(self.model, datamodule = self.datamodule, return_predictions = False)

#if __name__ == "__main__":
Inference().predict()
