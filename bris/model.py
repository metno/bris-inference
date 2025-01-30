import os
import math 
import logging 
import numpy as np
from abc import abstractmethod
from typing import Optional, Any, Iterable

import torch 
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.distributed.distributed_c10d import ProcessGroup

from .forcings import get_dynamic_forcings
from .checkpoint import Checkpoint
from .utils import check_anemoi_training

LOGGER = logging.getLogger(__name__)

class BasePredictor(pl.LightningModule):
    def __init__(
            self, 
            *args: Any, 
            checkpoint: Checkpoint,
            hardware_config,
            **kwargs : Any
            ):
        """ 
            Base predictor class, overwrite all the class methods
    
        """
        


        super().__init__(*args, **kwargs)
        #Lazy init
        self.model_comm_group = None
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        
        if check_anemoi_training(checkpoint.metadata):
            self.legacy = False
        else:
            self.legacy = True

        if self.legacy:
            self.model_comm_group = None
            self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // hardware_config["num_gpus_per_model"]
            self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % hardware_config["num_gpus_per_model"]
            self.model_comm_num_groups = math.ceil(
            hardware_config["num_gpus_per_node"] * hardware_config["num_nodes"] / hardware_config["num_gpus_per_model"],
        )
        else:
            #Lazy init
            self.model_comm_group = None
            self.model_comm_group_id = 0
            self.model_comm_group_rank = 0
            self.model_comm_num_groups = 1            

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int = None,
        model_comm_group_rank: int = None,
        model_comm_num_groups: int = None,
        model_comm_group_size: int = None,
    ) -> None:
        self.model_comm_group = model_comm_group
        if not self.legacy:
            self.model_comm_group_id = model_comm_group_id
            self.model_comm_group_rank = model_comm_group_rank
            self.model_comm_num_groups = model_comm_num_groups
            self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    @abstractmethod
    def get_static_forcings(self, datareader):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass 

    @abstractmethod
    def advance_input_predict(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        pass 

    @abstractmethod
    def predict_step(self, batch: torch.Tensor, batch_idx : int) -> torch.Tensor:
        pass 

    @abstractmethod
    def set_variable_indices(self, required_variables: list):
        pass

    
class BrisPredictor(BasePredictor):
    def __init__(
            self,
            *args,
            checkpoint: Checkpoint,
            data_reader: Iterable,
            forecast_length: int,
            required_variables: list,
            release_cache: bool=False,
            **kwargs
            ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

        self.model=checkpoint.model
        self.data_indices = self.model.data_indices
        self.metadata = checkpoint.metadata

        #TODO: where should these come from, add asserts?
        self.frequency = self.metadata.config.data.frequency #["config"]["data"]["frequency"]
        if isinstance(self.frequency, str) and self.frequency[-1] == 'h':
            self.frequency = int(self.frequency[0:-1])

        self.forecast_length = forecast_length
        self.latitudes = data_reader.latitudes
        self.longitudes = data_reader.longitudes

        # this makes it backwards compatible with older 
        # anemoi-models versions. I.e legendary gnome, etc..
        if (
            hasattr(self.data_indices, "internal_model") and hasattr(self.data_indices,"internal_data")
            ):
            self.internal_model = self.data_indices.internal_model
            self.internal_data = self.data_indices.internal_data
        else:
            self.internal_model = self.data_indices.model
            self.internal_data = self.data_indices.data
        self.set_variable_indices(required_variables)
        self.set_static_forcings(data_reader, self.metadata.config.data.forcing)
        
        self.model.eval()
        self.release_cache = release_cache


    def set_static_forcings(self, data_reader, selection) -> None:

        self.static_forcings = {}
        data = torch.from_numpy(data_reader[0].squeeze(axis=1).swapaxes(0,1))
        data_normalized = self.model.pre_processors(data, in_place=True)

        # np.ndarray are by default set to np.float64 and torch tensor torch.float32
        # without explicit converting and casting to torch.float32
        # appending an numpy array to torch.tensor might not automatically cast np.ndarray to torch.float32
        # i.e the new updated x tensor internally will have torch.float64, resulting in memory increase
        # both CPU/GPU RAM

        if "cos_latitude" in selection:
            self.static_forcings["cos_latitude"] = torch.from_numpy(np.cos(data_reader.latitudes * np.pi / 180.)).float()

        if "sin_latitude" in selection:    
            self.static_forcings["sin_latitude"] = torch.from_numpy(np.sin(data_reader.latitudes * np.pi / 180.)).float()
            
        if "cos_longitude" in selection:
            self.static_forcings["cos_longitude"] = torch.from_numpy(np.cos(data_reader.longitudes * np.pi / 180. )).float()
        
        if "sin_longitude" in selection:
            self.static_forcings["sin_longitude"] = torch.from_numpy(np.sin(data_reader.longitudes * np.pi / 180.)).float()

        if "lsm" in selection:
            self.static_forcings["lsm"] = data_normalized[..., data_reader.name_to_index["lsm"]].float()

        if "z" in selection:
            self.static_forcings["z"] = data_normalized[..., data_reader.name_to_index["z"]].float()

        del data_normalized

    def set_variable_indices(self, required_variables: list) -> None:
        required_variables = required_variables[0] #Assume one decoder
        variable_indices_input = list()
        variable_indices_output = list()
        for name in required_variables:
            index_input = self.internal_data.input.name_to_index[name]
            variable_indices_input += [index_input]
            index_output = self.internal_model.output.name_to_index[name]
            variable_indices_output += [index_output]

        self.variable_indices_input = variable_indices_input
        self.variable_indices_output = variable_indices_output


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.model(x, self.model_comm_group)
    
    def advance_input_predict(self, x: torch.Tensor, y_pred: torch.Tensor, time: np.datetime64) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        #Get prognostic variables:
        x[:, -1, :, :, self.internal_model.input.prognostic] = y_pred[..., self.internal_model.output.prognostic]

        forcings = get_dynamic_forcings(time, self.latitudes, self.longitudes, self.metadata.config.data.forcing)
        forcings.update(self.static_forcings)
        
        for forcing, value in forcings.items():
            if isinstance(value, np.ndarray): 
                x[:, -1, :, :, self.internal_model.input.name_to_index[forcing]] = torch.from_numpy(value).to(dtype=x.dtype)#, device=x.device)
            else:
                x[:, -1, :, :, self.internal_model.input.name_to_index[forcing]] = value #torch.from_numpy(np.array(value)).to(dtype=x.dtype, device=x.device)
        return x

    @torch.inference_mode
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:

        multistep = self.metadata.config.training.multistep_input

        batch = self.allgather_batch(batch)
        
        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0], 'h') #Consider not forcing 'h' here and instead generalize time + self.frequency
        times = [time]
        y_preds = torch.empty((batch.shape[0], self.forecast_length, batch.shape[-2], len(self.variable_indices_input)), dtype=batch.dtype, device="cpu")#.cpu()

        #Insert analysis for t=0
        y_analysis = batch[:,multistep-1,...].cpu()
        y_analysis[...,self.internal_data.output.diagnostic] = 0. #Set diagnostic variables to zero
        y_preds[:,0,...] = y_analysis[...,self.variable_indices_input]

        #Possibly have to extend this to handle imputer, see _step in forecaster.
        batch = self.model.pre_processors(batch, in_place=True)
        x = batch[..., self.internal_data.input.full]


        for fcast_step in range(self.forecast_length-1):
            y_pred = self(x)
            time += self.frequency
            x = self.advance_input_predict(x, y_pred, time)
            y_preds[:, fcast_step+1] = self.model.post_processors(y_pred, in_place=True)[:,0,:,self.variable_indices_output].cpu()

            times.append(time)
            if self.release_cache:
                del y_pred
                torch.cuda.empty_cache()
        return {"pred": [y_preds.to(torch.float32).numpy()], "times": times, "group_rank": self.model_comm_group_rank, "ensemble_member": 0}

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch #Not implemented properly
                  

class NetatmoPredictor(BasePredictor):
    #TODO: FIX VARIABLE INDICES
    def __init__(
            self,
            *args,
            checkpoint: Checkpoint,
            data_reader: Iterable,
            forecast_length: int,
            variable_indices: list,
            release_cache: bool=False,
            **kwargs
            ) -> None:
        super().__init__(
            *args, checkpoint=checkpoint, **kwargs)
        
        self.model=checkpoint.model
        self.metadata = checkpoint.metadata

        self.frequency = self.metadata["config"]["data"]["frequency"]
        if isinstance(self.frequency, str) and self.frequency[-1] == 'h':
            self.frequency = int(self.frequency[0:-1])

        self.forecast_length = forecast_length
        self.latitudes = data_reader.latitudes
        self.longitudes = data_reader.longitudes
        self.variable_indices = variable_indices
        
        self.set_static_forcings(data_reader, self.metadata["config"]["data"]["zip"])

    def set_variable_indices(self, required_variables: list) -> None:
        variable_indices_input = [() for _ in required_variables]
        variable_indices_output = [() for _ in required_variables]


        for dec_index, required_vars_dec in enumerate(required_variables):
            _variable_indices_input = list()
            _variable_indices_output = list()
            for name in required_vars_dec:
                index_input = self.data_indices[dec_index].internal_data.input.name_to_index[name]
                _variable_indices_input += [index_input]
                index_output = self.data_indices[dec_index].internal_model.output.name_to_index[name]
                _variable_indices_output += [index_output]
            variable_indices_input[dec_index] = _variable_indices_input
            variable_indices_output[dec_index] = _variable_indices_output

        self.variable_indices_input = variable_indices_input
        self.variable_indices_output = variable_indices_output

    def set_static_forcings(self, data_reader, zip_config):

        data = data_reader[0]
        num_dsets = len(data)
        data = [torch.from_numpy(x.squeeze(axis=1).swapaxes(0,1)) for x in data]
        data_normalized = self.model.pre_processors(data, in_place=False)

        self.static_forcings = [{} for _ in range(num_dsets)]
        for dset in range(num_dsets):
            selection = zip_config[dset]["forcing"]
            if "cos_latitude" in selection:
                self.static_forcings[dset]["cos_latitude"] = np.cos(data_reader.latitudes[dset] * np.pi / 180.)

            if "sin_latitude" in selection:    
                self.static_forcings[dset]["sin_latitude"] = np.sin(data_reader.latitudes[dset] * np.pi / 180.)
                
            if "cos_longitude" in selection:
                self.static_forcings[dset]["cos_longitude"] = np.cos(data_reader.longitudes[dset] * np.pi / 180. )
            
            if "sin_longitude" in selection:
                self.static_forcings[dset]["sin_longitude"] = np.sin(data_reader.longitudes[dset] * np.pi / 180.)

            if "lsm" in selection:
                self.static_forcings[dset]["lsm"] = data_normalized[dset][..., data_reader.name_to_index[dset]["lsm"]]

            if "z" in selection:
                self.static_forcings[dset]["z"] = data_normalized[dset][..., data_reader.name_to_index[dset]["z"]]
    
    def forward(self, x: torch.Tensor)-> list[torch.Tensor]:
        return self.model(x, self.model_comm_group)
    
    def advance_input_predict(self, x, y_pred, time):
        data_indices = self.model.data_indices

        for i in range(len(x)):

            x[i] = x[i].roll(-1,dims=1)
            #Get prognostic variables:
            x[i][:, -1, :, :, data_indices[i].internal_model.input.prognostic] = y_pred[i][..., data_indices[i].internal_model.output.prognostic]
    
            forcings = get_dynamic_forcings(time, self.latitudes[i], self.longitudes[i], self.metadata["config"]["data"]["zip"][i]["forcing"])
            forcings.update(self.static_forcings[i])

            for forcing, value in forcings.items():
                if type(value) == np.ndarray:
                    x[i][:, -1, :, :, data_indices[i].internal_model.input.name_to_index[forcing]] = torch.from_numpy(value)
                else:
                    x[i][:, -1, :, :, data_indices[i].internal_model.input.name_to_index[forcing]] = value

        return x  


    @torch.inference_mode
    def predict_step(self, batch: list, batch_idx: int) -> list:
        num_dsets = len(batch)
        data_indices = self.model.data_indices
        multistep = self.metadata["config"]["training"]["multistep_input"]

        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0], 'h') #Consider not forcing 'h' here and instead generalize time + self.frequency
        times = [time]
        y_preds = [np.zeros((batch[i].shape[0], self.forecast_length, batch[i].shape[-2], len(self.variable_indices_input[i]))) for i in range(num_dsets)]
        #Insert analysis for t=0
        for i in range(num_dsets):
            y_analysis = batch[i][:,multistep-1,0,...]
            y_analysis[...,data_indices[i].internal_data.output.diagnostic] = 0. 
            y_preds[i][:,0,...] = y_analysis[...,self.variable_indices_input[i]].cpu().to(torch.float32).numpy()

        batch = self.model.pre_processors(batch, in_place=False)
        x = [batch[i][..., data_indices[i].internal_data.input.full] for i in range(num_dsets)]
        for fcast_step in range(self.forecast_length-1):
            y_pred = self(x)
            time += self.frequency
            x = self.advance_input_predict(x, y_pred, time)
            y_pp = self.model.post_processors(y_pred, in_place=False)
            for i in range(num_dsets):
                y_preds[i][:, fcast_step+1, ...] = y_pp[i][:,0,...,self.variable_indices_output[i]].cpu().to(torch.float32).numpy() 
            times.append(time)

        return {"pred": y_preds, "times": times, "group_rank": self.model_comm_group_rank, "ensemble_member": 0}
    
    def set_model_comm_group(self, model_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group

