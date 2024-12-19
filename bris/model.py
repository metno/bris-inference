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

LOGGER = logging.getLogger(__name__)

class BasePredictor(pl.LightningModule):
    def __init__(
            self, 
            *args: Any, 
            **kwargs : Any
            ):
        """ 
            THIS IS NOT THE FINALIZED SETUP FOR THE BASEFORACASTER,
            CHANGES MAY OCCUR.
            
            Generic forecaster which handles multi or single
            GPU/Node setup. Includes abstract methods which
            needs to be overwritten by custom forecasters i.e
            interpolator, multi-enc-dec, ensemble, etc..
    
        """
        


        super().__init__(*args, **kwargs)

        #assert num_device_per_model >= 1, f"Number of device per model is not greater or equal to 1. Got: {num_device_per_model}"
        #assert num_devices_per_nodes >= 1, f"Number of device per node is not greater or equal to 1. Got: {num_device_per_model}"
        #assert num_nodes >= 1, f"Number of nodes is not greater or equal to 1. Got : {num_nodes}" 


        #self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // num_device_per_model
        #self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % num_device_per_model
        #self.model_comm_num_groups = math.ceil(
        #    num_devices_per_nodes * num_nodes / num_device_per_model
        #)

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group = None
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

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

    
class BrisPredictor(BasePredictor):
    def __init__(
            self,
            *args,
            checkpoint: Checkpoint,
            data_reader: Iterable,
            forecast_length: int,
            select_indices,
            **kwargs
            ) -> None:
        super().__init__(
            *args,**kwargs)
        
        self.model=checkpoint.model
        self.metadata = checkpoint.metadata

        #TODO: where should these come from, add asserts?
        self.frequency = self.metadata["config"]["data"]["frequency"]
        if isinstance(self.frequency, str) and self.frequency[-1] == 'h':
            self.frequency = int(self.frequency[0:-1])

        self.forecast_length = forecast_length
        self.latitudes = data_reader.latitudes
        self.longitudes = data_reader.longitudes
        self.select_indices = select_indices
        

        self.set_static_forcings(data_reader, self.metadata["config"]["data"]["forcing"])
        
    def set_static_forcings(self, data_reader, selection):

        self.static_forcings = {}
        data = torch.from_numpy(data_reader[0].squeeze(axis=1).swapaxes(0,1))
        data_normalized = self.model.pre_processors(data)

        if "cos_latitude" in selection:
            self.static_forcings["cos_latitude"] = np.cos(data_reader.latitudes)

        if "sin_latitude" in selection:    
            self.static_forcings["sin_latitude"] = np.sin(data_reader.latitudes)
            
        if "cos_longitude" in selection:
            self.static_forcings["cos_longitude"] = np.cos(data_reader.longitudes)
        
        if "sin_longitude" in selection:
            self.static_forcings["sin_longitude"] = np.sin(data_reader.longitudes)

        if "lsm" in selection:
            self.static_forcings["lsm"] = data_normalized[..., data_reader.name_to_index["lsm"]]

        if "z" in selection:
            self.static_forcings["z"] = data_normalized[..., data_reader.name_to_index["z"]]


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.model(x, self.model_comm_group)
    
    def advance_input_predict(self, x, y_pred, time):
        data_indices = self.model.data_indices

        x = x.roll(-1, dims=1)

        #Get prognostic variables:
        x[:, -1, :, :, data_indices.internal_model.input.prognostic] = y_pred[..., data_indices.internal_model.output.prognostic]

        forcings = get_dynamic_forcings(time, self.latitudes, self.longitudes, self.metadata["config"]["data"]["forcing"])
        forcings.update(self.static_forcings)

        for forcing, value in forcings.items():
            if type(value) == np.ndarray:
                x[:, -1, :, :, data_indices.internal_model.input.name_to_index[forcing]] = torch.from_numpy(value)
            else:
                x[:, -1, :, :, data_indices.internal_model.input.name_to_index[forcing]] = value

        return x

    @torch.inference_mode
    def predict_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        data_indices = self.model.data_indices
        multistep = self.metadata["config"]["training"]["multistep_input"]

        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0], 'h') #Consider not forcing 'h' here and instead generalize time + self.frequency
        times = [time]
        y_preds = np.zeros((batch.shape[0], self.forecast_length + 1, batch.shape[-2], len(self.select_indices)))

        #Insert analysis for t=0
        y_analysis = batch[:,multistep-1,0,...]
        y_analysis[...,data_indices.internal_data.output.diagnostic] = 0. #Set diagnostic variables to zero
        y_preds[:,0,...] = y_analysis[...,self.select_indices].cpu().to(torch.float32).numpy()

        #Possibly have to extend this to handle imputer, see _step in forecaster.
        batch = self.model.pre_processors(batch, in_place=False)
        x = batch[..., data_indices.internal_data.input.full]
        with torch.amp.autocast(device_type= "cuda", dtype=torch.bfloat16):
            for fcast_step in range(self.forecast_length):
                y_pred = self(x)
                print(type(time))
                print(time)
                print(type(self.frequency))
                print(self.frequency)
                time += self.frequency
                print(time)
                x = self.advance_input_predict(x, y_pred, time)
                y_preds[:, fcast_step+1, ...] = self.model.post_processors(y_pred, in_place=False)[:,0,...,self.select_indices].cpu().to(torch.float32).numpy() 
                times.append(time)
        return {"pred": [y_preds], "times": times, "group_rank": self.model_comm_group_rank, "ensemble_member": 0}
                  

class NetatmoPredictor(BasePredictor):
    def __init__(
            self,
            *args,
            model: torch.nn.Module, 
            metadata: DictConfig, 
            datareader: Iterable,
            **kwargs
            ) -> None:
        super().__init__(
            *args,**kwargs)
        self.model = model
        self.metadata = metadata
        self.config = config
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self(x, self.model_comm_group)
    
    def advance_input_predict(self, x, y_pred):
        return super().advance_input_predict(x, y_pred)
    
    @torch.inference_mode
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        pass