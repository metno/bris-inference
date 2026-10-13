import numpy as np
import torch

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..utils import timedelta64_from_timestep
from .basepredictor import BasePredictor
from .wrapper import ModelWrapper


class BrisPredictor(BasePredictor):
    """
    Custom Bris predictor: autoregressive rollout of a forecaster over one or more datasets.

    Methods
    -------

    __init__

    forward: Forward pass through the model.

    initial_state: The model input (data order, physical units) from a batch.

    predict_step: Run the rollout for one batch.
    """

    def __init__(
        self,
        *args,
        checkpoints: dict[str, Checkpoint],
        datamodule: DataModule,
        checkpoints_config: dict,
        required_variables: dict,
        release_cache: bool = False,
        fcstep_const: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the BrisPredictor.

        Args:
            checkpoints
                Example: {"forecaster": checkpoint_object}

            datamodule
                Data loader containing the dataset, from one or more datasets. Loaded in config as for example:

                    dataset: /home/larsfp/nobackup/bris_random_data.zarr
                    dataloader:
                        datamodule:
                            _target_: bris.data.dataset.NativeGridDataset

            checkpoints_config
                config.checkpoints; "forecaster" needs "timestep" and "leadtimes" (the forecast length).

            required_variables
                Dictionary of datasets with list of required variables for each dataset. Example:
                    {"data": ['2d', '2t']}

            release_cache
                Release cache (torch.cuda.empty_cache()) after each prediction step. This is useful for large models,
                but may slow down the prediction.

            fcstep_const
                For inference on non-rollout trained ensemble models. Keep fcstep constant at 0 as in training if True.
        """

        super().__init__(
            *args,
            checkpoints=checkpoints,
            release_cache=release_cache,
            fcstep_const=fcstep_const,
            **kwargs,
        )

        checkpoint = checkpoints["forecaster"]
        self.model = ModelWrapper(checkpoint)
        self.metadata = checkpoint.metadata

        self.timestep = timedelta64_from_timestep(
            checkpoints_config["forecaster"]["timestep"]
        )
        self.multistep = self.model.multistep
        self.forecast_length = checkpoints_config["forecaster"]["leadtimes"]
        self.dataset_names = self.model.dataset_names

        assert self.dataset_names == datamodule.dataset_names, (
            f" Dataset names of the input data {datamodule.dataset_names} do not match expected dataset names {self.dataset_names} of the model."
        )

        for ds in self.dataset_names:
            self.model.add_dataset(
                ds,
                required_variables=required_variables.get(ds, []),
                dataset_variables=datamodule.data_readers[ds].variables,
                latitudes=datamodule.latitudes[ds],
                longitudes=datamodule.longitudes[ds],
            )

        # Set per forecast in predict_step, from the starting state of the forecast (the batch),
        # like the prognostic variables; only the dynamic forcings are computed on the fly. The
        # first time step of the dataset is not representative for long datasets, where z changes
        # with model updates.
        self.static_forcings = {ds: {} for ds in self.dataset_names}

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        Perform a forward pass through the model.
        Args:
            x (dict): Model inputs keyed by dataset name.
        Returns:
            dict: Model outputs keyed by dataset name.
        """
        return self.model(x, model_comm_group=self.model_comm_group, **kwargs)

    def initial_state(
        self, dataset_name: str, batch: torch.Tensor, time: np.datetime64
    ) -> torch.Tensor:
        """
        The model input of a dataset in data order (physical units) from a batch: prognostic
        variables and static forcings from the batch, dynamic forcings for each input time.

        Args:
            dataset_name (str): Dataset name.
            batch (torch.Tensor): Input states, shape (batch, multistep, ensemble, points, variables).
            time (np.datetime64): Time of the last input state (the start of the forecast).
        """
        times = [
            time - (self.multistep - 1 - time_index) * self.timestep
            for time_index in range(self.multistep)
        ]
        return self.model.inputs[dataset_name].build_state(batch, times)

    @torch.inference_mode
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Perform a prediction step using the model.
        Args:
            batch (tuple): Input batch containing the data.
            batch_idx (int): Index of the batch.
        Returns:
            dict: Dictionary containing the predicted output, time stamps, group rank, and ensemble member.
        """
        batch = self.allgather_batch(batch)

        batch, time_stamp = batch
        time = np.datetime64(time_stamp[0])
        times = [time]

        y_preds = {}
        x = {}
        for ds in self.dataset_names:
            inputs = self.model.inputs[ds]
            y_preds[ds] = torch.empty(
                (
                    batch[ds].shape[0],
                    self.forecast_length,
                    batch[ds].shape[-2],
                    len(inputs.mapping.variables_output),
                ),
                dtype=batch[ds].dtype,
                device="cpu",
            )

            state = self.initial_state(ds, batch[ds], time)
            y_preds[ds][:, 0] = inputs.initial_output(state, self.multistep - 1)

            # Possibly have to extend this to handle imputer, see _step in forecaster.
            state = self.model.pre_process(state, ds, in_place=True)
            self.static_forcings[ds] = inputs.static_forcings(
                state[:, self.multistep - 1]
            )
            x[ds] = inputs.to_model_order(state)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for forecast_step in range(self.forecast_length - 1):
                y_pred = self(x, fcstep=self.fcstep(forecast_step))

                time += self.timestep
                for ds in self.dataset_names:
                    inputs = self.model.inputs[ds]
                    x[ds] = inputs.advance(
                        x[ds], y_pred[ds], time, self.static_forcings[ds]
                    )
                    y_preds[ds][:, forecast_step + 1] = inputs.output(
                        self.model.post_process(y_pred[ds], ds, in_place=True)
                    )

                times.append(time)
                if self.release_cache:
                    del y_pred
                    torch.cuda.empty_cache()

        return self.prediction(y_preds, times)
