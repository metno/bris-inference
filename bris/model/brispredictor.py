import numpy as np
import torch

from ..checkpoint import Checkpoint
from ..data.datamodule import DataModule
from ..utils import timedelta64_from_timestep
from .basepredictor import BasePredictor, ForecastOutputs
from .wrapper import ModelWrapper


class BrisPredictor(BasePredictor):
    """
    Custom Bris predictor: autoregressive rollout of a forecaster over one or more datasets.

    Methods
    -------

    __init__

    forward: Forward pass through the model.

    initial_state: The model input (data order, physical units) from a batch.

    forecast: Run the rollout for one batch.
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
        self.model = ModelWrapper(
            checkpoint,
            timestep=timedelta64_from_timestep(
                checkpoints_config["forecaster"]["timestep"]
            ),
        )
        self.metadata = checkpoint.metadata
        self.timestep = self.model.timestep
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
        return self.model.input_builders[dataset_name].build_state(
            batch, self.model.input_times(time)
        )

    def forecast(
        self, batch: dict[str, torch.Tensor], time: np.datetime64
    ) -> ForecastOutputs:
        """
        The rollout of the forecaster from one batch, see BasePredictor.forecast.
        """
        # (valid time, {dataset: output}) per lead time
        outputs = [(time, {})]
        x = {}
        for ds in self.dataset_names:
            builder = self.model.input_builders[ds]
            state = self.initial_state(ds, batch[ds], time)
            outputs[0][1][ds] = builder.initial_output(state, -1)

            # Possibly have to extend this to handle imputer, see _step in forecaster.
            state = self.model.pre_process(state, ds, in_place=True)
            # Static forcings of this forecast (lsm, z, ...) from its starting state, like the
            # prognostic variables; only the dynamic forcings are computed on the fly.
            builder.set_static_forcings_from_state(state[:, -1])
            x[ds] = builder.to_model_order(state)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for forecast_step in range(self.forecast_length - 1):
                y_pred = self(x, fcstep=self.fcstep(forecast_step))

                time += self.timestep
                output = {}
                for ds in self.dataset_names:
                    builder = self.model.input_builders[ds]
                    x[ds] = builder.advance(x[ds], y_pred[ds], time)
                    output[ds] = builder.output(
                        self.model.post_process(y_pred[ds], ds, in_place=True)
                    )
                outputs.append((time, output))

                if self.release_cache:
                    del y_pred
                    torch.cuda.empty_cache()

        return outputs
