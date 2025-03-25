import os

import pytest
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bris.checkpoint
import bris.model
import bris.routes
from bris.data.datamodule import DataModule


def test_bris_predictor():
    """Set up a default configuration and do a simple test of the BrisPredictor class.
    Test will be skipped if the required dataset is not available."""
    dataset_path = "./bris_random_data.zarr"
    if os.environ.get("TOX_WORK_DIR"):
        dataset_path = os.environ.get("TOX_WORK_DIR") + "/bris_random_data.zarr"

    if not os.path.exists(dataset_path):
        pytest.skip(
            "Skipping test_bris_predictor, as the required dataset is not available. Run `tox -e trainingdata`."
        )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"File {dataset_path} not found at {os.path.abspath(dataset_path)}"
        )

    checkpoint_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/checkpoint.ckpt"
    )
    checkpoint = bris.checkpoint.Checkpoint(path=checkpoint_path)

    # Create test config
    config = {
        "start_date": "2022-01-01T00:00:00",
        "end_date": "2022-01-02T00:00:00",
        "checkpoint_path": os.path.dirname(os.path.abspath(__file__))
        + "/files/checkpoint.ckpt",
        "leadtimes": 2,
        "timestep": "6h",
        "frequency": "6h",
        "release_cache": True,
        "inference_num_chunks": 32,
        "dataset": dataset_path,
        "workdir": "/tmp",
        "dataloader": {
            "prefetch_factor": 2,
            "num_workers": 1,
            "pin_memory": True,
            "datamodule": {
                "_target_": "bris.data.dataset.NativeGridDataset",
                "_convert_": "all",
            },
        },
        "hardware": {"num_gpus_per_node": 1, "num_gpus_per_model": 1, "num_nodes": 1},
        "hardware_config": {
            "num_gpus_per_node": 1,
            "num_gpus_per_model": 1,
            "num_nodes": 1,
        },
        "model": {"_target_": "bris.model.BrisPredictor", "_convert_": "all"},
        "routing": [
            {
                "decoder_index": 0,
                "domain_index": 0,
                "domain": 0,
                "outputs": [
                    {
                        "netcdf": {
                            "filename_pattern": "meps_pred_%Y%m%dT%HZ.nc",
                            "variables": ["2t", "2d"],
                        }
                    }
                ],
            }
        ],
    }
    args_dict = {
        "debug": False,
        "config": "config/tox_test_inference.yaml",
        "dataset_path": None,
        "dataset_path_cutout": None,
    }
    config = OmegaConf.merge(config, OmegaConf.create(args_dict))
    config.dataset = {
        "dataset": config.dataset,
        "start": config.start_date,
        "end": config.end_date,
        "frequency": config.frequency,
    }

    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoint,
        timestep=config.timestep,
        frequency=config.frequency,
    )

    required_variables = bris.routes.get_required_variables(
        config["routing"], checkpoint
    )

    # Forecaster must know about what leadtimes to output
    _model = instantiate(
        config.model,
        checkpoint=checkpoint,
        hardware_config=config.hardware,
        datamodule=datamodule,
        forecast_length=config.leadtimes,
        required_variables=required_variables,
        release_cache=config.release_cache,
    )

    _bp = bris.model.BrisPredictor(
        checkpoint=checkpoint,
        datamodule=datamodule,
        forecast_length=1,
        required_variables=required_variables,
        hardware_config=DotDict(config.hardware_config),
    )


if __name__ == "__main__":
    test_bris_predictor()
