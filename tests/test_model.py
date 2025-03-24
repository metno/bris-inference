import os

from anemoi.utils.config import DotDict
from omegaconf import OmegaConf

import bris.checkpoint
import bris.model
from bris.data.datamodule import DataModule


def test_bris_predictor():
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
        "dataset_path": "/home/larsfp/nobackup/bris_random_data.zarr",
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
        # "config": "config/larsfp.yaml",
        # "checkpoint_path": "/home/larsfp/nobackup/output/checkpoint/1aece27b-be55-496a-b4f5-72649019f1d4/inference-last.ckpt",
        # "start_date": "2022-01-01T00:00:00",
        # "end_date": "2022-01-02T00:00:00",
        # "dataset_path": None,
        # "workdir": "/tmp",
        "dataset_path_cutout": None,
        # "frequency": "6h",
        # "leadtimes": 2,
    }
    config = OmegaConf.merge(config, OmegaConf.create(args_dict))

    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoint,
    )

    bp = bris.model.BrisPredictor(
        checkpoint=checkpoint,
        datamodule=datamodule,
        forecast_length=1,
        required_variables={},
        hardware_config=DotDict(config.hardware_config),
    )

    print(bp)


if __name__ == "__main__":
    test_bris_predictor()
