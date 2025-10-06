import os
from datetime import datetime, timedelta

import pytest
from anemoi.utils.config import DotDict
from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bris.checkpoint
import bris.routes
from bris.checkpoint import Checkpoint
from bris.data.datamodule import DataModule
from bris.inference import Inference
from bris.model.brispredictor import BrisPredictor
from bris.utils import (
    create_config,
    get_all_leadtimes,
    parse_args,
    set_base_seed,
    set_encoder_decoder_num_chunks,
)


def test_bris_predictor(benchmark):
    """Set up configuration and do a simple test run of the BrisPredictor class.
    Test will be skipped if the required dataset is not available. Also touches
    DataModule."""
    dataset_path = "./bris_random_data.zarr"
    if os.environ.get("TOX_ENV_DIR"):
        dataset_path = os.environ.get("TOX_ENV_DIR") + "/tmp/bris_random_data.zarr"
    if not os.path.exists(dataset_path):
        pytest.skip(
            "Skipping test_bris_predictor, as the required dataset is not available. Run `tox -e trainingdata`."
        )

    checkpoint_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/checkpoint.ckpt"
    )

    # Create test config
    config = bris.utils.create_config(
        "config/tox_test_inference.yaml",
        {
            "leadtimes": 2,
            "timestep": "6h",
            "checkpoints": {"forecaster": {"checkpoint_path": checkpoint_path}},
            "dataset": dataset_path,
        },
    )

    models = list(config.checkpoints.keys())
    checkpoints = {
        model: bris.checkpoint.Checkpoint(
            config.checkpoints[model].checkpoint_path,
            getattr(config.checkpoints[model], "switch_graph", None),
        )
        for model in models
    }
    set_encoder_decoder_num_chunks(getattr(config, "inference_num_chunks", 1))
    if "release_cache" not in config or not isinstance(config["release_cache"], bool):
        config["release_cache"] = False

    set_base_seed()

    # Get timestep from checkpoint. Also store a version in seconds for local use.
    for model in models:
        config.checkpoints[model].timestep = None
        try:
            config.checkpoints[model].timestep = checkpoints[model].config.data.timestep
        except KeyError as err:
            raise RuntimeError(
                f"Error getting timestep from {model} checkpoint (checkpoint.config.data.timestep)"
            ) from err
        config.checkpoints[model].timestep_seconds = frequency_to_seconds(
            config.checkpoints[model].timestep
        )

    # Get multistep. A default of 2 to ignore multistep in start_date calculation if not set.
    multistep = 2
    multistep = checkpoints["forecaster"].config.training.multistep_input

    # If no start_date given, calculate as end_date-((multistep-1)*timestep)
    if "start_date" not in config or config.start_date is None:
        config.start_date = datetime.strftime(
            datetime.strptime(config.end_date, "%Y-%m-%dT%H:%M:%S")
            - timedelta(
                seconds=(multistep - 1) * config.checkpoints.forecaster.timestep_seconds
            ),
            "%Y-%m-%dT%H:%M:%S",
        )
    else:
        config.start_date = datetime.strftime(
            datetime.strptime(config.start_date, "%Y-%m-%dT%H:%M:%S")
            - timedelta(
                seconds=(multistep - 1) * config.checkpoints.forecaster.timestep_seconds
            ),
            "%Y-%m-%dT%H:%M:%S",
        )

    config.dataset = {
        "dataset": config.dataset,
        "start": config.start_date,
        "end": config.end_date,
        "frequency": config.frequency,
    }

    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoints["forecaster"],
        timestep="6h",
        frequency=config.frequency,
    )

    required_variables = bris.routes.get_required_variables(
        config["routing"], checkpoints
    )

    # Set hydra defaults
    config.defaults = [
        {"override hydra/job_logging": "none"},  # disable config parsing logs
        {"override hydra/hydra_logging": "none"},  # disable config parsing logs
        "_self_",
    ]
    num_members_in_parallel = 1

    # Forecaster must know about what leadtimes to output
    _model = instantiate(
        config.model,
        checkpoints=checkpoints,
        hardware_config=config.hardware,
        datamodule=datamodule,
        forecast_length=config.leadtimes,
        required_variables=required_variables,
        release_cache=config.release_cache,
        num_members_in_parallel=num_members_in_parallel,
    )

    result = benchmark(bris.model.brispredictor.BrisPredictor,
            checkpoints=checkpoints,
            datamodule=datamodule,
            forecast_length=1,
            required_variables=required_variables,
            hardware_config=DotDict(config.hardware),
            num_members_in_parallel=num_members_in_parallel,
        )


def test_multiencdec_predictor():
    """Set up a configuration and do a simple test run of the MultiEncDecPredictor class.
    Test will be skipped if the required dataset is not available."""
    dataset_path = "./bris_random_data.zarr"
    if os.environ.get("TOX_ENV_DIR"):
        dataset_path = os.environ.get("TOX_ENV_DIR") + "/tmp/bris_random_data.zarr"
    if not os.path.exists(dataset_path):
        pytest.skip(
            "Skipping test_multiencdec_predictor, as the required dataset is not available. Run `tox -e trainingdata`."
        )

    checkpoint_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/multiencdec.ckpt"
    )

    # Create test config
    config = bris.utils.create_config(
        "config/tox_test_inference_multi.yaml",
        {
            "leadtimes": 2,
            "timestep": "6h",
            "checkpoints": {"forecaster": {"checkpoint_path": checkpoint_path}},
            "dataset": {
                "zip": [
                    {"dataset": dataset_path},
                    {"dataset": dataset_path, "select": ["tp", "2t"]},
                ],
                "adjust": ["start", "end"],
            },
        },
    )
    models = list(config.checkpoints.keys())
    checkpoints = {
        model: bris.checkpoint.Checkpoint(
            config.checkpoints[model].checkpoint_path,
            getattr(config.checkpoints[model], "switch_graph", None),
        )
        for model in models
    }
    set_encoder_decoder_num_chunks(getattr(config, "inference_num_chunks", 1))
    if "release_cache" not in config or not isinstance(config["release_cache"], bool):
        config["release_cache"] = False

    set_base_seed()

    # Get timestep from checkpoint. Also store a version in seconds for local use.
    for model in models:
        config.checkpoints[model].timestep = None
        try:
            config.checkpoints[model].timestep = checkpoints[model].config.data.timestep
        except KeyError as err:
            raise RuntimeError(
                f"Error getting timestep from {model} checkpoint (checkpoint.config.data.timestep)"
            ) from err
        config.checkpoints[model].timestep_seconds = frequency_to_seconds(
            config.checkpoints[model].timestep
        )

    # Get multistep. A default of 2 to ignore multistep in start_date calculation if not set.
    multistep = 2
    multistep = checkpoints["forecaster"].config.training.multistep_input

    # If no start_date given, calculate as end_date-((multistep-1)*timestep)
    if "start_date" not in config or config.start_date is None:
        config.start_date = datetime.strftime(
            datetime.strptime(config.end_date, "%Y-%m-%dT%H:%M:%S")
            - timedelta(
                seconds=(multistep - 1) * config.checkpoints.forecaster.timestep_seconds
            ),
            "%Y-%m-%dT%H:%M:%S",
        )
    else:
        config.start_date = datetime.strftime(
            datetime.strptime(config.start_date, "%Y-%m-%dT%H:%M:%S")
            - timedelta(
                seconds=(multistep - 1) * config.checkpoints.forecaster.timestep_seconds
            ),
            "%Y-%m-%dT%H:%M:%S",
        )

    config.dataset = {
        "dataset": config.dataset,
        "start": config.start_date,
        "end": config.end_date,
        "frequency": config.frequency,
    }

    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoints["forecaster"],
        timestep="6h",
        frequency=config.frequency,
    )

    required_variables = bris.routes.get_required_variables(
        config["routing"], checkpoints
    )

    # Set hydra defaults
    config.defaults = [
        {"override hydra/job_logging": "none"},  # disable config parsing logs
        {"override hydra/hydra_logging": "none"},  # disable config parsing logs
        "_self_",
    ]
    num_members_in_parallel = 1

    # Forecaster must know about what leadtimes to output
    _model = instantiate(
        config.model,
        checkpoints=checkpoints,
        hardware_config=config.hardware,
        datamodule=datamodule,
        forecast_length=config.leadtimes,
        required_variables=required_variables,
        release_cache=config.release_cache,
        num_members_in_parallel=num_members_in_parallel,
    )

    _bp = bris.model.multiencdecpredictor.MultiEncDecPredictor(
        checkpoints=checkpoints,
        datamodule=datamodule,
        forecast_length=1,
        required_variables=required_variables,
        hardware_config=DotDict(config.hardware),
        num_members_in_parallel=num_members_in_parallel,
    )
