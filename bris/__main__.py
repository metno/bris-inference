import logging
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta

from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate
import torch

import bris.routes
from bris.data.datamodule import DataModule

from .checkpoint import Checkpoint
from .inference import Inference
from .utils import (
    create_config,
    get_all_leadtimes,
    parse_args,
    set_base_seed,
    set_encoder_decoder_num_chunks,
)
from .writer import CustomWriter

LOGGER = logging.getLogger(__name__)


def main(arg_list: list[str] | None = None):
    args = parse_args(arg_list)
    config = create_config(args["config"], args)

    models = list(config.checkpoints.keys())
    checkpoints = {
        model: Checkpoint(
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

    num_members = 1
    try:
        num_members = int(
            config['hardware']['num_members']
        )
    except:
        LOGGER.debug("Key 'num_members' was not found in config")

    # Distribute ensemble members across GPUs, run in sequence if not enough GPUs
    num_gpus = config["hardware"]["num_gpus_per_node"] * config["hardware"]["num_nodes"]
    num_gpus_per_model = config["hardware"]["num_gpus_per_model"]
    num_gpus_per_ensemble = num_gpus_per_model * num_members

    if num_gpus_per_ensemble > num_gpus:
        assert num_gpus_per_ensemble % num_gpus == 0, (
            f"Number of gpus per ensemble ({num_gpus_per_ensemble}) needs to be divisible by num_gpus ({num_gpus}). "
            f"num_gpus_per_ensemble = num_gpus_per_model * num_members"
        )
        num_members_in_sequence = int(num_gpus_per_ensemble / num_gpus)
        num_members_in_parallel = int(num_gpus / num_gpus_per_model)
        num_gpus_per_ensemble = num_gpus
    else:
        num_members_in_sequence = 1
        num_members_in_parallel = num_members
    

    # Get multistep. A default of 2 to ignore multistep in start_date calculation if not set.
    multistep = 2
    try:
        multistep = checkpoints["forecaster"].config.training.multistep_input
    except KeyError:
        LOGGER.debug("Multistep not found in checkpoint")

    # If no start_date given, calculate as end_date-((multistep-1)*timestep)
    if "start_date" not in config or config.start_date is None:
        config.start_date = datetime.strftime(
            datetime.strptime(config.end_date, "%Y-%m-%dT%H:%M:%S")
            - timedelta(
                seconds=(multistep - 1) * config.checkpoints.forecaster.timestep_seconds
            ),
            "%Y-%m-%dT%H:%M:%S",
        )
        LOGGER.info(
            "No start_date given, setting %s based on start_date and timestep.",
            config.start_date,
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
        timestep=config.checkpoints.forecaster.timestep,
        frequency=config.frequency,
        num_members_in_sequence=num_members_in_sequence,
    )

    # Get outputs and required_variables of each decoder
    if hasattr(config.checkpoints, "interpolator"):
        leadtimes = get_all_leadtimes(
            config.checkpoints.forecaster.leadtimes,
            config.checkpoints.forecaster.timestep_seconds,
            config.checkpoints.interpolator.leadtimes,
            config.checkpoints.intepoltor.timestep_seconds,
        )
    else:
        leadtimes = get_all_leadtimes(
            config.checkpoints.forecaster.leadtimes,
            config.checkpoints.forecaster.timestep_seconds,
        )

    decoder_outputs = bris.routes.get(
        config["routing"],
        leadtimes,
        num_members,
        datamodule,
        checkpoints,
        config.workdir,
    )
    required_variables = bris.routes.get_required_variables_all_checkpoints(
        config["routing"], checkpoints
    )
    writer = CustomWriter(decoder_outputs, write_interval="batch")

    # Set hydra defaults
    config.defaults = [
        {"override hydra/job_logging": "none"},  # disable config parsing logs
        {"override hydra/hydra_logging": "none"},  # disable config parsing logs
        "_self_",
    ]

    # Forecaster must know about what leadtimes to output
    model = instantiate(
        config.model,
        checkpoints=checkpoints,
        hardware_config=config.hardware,
        datamodule=datamodule,
        forecast_length=config.checkpoints.forecaster.leadtimes,
        required_variables=required_variables,
        release_cache=config.release_cache,
        num_members_in_parallel=num_members_in_parallel,
    )

    callbacks = [writer]

    inference = Inference(
        config=config,
        model=model,
        callbacks=callbacks,
        datamodule=datamodule,
        num_gpus_per_ensemble=num_gpus_per_ensemble,
    )
    inference.run()

    # Finalize all output, so they can flush to disk if needed
    is_main_thread = ("SLURM_PROCID" not in os.environ) or (
        os.environ["SLURM_PROCID"] == "0"
    )
    if is_main_thread:
        for decoder_output in decoder_outputs:
            for output in decoder_output["outputs"]:
                output.finalize()

        print("Model run completed. 🤖")


if __name__ == "__main__":
    main()
