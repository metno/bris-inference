import os
import time
from concurrent.futures import Future
from datetime import datetime, timedelta

from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate

import bris.routes
from bris.data.datamodule import DataModule

from .checkpoint import Checkpoint
from .inference import Inference
from .utils import (
    LOGGER,
    create_config,
    get_all_leadtimes,
    parse_args,
    set_base_seed,
    set_encoder_decoder_num_chunks,
    setup_logging,
    get_dataset_config,
)
from .writer import CustomWriter


def main(arg_list: list[str] | None = None):
    t0 = time.perf_counter()
    args = parse_args(arg_list)
    config = create_config(args["config"], args)

    setup_logging(config)

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

    # Compute timestep_seconds for each checkpoint
    config.checkpoints.forecaster.timestep = checkpoints[
        "forecaster"
    ].config.data.timestep
    config.checkpoints.forecaster.timestep_seconds = frequency_to_seconds(
        config.checkpoints.forecaster.timestep
    )
    if "interpolator" in checkpoints:
        config.checkpoints.interpolator.timestep_seconds = int(
            config.checkpoints.forecaster.timestep_seconds
            / (
                len(checkpoints["interpolator"].config.training.explicit_times.target)
                + 1
            )
        )

    num_members = config["hardware"].get("num_members", 1)

    # Distribute ensemble members across GPUs, run in sequence if not enough GPUs
    num_gpus = config["hardware"]["num_gpus_per_node"] * config["hardware"]["num_nodes"]
    num_gpus_per_model = config["hardware"].get("num_gpus_per_model", 1)
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
        LOGGER.warning(
            "No start_date given, setting %s based on end_date and timestep.",
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
    # Get dataset config with backwards comapatibility for single dataset config setup
    config.datasets = get_dataset_config(config)

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
            config.checkpoints.interpolator.timestep_seconds,
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

    # List of background write processes
    write_process_list: list[Future] | None = []

    if "background_write" in config and not config["background_write"]:
        write_process_list = None

    max_processes = os.cpu_count() - config["dataloader"].get("num_workers", 1) - 1
    LOGGER.debug(
        f"cpus available {os.cpu_count()}, max writer processes {max_processes}"
    )
    writer = CustomWriter(
        decoder_outputs,
        process_list=write_process_list,
        max_processes=max_processes,
    )

    model = instantiate(
        config.model,
        checkpoints=checkpoints,
        hardware_config=config.hardware,
        datamodule=datamodule,
        checkpoints_config=config.checkpoints,
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

    # Wait for all writer processes to finish
    if write_process_list is not None:
        while len(write_process_list) > 0:
            t2 = time.perf_counter()
            p = write_process_list.pop()
            p.result()
            LOGGER.debug(f"Waited {time.perf_counter() - t2:.1f}s for {p} to complete.")

    # Finalize all outputs, so they can flush to disk if needed
    is_main_thread = ("SLURM_PROCID" not in os.environ) or (
        os.environ["SLURM_PROCID"] == "0"
    )
    if is_main_thread:
        LOGGER.debug("Starting finalizing all outputs.")
        t1 = time.perf_counter()
        for decoder_output in decoder_outputs:
            for output in decoder_output["outputs"]:
                output.finalize()
        LOGGER.debug(f"Finalized all outputs in {time.perf_counter() - t1:.1f}s.")
        LOGGER.info(f"Bris main completed in {time.perf_counter() - t0:.1f}s. 🤖")
    else:
        LOGGER.info(f"Bris instance completed in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
