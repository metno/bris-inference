import os
import time
from concurrent.futures import Future
from datetime import datetime, timedelta

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


def main(arg_list: list[str] | None = None):
    t0 = time.perf_counter()
    args = parse_args(arg_list)
    config = create_config(args["config"], args)
    requested_forecast_start_date = config.get("start_date", None)
    requested_forecast_end_date = config.get("end_date", None)

    setup_logging(config)
    LOGGER.info("Bris startup: config loaded in %.1fs.", time.perf_counter() - t0)

    LOGGER.info("Importing runtime modules.")
    t_import = time.perf_counter()
    t_step = time.perf_counter()
    from anemoi.utils.dates import frequency_to_seconds
    LOGGER.info("Imported anemoi.utils.dates in %.1fs.", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    from hydra.utils import instantiate
    LOGGER.info("Imported hydra.utils in %.1fs.", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    import bris.routes
    LOGGER.info("Imported bris.routes in %.1fs.", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    from bris.data.datamodule import DataModule
    LOGGER.info("Imported bris.data.datamodule in %.1fs.", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    from .checkpoint import Checkpoint
    LOGGER.info("Imported bris.checkpoint in %.1fs.", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    from .inference import Inference
    LOGGER.info("Imported bris.inference in %.1fs.", time.perf_counter() - t_step)

    t_step = time.perf_counter()
    from .writer import CustomWriter
    LOGGER.info("Imported bris.writer in %.1fs.", time.perf_counter() - t_step)

    LOGGER.info("Imported runtime modules in %.1fs.", time.perf_counter() - t_import)

    models = list(config.checkpoints.keys())

    checkpoints = {}
    for model in models:
        LOGGER.info("Loading %s checkpoint from %s.", model, config.checkpoints[model].checkpoint_path)
        t_checkpoint = time.perf_counter()
        checkpoints[model] = Checkpoint(
            config.checkpoints[model].checkpoint_path,
            getattr(config.checkpoints[model], "switch_graph", None),
        )
        LOGGER.info("Loaded %s checkpoint in %.1fs.", model, time.perf_counter() - t_checkpoint)

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

    history_steps = multistep - 1
    dataset_input_time_indices = getattr(
        getattr(checkpoints["forecaster"].model, "model", None),
        "dataset_input_time_indices",
        None,
    )
    sparse_forecaster_history = False
    if isinstance(dataset_input_time_indices, dict):
        output_dataset_names = [
            route["decoder_name"]
            for route in config.get("routing", [])
            if route.get("decoder_name") in dataset_input_time_indices
        ]
        if output_dataset_names:
            sparse_forecaster_history = True
            history_steps = max(
                max(int(offset) for offset in dataset_input_time_indices[dataset_name])
                for dataset_name in output_dataset_names
            )

    if sparse_forecaster_history:
        if requested_forecast_start_date is None:
            requested_forecast_start_date = requested_forecast_end_date
        if requested_forecast_end_date is None:
            requested_forecast_end_date = requested_forecast_start_date
        config.dataloader.datamodule.forecast_start_date = requested_forecast_start_date
        config.dataloader.datamodule.forecast_end_date = requested_forecast_end_date

    # If no start_date given, calculate as end_date-((multistep-1)*timestep)
    if "start_date" not in config or config.start_date is None:
        config.start_date = datetime.strftime(
            datetime.strptime(config.end_date, "%Y-%m-%dT%H:%M:%S")
            - timedelta(
                seconds=history_steps * config.checkpoints.forecaster.timestep_seconds
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
                seconds=history_steps * config.checkpoints.forecaster.timestep_seconds
            ),
            "%Y-%m-%dT%H:%M:%S",
        )
    # Get dataset config with backwards comapatibility for single dataset config setup
    config.datasets = get_dataset_config(config)

    LOGGER.info("Building datamodule.")
    t_datamodule = time.perf_counter()
    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoints["forecaster"],
        timestep=config.checkpoints.forecaster.timestep,
        frequency=config.frequency,
        num_members_in_sequence=num_members_in_sequence,
    )
    LOGGER.info("Built datamodule in %.1fs.", time.perf_counter() - t_datamodule)

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

    LOGGER.info("Resolving routing and output writers.")
    t_routes = time.perf_counter()
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
    LOGGER.info("Resolved routing and output writers in %.1fs.", time.perf_counter() - t_routes)

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

    LOGGER.info("Instantiating predictor model.")
    t_model = time.perf_counter()
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
    LOGGER.info("Instantiated predictor model in %.1fs.", time.perf_counter() - t_model)

    callbacks = [writer]

    LOGGER.info("Starting inference run.")
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
