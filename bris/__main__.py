import argparse
import logging

import omegaconf
import yaml

import bris.output
import bris.utils

from .checkpoint import Checkpoint
from .datamodule import DataModule
from .forecaster import BrisForecaster
from .inference import Inference
from .predict_metadata import PredictMetadata
from .writer import CustomWriter

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # TODO: Come up with argument list
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-c", type=str, dest="checkpoint_path", required=True)
    parser.add_argument("-sd", type=int, dest="start_date", default=None, const=None)
    parser.add_argument("-ed", type=int, dest="end_date", default=None, const=None)

    parser.add_argument(
        "-p",
        type=str,
        dest="paths",
        nargs="*",
        help="List of paths for input data",
        default=None,
        const=None,
    )
    parser.add_argument("-f", type=str, dest="frequency", default=None, const=None)
    parser.add_argument("-s", type=str, dest="timestep", default=None, const=None)
    args = parser.parse_args()

    try:
        config = omegaconf.OmegaConf.load(args.config)
        LOGGER.debug(f"config file from {args.config} is loaded")
    except Exception as e:
        raise e

    # TODO: find a way to use start_date and end_date
    start_date = config.start_date if not args.start_date else args.start_date
    end_date = config.end_date if not args.end_date else args.end_date
    paths = None if not args.paths else args.paths
    # Load checkpoint, and patch it if needed
    checkpoint = Checkpoint(args.checkpoint_path)

    if hasattr(config.model.graph, "path"):
        LOGGER.info("Update graph is enabled. Proceeding to change internal graph")
        checkpoint.update_graph(
            config.model.graph.path
        )  # Pass in a new graph if needed

    # TODO: should frequency and timestep be input args? conf based or both?
    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoint,
        paths=paths,
        frequency=args.frequency,
        timestep=args.timestep,
        graph=checkpoint.graph,
    )
    # Assemble outputs
    outputs = dict()
    for name, c in config["outputs"]:
        assert name in datamodule.grids

        start = datamodule.grids[name]["start"]
        end = datamodule.grids[name]["end"]

        outputs[name] = {"start": start, "end": end, "outputs": list()}

        # Fetch the metadata that describes what predict_step gives
        # TODO: Figure out what the leadtimes are based on the config
        leadtimes = range(0, 66)
        # TODO: Get variables, lats, lons from the data module?
        variables = []
        lats = []
        lons = []
        pm = PredictMetadata(variables, lats, lons, leadtimes)

        for o, args in c.items():
            workdir = bris.utils.get_workdir(args["path"])
            output = bris.output.instantiate(o, pm, workdir, args)
            outputs[name]["outputs"] += [output]

    writer = CustomWriter(outputs, write_interval="batch")

    # Forecaster must know about what leadtimes to output
    model = BrisForecaster()

    callbacks = list()
    callbacks += [writer]

    inference = Inference(model, datamodule, callbacks)
    inference.run()

    # Finalize all output, so they can flush to disk if needed
    # TODO: Only do this on rank 0 (maybe this is already the case at this stage of the code?
    for name, o in outputs.items():
        for output in o["outputs"]:
            output.finalize()

    print("Hello world")


def load_config(filename):
    with open(filename) as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
