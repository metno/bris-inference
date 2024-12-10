import argparse
import yaml

from .inference import Inference
from .checkpoint import Checkpoint
from .datamodule import DataModule
from .writer import CustomWriter


def main():
    parser = argparse.ArgumentParser()

    # TODO: Come up with argument list
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-c", type=str, dest="checkpoint_path", required=True)
    parser.add_argument("-sd", type=int, dest="start_date")
    parser.add_argument("-ed", type=int, dest="end_date")
    parser.add_argument("-f", type=int, dest="frequency")

    args = parser.parse_args()

    config = load_config(args.config)

    # Load checkpoint, and patch it if needed
    checkpoint = Checkpoint(args.checkpoint_path)
    checkpoint.update_paths(config.paths)
    checkpoint.update_graph() # Pass in a new graph if needed

    datamodule = DataModule(checkpoint.graph, checkpoint.dataset_config)

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

        for o in c:
            output = output.instantiate(o, pm)
            outputs[name]["outputs"] += [output]

    writer = CustomWriter(outputs, write_interval="batch")

    # Forecaster must know about what leadtimes to output
    model = BrisForecaster()

    callbacks = list()
    callbacks += [write]

    inference = Inference(model, datamodule, callbacks)
    inference.run()

    print("Hello world")


def load_config(filename):
    with open(filename) as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
