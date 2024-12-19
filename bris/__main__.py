import argparse
import logging

from argparse import ArgumentParser
from hydra.utils import instantiate

import bris.utils

from .checkpoint import Checkpoint
<<<<<<< HEAD
from .data.datamodule import DataModule
=======
from bris.data.datamodule import DataModule
>>>>>>> 3656dfa (Parser merge with config)
from .inference import Inference
from .predict_metadata import PredictMetadata
from .writer import CustomWriter
from .utils import create_config


LOGGER = logging.getLogger(__name__)


def main():
    
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", type=str, required=True)   
    
    config = create_config(parser)

    # Load checkpoint, and patch it if needed
    '''
    #checkpoint = Checkpoint(args.checkpoint_path)

    '''
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

    run_name = "legendary_gnome"
    workdir = "testdir"
    # TODO: Figure out what the leadtimes are based on the config
    leadtimes = range(0, 66)
    # TODO: Get this from the config
    num_members = 2

    # Get outputs and required_variables of each decoder
    decoder_outputs = bris.routes.get(config["routing"], leadtimes, num_members, data_module, run_name, workdir)
    decoder_variables = bris.routes.get_required_variables(config["routing"])

    writer = CustomWriter(decoder_outputs, write_interval="batch")

    # Forecaster must know about what leadtimes to output
    #model = BrisPredictor(config, model, metadata, data_reader, decoder_variables)
    model = instantiate(config.model, 
                        checkpoint = checkpoint,
                        data_reader = datamodule.data_reader,
                        forecast_length = config.forecast_length,
                        select_indices = [0,1,2,3] #TODO: fix
    )    

    callbacks = list()
    callbacks += [writer]

    inference = Inference(model, datamodule, callbacks)
    inference.run()

    # Finalize all output, so they can flush to disk if needed
    # TODO: Only do this on rank 0 (maybe this is already the case at this stage of the code?
    for decoder_output in decoder_outputs:
        for output in decoder_output:
            output.finalize()

    print("Hello world")
    '''

if __name__ == "__main__":
    main()
