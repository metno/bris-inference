import logging

from argparse import ArgumentParser
from hydra.utils import instantiate

import bris.utils
import bris.routes

from .checkpoint import Checkpoint
from bris.data.datamodule import DataModule
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
    checkpoint = Checkpoint(config.checkpoint_path)
    if hasattr(config.model, "graph"):
        LOGGER.info("Update graph is enabled. Proceeding to change internal graph")
        checkpoint.update_graph(
            config.model.graph
        )  # Pass in a new graph if needed

    datamodule = DataModule(
        config=config,
        checkpoint_object=checkpoint,
    )

    # Assemble outputs
    run_name = "legendary_gnome"
    workdir = "testdir"
    # TODO: Figure out what the leadtimes are based on the config
    # TODO: Get this from the config
    num_members = 1

    # Get outputs and required_variables of each decoder
    decoder_outputs = bris.routes.get(config["routing"], config.leadtimes, num_members, datamodule, run_name, workdir) #get num_leadtimes from config.leadtimes
#    decoder_variables = bris.routes.get_required_variables(config["routing"])

    writer = CustomWriter(decoder_outputs, write_interval="batch")

    # Forecaster must know about what leadtimes to output
    #model = BrisPredictor(config, model, metadata, data_reader, decoder_variables)
    model = instantiate(config.model, 
                        checkpoint = checkpoint,
                        data_reader = datamodule.data_reader,
                        forecast_length = config.leadtimes,
                        select_indices = [3,4] #TODO: fix
    )    

    callbacks = list()
    callbacks += [writer]

    inference = Inference(config=config,
                          model=model,
                          callbacks=callbacks,
                          checkpoint=checkpoint,
                          datamodule=datamodule,
                          )
    inference.run()

    # Finalize all output, so they can flush to disk if needed
    # TODO: Only do this on rank 0 (maybe this is already the case at this stage of the code?
    for decoder_output in decoder_outputs:
        for output in decoder_output:
            output.finalize()

    print("Hello world")

if __name__ == "__main__":
    main()
