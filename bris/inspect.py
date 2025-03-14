import logging
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta
import json

import numpy as np
from anemoi.utils.dates import frequency_to_seconds
from hydra.utils import instantiate

import bris.routes
import bris.utils
from bris.data.datamodule import DataModule

from .checkpoint import Checkpoint
from .inference import Inference

LOGGER = logging.getLogger(__name__)


def inspect():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "-c", type=str, dest="checkpoint_path", required=True, help="Path to checkpoint"
    )
    args, _ = parser.parse_known_args()

    # Load checkpoint
    checkpoint = Checkpoint(args.checkpoint_path)

    # Print data
    print("Checkpoint version", checkpoint.metadata.version)
    print("checkpoint run_id", checkpoint.metadata.run_id)
    print("checkpoint timestamp", checkpoint.metadata.timestamp)
    print("checkpoint multistep", checkpoint.multistep)
    print("checkpoint variables", json.dumps(checkpoint.index_to_name, indent=4))

    # print("Checkpoint metadata", checkpoint.metadata)


if __name__ == "__main__":
    inspect()
