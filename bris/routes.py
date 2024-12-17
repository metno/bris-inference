import numpy as np

import bris.output
from bris import utils
from bris.datamodule import DataModule
from bris.predict_metadata import PredictMetadata


def get(routing_config: dict, data_module: DataModule, run_name: str, workdir: str):
    """
    Args:
        routing_config: Dictionary from config file
        data_module: Data module
        run_name: Name of this run used by outputs to set filenames
    Returns:
        list of dicts:
            decoder_index:
            start_gridpoint:
            end_gridpoint)
            outputs (list): List of outputs
    """
    ret = list()
    for config in routing_config:
        decoder_index = config["decoder_index"]
        domain_index = config["domain"]

        # TODO: Get these from data_module
        start_gridpoint = 0
        end_gridpoint = 10
        key = (decoder_index, start_gridpoint, end_gridpoint)

        # TODO: Get this from data_module
        variables = ["u_800", "u_600", "2t", "v_500", "10u"]
        lats = [1, 2]
        lons = [2, 3]
        leadtimes = [0, 6, 12, 18]
        num_members = 2

        pm = PredictMetadata(variables, lats, lons, leadtimes, num_members)

        outputs = list()
        for output_type, output_config in config["outputs"].items():
            args = output_config
            if "filename" in args:
                args["filename"] = expand_run_name(args["filename"], run_name)

            curr_workdir = utils.get_workdir(workdir)
            output = bris.output.instantiate(output_type, pm, curr_workdir, args)

        ret += [
            dict(
                decoder_index=decoder_index,
                start_gridpoint=start_gridpoint,
                end_gridpoint=end_gridpoint,
                outputs=outputs,
            )
        ]

    return ret


def expand_run_name(string, run_name):
    return string.replace("%R", run_name)


def expand_variable(string, variable):
    return string.replace("%V", variable)
