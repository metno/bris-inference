from collections import defaultdict
import numpy as np

import bris.outputs
from bris import utils
from bris.data.datamodule import DataModule
from bris.predict_metadata import PredictMetadata


def get(routing_config: dict, num_leadtimes: list, num_members: int, data_module: DataModule, run_name: str, workdir: str):
    """Returns outputs for each decoder and domain

    Args:
        routing_config: Dictionary from config file
        num_leadtimes: Number of leadtimes in outptu
        data_module: Data module
        run_name: Name of this run used by outputs to set filenames
    Returns:
        list of dicts:
            decoder_index (int)
            start_gridpoint (int)
            end_gridpoint (int)
            outputs (list)
    """
    ret = list()
    required_variables = get_required_variables(routing_config)
    for config in routing_config:
        decoder_index = config["decoder_index"]
        domain_index = config["domain"]

        curr_grids = data_module.grids[decoder_index]
        if isinstance(curr_grids, int): #ugly fix
            curr_grids = (curr_grids,)

        if domain_index == 0:
            start_gridpoint = 0
            end_gridpoint = curr_grids[domain_index]
        else:
            start_gridpoint = np.sum(curr_grids[0:domain_index])
            end_gridpoint = start_gridpoint + curr_grids[domain_index]

        outputs = list()
        for oc in config["outputs"]:
            # TODO: Get this from data_module
            variables = ["u_800", "u_600", "2t", "v_500", "10u"]
            lats = [1, 2]
            lons = [2, 3]
            field_shape = None

            pm = PredictMetadata(required_variables[decoder_index], lats, lons, num_leadtimes,
                    num_members, field_shape)

            for output_type, args in oc.items():
                if "filename" in args:
                    args["filename"] = expand_run_name(args["filename"], run_name)

                curr_workdir = utils.get_workdir(workdir)
                output = bris.outputs.instantiate(output_type, pm, curr_workdir, args)
                outputs += [output]

        ret += [
            dict(
                decoder_index=decoder_index,
                start_gridpoint=start_gridpoint,
                end_gridpoint=end_gridpoint,
                required_variables=required_variables,
                outputs=outputs,
            )
        ]

    return ret

def get_required_variables(routing_config: dict):
    """Returns a list of required variables for each decoder"""
    required_variables = defaultdict(list)
    for rc in routing_config:
        l = list()
        for oc in rc["outputs"]:
            for output_type, args in oc.items():
                l += bris.outputs.get_required_variables(output_type, args)
        required_variables[rc["decoder_index"]] += l
    for k,v in required_variables.items():
        required_variables[k] = sorted(list(set(v)))
    return required_variables

def expand_run_name(string, run_name):
    return string.replace("%R", run_name)


def expand_variable(string, variable):
    return string.replace("%V", variable)
