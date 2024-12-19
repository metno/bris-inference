from collections import defaultdict
import numpy as np

import bris.outputs
from bris import utils
from bris.data.datamodule import DataModule
from bris.predict_metadata import PredictMetadata


def get(routing_config: dict, num_leadtimes: int, num_members: int, data_module: DataModule, run_name: str, workdir: str):
    """Returns outputs for each decoder and domain

    Args:
        routing_config: Dictionary from config file
        num_leadtimes: Number of leadtimes that the model will produce
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
            lats = data_module.latitudes[decoder_index][start_gridpoint:end_gridpoint]
            lons = data_module.longitudes[decoder_index][start_gridpoint:end_gridpoint]
            field_shape = data_module.field_shape[decoder_index][domain_index]

            curr_required_variables = required_variables[decoder_index]
            if curr_required_variables is None:
                # Convert None to all available variables
                # TODO: This is not tested yet
                name_to_index = data_module.data_reader.name_to_index[decoder_index]
                available_names = name_to_index.keys()
                curr_required_variables = [i for i in range(len(available_names))]
                for name, index in name_to_index.items():
                    curr_required_variables[index] = name

            pm = PredictMetadata(curr_required_variables, lats, lons, num_leadtimes,
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
        if None in v:
            required_variables[k] = None
        else:
            required_variables[k] = sorted(list(set(v)))

    return required_variables

def expand_run_name(string, run_name):
    return string.replace("%R", run_name)


def expand_variable(string, variable):
    return string.replace("%V", variable)
