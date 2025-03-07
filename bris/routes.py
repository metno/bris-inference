from collections import defaultdict

import numpy as np

import bris.outputs
from bris import utils
from bris.data.datamodule import DataModule
from bris.predict_metadata import PredictMetadata


def get(
    routing_config: dict,
    leadtimes: list,
    num_members: int,
    data_module: DataModule,
    workdir: str,
):
    """Returns outputs for each decoder and domain

    This is used by the CustomWriter

    Args:
        routing_config: Dictionary from config file
        leadtimes: Which leadtimes that the model will produce
        data_module: Data module
    Returns:
        list of dicts:
            decoder_index (int)
            domain_index (int)
            start_gridpoint (int)
            end_gridpoint (int)
            outputs (list)
        dicts:
            decoder_index -> variable_indices

    """
    ret = list()
    required_variables = get_required_variables(routing_config, data_module)

    count = 0
    for config in routing_config:
        decoder_index = config["decoder_index"]
        domain_index = config["domain_index"]

        curr_grids = data_module.grids[decoder_index]
        if domain_index == 0:
            start_gridpoint = 0
            end_gridpoint = curr_grids[domain_index]
        else:
            start_gridpoint = np.sum(curr_grids[0:domain_index])
            end_gridpoint = start_gridpoint + curr_grids[domain_index]

        outputs = list()
        for oc in config["outputs"]:
            lats = data_module.latitudes[decoder_index][start_gridpoint:end_gridpoint]
            lons = data_module.longitudes[decoder_index][start_gridpoint:end_gridpoint]
            altitudes = None
            if data_module.altitudes[decoder_index] is not None:
                altitudes = data_module.altitudes[decoder_index][
                    start_gridpoint:end_gridpoint
                ]
            field_shape = data_module.field_shape[decoder_index][domain_index]

            curr_required_variables = required_variables[decoder_index]

            pm = PredictMetadata(
                curr_required_variables,
                lats,
                lons,
                altitudes,
                leadtimes,
                num_members,
                field_shape,
            )

            for output_type, args in oc.items():
                curr_workdir = utils.get_workdir(workdir) + "_" + str(count)
                count += 1
                output = bris.outputs.instantiate(output_type, pm, curr_workdir, args)
                outputs += [output]

        # We don't need to pass out domain_index, since this is only used to get start/end
        # gridpoints and is not used elsewhere in the code
        ret += [
            dict(
                decoder_index=decoder_index,
                start_gridpoint=start_gridpoint,
                end_gridpoint=end_gridpoint,
                outputs=outputs,
            )
        ]

    return ret


def get_variable_indices(routing_config: dict, data_module: DataModule):
    """Returns a list of variable indices for each decoder

    This is used by Model
    """
    required_variables = get_required_variables(routing_config, data_module)

    variable_indices = dict()
    for decoder_index, _r in required_variables.items():
        variable_indices[decoder_index] = list()
        for name in required_variables[decoder_index]:
            index = data_module.name_to_index[decoder_index][name]
            variable_indices[decoder_index] += [index]

    return variable_indices


def get_required_variables(routing_config: dict, data_module: DataModule):
    """Returns a list of required variables for each decoder"""
    required_variables = defaultdict(list)
    for rc in routing_config:
        var_list = []
        for oc in rc["outputs"]:
            for output_type, args in oc.items():
                var_list += bris.outputs.get_required_variables(output_type, args)
        required_variables[rc["decoder_index"]] += var_list

    for decoder_index, v in required_variables.items():
        if None in v:
            name_to_index = data_module.name_to_index[decoder_index]

            # Pre-initialize list
            required_variables[decoder_index] = list(name_to_index.keys())

            for name, index in name_to_index.items():
                assert index < len(name_to_index)

                required_variables[decoder_index][index] = name
        else:
            required_variables[decoder_index] = sorted(list(set(v)))

    return required_variables


def expand_variable(string, variable):
    return string.replace("%V", variable)
