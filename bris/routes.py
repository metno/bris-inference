from collections import defaultdict

import numpy as np

import bris.outputs
from bris import utils
from bris.checkpoint import Checkpoint
from bris.data.datamodule import DataModule
from bris.predict_metadata import PredictMetadata


def get(
    routing_config: dict,
    leadtimes: list,
    num_members: int,
    data_module: DataModule,
    checkpoints: dict[str, Checkpoint],
    workdir: str,
):
    """Returns outputs for each decoder and domain

    This is used by the CustomWriter

    Args:
        routing_config: Dictionary from config file
        leadtimes: Which leadtimes that the model will produce
        data_module: Data module
        checkpoints: Dictionary with checkpoints
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

    ret = []
    required_variables = get_required_variables_all_checkpoints(
        routing_config, checkpoints
    )

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

        outputs = []
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
            {
                "decoder_index": decoder_index,
                "start_gridpoint": start_gridpoint,
                "end_gridpoint": end_gridpoint,
                "outputs": outputs,
            }
        ]

    return ret


def get_required_variables_all_checkpoints(
    routing_config: dict, checkpoints: dict[str, Checkpoint]
) -> dict[int, list[str]]:
    """Returns a list of required variables for each decoder from all checkpoints. Will return the union if one checkpoint has more outputs than the others"""

    required_variables_per_model = {
        model: get_required_variables(routing_config, checkpoint)
        for model, checkpoint in checkpoints.items()
    }
    required_variables_full = defaultdict(set)
    for _, _required_variables in required_variables_per_model.items():
        for key, variable_list in _required_variables.items():
            required_variables_full[key].update(variable_list)

    required_variables = {
        key: list(values) for key, values in required_variables_full.items()
    }
    return required_variables


def get_required_variables(
    routing_config: dict, checkpoint_object: Checkpoint
) -> dict[int, list[str]]:
    """Returns a list of required variables for each decoder"""
    required_variables: dict[int, list[str]] = defaultdict(list)
    for rc in routing_config:
        var_list = []
        for oc in rc["outputs"]:
            for output_type, args in oc.items():
                var_list += bris.outputs.get_required_variables(output_type, args)
        required_variables[rc["decoder_index"]] += var_list

    for decoder_index, v in required_variables.items():
        if None in v:
            name_to_index = checkpoint_object.model_output_name_to_index[decoder_index]
            required_variables[decoder_index] = sorted(list(set(name_to_index.keys())))
        else:
            required_variables[decoder_index] = sorted(list(set(v)))

    return required_variables


def expand_variable(string: str, variable: str) -> str:
    return string.replace("%V", variable)
