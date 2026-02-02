import os
from collections import defaultdict
from typing import Any, Literal

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
            decoder_name (str)
            start_gridpoint (int)
            end_gridpoint (int)
            outputs (list)
        dicts:
            decoder_name -> variable_indices

    """

    ret = []
    required_variables = get_required_variables_all_checkpoints(
        routing_config, checkpoints
    )
    count = 0
    for config in routing_config:
        decoder_name = config["decoder_name"]
        domain_index = config.get("domain_index", None)

        curr_grids = data_module.grids[decoder_name]

        if domain_index is None:
            start_gridpoint = 0
            end_gridpoint = np.sum(curr_grids)
        elif domain_index == 0:
            start_gridpoint = 0
            end_gridpoint = curr_grids[domain_index]
        else:
            start_gridpoint = np.sum(curr_grids[0:domain_index])
            end_gridpoint = start_gridpoint + curr_grids[domain_index]

        outputs = []
        for oc in config["outputs"]:
            # If outputing netcdf, add global_attributes with checkpoint name
            if "netcdf" in oc:
                oc = add_checkpoint_name_to_attrs(oc, checkpoints)

            lats = data_module.latitudes[decoder_name][start_gridpoint:end_gridpoint]
            lons = data_module.longitudes[decoder_name][start_gridpoint:end_gridpoint]
            altitudes = None
            if data_module.altitudes[decoder_name] is not None:
                altitudes = data_module.altitudes[decoder_name][
                    start_gridpoint:end_gridpoint
                ]

            field_shape = data_module.field_shape[decoder_name][domain_index]

            curr_required_variables = required_variables[decoder_name]

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
                "decoder_name": decoder_name,
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
        key: sorted(list(values)) for key, values in required_variables_full.items()
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
        required_variables[rc["decoder_name"]] += var_list

    for decoder_name, v in required_variables.items():
        if None in v:
            model_output = checkpoint_object.data_indices[decoder_name].model.output.includes
            required_variables[decoder_name] = sorted(model_output)
        else:
            required_variables[decoder_name] = sorted(list(set(v)))

    return required_variables


def expand_variable(string: str, variable: str) -> str:
    return string.replace("%V", variable)


def add_checkpoint_name_to_attrs(
    oc: dict[Literal["netcdf"], dict[str, Any]], checkpoints: dict[str, Checkpoint]
) -> dict[Literal["netcdf"], dict[str, Any]]:
    """Add checkpoint name"""
    # oc {'netcdf': {'filename_pattern': './tox_test_inference.nc', 'variables': ['2t', '2d']}}
    ckpt_str = "Checkpoints used: "
    if "global_attributes" not in oc["netcdf"]:
        oc["netcdf"]["global_attributes"] = {}
    if "source" in oc["netcdf"]["global_attributes"]:
        ckpt_str = f"{oc['netcdf']['global_attributes']['source']} {ckpt_str}"
    for type, checkpoint in checkpoints.items():
        ckpt_path = os.path.abspath(checkpoint.path)
        ckpt_str += f"{type}:{ckpt_path}, "
    oc["netcdf"]["global_attributes"]["source"] = f"{ckpt_str}"
    return oc
