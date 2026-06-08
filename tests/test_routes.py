import os

from anemoi.utils.config import DotDict

import bris.routes
from bris.checkpoint import Checkpoint


class FakeDataModule:
    def __init__(self):
        self.field_shape = {
            "data1": [None, [1, 2]],
            "data2": [None],
        }

    @property
    def grids(self):
        ret = dict()
        ret["data1"] = [1, 2]
        ret["data2"] = [1]
        return ret

    @property
    def latitudes(self):
        return {"data1": [1, 1, 2], "data2": [1]}

    @property
    def longitudes(self):
        return self.latitudes

    @property
    def altitudes(self):
        return {
            "data1": [0, 100, 200],
            "data2": [300],
        }

    @property
    def name_to_index(self):
        return {
            "data1": {"2t": 0, "10u": 1, "10v": 2},
            "data2": {"100v": 0, "100u": 1},
        }


class FakeCheckpointObject:
    path = "FakeCheckpointObject_path"

    @property
    def data_indices(self):
        data_indices = {}
        data_indices["data1"] = DotDict(
            {
                "model": {
                    "output": {
                        "includes": ["2t", "10u", "10v"],
                    }
                }
            }
        )
        data_indices["data2"] = DotDict(
            {
                "model": {
                    "output": {
                        "includes": ["100v", "100u"],
                    }
                }
            }
        )

        return data_indices


#    @property
#    def model_output_name_to_index(self):
#        return [{"2t": 0, "10u": 1, "10v": 2}, {"100v": 0, "100u": 1}]


def test_get():
    config = list()
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input.nc"
    config += [
        {
            "decoder_name": "data1",
            "domain_index": 0,
            "outputs": [
                {
                    "verif": {
                        "filename": "nordic/2t/%R.nc",
                        "variable": "2t",
                        "units": "C",
                        "thresholds": [0, 10, 20],
                        "quantile_levels": [0.1, 0.9],
                        "obs_sources": [{"verif": {"filename": filename}}],
                    }
                }
            ],
        },
        {
            "decoder_name": "data1",
            "domain_index": 1,
            "outputs": [
                {
                    "netcdf": {
                        "filename_pattern": "%Y%m%d.nc",
                    }
                }
            ],
        },
        {
            "decoder_name": "data2",
            "domain_index": 0,
            "outputs": [
                {
                    "netcdf": {
                        "filename_pattern": "%Y%m%d.nc",
                        "variables": ["100u"],
                    }
                }
            ],
        },
    ]
    data_module = FakeDataModule()
    checkpoint_object = FakeCheckpointObject()
    checkpoints = {"forecaster": checkpoint_object}
    workdir = "testdir"
    leadtimes = range(66)
    num_members = 2

    required_variables = bris.routes.get_required_variables(config, checkpoint_object)
    correct_variables = {"data1": ["2t", "10u", "10v"], "data2": ["100u"]}
    for key in required_variables:
        assert set(required_variables[key]) == set(correct_variables[key])

    _ = bris.routes.get(
        config, len(leadtimes), num_members, data_module, checkpoints, workdir
    )


def test_add_checkpoint_name_to_attrs():
    test_oc = {
        "netcdf": {
            "filename_pattern": "./tox_test_inference.nc",
            "variables": ["2t", "2d"],
        }
    }
    test_ckpts = {"testchk": Checkpoint("./tests/files/checkpoint_single.ckpt")}
    new_oc = bris.routes.add_checkpoint_name_to_attrs(test_oc, test_ckpts)
    assert "testchk" in new_oc["netcdf"]["global_attributes"]["source"]
