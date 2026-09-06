import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from bris.outputs import AggregateVerif
from bris.predict_metadata import PredictMetadata
from bris.sources import Verif as VerifInput


@pytest.fixture
def setup():
    stuff = 1
    yield stuff


def test_1():
    filename = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input_with_units.nc"
    )
    sources = [VerifInput(filename)]

    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.arange(50, 70)
    lons = np.arange(5, 15)
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 2

    field_shape = [len(lats), len(lons)]

    lats, lons = np.meshgrid(lats, lons)
    lats = lats.flatten()
    lons = lons.flatten()

    with tempfile.TemporaryDirectory() as temp_dir:
        ofilename = os.path.join(temp_dir, "otest.nc")
        workdir = os.path.join(temp_dir, "verif_workdir")
        frt = 1672552800
        for altitudes in [np.arange(len(lats)), None]:
            pm = PredictMetadata(
                variables, lats, lons, altitudes, leadtimes, num_members, field_shape
            )
            elev_gradient = None
            for max_distance in [None, 100000]:
                output = AggregateVerif(
                    predict_metadata=pm,
                    workdir=workdir,
                    filename=ofilename,
                    variable="2t",
                    obs_sources=sources,
                    units="K",
                    elev_gradient=elev_gradient,
                    max_distance=max_distance,
                )

                times = frt + leadtimes
                for member in range(num_members):
                    pred = np.random.rand(*pm.shape)
                    output.add_forecast(times, member, pred)

                output.finalize()
                check_expected_variable(ofilename)

        altitudes = np.arange(len(lats))
        pm = PredictMetadata(
            variables, lats, lons, altitudes, leadtimes, num_members, field_shape
        )
        elev_gradient = 0
        for max_distance in [None, 100000]:
            output = AggregateVerif(
                predict_metadata=pm,
                workdir=workdir,
                filename=ofilename,
                variable="2t",
                obs_sources=sources,
                units="C",
                elev_gradient=elev_gradient,
                max_distance=max_distance,
            )

            times = frt + leadtimes
            for member in range(num_members):
                pred = np.random.rand(*pm.shape)
                output.add_forecast(times, member, pred)

            output.finalize()
            check_expected_variable(ofilename)


def check_expected_variable(filename):
    expected_variables = [
        "fcst",
        "obs",
        "rmse",
        "bias",
    ]

    with xr.open_dataset(filename) as file:
        for variable in expected_variables:
            assert variable in file, variable


if __name__ == "__main__":
    test_1()
