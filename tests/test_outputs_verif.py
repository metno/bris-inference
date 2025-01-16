import os

import numpy as np
import pytest
from bris.outputs import Verif
from bris.predict_metadata import PredictMetadata
from bris.sources import Verif as VerifInput


@pytest.fixture
def setup():
    stuff = 1
    yield stuff


def test_1():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input.nc"
    sources = [VerifInput(filename)]

    variables = ["u_800", "u_600", "2t", "v_500", "10u"]
    lats = np.arange(50, 70)
    lons = np.arange(5, 15)
    leadtimes = np.arange(0, 3600 * 4, 3600)
    num_members = 2
    thresholds = [0.2, 0.5]
    quantile_levels = [0.1, 0.9]

    field_shape = [len(lats), len(lons)]

    lats, lons = np.meshgrid(lats, lons)
    lats = lats.flatten()
    lons = lons.flatten()
    altitudes = np.arange(len(lats))

    pm = PredictMetadata(variables, lats, lons, altitudes, leadtimes, num_members, field_shape)
    ofilename = "otest.nc"
    workdir = "verif_workdir"
    frt = 1672552800
    for elev_gradient in [None, 0]:
        output = Verif(
            pm,
            workdir,
            ofilename,
            "2t",
            sources,
            "K",
            thresholds=thresholds,
            quantile_levels=quantile_levels,
            elev_gradient=elev_gradient,
        )

        times = frt + leadtimes
        for member in range(num_members):
            pred = np.random.rand(*pm.shape)
            output.add_forecast(times, member, pred)

        output.finalize()


if __name__ == "__main__":
    test_1()
