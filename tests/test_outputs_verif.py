import os
import tempfile
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
    filename = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input_with_units.nc"
    )
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
                output = Verif(
                    predict_metadata=pm,
                    workdir=workdir,
                    filename=ofilename,
                    variable="2t",
                    obs_sources=sources,
                    units="K",
                    thresholds=thresholds,
                    quantile_levels=quantile_levels,
                    elev_gradient=elev_gradient,
                    max_distance=max_distance,
                )

                times = frt + leadtimes
                for member in range(num_members):
                    pred = np.random.rand(*pm.shape)
                    output.add_forecast(times, member, pred)

                output.finalize()

        altitudes = np.arange(len(lats))
        pm = PredictMetadata(
            variables, lats, lons, altitudes, leadtimes, num_members, field_shape
        )
        elev_gradient = 0
        for max_distance in [None, 100000]:
            output = Verif(
                predict_metadata=pm,
                workdir=workdir,
                filename=ofilename,
                variable="2t",
                obs_sources=sources,
                units="C",
                thresholds=thresholds,
                quantile_levels=quantile_levels,
                elev_gradient=elev_gradient,
                max_distance=max_distance,
            )

            times = frt + leadtimes
            for member in range(num_members):
                pred = np.random.rand(*pm.shape)
                output.add_forecast(times, member, pred)

            output.finalize()


def test_2():
    """Test verif output with logits and probability variables."""
    filename = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input_logits.nc"
    )
    sources = [VerifInput(filename)]

    variables = ["te"]
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

    with tempfile.TemporaryDirectory() as temp_dir:
        ofilename = os.path.join(temp_dir, "otest.nc")
        workdir = os.path.join(temp_dir, "verif_workdir")
        altitudes = np.arange(len(lats))
        pm = PredictMetadata(
            variables, lats, lons, altitudes, leadtimes, num_members, field_shape
        )
        elev_gradient = 0
        # Test for logits:
        for max_distance in [None, 100000]:
            output = Verif(
                predict_metadata=pm,
                workdir=workdir,
                filename=ofilename,
                variable="te",
                variable_type="logit",
                obs_sources=sources,
                units="probability",
                thresholds=thresholds,
                quantile_levels=quantile_levels,
                elev_gradient=elev_gradient,
                max_distance=max_distance,
            )

            output.finalize()

        # Test for threshold_probability:
        for max_distance in [None, 100000]:
            output = Verif(
                predict_metadata=pm,
                workdir=workdir,
                filename=ofilename,
                variable="te",
                variable_type="threshold_probability",
                obs_sources=sources,
                units="probability",
                thresholds=thresholds,
                quantile_levels=quantile_levels,
                elev_gradient=elev_gradient,
                max_distance=max_distance,
            )

            output.finalize()


if __name__ == "__main__":
    test_1()
    test_2()
