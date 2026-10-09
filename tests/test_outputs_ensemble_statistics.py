import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from bris.conventions.variable_list import VariableList
from bris.outputs import EnsembleStatistics
from bris.predict_metadata import PredictMetadata

VARIABLES = ["2t", "10u", "10v", "tp"]
EXTRA_VARIABLES = ["ws"]
NUM_MEMBERS = 4
LEADTIMES = np.arange(0, 3600 * 3, 3600)
FRT = 1672552800


def get_pm(gridded: bool, num_members: int = NUM_MEMBERS) -> PredictMetadata:
    if gridded:
        lats = np.arange(50, 55)
        lons = np.arange(5, 8)
        field_shape = [len(lats), len(lons)]
        lats, lons = np.meshgrid(lats, lons, indexing="ij")
        lats = lats.flatten()
        lons = lons.flatten()
    else:
        lats = np.array([50, 51, 52, 53, 54, 55, 56])
        lons = np.array([5, 6, 7, 8, 9, 10, 11])
        field_shape = (len(lats),)
    return PredictMetadata(
        VARIABLES, lats, lons, None, LEADTIMES, num_members, field_shape
    )


def get_predictions(pm: PredictMetadata, seed: int = 0) -> np.ndarray:
    """Random forecasts with dimensions (member, leadtime, location, variable)"""
    rng = np.random.default_rng(seed)
    pred = rng.random((pm.num_members, *pm.shape)).astype(np.float32)
    pred[..., 0] += 273  # 2t in K
    return pred


def expected_statistics(pred: np.ndarray, ddof: int = 0) -> dict:
    """Expected statistics with dimensions (leadtime, location, variable) including ws"""
    ws = np.sqrt(pred[..., [1]] ** 2 + pred[..., [2]] ** 2)
    pred = np.concatenate([pred, ws], axis=-1)
    return {
        "mean": pred.mean(axis=0),
        "std": pred.std(axis=0, ddof=ddof),
        "min": pred.min(axis=0),
        "max": pred.max(axis=0),
    }


def check_file(filename: str, expected: np.ndarray, pm: PredictMetadata, statistic):
    """Checks that the written statistic matches the expected (leadtime, location, variable)"""
    variable_list = VariableList(VARIABLES + EXTRA_VARIABLES)
    with xr.open_dataset(filename) as ds:
        assert "ensemble_member" not in ds.dims
        for v, variable in enumerate(VARIABLES + EXTRA_VARIABLES):
            ncname = variable_list.get_ncname_from_anemoi_name(variable)
            assert ncname in ds, ncname
            values = ds[ncname].values
            assert ds[ncname].attrs["cell_methods"].startswith("realization: ")
            if pm.is_gridded:
                values = values.reshape(pm.num_leadtimes, -1, pm.num_points)
            else:
                values = values.reshape(pm.num_leadtimes, -1, pm.num_points)
            # Squeeze out the level dimension, if any
            values = values[:, 0, :] if values.shape[1] == 1 else values
            curr = expected[..., v]
            if variable == "tp":
                # Anemoi uses Mg/m^2, output is kg/m^2. Both mean and std are scaled, since
                # the conversion is linear
                curr = curr * 1000
            np.testing.assert_allclose(values, curr, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("gridded", [True, False])
@pytest.mark.parametrize("ddof", [0, 1])
def test_sequential_members(gridded, ddof):
    """Members added one at a time through add_forecast (the non-reduced path)"""
    pm = get_pm(gridded)
    pred = get_predictions(pm)
    statistics = ["mean", "std", "min", "max"]

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "ens_{statistic}_%Y%m%dT%HZ.nc")
        output = EnsembleStatistics(
            pm,
            os.path.join(temp_dir, "workdir"),
            pattern,
            variables=VARIABLES,
            extra_variables=EXTRA_VARIABLES,
            statistics=statistics,
            ddof=ddof,
        )
        times = [np.datetime64(FRT, "s") + lt for lt in LEADTIMES]
        for member in range(NUM_MEMBERS):
            output.add_forecast(times, member, pred[member])
        output.finalize()

        expected = expected_statistics(pred, ddof)
        for statistic in statistics:
            filename = output.writers[statistic].get_filename(FRT)
            assert os.path.exists(filename), filename
            assert f"ens_{statistic}_2023" in filename
            check_file(filename, expected[statistic], pm, statistic)

        # Intermediate is cleaned up
        assert not os.path.exists(os.path.join(temp_dir, "workdir"))


def test_reduced_members():
    """Members reduced in two groups (as when members run both in parallel and in sequence),
    over several forecast reference times"""
    pm = get_pm(gridded=True)
    statistics = ["mean", "std"]
    frts = [FRT, FRT + 6 * 3600]

    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "ens_{statistic}_%Y%m%dT%HZ.nc")
        output = EnsembleStatistics(
            pm,
            os.path.join(temp_dir, "workdir"),
            pattern,
            variables=VARIABLES,
            extra_variables=EXTRA_VARIABLES,
            statistics=statistics,
        )
        preds = {}
        for i, frt in enumerate(frts):
            preds[frt] = get_predictions(pm, seed=i)
            times = [np.datetime64(frt, "s") + lt for lt in LEADTIMES]
            # Two groups of two members, reduced like the writer would do it
            for members in [[0, 1], [2, 3]]:
                contributions = [
                    output.member_contribution(times, m, preds[frt][m]) for m in members
                ]
                reduced = {
                    "sum": sum(c["sum"] for c in contributions),
                    "sum_sq": sum(c["sum_sq"] for c in contributions),
                    "min": np.minimum(*[c["min"] for c in contributions]),
                    "max": np.maximum(*[c["max"] for c in contributions]),
                }
                output.add_reduced_forecast(times, reduced, len(members))
            assert output.intermediate.get_num_members(times[0]) == NUM_MEMBERS

        output.finalize()

        for frt in frts:
            expected = expected_statistics(preds[frt])
            for statistic in statistics:
                filename = output.writers[statistic].get_filename(frt)
                check_file(filename, expected[statistic], pm, statistic)


def test_member_contribution_shapes():
    pm = get_pm(gridded=True)
    pred = get_predictions(pm)
    with tempfile.TemporaryDirectory() as temp_dir:
        output = EnsembleStatistics(
            pm,
            os.path.join(temp_dir, "workdir"),
            os.path.join(temp_dir, "mean.nc"),
            variables=["2t"],
            statistics=["mean"],
        )
        times = [np.datetime64(FRT, "s") + lt for lt in LEADTIMES]
        contributions = output.member_contribution(times, 0, pred[0])
        assert set(contributions.keys()) == set(output.ensemble_reductions.keys())
        for value in contributions.values():
            # Only the requested variables are accumulated
            assert value.shape == (pm.num_leadtimes, pm.num_points, 1)
            assert value.dtype == np.float64
        np.testing.assert_allclose(contributions["sum"][..., 0], pred[0][..., 0])
        np.testing.assert_allclose(
            contributions["sum_sq"][..., 0], pred[0][..., 0].astype(np.float64) ** 2
        )


def test_invalid_config():
    pm = get_pm(gridded=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        workdir = os.path.join(temp_dir, "workdir")
        # Several statistics need the {statistic} token
        with pytest.raises(ValueError):
            EnsembleStatistics(
                pm,
                workdir,
                os.path.join(temp_dir, "out.nc"),
                statistics=["mean", "std"],
            )
        with pytest.raises(ValueError):
            EnsembleStatistics(
                pm, workdir, os.path.join(temp_dir, "out.nc"), statistics=["median"]
            )
        with pytest.raises(ValueError):
            EnsembleStatistics(
                pm, workdir, os.path.join(temp_dir, "out.nc"), variables=["2d"]
            )
        # A single statistic does not need the token
        EnsembleStatistics(
            pm, workdir, os.path.join(temp_dir, "out.nc"), statistics=["mean"]
        )


if __name__ == "__main__":
    _ = pytest.main([__file__])
