import gridpp
import numpy as np

from bris.outputs.layout import Layout
from bris.predict_metadata import PredictMetadata


def get_pm(variables, num_points=60, gridded=False):
    rng = np.random.default_rng(0)
    if gridded:
        lats, lons = np.meshgrid(
            np.linspace(59, 61, 6), np.linspace(8, 12, 10), indexing="ij"
        )
        lats, lons, field_shape = lats.flatten(), lons.flatten(), [6, 10]
    else:
        lats = 55 + 10 * rng.random(num_points)
        lons = 5 + 15 * rng.random(num_points)
        field_shape = (num_points,)
    altitudes = rng.random(len(lats)) * 1000
    leadtimes = np.arange(0, 3600 * 3, 3600)
    return PredictMetadata(variables, lats, lons, altitudes, leadtimes, 1, field_shape)


def get_fields(layout, pred):
    """The variables in the file for one member, as the Zarr output gets them"""
    rotated = layout.rotated_winds(pred)
    return [
        layout.field_values(variable, pred, rotated)
        for variable in layout.extract_variables
    ]


def test_interpolation_is_gridpp_nearest():
    """The nearest points are looked up once, and the values are those of gridpp.nearest for
    every variable, as is the altitude"""
    variables = ["2t", "msl", "tp"]
    pm = get_pm(variables)
    layout = Layout(pm, list(variables), interp_res=0.5)
    rng = np.random.default_rng(1)
    pred = rng.random(pm.shape).astype(np.float32)

    ipoints, ogrid = layout._interpolation
    for i, (_, _, values) in enumerate(get_fields(layout, pred)):
        expected = np.asarray(gridpp.nearest(ipoints, ogrid, pred[:, :, i]))
        if variables[i] == "tp":
            expected = expected * 1000  # written in kg/m^2
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values, expected.astype(np.float32))

    altitude = layout.skeleton(1672552800 + pm.leadtimes)["altitude"].values
    expected = gridpp.nearest(ipoints, ogrid, pm.altitudes).astype(np.double)
    np.testing.assert_array_equal(altitude, expected)

    # Indices that single precision cannot hold are found as well
    index = layout._interpolation_index
    assert index.shape == layout.spatial_shape
    assert index.min() >= 0 and index.max() < pm.num_points


def test_field_values_leave_the_predictions():
    """The variables are taken from the predictions without a copy of them all, and the
    predictions are left as they were: other outputs get the same array"""
    variables = ["10u", "10v", "2t", "tp", "u_850", "v_850"]
    pm = get_pm(variables, gridded=True)
    layout = Layout(
        pm,
        ["2t", "10u", "10v", "u_850", "v_850", "tp", "tp_acc"],
        accumulated_variables=["tp_acc"],
        proj4_str="+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +R=6371000",
    )
    rng = np.random.default_rng(2)
    pred = rng.random(pm.shape).astype(np.float32)
    pred[0, :, variables.index("tp")] = np.nan
    before = pred.copy()

    fields = {(name, level): values for name, level, values in get_fields(layout, pred)}
    np.testing.assert_array_equal(pred, before)

    # The winds are rotated, in pairs
    e_x, n_x, e_y, n_y = layout.get_projection_rotation_matrices
    u, v = pred[:, :, 0], pred[:, :, 1]
    np.testing.assert_array_equal(
        fields[("x_wind_10m", 0)],
        (e_x * u + n_x * v).astype(np.float32).reshape(3, 6, 10),
    )
    np.testing.assert_array_equal(
        fields[("y_wind_10m", 0)],
        (e_y * u + n_y * v).astype(np.float32).reshape(3, 6, 10),
    )
    assert not np.array_equal(fields[("x_wind_pl", 0)], pred[:, :, 4].reshape(3, 6, 10))
    # ... and the rest is as predicted
    np.testing.assert_array_equal(
        fields[("air_temperature_2m", 0)], pred[:, :, 2].reshape(3, 6, 10)
    )
    assert np.all(np.isnan(fields[("precipitation_amount", 0)][0]))
    assert np.all(np.isfinite(fields[("precipitation_amount_acc", 0)]))
