import os

import numpy as np

from bris import sources


def test_read():
    # Simple test to see if no errors occur when reading a Verif as input
    for f in ["verif_input", "verif_input_with_units"]:
        filename = os.path.dirname(os.path.abspath(__file__)) + f"/files/{f}.nc"
        source = sources.Verif(filename)
        variable = "test"

        start_time = 1672768800  # 20230103 18:00:00
        end_time = 1672768800 + 6 * 3600  # 20230104 00:00:00

        result = source.get(variable, start_time, end_time, 3600)
        assert len(result.times) == 7

        values = result.get_data(variable, end_time)
        expected = [-7.3, 0.4, -3.6, -8.3, -8.6, -11]
        np.testing.assert_array_almost_equal(values, expected)


def test_read_no_leadtime():
    filename = (
        os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input_no_leadtime.nc"
    )
    source = sources.Verif(filename)
    variable = "test"

    start_time = 1672552800  # 20230101 06:00:00
    end_time = 1672574400  # 20230101 12:00:00

    result = source.get(variable, start_time, end_time, 3600)
    assert len(result.times) == 7

    values = result.get_data(variable, start_time)
    expected = [-2.9, 2.8, -7.9, 2.1, -5.3, -5.4]
    np.testing.assert_array_almost_equal(values, expected)
