import os

import numpy as np
from bris.sources.verif import Verif


def test_read():
    # Simple test to see if no errors occur when reading a Verif as input
    filename = os.path.dirname(os.path.abspath(__file__)) + "/files/verif_input.nc"
    source = Verif(filename)
    variable = "test"

    start_time = 1672552800  # 20230101 06:00:00
    end_time = 1672574400  # 20230101 12:00:00

    result = source.get(variable, start_time, end_time, 3600)
    assert len(result.times) == 7

    values = result.get_data(variable, start_time)
    expected = [-2.9, 2.8, -7.9, 2.1, -5.3, -5.4]
    np.testing.assert_array_almost_equal(values, expected)


if __name__ == "__main__":
    test_read()
