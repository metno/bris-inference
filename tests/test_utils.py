import os
import tempfile

import bris.utils
import numpy as np
import pytest


def test_expand_time_tokens():
    filename = "/test%Y%m%dtest"
    unixtime = 1704067200  # 2024-01-01

    output = bris.utils.expand_time_tokens(filename, unixtime)
    assert output == "/test20240101test", output

    # Invalid arguments
    invalid_arguments = [np.nan, "test", [], {}]
    for invalid_argument in invalid_arguments:
        with pytest.raises(ValueError):
            bris.utils.expand_time_tokens(filename, invalid_argument)


def test_is_number():
    valid_numbers = [-1, 0, 1, np.nan, np.float32(1)]
    invalid_numbers = ["test", [], {}, "1", None]

    for valid_number in valid_numbers:
        assert bris.utils.is_number(valid_number), valid_number

    for invalid_number in invalid_numbers:
        assert not bris.utils.is_number(invalid_number), invalid_number


if __name__ == "__main__":
    test_expand_time_tokens()
    test_is_number()
