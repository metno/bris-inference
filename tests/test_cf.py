import os

import bris.conventions.cf as cf


def test_get_metadata():
    test = cf.get_attributes("air_pressure")
    assert test["standard_name"] == "air_pressure"
    assert test["units"] == "hPa"
    assert test["description"] == "pressure"
    assert test["positive"] == "up"


def test_get_attributes():
    test = cf.get_attributes("air_pressure")
    assert test["standard_name"] == "air_pressure"
    assert test["units"] == "hPa"


if __name__ == "__main__":
    test_get_attributes()
