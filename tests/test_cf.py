import os

import bris.conventions.cf as cf


def test_get_metadata():
    test = cf.get_attributes("air_pressure")
    assert test["standard_name"] == "air_pressure"
    assert test["units"] == "hPa"
    assert test["description"] == "pressure"
    assert test["positive"] == "up"


def test_get_attributes():
    attr = cf.get_attributes("non_existant")
    assert attr == {}

    attr = cf.get_attributes("air_pressure")
    assert attr["standard_name"] == "air_pressure"
    assert attr["units"] == "hPa"

    attr = cf.get_attributes("realization")
    assert attr["standard_name"] == "realization"
    assert len(attr) == 1

    attr = cf.get_attributes("thunder_event")
    assert attr["standard_name"] == "thunderstorm_probability"
