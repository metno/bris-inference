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


def test_vertical_velocity():
    # Anemoi "w" on pressure levels is omega (Pa/s), not a geometric velocity
    md = cf.get_metadata("w_500")
    assert md["cfname"] == "lagrangian_tendency_of_air_pressure"
    assert md["leveltype"] == "air_pressure"
    assert md["level"] == 500
    assert cf.get_attributes(md["cfname"])["units"] == "Pa/s"

    # wz_<level> is the derived geometric vertical velocity
    md = cf.get_metadata("wz_50")
    assert md["cfname"] == "upward_air_velocity"
    assert md["leveltype"] == "air_pressure"
    assert md["level"] == 50
    assert cf.get_attributes(md["cfname"])["units"] == "m/s"
