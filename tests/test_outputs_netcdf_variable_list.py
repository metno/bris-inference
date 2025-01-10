from bris.outputs import netcdf


def test_empty():
    vl = netcdf.VariableList([])
    assert len(vl.dimensions) == 0


def test_1():
    vl = netcdf.VariableList(["u_800"])
    assert vl.dimensions == {"pressure": ("air_pressure", [800])}
    assert vl.get_level_dimname("x_wind_pl") == "pressure"
    assert vl.get_level_index("u_800") == 0


def test_2():
    vl = netcdf.VariableList(["u_800", "u_700", "v_700", "2t", "10u"])
    assert len(vl.dimensions) == 4
    dimname = vl.get_level_dimname("x_wind_pl")
    assert vl.dimensions[dimname] == ("air_pressure", [700, 800])
    assert vl.get_level_index("u_700") == 0
    assert vl.get_level_index("u_800") == 1

    dimname = vl.get_level_dimname("y_wind_pl")
    assert vl.dimensions[dimname] == ("air_pressure", [700])
    assert vl.get_level_index("v_700") == 0

    dimname = vl.get_level_dimname("air_temperature_2m")
    assert vl.dimensions[dimname] == ("height", [2])
    assert vl.get_level_index("2t") == 0

    dimname = vl.get_level_dimname("x_wind_10m")
    assert vl.dimensions[dimname] == ("height", [10])
    assert vl.get_level_index("10u") == 0


def test_3():
    # This checks that with repeated variables, we get the right matching of dimensions
    vl = netcdf.VariableList(["10u", "10v", "2t", "2d"])
    assert vl.dimensions == {"height": ("height", [10]), "height1": ("height", [2])}


def test_4():
    vl = netcdf.VariableList(["cos_julian_day"])
    assert vl.dimensions == {}
    assert vl.get_level_dimname("cos_julian_day") is None
    assert vl.get_level_index("cos_julian_day") is None


if __name__ == "__main__":
    test_3()
