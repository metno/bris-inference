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
