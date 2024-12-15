from collections import defaultdict

variable_metadata = {
        "2t": {
            "cfname": "air_temperature",
            "leveltype": "height",
            "level": 2,
            "ncname": f"air_temperature_2m",
            },
        }

def get_variable_metadata(variable: str) -> tuple:
    """Extract metadata about a variable
        Args:
            variable: Anemoi variable name (e.g. u_800)
        Returns:
            ncname: Name of variable in NetCDF
            cfname: CF-standard name of variable
            leveltype: e.g. pressure, height
            level: e.g 800
    """
    if variable == "2t":
        cfname = "air_temperature"
        leveltype = "height"
        level = 2
        ncname = f"{cfname}_2m"
    elif variable == "2d":
        cfname = "dew_point_temperature"
        leveltype = "height"
        level = 2
        ncname = f"{cfname}_2m"
    elif variable == "10u":
        cfname = "x_wind"
        leveltype = "height"
        level = 10
        ncname = f"{cfname}_10m"
    else:
        name, level = variable.split('_')
        level = int(level)
        if name == "t":
            cfname = "air_temperature"
        elif name == "u":
            cfname = "x_wind"
        elif name == "v":
            cfname = "y_wind"
        elif name == "z":
            cfname = "geopotential"
        leveltype = "pressure"
        ncname = f"{cfname}_pl"

    return ncname, cfname, leveltype, level

def get_ncname(cfname: str, leveltype: str, levels: list):
    """Gets the name of a NetCDF variable given level information"""
    if leveltype == "height":
        if len(levels) == 1:
            # e.g. air_temperature_2m
            ncname = f"{cfname}_{levels[0]:d}m"
        else:
            raise NotImplementedError()
    elif leveltype == "pressure":
        ncname = f"{cfname}_pl"
    else:
        print(cfname, leveltype, levels)
        raise NotImplementedError()

    return ncname

def get_attributes_from_leveltype(leveltype):
    if leveltype == "pressure":
        return {
                "units": "hPa",
                "description": "pressure",
                "standard_name": "air_pressure",
                "positive": "up",
            }
    elif leveltype == "height":
        return {"units": "m", "description": "height above ground", "long_name": "height", "positive": "up"}

