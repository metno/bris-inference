"""This module converts anemoi variable names to CF-standard names"""


def get_metadata(anemoi_variable: str) -> dict:
    """Extract metadata about a variable
    Args:
        variable: Anemoi variable name (e.g. u_800)
    Returns:
        dict with:
            ncname: Name of variable in NetCDF
            cfname: CF-standard name of variable
            leveltype: e.g. pressure, height
            level: e.g 800
    """
    if anemoi_variable == "2t":
        cfname = "air_temperature"
        leveltype = "height"
        level = 2
    # This is problematic for metno conventions, since this would put skt into the same
    # variable as 2m temperature, which we don't want.
    # elif anemoi_variable == "skt":
    #     cfname = "air_temperature"
    #     leveltype = "height"
    #     level = 0
    elif anemoi_variable == "2d":
        cfname = "dew_point_temperature"
        leveltype = "height"
        level = 2
    elif anemoi_variable == "10u":
        cfname = "x_wind"
        leveltype = "height"
        level = 10
    elif anemoi_variable == "10v":
        cfname = "y_wind"
        leveltype = "height"
        level = 10
    elif anemoi_variable == "10si":
        cfname = "wind_speed"
        leveltype = "height"
        level = 10
    elif anemoi_variable == "10fg":
        cfname = "wind_speed_of_gust"
        leveltype = "height"
        level = 10
    elif anemoi_variable == "100u":
        cfname = "x_wind"
        leveltype = "height"
        level = 100
    elif anemoi_variable == "100v":
        cfname = "y_wind"
        leveltype = "height"
        level = 100
    elif anemoi_variable == "msl":
        cfname = "air_pressure_at_sea_level"
        leveltype = "height_above_msl"
        level = 0
    elif anemoi_variable == "tp":
        cfname = "precipitation_amount"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "z":
        cfname = "surface_geopotential"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "lsm":
        cfname = "land_sea_mask"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "sp":
        cfname = "surface_air_pressure"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "vis":
        cfname = "visibility_in_air"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "cbh":
        cfname = "cloud_base_altitude"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "10si":
        cfname = "wind_speed"
        leveltype = "height"
        level = 10
    elif anemoi_variable == "fog":
        cfname = "fog_type_cloud_area_fraction"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "hcc":
        cfname = "high_type_cloud_area_fraction"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "lcc":
        cfname = "low_type_cloud_area_fraction"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "mcc":
        cfname = "medium_type_cloud_area_fraction"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "tcc":
        cfname = "cloud_area_fraction"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "ssrd":
        cfname = "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time"
        leveltype = "height"
        level = 0
    elif anemoi_variable == "strd":
        cfname = "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time"
        leveltype = "height"
        level = 0
    else:

        words = anemoi_variable.split("_")
        if len(words) == 2 and words[0] in ["t", "u", "v", "z", "q", "w"]:
            name, level = words
            level = int(level)
            if name == "t":
                cfname = "air_temperature"
            elif name == "u":
                cfname = "x_wind"
            elif name == "v":
                cfname = "y_wind"
            elif name == "z":
                cfname = "geopotential"
            elif name == "w":
                cfname = "vertical_velocity"
            elif name == "q":
                cfname = "specific_humidity"
            else:
                raise ValueError()
            leveltype = "air_pressure"
        else:
            # Forcing parameters
            level = None
            leveltype = None
            cfname = anemoi_variable

    return dict(cfname=cfname, leveltype=leveltype, level=level)


def get_attributes_from_leveltype(leveltype):
    if leveltype == "air_pressure":
        return {
            "units": "hPa",
            "description": "pressure",
            "standard_name": "air_pressure",
            "positive": "up",
        }
    elif leveltype == "height":
        return {
            "units": "m",
            "description": "height above ground",
            "long_name": "height",
            "positive": "up",
        }
    elif leveltype == "height_above_msl":
        return {
            "units": "m",
            "description": "height above MSL",
            "long_name": "height",
            "positive": "up",
        }


def get_attributes(cfname):
    ret = {"standard_name": cfname}

    # Coordinate variables
    if cfname == "forecast_reference_time":
        ret["units"] = "seconds since 1970-01-01 00:00:00 +00:00"
    elif cfname == "time":
        ret["units"] = "seconds since 1970-01-01 00:00:00 +00:00"
    elif cfname == "latitude":
        ret["units"] = "degrees_north"
    elif cfname == "surface_altitude":
        ret["units"] = "m"
    elif cfname == "longitude":
        ret["units"] = "degrees_east"
    elif cfname == "projection_x_coordinate":
        ret["units"] = "m"
    elif cfname == "projection_y_coordinate":
        ret["units"] = "m"
    elif cfname == "realization":
        pass
    elif cfname == "air_pressure":
        ret["units"] = "hPa"
        ret["description"] = "pressure"
        ret["positive"] = "up"
    elif cfname == "height":
        ret["units"] = "m"
        ret["description"] = "height above ground"
        ret["long_name"] = "height"
        ret["positive"] = "up"

    # Data variables
    elif cfname in [
        "x_wind",
        "y_wind",
        "wind_speed",
        "wind_speed_of_gust",
        "vertical_velocity",
        "wind_speed_of_gust",
    ]:
        ret["units"] = "m/s"
    elif cfname in ["air_temperature", "dew_point_temperature"]:
        ret["units"] = "K"
    elif cfname == "land_sea_mask":
        ret["units"] = "1"
    elif cfname in ["geopotential", "surface_geopotential"]:
        ret["units"] = "m^2/s^2"
    elif cfname in ["precipitation_amount", "precipitation_amount_acc"]:
        ret["units"] = "kg/m^2"
    elif cfname in ["air_pressure_at_sea_level", "surface_air_pressure"]:
        ret["units"] = "Pa"
    elif cfname in ["specific_humidity"]:
        ret["units"] = "kg/kg"
    elif cfname in ["cloud_base_altitude", "visibility_in_air"]:
        ret["units"] = "m"
    elif "area_fraction" in cfname:
        ret["units"] = "1"
    elif cfname in ["integral_of_surface_downwelling_longwave_flux_in_air_wrt_time", "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time"]:
        ret["units"] = "J/m^2"

    # Unknown cfname, let's not write any attributes
    else:
        ret = {}

    return ret
