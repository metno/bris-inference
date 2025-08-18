"""This module converts anemoi variable names to CF-standard names"""


def get_metadata(anemoi_variable: str) -> dict:
    """Extract metadata about a variable
    Args:
        variable: Anemoi variable name (e.g. u_800)
    Returns:
        dict with:
            ncname: Name of variable in NetCDF
            cfname: CF-standard name of variable (or the original anemoi_variable name if unknown)
            leveltype: e.g. pressure, height
            level: e.g 800
    """
    variable_mapping = {
        "2t": ("air_temperature", "height", 2),
        # This is problematic for met.no conventions, since this would put skt into the same
        # variable as 2m temperature, which we don't want.
        # "skt": ("air_temperature", "height", 0),
        "2d": ("dew_point_temperature", "height", 2),
        "10u": ("x_wind", "height", 10),
        "10v": ("y_wind", "height", 10),
        "10si": ("wind_speed", "height", 10),
        "10fg": ("wind_speed_of_gust", "height", 10),
        "100u": ("x_wind", "height", 100),
        "100v": ("y_wind", "height", 100),
        "msl": ("air_pressure_at_sea_level", "height_above_msl", 0),
        "tp": ("precipitation_amount", "height", 0),
        "z": ("surface_geopotential", "height", 0),
        "lsm": ("land_sea_mask", "height", 0),
        "sp": ("surface_air_pressure", "height", 0),
        "vis": ("visibility_in_air", "height", 0),
        "cbh": ("cloud_base_altitude", "height", 0),
        "ws": ("wind_speed", "height", 10),
        "fog": ("fog_type_cloud_area_fraction", None, None),
        "hcc": ("high_type_cloud_area_fraction", None, None),
        "lcc": ("low_type_cloud_area_fraction", None, None),
        "mcc": ("medium_type_cloud_area_fraction", None, None),
        "tcc": ("cloud_area_fraction", None, None),
        "ssrd": (
            "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
            "height",
            0,
        ),
        "strd": (
            "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
            "height",
            0,
        ),
    }

    if anemoi_variable in variable_mapping:
        cfname, leveltype, level = variable_mapping[anemoi_variable]
    else:
        words = anemoi_variable.split("_")
        if len(words) == 2 and words[0] in ["t", "u", "v", "z", "q", "w"]:
            name, level = words[0], int(words[1])
            cfname = {  # noqa: SIM910 - None is explicitly handled in the following code block
                "t": "air_temperature",
                "u": "x_wind",
                "v": "y_wind",
                "z": "geopotential",
                "w": "vertical_velocity",
                "q": "specific_humidity",
            }.get(name, "unknown")
            if cfname == "unknown":
                raise ValueError(f"Unknown variable name: {name}")
            leveltype = "air_pressure"
        else:
            # Forcing parameters
            level = None
            leveltype = None
            cfname = anemoi_variable

    return {"cfname": cfname, "leveltype": leveltype, "level": level}


# def get_attributes_from_leveltype(leveltype):
#     if leveltype == "air_pressure":
#         return {
#             "units": "hPa",
#             "description": "pressure",
#             "standard_name": "air_pressure",
#             "positive": "up",
#         }
#     if leveltype == "height":
#         return {
#             "units": "m",
#             "description": "height above ground",
#             "long_name": "height",
#             "positive": "up",
#         }
#     if leveltype == "height_above_msl":
#         return {
#             "units": "m",
#             "description": "height above MSL",
#             "long_name": "height",
#             "positive": "up",
#         }
#     raise ValueError(f"Unknown leveltype: {leveltype}")


def get_attributes(cfname: str) -> dict[str, str]:
    ret = {"standard_name": cfname}

    # Coordinate variables
    if cfname in ["forecast_reference_time", "time"]:
        ret["units"] = "seconds since 1970-01-01 00:00:00 +00:00"
    elif cfname == "latitude":
        ret["units"] = "degrees_north"
    elif cfname in [
        "surface_altitude",
        "projection_x_coordinate",
        "projection_y_coordinate",
    ]:
        ret["units"] = "m"
    elif cfname == "longitude":
        ret["units"] = "degrees_east"
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
    elif cfname in ["thunder_event"]:
        ret["standard_name"] = "thunderstorm_probability"
        ret["units"] = "1"

    # Data variables
    elif cfname in [
        "x_wind",
        "y_wind",
        "wind_speed",
        "wind_speed_of_gust",
        "vertical_velocity",
    ]:
        ret["units"] = "m/s"
    elif cfname in ["air_temperature", "dew_point_temperature"]:
        ret["units"] = "K"
    elif cfname in ["land_sea_mask", "area_fraction"]:
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
    elif cfname in [
        "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
        "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
    ]:
        ret["units"] = "J/m^2"

    else:  # Handle unknown `cfname` by returning an empty dictionary
        return {}

    return ret
