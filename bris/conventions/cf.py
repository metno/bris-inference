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
        "skt": ("air_temperature", "height", 0),
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
        "fog": ("fog_type_cloud_area_fraction", "height", 0),
        "hcc": ("high_type_cloud_area_fraction", "height", 0),
        "lcc": ("low_type_cloud_area_fraction", "height", 0),
        "mcc": ("medium_type_cloud_area_fraction", "height", 0),
        "tcc": ("cloud_area_fraction", "height", 0),
        "tcw": ("atmosphere_mass_content_of_water", "height", 0),
        "tp_acc": ("precipitation_amount_acc", "height", 0),
        "ssrd_acc": (
            "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
            "height",
            0,
        ),
        "strd_acc": (
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


def get_attributes(cfname: str) -> dict[str, str] | dict:
    # Define a mapping of cfname to their attributes
    attribute_mapping = {
        # Coordinate variables
        "forecast_reference_time": {
            "units": "seconds since 1970-01-01 00:00:00 +00:00"
        },
        "fog_type_cloud_area_fraction": {"units": "m"},
        "high_type_cloud_area_fraction": {"units": "m"},
        "low_type_cloud_area_fraction": {"units": "m"},
        "medium_type_cloud_area_fraction": {"units": "m"},
        "cloud_area_fraction": {"units": "m"},
        "time": {"units": "seconds since 1970-01-01 00:00:00 +00:00"},
        "latitude": {"units": "degrees_north"},
        "longitude": {"units": "degrees_east"},
        "surface_altitude": {"units": "m"},
        "projection_x_coordinate": {"units": "m"},
        "projection_y_coordinate": {"units": "m"},
        "air_pressure": {
            "units": "hPa",
            "description": "pressure",
            "positive": "up",
        },
        "height": {
            "units": "m",
            "description": "height above ground",
            "long_name": "height",
            "positive": "up",
        },
        "thunder_event": {
            "standard_name": "thunderstorm_probability",
            "units": "1",
        },
        "realization": {},
        # Data variables
        "x_wind": {"units": "m/s"},
        "y_wind": {"units": "m/s"},
        "wind_speed": {"units": "m/s"},
        "wind_speed_of_gust": {"units": "m/s"},
        "vertical_velocity": {"units": "m/s"},
        "air_temperature": {"units": "K"},
        "dew_point_temperature": {"units": "K"},
        "land_sea_mask": {"units": "1"},
        "area_fraction": {"units": "1"},
        "geopotential": {"units": "m^2/s^2"},
        "surface_geopotential": {"units": "m^2/s^2"},
        "precipitation_amount": {"units": "kg/m^2"},
        "precipitation_amount_acc": {
            "units": "kg/m^2",
            "standard_name": "precipitation_amount",
        },
        "air_pressure_at_sea_level": {"units": "Pa"},
        "surface_air_pressure": {"units": "Pa"},
        "specific_humidity": {"units": "kg/kg"},
        "cloud_base_altitude": {"units": "m"},
        "visibility_in_air": {"units": "m"},
        "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time": {
            "units": "J/m^2"
        },
        "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": {
            "units": "J/m^2"
        },
        "atmosphere_mass_content_of_water": {
            "long_name": "Total column water contents (TCW)",
            "units": "kg/m^2",
        },
        "surface_temperature": {
            "long_name": "Surface (skin) temperature (SKT)",
            "units": "K",
        },
    }

    # Return empty dictionary if unknown
    if cfname not in attribute_mapping:
        return {}

    # Add standard_name if it doesn't exist, return attributes for the given cfname
    return {"standard_name": cfname} | attribute_mapping[cfname]
