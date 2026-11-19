"""Grib (edition 2) codes for the variables, levels and grids that bris writes

The names are the ones in the NetCDF layout (see bris.conventions.metno), which the Zarr
stores share and bris.convert.to_grib reads.
"""

# Map level type name to code
# https://codes.ecmwf.int/grib/format/grib2/ctables/4/5/
LEVEL_TYPES = {
    "surface": 1,
    "air_pressure": 100,  # Isobaric surface (Pa)
    "height_above_msl": 102,  # Specific altitude above mean sea level (m)
    "height": 103,  # Specified height level above ground (m)
}

# Stands in for the points without a value when a field is encoded with a bitmap
MISSING_VALUE = 9999.0

GRID_TEMPLATES = {
    "latlon": 0,
    "latitude_longitude": 0,
    "rotated_ll": 1,
    "mercator": 10,
    "polar_stereographic": 20,
    "lambert_conformal_conic": 30,
    "gaussian": 40,
    "rotated_gaussian": 41,
    "spherical_harmonic": 50,
}

# Map parameter name to:
# (discipline, parameter category, parameter number, type of statistical processing,
#  product definition template number, product definition template number in an ensemble)
# https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-0.shtml
PARAMETERS = {
    "air_temperature_0m": (0, 0, 0, None, 0, 11),
    "air_temperature_2m": (0, 0, 0, None, 0, 11),
    "air_temperature_pl": (0, 0, 0, None, 0, 11),
    "geopotential_pl": (0, 3, 4, None, 0, 11),
    "relative_humidity_2m": (0, 1, 192, None, 0, 11),
    "relative_humidity_pl": (0, 1, 192, None, 0, 11),
    "x_wind_10m": (0, 2, 2, None, 0, 11),
    "x_wind_pl": (0, 2, 2, None, 0, 11),
    "y_wind_10m": (0, 2, 3, None, 0, 11),
    "y_wind_pl": (0, 2, 3, None, 0, 11),
    "wind_speed_10m": (0, 2, 1, None, 0, 11),
    "atmosphere_boundary_layer_thickness": (0, 19, 3, None, 0, 11),
    "wind_speed_of_gust": (0, 2, 22, 2, 8, 11),
    "air_pressure_at_sea_level": (0, 3, 0, None, 0, 1),
    "surface_air_pressure": (0, 3, 0, None, 0, 11),
    "specific_humidity_pl": (0, 1, 0, None, 0, 11),
    "upward_air_velocity_pl": (0, 2, 9, None, 0, 11),
    "dew_point_temperature_2m": (0, 0, 6, None, 0, 11),
    "lagrangian_tendency_of_air_pressure_pl": (0, 2, 8, None, 0, 11),
    "tcw": (0, 1, 3, None, 0, 11),
    "skt": (0, 0, 17, None, 0, 11),
    "precipitation_amount": (0, 1, 49, None, 8, 11),
    "cp": (0, 1, 10, None, 0, 11),
}
