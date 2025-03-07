"""Unit conversion module

This module handles conversion between different units and provides a set of preferred units that
should be used internally in yrprod.
"""

import numbers

import numpy as np

default_units = ["K", "m/s", "1", "kg/m^2", "degree", "Pa", "text"]
"""Preferred units (must all be strings)"""


def get_default_units(iunits):
    """Provides the default (preferred) units.

    Args:
        iunits (str): Input units

    Returns:
        str: The default units. Returns None, if no default unit is available
    """
    if iunits in default_units:
        return iunits

    iunits = find_common_name(iunits)
    if iunits in default_units:
        return iunits

    linear_convert = get_conversion_map()
    for c in linear_convert:
        if iunits == c[0] and c[1] in default_units:
            return c[1]
    return None


def get_conversion_map():
    """Returns conversion map of all possible unit conversions

    Returns:
        dict: Dictionary of the form: (input_units, output_units) -> (multiplicative_factor, additive_factor)
    """
    # Linear conversion map out = in * v[0] + v[1]
    # Key: (from_unit, to_unit)
    # Value: (multiplicative, additive)
    linear_convert = dict()
    linear_convert["celsius", "K"] = (1, 273.15)
    linear_convert["m/s", "km/h"] = (3.6, 0)
    linear_convert["kg/m^2", "mm"] = (1, 0)
    linear_convert["kg/m^2", "Mg/m^2"] = (0.001, 0)
    linear_convert["1", "%"] = (100, 0)
    linear_convert["1", "percent"] = (100, 0)
    linear_convert["Pa", "hPa"] = (0.01, 0)
    linear_convert["octas", "1"] = (0.125, 0)
    linear_convert["octas", "%"] = (12.5, 0)

    # ** means the same as ^
    for key, c in dict(linear_convert).items():
        found = False
        key0 = key[0]
        key1 = key[1]
        if key[0].find("^") > 0:
            found = True
            key0 = key[0].replace("^", "**")
        if key[1].find("^") > 0:
            found = True
            key1 = key[1].replace("^", "**")
        if found:
            linear_convert[(key0, key1)] = (1, 0)

    # Create inverse map
    for key, c in dict(linear_convert).items():
        linear_convert[(key[1], key[0])] = (1 / c[0], -c[1] / c[0])

    return linear_convert


def find_common_name(units):
    """Finds a more common name for a strange unit, e.g. celsius instead of C or degC

    Args:
        units (str): Input units

    Returns:
        str: Common name corresponding to input units. Returns the input units if no common name found.
    """

    if units is None:
        return None
    if isinstance(units, str):
        units = units.replace("**", "^")

    # Key: common name, value: List of alternative names
    identical = {
        "celsius": ["C", "degC"],
        "km/h": ["kmh"],
        "kg/m^2": ["mm", "Kg/m^2", "Kg/m2", "kg/m2"],
        "degree": ["degrees"],
        "%": ["percent"],
    }
    if units in identical:
        return units
    for key, value in identical.items():
        if units in value:
            return key

    return units


def convert(array, iunits, ounits=None, inplace=False):
    """Converts data from one unit to another

    Args:
        array (np.array or float): Values to convert
        iunits (str): Input units
        ounits (str): Output units (if None, convert to default units)
        inplace (bool): Edits array in place. Requires array to be of type np.array.

    Raises:
        ValueError: On invalid input arguments

    Returns:
        np.array or float: Converted values (if inplace=False)
        units (str): New units
    """
    original_ounits = ounits
    iunits = find_common_name(iunits)
    ounits = find_common_name(ounits)
    if original_ounits is None:
        original_ounits = ounits

    if inplace:
        if not isinstance(array, np.ndarray):
            raise ValueError(
                "Input array is not a numpy array, cannot edit values in place"
            )

    if iunits == ounits:
        if inplace:
            return original_ounits
        else:
            return array, original_ounits

    if ounits is None:
        # Use default units
        if iunits in default_units:
            if inplace:
                return iunits
            else:
                return array, iunits

    if isinstance(array, np.ndarray):
        if not issubclass(array.dtype.type, np.floating):
            raise ValueError("Input array is not a floating point numpy array")
    elif isinstance(array, list):
        if isinstance(array, list):
            if any([not isinstance(a, numbers.Number) for a in array]):
                raise ValueError(
                    "Input list contains one or more non-numerical values: ", array
                )
    else:
        if not isinstance(array, numbers.Number):
            raise ValueError("Input is not np.array, list, or number")

    linear_convert = get_conversion_map()

    if ounits is None:
        for default_unit in default_units:
            key = (iunits, default_unit)
            if key in linear_convert:
                if inplace:
                    convert(array, iunits, default_unit, inplace)
                    return default_unit
                else:
                    return convert(array, iunits, default_unit)

        raise ValueError(
            "Cannot convert units. Cannot find a default unit to convert '%s' to"
            % iunits
        )

    else:
        key = (iunits, ounits)
        if key in linear_convert:
            if inplace:
                array *= linear_convert[key][0]
                array += linear_convert[key][1]
                return original_ounits
            else:
                return (
                    array * linear_convert[key][0] + linear_convert[key][1],
                    original_ounits,
                )
    raise ValueError(
        "Unrecognized input unit conversion '%s'->'%s'" % (iunits, original_ounits)
    )


def get_time_units_from_unix_time_steps(t0, t1):
    """Compute the unit of the time step: seconds, minutes, hours, days or weeks.
        If none of the above, return units=1.

    Args:
        t0 (float): Timestep 0 in unixtime
        t1 (float): Timestep 1 in unixtime

    Returns:
        string: unit of the timestep between t0 and t1.
    """
    dt = t1 - t0
    if dt == 1.0:
        units = "sec"
    elif dt / 60 == 1.0:
        units = "min"
    elif dt / 3600 == 1.0:
        units = "hour"
    elif dt / (3600 * 24) == 1.0:
        units = "day"
    elif dt / (3600 * 24 * 7) == 1.0:
        units = "week"
    else:
        units = "1"

    return units
