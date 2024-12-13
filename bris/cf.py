from collections import defaultdict


def get_list(variable_names):
    """ Converts Anemoi variable names to CF-standard metadata for MET Norway's NetCDF files
        returns:
            dimension_definitions: {height1: (height, [2])}
            variable_definitions: {air_temperature_2m: height1}
            variable_to_ncname_and_index: {2t: (air_temperature_2m, 0)}
    """
    # Get the levels for each leveltype for each cfname
    cfname_to_levels = dict()
    for v, variable in enumerate(variable_names):
        _, cfname, leveltype, level = get_variable_metadata(variable)
        if cfname not in cfname_to_levels:
            cfname_to_levels[cfname] = dict()
        if leveltype not in cfname_to_levels[cfname]:
            cfname_to_levels[cfname][leveltype] = list()
        cfname_to_levels[cfname][leveltype] += [level]
    # Sort levels
    for cfname, v in cfname_to_levels.items():
        for leveltype, vv in v.items():
            v[leveltype] = sorted(vv)
    # air_temperature -> pressure -> [1000, 925, 800, 700]

    # Determine unique dimensions to add
    dims_to_add = dict()   # height1 -> [height, [2]]
    ncname_to_level_dim = dict()
    for cfname, v in cfname_to_levels.items():
        for leveltype, levels in v.items():
            ncname = get_ncname(cfname, leveltype, levels)

            if (leveltype, levels) in dims_to_add.values():
                # Reuse
                pass
            else:
                count = 0
                for curr_leveltype, _ in dims_to_add.values():
                    if curr_leveltype == leveltype:
                        count += 1
                if count == 0:
                    name = leveltype  # height
                else:
                    name = f"{leveltype}{count}" # height1
            dims_to_add[name] = (leveltype, levels)
            ncname_to_level_dim[ncname] = name

    # Determine what ncname and index each variable belongs to
    variable_to_ncname_and_index = dict()  # air_temperature_2m 0
    for v, variable in enumerate(variable_names):
        ncname, cfname, leveltype, level = get_variable_metadata(variable)
        # Find the name of the level dimension
        dimname = ncname_to_level_dim[ncname]
        # Find the index in this dimension
        index = dims_to_add[dimname][1].index(level)
        variable_to_ncname_and_index[variable] = (ncname, index)

    return dims_to_add, ncname_to_level_dim, variable_to_ncname_and_index

def get_variable_metadata(variable: str): -> tuple
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

def get_units_from_leveltype(self, leveltype):
    if leveltype == "pressure":
        return "hPa"
    elif leveltype == "height":
        return "m"
