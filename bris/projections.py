import netCDF4
import pyproj
import numpy as np


def get_proj4_str(name: str) -> str:
    """Returns a proj4 string based on a name"""
    return {
        "meps": "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs",
        "arome_arctic": "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs",
        "norkyst_v3": "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=70 +x_0=3369600 +y_0=1844800 +a=6378137 +b=6356752.3142 +units=m +no_defs +type=crs",
        "nordic_analysis": "+proj=lcc +lat_0=63.0 +lon_0=15 +lat_1=63.0 +lat_2=63.0 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs",
    }[name]


def get_xy(lats, lons, proj_str):
    """Reverse engineer x and y vectors from lats and lons"""

    proj_from = pyproj.Proj("proj+=longlat")
    proj_to = pyproj.Proj(proj_str)

    transformer = pyproj.transformer.Transformer.from_proj(proj_from, proj_to)

    xx, yy = transformer.transform(lons, lats)
    x = xx[0, :]
    y = yy[:, 0]

    return x, y


def get_proj_attributes(proj_str):
    crs = pyproj.CRS.from_proj4(proj_str)
    attrs = crs.to_cf()

    del attrs["crs_wkt"]
    attrs = {k: v for k, v in attrs.items() if v != "unknown"}

    return attrs


def proj_from_ncfile(filename):
    """Returns a projection object based on information in a netCDF file

    Args:
        filename (str|netCDF4.Dataset): File to get projection from

    Returns:
        Proj: A projection object
    """
    file = netCDF4.Dataset(filename)

    # Find variable with grid_mapping_name attribute
    for variable in file.variables:
        var = file.variables[variable]
        if hasattr(var, "grid_mapping_name"):
            attributes = {}
            for attr in var.ncattrs():
                attributes[attr] = getattr(var, attr)
            try:
                crs = pyproj.CRS.from_cf(attributes)
                proj_str = crs.to_proj4()
            except (pyproj.exceptions.CRSError, KeyError) as e:
                print(e)
                print("Invalid projection")
                continue
            break

    file.close()

    return proj_str


def rotate_wind_to_projected_coords(
    x_wind, y_wind, lats, lons, proj_str_to, proj_str_from="proj+=longlat"
):
    """Rotates winds in projection to another projection. Based loosely on
    https://github.com/SciTools/cartopy/blob/main/lib/cartopy/crs.py#L429

    Args:
        x_wind (np.array or list): Array of winds in x-direction [m]
        y_wind (np.array or list): Array of winds in y-direction [m]
        lats (np.array or list): Array of latitudes [degrees]
        lons (np.array or list): Array of longitudes [degrees]
        proj_str_from (str): Projection string to convert from
        proj_str_to (str): Projection string to convert to

    Returns:
        new_x_wind (np.array): Array of winds in x-direction in new projection [m]
        new_y_wind (np.array): Array of winds in y-direction in new projection [m]

    Todo:
        Deal with perturbations that end up outside the domain of the transformation
        Deal with any issues related to directions on the poles
    """

    if np.shape(x_wind) != np.shape(y_wind):
        raise ValueError(
            f"x_wind {np.shape(x_wind)} and y_wind {np.shape(y_wind)} arrays must be the same size"
        )
    if len(lats.shape) != 1:
        raise ValueError(f"lats {np.shape(lats)} must be 1D")

    if np.shape(lats) != np.shape(lons):
        raise ValueError(
            f"lats {np.shape(lats)} and lats {np.shape(lons)} must be the same size"
        )

    if len(np.shape(x_wind)) == 1:
        if np.shape(x_wind) != np.shape(lats):
            raise ValueError(
                f"x_wind {len(x_wind)} and lats {len(lats)} arrays must be the same size"
            )
    elif len(np.shape(x_wind)) == 2:
        if x_wind.shape[1] != len(lats):
            raise ValueError(
                f"Second dimension of x_wind {x_wind.shape[1]} must equal number of lats {len(lats)}"
            )
    else:
        raise ValueError(f"x_wind {np.shape(x_wind)} must be 1D or 2D")

    proj_from = pyproj.Proj(proj_str_from)
    proj_to = pyproj.Proj(proj_str_to)

    transformer = pyproj.transformer.Transformer.from_proj(proj_from, proj_to)

    # To compute the new vector components:
    # 1) perturb each position in the direction of the winds
    # 2) convert the perturbed positions into the new coordinate system
    # 3) measure the new x/y components.
    #
    # A complication occurs when using the longlat "projections", since this is not a cartesian grid
    # (i.e. distances in each direction is not consistent), we need to deal with the fact that the
    # width of a longitude varies with latitude
    orig_speed = np.sqrt(x_wind**2 + y_wind**2)

    x0, y0 = proj_from(lons, lats)
    if proj_from.name != "longlat":
        x1 = x0 + x_wind
        y1 = y0 + y_wind
    else:
        # Reduce the perturbation, since x_wind and y_wind are in meters, which would create
        # large perturbations in lat, lon. Also, deal with the fact that the width of longitude
        # varies with latitude.
        factor = 3600000.0
        x1 = x0 + x_wind / factor / np.cos(lats * 3.14159265 / 180)
        y1 = y0 + y_wind / factor

    X0, Y0 = transformer.transform(x0, y0)
    X1, Y1 = transformer.transform(x1, y1)

    new_x_wind = X1 - X0
    new_y_wind = Y1 - Y0
    if proj_to.name == "longlat":
        new_x_wind *= np.cos(lats * 3.14159265 / 180)
    else:
        print('proj_to.name == "longlat" is False')
        print("proj_to.name: ", proj_to.name)

    if proj_to.name == "longlat" or proj_from.name == "longlat":
        # Ensure the wind speed is not changed (which might not the case since the units in longlat
        # is degrees, not meters)
        curr_speed = np.sqrt(new_x_wind**2 + new_y_wind**2)
        new_x_wind *= orig_speed / curr_speed
        new_y_wind *= orig_speed / curr_speed
    else:
        print('proj_to.name == "longlat" or proj_from.name == "longlat" is False')

    return new_x_wind, new_y_wind
