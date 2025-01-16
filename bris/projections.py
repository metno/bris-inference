import netCDF4
import numpy as np
import pyproj


def get_proj4_str(name):
    if name == "meps":
        return "+proj=lcc +lat_0=63.3 +lon_0=15 +lat_1=63.3 +lat_2=63.3 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs"
    elif name == "arome_arctic":
        return "+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs +type=crs"


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
    attrs = {k:v for k,v in attrs.items() if v != "unknown"}

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
            attributes = dict()
            for attr in var.ncattrs():
                attributes[attr] = getattr(var, attr)
            try:
                crs = pyproj.CRS.from_cf(attributes)
                proj_str = crs.to_proj4()
            except (pyproj.exceptions.CRSError, KeyError) as e:
                print(e)
                print(f"Invalid projection")
                continue
            break

    file.close()

    return proj_str
