import netCDF4
import numpy as np
import pyproj


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


def compute_local_mapping_from_lonlat(
    lon, lat, proj_str, dist=1.0, return_matrix=False
):
    """
    Compute local 2x2 mapping M such that:
        [Vx, Vy]^T = M @ [u_east, v_north]^T

    Args:
        lon, lat : arrays (shape (...)) in degrees
        proj_str : projection string or anything accepted by pyproj.CRS.from_user_input
        dist : step distance in metres used to estimate east/north displacements (default 1.0)
        return_matrix : if True, return array shaped (..., 2, 2)

    Returns:
        If return_matrix is False:
            e_x, n_x, e_y, n_y  (each shaped like lon/lat)
            where M = [[e_x, n_x],
                       [e_y, n_y]]
        If return_matrix is True:
            M : array shaped (..., 2, 2)
    Notes:
        - Use dist=1.0 if u,v are in m/s (so matrix maps m/s east/north -> m/s proj).
        - lon,lat may be any shape; results have same shape.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)

    # CRS/transformers
    crs_proj = pyproj.CRS.from_user_input(proj_str)
    to_proj = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326), crs_proj, always_xy=True
    )

    # geodetic forward: compute points dist metres East (az=90) and North (az=0)
    geod = pyproj.Geod(ellps="WGS84")
    lon_east, lat_east, _ = geod.fwd(
        lon, lat, np.full_like(lon, 90.0), np.full_like(lon, dist)
    )  # 90° = east
    lon_north, lat_north, _ = geod.fwd(
        lon, lat, np.full_like(lon, 0.0), np.full_like(lon, dist)
    )  # 0° = north

    # project original and offset points into projection coordinates
    x, y = to_proj.transform(lon, lat)
    x_east, y_east = to_proj.transform(lon_east, lat_east)
    x_north, y_north = to_proj.transform(lon_north, lat_north)

    # compute projected displacements per metre geographic displacement
    e_x = (x_east - x) / dist
    e_y = (y_east - y) / dist
    n_x = (x_north - x) / dist
    n_y = (y_north - y) / dist

    if return_matrix:
        # stack into shape (..., 2, 2) where M[...,0,0]=e_x, M[...,0,1]=n_x, M[...,1,0]=e_y, M[...,1,1]=n_y
        M = np.stack(
            [np.stack([e_x, n_x], axis=-1), np.stack([e_y, n_y], axis=-1)], axis=-2
        )
        return M

    return e_x, n_x, e_y, n_y
