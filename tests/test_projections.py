import pyproj

from bris import projections


def test_polar_stereographic_has_latitude_of_projection_origin():
    """pyproj drops latitude_of_projection_origin for +proj=stere with lat_ts; CF (and DIANA) need it."""
    for name in ("carra2", "norkyst_v3"):
        attrs = projections.get_proj_attributes(projections.get_proj4_str(name))
        assert attrs["grid_mapping_name"] == "polar_stereographic"
        assert attrs["latitude_of_projection_origin"] == 90.0
        # the added attribute must not change the projection (CRS objects differ in names/axis metadata,
        # so compare the proj parameters)
        back = pyproj.CRS.from_cf(attrs).to_dict()
        orig = pyproj.CRS.from_proj4(projections.get_proj4_str(name)).to_dict()
        for key in ("proj", "lat_0", "lat_ts", "lon_0", "x_0", "y_0"):
            assert back[key] == orig[key], key


def test_south_polar_stereographic():
    attrs = projections.get_proj_attributes(
        "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +R=6371229 +units=m +no_defs"
    )
    assert attrs["latitude_of_projection_origin"] == -90.0


def test_lambert_unchanged():
    attrs = projections.get_proj_attributes(projections.get_proj4_str("meps"))
    assert attrs["grid_mapping_name"] == "lambert_conformal_conic"
    assert attrs["latitude_of_projection_origin"] == 63.3
