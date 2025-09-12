"""Met-Norway's conventions for writing NetCDF files

In particular, the naming of variables (which cannot follow CF standard, since
these are not unique (e.g. air_temperature_pl vs air_temperature_2m).

Additionally, the names of some dimension-variables do not use CF-names
"""

from bris.utils import LOGGER


class Metno:
    cf_to_metno = {
        "projection_y_coordinate": "y",
        "projection_x_coordinate": "x",
        "realization": "ensemble_member",
        "air_pressure": "pressure",
        "surface_altitude": "altitude",
        "tcw": "atmosphere_mass_content_of_water",
    }

    def get_ncname(self, cfname: str, leveltype: str, level: int):
        """Gets the name of a NetCDF variable given level information"""
        if cfname in [
            "precipitation_amount",
            "surface_air_pressure",
            "air_pressure_at_sea_level",
            "wind_speed_of_gust",
            "land_sea_mask",
            "fog_type_cloud_area_fraction",
            "high_type_cloud_area_fraction",
            "low_type_cloud_area_fraction",
            "medium_type_cloud_area_fraction",
            "cloud_area_fraction",
            "atmosphere_mass_content_of_water",
        ]:
            # Prevent _0m from being added at the end of variable name
            ncname = f"{cfname}"
        elif leveltype == "height":
            # e.g. air_temperature_2m
            ncname = f"{cfname}_{level:d}m"
        elif leveltype == "height_above_msl":
            # e.g. air_pressure_at_sea_level
            ncname = f"{cfname}"
        elif leveltype == "air_pressure":
            ncname = f"{cfname}_pl"
        elif leveltype is None and level is None:
            # This is likely a forcing variable
            return cfname
        else:
            LOGGER.error(cfname, leveltype, level)
            raise NotImplementedError()

        return ncname

    def is_single_level(self, cfname: str, leveltype: str) -> bool:
        """Returns true if there should only be a single level in the level dimension for this
        variable.

        Args:
            cfname: e.g. air_temperature
            leveltype: e.g. height

        E.g. air_temperature_2m and air_temperature_0m should not share a height dimension, and
        therefore these should return True
        """
        return cfname in [
            "air_temperature",
            "x_wind",
            "y_wind",
            "wind_speed",
        ] and leveltype in ["height"]

    def get_name(self, cfname: str) -> str:
        """Get MetNorway's dimension name from cf standard name"""
        if cfname in self.cf_to_metno:
            return self.cf_to_metno[cfname]
        return cfname

    def get_cfname(self, ncname) -> str:
        """Get the CF-standard name from a given MetNo name"""
        for k, v in self.cf_to_metno.items():
            if v == ncname:
                return k
        return ncname
