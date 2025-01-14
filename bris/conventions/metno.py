"""Met-Norway's conventions for writing NetCDF files

In particular, the naming of variables (which cannot follow CF standard, since
these are not unique (e.g. air_temperature_pl vs air_temperature_2m).

Additionally, the names of some dimension-variables do not use CF-names
"""


class Metno:
    cf_to_metno = {
        "projection_y_coordinate": "y",
        "projection_x_coordinate": "x",
        "realization": "ensemble_member",
        "air_pressure": "pressure",
        "surface_altitude": "altitude",
    }

    def get_ncname(self, cfname: str, leveltype: str, level: int):
        """Gets the name of a NetCDF variable given level information"""
        if leveltype == "height":
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
            print(cfname, leveltype, level)
            raise NotImplementedError()

        return ncname

    def get_name(self, cfname: str):
        """Get MetNorway's dimension name from cf standard name"""
        if cfname in self.cf_to_metno:
            return self.cf_to_metno[cfname]
        else:
            return cfname

    def get_cfname(self, ncname):
        """Get the CF-standard name from a given MetNo name"""
        for k, v in self.cf_to_metno.items():
            if v == ncname:
                return k
        return ncname
