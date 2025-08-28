from bris.conventions import cf
from bris.conventions.metno import Metno


class VariableList:
    """This class keeps track of levels are available for each cf-variables
    and determines a unique name of the level dimension, if there are multiple definitions of the
    same dimension (e.g. two variables with a different set of pressure levels)
    """

    def __init__(self, anemoi_names: list, conventions = None):
        """Args:
        anemoi_names: A list of variables names used in Anemoi (e.g. u10)
        conventions: What NetCDF naming convention to use
        """
        self.anemoi_names = anemoi_names
        self.conventions = conventions if conventions is not None else Metno()

        self._dimensions, self._ncname_to_level_dim = self.load_dimensions()

    @property
    def dimensions(self) -> dict:
        """A diction of dimension names needed to represent the variable list

        The key is the dimension name, the value is a tuple of (leveltype, levels)
        E.g. for a dataset with 2m temperature: {height1: (height, 2)}
        """
        return self._dimensions

    def load_dimensions(self):
        cfname_to_levels = {}
        for _v, variable in enumerate(self.anemoi_names):
            metadata = cf.get_metadata(variable)
            cfname = metadata["cfname"]
            leveltype = metadata["leveltype"]
            level = metadata["level"]

            if leveltype is None:
                # This variable (likely a forcing parameter) does not need a level dimension
                continue

            if cfname not in cfname_to_levels:
                cfname_to_levels[cfname] = {}
            if leveltype not in cfname_to_levels[cfname]:
                cfname_to_levels[cfname][leveltype] = []
            cfname_to_levels[cfname][leveltype] += [level]
        # Sort levels
        for cfname, v in cfname_to_levels.items():
            for leveltype, vv in v.items():
                if leveltype == "height" and len(vv) > 1:
                    raise ValueError(
                        f"A variable {cfname} with height leveltype should only have one level"
                    )
                v[leveltype] = sorted(vv)
        # air_temperature -> pressure -> [1000, 925, 800, 700]

        # Determine unique dimensions to add
        dims_to_add = {}  # height1 -> [height, [2]]
        ncname_to_level_dim = {}
        for cfname, v in cfname_to_levels.items():
            for leveltype, levels in v.items():
                ncname = self.conventions.get_ncname(cfname, leveltype, levels[0])
                dimname = self.conventions.get_name(leveltype)

                if (leveltype, levels) in dims_to_add.values():
                    # Reuse an existing dimension
                    i = list(dims_to_add.values()).index((leveltype, levels))
                    dimname = list(dims_to_add.keys())[i]
                else:
                    count = 0
                    for curr_leveltype, _ in dims_to_add.values():
                        if curr_leveltype == leveltype:
                            count += 1
                    if count == 0:
                        pass  # height
                    else:
                        dimname = f"{dimname}{count}"  # height1
                dims_to_add[dimname] = (leveltype, levels)
                ncname_to_level_dim[ncname] = dimname
        return dims_to_add, ncname_to_level_dim

    def get_level_dimname(self, ncname) -> dict[str, str]|None:
        """Get the name of the level dimension for given NetCDF variable"""
        if ncname not in self._ncname_to_level_dim:
            return None
        return self._ncname_to_level_dim[ncname]

    def get_level_index(self, anemoi_name) -> int|None:
        """Get the index into the level dimension that this anemoi variable belongs to"""
        # Determine what ncname and index each variable belongs to
        metadata = cf.get_metadata(anemoi_name)

        # Find the name of the level dimension
        ncname = self.get_ncname_from_anemoi_name(anemoi_name)
        if ncname not in self._ncname_to_level_dim:
            return None
        dimname = self._ncname_to_level_dim[ncname]

        # Find the index in this dimension
        level = metadata["level"]
        if level is None:
            return None
        index = self.dimensions[dimname][1].index(level)
        return index

    def get_ncname_from_anemoi_name(self, anemoi_name) -> str:
        """Get the NetCDF variable name corresponding to this anemoi variable name"""
        # Determine what ncname and index each variable belongs to
        metadata = cf.get_metadata(anemoi_name)
        ncname = self.conventions.get_ncname(**metadata)
        return ncname
