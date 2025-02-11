import sys
import os
from datetime import datetime, timedelta

import eccodes as ecc
import gridpp
import numpy as np
import xarray as xr
from bris import utils
from bris.conventions import cf
from bris.conventions.metno import Metno
from bris.outputs import Output
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata


class Grib(Output):
    """Write predictions to Grib, using CF-standards and local conventions"""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename_pattern: str,
        variables=None,
        interp_res=None,
    ):
        """
        Args:
            filename_pattern: Save predictions to this filename after time tokens are expanded
            interp_res: Interpolate to this resolution [degrees] on a lat/lon grid
            variables: If None, predict all variables
        """
        super().__init__(predict_metadata)

        self.filename_pattern = filename_pattern
        if variables is None:
            self.extract_variables = predict_metadata.variables
        else:
            self.extract_variables = variables

        self.intermediate = None
        if self.pm.num_members > 1:
            self.intermediate = Intermediate(predict_metadata, workdir)

        self.variable_list = VariableList(self.extract_variables)

        # Conventions specify the names of variables in the output
        # CF-standard names are added in the standard_name attributes
        self.conventions = Metno()
        self.interp_res = interp_res

    def _add_forecast(self, times: list, ensemble_member: int, pred: np.array):
        if self.pm.num_members > 1:
            # Cache data with intermediate
            self.intermediate.add_forecast(times, ensemble_member, pred)
            return
        else:
            assert ensemble_member == 0

            forecast_reference_time = times[0].astype("datetime64[s]").astype("int")

            filename = self.get_filename(forecast_reference_time)

            # Add ensemble dimension to the last
            self.write(filename, times, pred[..., None])

    def get_filename(self, forecast_reference_time):
        return utils.expand_time_tokens(self.filename_pattern, forecast_reference_time)

    @property
    def _is_gridded(self):
        """Is the output gridded?"""
        return len(self.pm.field_shape) == 2 or self.interp_res is not None

    @property
    def _interpolate(self):
        return self.interp_res is None

    def write(self, filename: str, times: list, pred: np.array):

        # remove the ensemble dimension for now
        pred = pred.squeeze()

        with open(filename, "wb") as file_handle:
            print(f"Write to: {filename}")
            for time_index, numpy_dt in enumerate(times):
                dt = numpy_dt.astype(datetime)
                for variable_index in range(pred.shape[2]):
                    variable = self.pm.variables[variable_index]
                    ncname = self.variable_list.get_ncname_from_anemoi_name(variable)
                    metadata = cf.get_metadata(variable)

                    self.convert_to_grib(
                            file_handle,
                            dt,
                            metadata.get("level", 0),
                            metadata.get("leveltype", "height"),
                            ncname,
                            self.pm.field_shape[1],
                            self.pm.field_shape[0],
                            pred[time_index, :, variable_index]
                    )


    def level_type_to_id(self, level_type):
        # Map level type name to code
        # https://codes.ecmwf.int/grib/format/grib2/ctables/4/5/
        return {
            "pressure": 100,  # Isobaric surface (Pa)
            "height_above_msl": 102,  # Specific altitude above mean sea level (m)
            "height": 103,  # Specified height level above ground (m)
            "height1": 103,
            "height2": 103,
            "height6": 103,
            "height7": 103,
        }.get(level_type, 103)


    def param_to_id(self, param):
        # Map parameter name to:
        # (product definition template number, discipline, parameter category, parameter number, type of statistical processing)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-0.shtml
        return {
            "air_temperature_0m":                   (0, 0, 0, 0, None),
            "air_temperature_2m":                   (0, 0, 0, 0, None),
            "air_temperature_pl":                   (0, 0, 0, 0, None),
            "geopotential_pl":                      (0, 0, 3, 4, None),
            "relative_humidity_2m":                 (0, 0, 1, 192, None),
            "relative_humidity_pl":                 (0, 0, 1, 192, None),
            "x_wind_10m":                           (0, 0, 2, 2, None),
            "x_wind_pl":                            (0, 0, 2, 2, None),
            "y_wind_10m":                           (0, 0, 2, 3, None),
            "y_wind_pl":                            (0, 0, 2, 3, None),
            "atmosphere_boundary_layer_thickness":  (0, 0, 19, 3, None),
            "wind_speed_of_gust":                   (8, 0, 2, 22, 2),
            "air_pressure_at_sea_level":            (0, 0, 3, 0, None),
            "surface_air_pressure":                 (0, 0, 3, 0, None),
            "specific_humidity_pl":                 (0, 0, 1, 0, None),
            "upward_air_velocity_pl":               (0, 0, 2, 9, None),
            "dew_point_temperature_2m":             (0, 0, 2, 9, None),
        }.get(param)


    def convert_to_grib(self, fp, validtime, level_value, level_type, parameter, nx, ny, values):
        pdtn, dis, cat, num, tosp = self.param_to_id(parameter)

        year = int(validtime.strftime("%Y"))
        month = int(validtime.strftime("%m"))
        day = int(validtime.strftime("%d"))
        hour = int(validtime.strftime("%H"))

        grib = ecc.codes_grib_new_from_samples("GRIB2")
        ecc.codes_set(grib, "tablesVersion", 21)
        ecc.codes_set(grib, "generatingProcessIdentifier", 5)
        ecc.codes_set(grib, "centre", 86)  # Reserved for other centres
        ecc.codes_set(grib, "subCentre", 255)  # Consensus
        ecc.codes_set(grib, "gridDefinitionTemplateNumber", 30) # Lambert conformal (Can be secant or tangent, conical or bipolar)
        ecc.codes_set(grib, "latitudeOfSouthernPole", -90000000)
        ecc.codes_set(grib, "longitudeOfSouthernPole", 0)

        ecc.codes_set(grib, "latitudeOfFirstGridPoint", 50319616)
        ecc.codes_set(grib, "longitudeOfFirstGridPoint", 278280)
        # Latitude where Dx and Dy are specified
        ecc.codes_set(grib, "LaD", 63300000)
        # is the longitude value of the meridian which is parallel to the Y-axis (or columns of the grid) along which latitude increases as the Y-coordinate increases (the orientation longitude may or may not appear on a particular grid).
        ecc.codes_set(grib, "LoV", 15000000)
        ecc.codes_set(grib, "DxInMetres", 10000)
        ecc.codes_set(grib, "DyInMetres", 10000)
        #ecc.codes_set(grib, "Dx", 10000)  # X-direction grid length
        #ecc.codes_set(grib, "Dy", 10000)  # Y-direction grid length
        ecc.codes_set(grib, "Nx", nx)
        ecc.codes_set(grib, "Ny", ny)
        # first/second latitude from the pole at which the secant cone cuts the sphere
        ecc.codes_set(grib, "Latin1", 63300000)
        ecc.codes_set(grib, "Latin2", 63300000)
        ecc.codes_set(grib, "resolutionAndComponentFlags", 56)
        ecc.codes_set(grib, "discipline", dis)
        ecc.codes_set(grib, "parameterCategory", cat)
        ecc.codes_set(grib, "parameterNumber", num)
        ecc.codes_set(grib, "significanceOfReferenceTime", 1) # Start of forecast ( https://codes.ecmwf.int/grib/format/grib2/ctables/1/2/ )
        ecc.codes_set(grib, "typeOfProcessedData", 2) # Analysis and Forecast Products ( https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table1-4.shtml )
        ecc.codes_set(grib, "productionStatusOfProcessedData", 0)  # Operational Products
        ecc.codes_set(grib, "shapeOfTheEarth", 6) # Earth assumed spherical with radius = 6,371,229.0 m
        ecc.codes_set(grib, "productDefinitionTemplateNumber", pdtn) # tämä on 0 jos suure on ei-aggregoitu ja 8 jos suure on aggregoitu (esim sade, puuska, en tiedä onko näitä edes bris ouputissa)
        ecc.codes_set(grib, "scanningMode", 64) 
        ecc.codes_set(grib, "NV", 132) # Number of coordinate values after template or number of information according to 3D vertical coordinate GRIB2 message
        ecc.codes_set(grib, "typeOfGeneratingProcess", 2) # Ensemble forecast == 4 , 2 == Forecast ( https://codes.ecmwf.int/grib/format/grib2/ctables/4/3/ )
        ecc.codes_set(grib, "indicatorOfUnitOfTimeRange", 1)
        ecc.codes_set(grib, "typeOfFirstFixedSurface", self.level_type_to_id(level_type))
        ecc.codes_set(grib, "level", level_value)  # TODO: set from data -> 0 == surface, q_850 == pressure_level(850)
        ecc.codes_set(grib, "year", year)
        ecc.codes_set(grib, "month", month)
        ecc.codes_set(grib, "day", day)
        ecc.codes_set(grib, "hour", hour)
        ecc.codes_set(grib, "forecastTime", 0)
        ecc.codes_set(grib, "bitsPerValue", 24)
        ecc.codes_set(grib, "packingType", "grid_ccsds")
        ecc.codes_set_values(grib, values)

        # Average, accumulation, extreme values or other statistically processed values at a horizontal level or in a horizontal layer in a continuous or non-continuous time interval.
        if pdtn == 8:
            ecc.codes_set(grib, "lengthOfTimeRange", 1)
            ecc.codes_set(grib, "typeOfStatisticalProcessing", tosp)
            ecc.codes_set(grib, "yearOfEndOfOverallTimeInterval", year)
            ecc.codes_set(grib, "monthOfEndOfOverallTimeInterval", month)
            ecc.codes_set(grib, "dayOfEndOfOverallTimeInterval", day)
            ecc.codes_set(grib, "hourOfEndOfOverallTimeInterval", hour)

        ecc.codes_write(grib, fp)
        ecc.codes_release(grib)


    def finalize(self):
        if self.intermediate is not None:
            # Load data from the intermediate and write to disk
            forecast_reference_times = self.intermediate.get_time_sets()
            for forecast_reference_time in forecast_reference_times:
                # Arange all ensemble members
                pred = np.zeros(self.pm.shape + [self.pm.num_members], np.float32)
                for m in range(self.pm.num_members):
                    curr = self.intermediate.get_forecast(forecast_reference_time, m)
                    if curr is not None:
                        pred[..., m] = curr

                filename = self.get_filename(forecast_reference_time)
                self.write(filename, forecast_reference_time, pred)

    def get_lower(self, array):
        m = np.min(array)
        return np.floor(m / self.interp_res) * self.interp_res

    def get_upper(self, array):
        m = np.max(array)
        return np.ceil(m / self.interp_res) * self.interp_res


class VariableList:
    """This class keeps track of levels are available for each cf-variables
    and determines a unique name of the level dimension, if there are multiple definitions of the
    same dimension (e.g. two variables with a different set of pressure levels)
    """

    def __init__(self, anemoi_names: list, conventions=Metno()):
        """Args:
        anemoi_names: A list of variables names used in Anemoi (e.g. u10)
        conventions: What NetCDF naming convention to use
        """
        self.anemoi_names = anemoi_names
        self.conventions = conventions

        self._dimensions, self._ncname_to_level_dim = self.load_dimensions()

    @property
    def dimensions(self):
        """A diction of dimension names needed to represent the variable list

        The key is the dimension name, the value is a tuple of (leveltype, levels)
        E.g. for a dataset with 2m temperature: {height1: (height, 2)}
        """
        return self._dimensions

    def load_dimensions(self):
        cfname_to_levels = dict()

        for v, variable in enumerate(self.anemoi_names):
            metadata = cf.get_metadata(variable)
            cfname = metadata["cfname"]
            leveltype = metadata["leveltype"]
            level = metadata["level"]

            if leveltype is None:
                # This variable (likely a forcing parameter) does not need a level dimension
                continue

            if cfname not in cfname_to_levels:
                cfname_to_levels[cfname] = dict()
            if leveltype not in cfname_to_levels[cfname]:
                cfname_to_levels[cfname][leveltype] = list()
            cfname_to_levels[cfname][leveltype] += [level]
        # Sort levels
        for cfname, v in cfname_to_levels.items():
            for leveltype, vv in v.items():
                if leveltype == "height" and len(vv) > 1:
                    raise Exception(
                        f"A variable {cfname} with height leveltype should only have one level"
                    )
                v[leveltype] = sorted(vv)
        # air_temperature -> pressure -> [1000, 925, 800, 700]

        # Determine unique dimensions to add
        dims_to_add = dict()  # height1 -> [height, [2]]
        ncname_to_level_dim = dict()
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

    def get_level_dimname(self, ncname):
        """Get the name of the level dimension for given NetCDF variable"""
        if ncname not in self._ncname_to_level_dim:
            return None
        return self._ncname_to_level_dim[ncname]

    def get_level_index(self, anemoi_name):
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

    def get_ncname_from_anemoi_name(self, anemoi_name):
        """Get the NetCDF variable name corresponding to this anemoi variable name"""
        # Determine what ncname and index each variable belongs to
        metadata = cf.get_metadata(anemoi_name)
        ncname = self.conventions.get_ncname(**metadata)
        return ncname
