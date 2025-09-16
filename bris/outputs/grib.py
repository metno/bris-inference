from datetime import datetime

import eccodes as ecc
import numpy as np

from bris import projections, utils
from bris.conventions import cf
from bris.conventions.metno import Metno
from bris.conventions.variable_list import VariableList
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
        grib_keys: dict = None,
        proj4_str=None,
        domain_name=None,
        extra_variables=None,
        remove_intermediate: bool = True,
    ):
        """
        Args:
            filename_pattern: Save predictions to this filename after time tokens are expanded
            interp_res: Interpolate to this resolution [degrees] on a lat/lon grid
            variables: If None, predict all variables
        """
        super().__init__(predict_metadata, extra_variables)

        self.filename_pattern = filename_pattern
        if variables is None:
            self.extract_variables = predict_metadata.variables
        else:
            if extra_variables is None:
                extra_variables = []
            self.extract_variables = variables + extra_variables

        self.intermediate = None
        if self.pm.num_members > 1:
            self.intermediate = Intermediate(predict_metadata, workdir, extra_variables)
        self.remove_intermediate = remove_intermediate
        self.variable_list = VariableList(self.extract_variables)

        # Conventions specify the names of variables in the output
        # CF-standard names are added in the standard_name attributes
        self.conventions = Metno()
        self.interp_res = interp_res
        self.grib_keys = grib_keys or {}

        if domain_name is not None:
            self.proj4_str = projections.get_proj4_str(domain_name)
        else:
            self.proj4_str = proj4_str

    def _add_forecast(self, times: list, ensemble_member: int, pred: np.array):
        if self.pm.num_members > 1:
            # Cache data with intermediate
            self.intermediate.add_forecast(times, ensemble_member, pred)
            return

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
        forecast_reference_time = times[0].astype(datetime)

        with open(filename, "wb") as file_handle:
            for time_index, numpy_dt in enumerate(times):
                dt = numpy_dt.astype(datetime)
                for variable in self.extract_variables:
                    variable_index = self.pm.variables.index(variable)
                    ncname = self.variable_list.get_ncname_from_anemoi_name(variable)
                    metadata = cf.get_metadata(variable)

                    # only one location dimension
                    if len(self.pm.field_shape) == 1:
                        ny = self.pm.field_shape[0]
                        nx = 1
                    else:
                        ny = self.pm.field_shape[0]
                        nx = self.pm.field_shape[1]
                    for ens in range(self.pm.num_members):
                        self.convert_to_grib(
                            file_handle,
                            forecast_reference_time,
                            dt,
                            metadata.get("level", 0) or 0,
                            metadata.get("leveltype", "height") or "surface",
                            ncname,
                            nx,
                            ny,
                            pred[time_index, :, variable_index, ens],
                            ensemble_member=ens,
                        )

    def level_type_to_id(self, level_type):
        # Map level type name to code
        # https://codes.ecmwf.int/grib/format/grib2/ctables/4/5/
        return {
            "surface": 1,
            "air_pressure": 100,  # Isobaric surface (Pa)
            "height_above_msl": 102,  # Specific altitude above mean sea level (m)
            "height": 103,  # Specified height level above ground (m)
            "height1": 103,
            "height2": 103,
            "height6": 103,
            "height7": 103,
        }.get(level_type, 103)

    def grib_definition_to_template_number(self, definition):
        return {
            "latlon": 0,
            "rotated_ll": 1,
            "mercator": 10,
            "polar_stereographic": 20,
            "lambert_conformal_conic": 30,
            "gaussian": 40,
            "rotated_gaussian": 41,
            "spherical_harmonic": 50,
        }.get(definition, 30)

    def param_to_id(self, param):
        # Map parameter name to:
        # (product definition template number, discipline, parameter category, parameter number, type of statistical processing)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-0.shtml
        return {
            "air_temperature_0m": (0, 0, 0, 0, None),
            "air_temperature_2m": (0, 0, 0, 0, None),
            "air_temperature_pl": (0, 0, 0, 0, None),
            "geopotential_pl": (0, 0, 3, 4, None),
            "relative_humidity_2m": (0, 0, 1, 192, None),
            "relative_humidity_pl": (0, 0, 1, 192, None),
            "x_wind_10m": (0, 0, 2, 2, None),
            "x_wind_pl": (0, 0, 2, 2, None),
            "y_wind_10m": (0, 0, 2, 3, None),
            "y_wind_pl": (0, 0, 2, 3, None),
            "wind_speed_10m": (0, 0, 2, 1, None),
            "atmosphere_boundary_layer_thickness": (0, 0, 19, 3, None),
            "wind_speed_of_gust": (8, 0, 2, 22, 2),
            "air_pressure_at_sea_level": (0, 0, 3, 0, None),
            "surface_air_pressure": (0, 0, 3, 0, None),
            "specific_humidity_pl": (0, 0, 1, 0, None),
            "upward_air_velocity_pl": (0, 0, 2, 9, None),
            "dew_point_temperature_2m": (0, 0, 0, 6, None),
            "vertical_velocity_pl": (0, 0, 2, 8, None),
            "tcw": (0, 0, 1, 3, None),
            "skt": (0, 0, 0, 17, None),
            "precipitation_amount": (8, 0, 1, 49, None),
            "cp": (0, 0, 1, 10, None),
        }.get(param)

    def param_to_id_ens(self, param):
        # Map parameter name to:
        # (product definition template number, discipline, parameter category, parameter number, type of statistical processing)
        # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-0.shtml
        return {
            "air_temperature_0m": (11, 0, 0, 0, None),
            "air_temperature_2m": (11, 0, 0, 0, None),
            "air_temperature_pl": (11, 0, 0, 0, None),
            "geopotential_pl": (11, 0, 3, 4, None),
            "relative_humidity_2m": (11, 0, 1, 192, None),
            "relative_humidity_pl": (11, 0, 1, 192, None),
            "x_wind_10m": (11, 0, 2, 2, None),
            "x_wind_pl": (11, 0, 2, 2, None),
            "y_wind_10m": (11, 0, 2, 3, None),
            "y_wind_pl": (11, 0, 2, 3, None),
            "wind_speed_10m": (11, 0, 2, 1, None),
            "atmosphere_boundary_layer_thickness": (11, 0, 19, 3, None),
            "wind_speed_of_gust": (11, 0, 2, 22, 2),
            "air_pressure_at_sea_level": (1, 0, 3, 0, None),
            "surface_air_pressure": (11, 0, 3, 0, None),
            "specific_humidity_pl": (11, 0, 1, 0, None),
            "upward_air_velocity_pl": (11, 0, 2, 9, None),
            "dew_point_temperature_2m": (11, 0, 0, 6, None),
            "vertical_velocity_pl": (11, 0, 2, 8, None),
            "tcw": (11, 0, 1, 3, None),
            "skt": (11, 0, 0, 17, None),
            "precipitation_amount": (11, 0, 1, 49, None),
            "cp": (11, 0, 1, 10, None),
        }.get(param)

    def set_geometry(self, grib, nx, ny):
        # get Dx/Dy in meters
        dx = None
        dy = None
        if self.proj4_str:
            lats = np.reshape(self.pm.lats, self.pm.field_shape).astype(np.double)
            lons = np.reshape(self.pm.lons, self.pm.field_shape).astype(np.double)
            x, y = projections.get_xy(lats, lons, self.proj4_str)
            dx = int(x[1] - x[0])
            dy = int(y[1] - y[0])

        # set geometry from proj attributes
        attrs = {}
        if self.proj4_str:
            attrs = projections.get_proj_attributes(self.proj4_str)

        ecc.codes_set(
            grib,
            "gridDefinitionTemplateNumber",
            self.grib_definition_to_template_number(attrs.get("grid_mapping_name")),
        )
        ecc.codes_set(grib, "latitudeOfSouthernPole", -90000000)
        ecc.codes_set(grib, "longitudeOfSouthernPole", 0)
        ecc.codes_set(
            grib, "latitudeOfFirstGridPoint", int(self.pm.lats[0] * 1_000_000)
        )
        ecc.codes_set(
            grib, "longitudeOfFirstGridPoint", int(self.pm.lons[0] * 1_000_000)
        )
        ecc.codes_set(
            grib, "LaD", attrs.get("latitude_of_projection_origin", 63.3) * 1_000_000
        )
        ecc.codes_set(
            grib, "LoV", attrs.get("longitude_of_central_meridian", 15.0) * 1_000_000
        )
        if dx and dy:
            ecc.codes_set(grib, "DxInMetres", dx)
            ecc.codes_set(grib, "DyInMetres", dy)
        ecc.codes_set(grib, "Nx", nx)
        ecc.codes_set(grib, "Ny", ny)
        ecc.codes_set(grib, "Latin1", 63300000)
        ecc.codes_set(grib, "Latin2", 63300000)
        ecc.codes_set(grib, "shapeOfTheEarth", 6)

        return grib

    def convert_to_grib(
        self,
        fp,
        origintime,
        validtime,
        level_value,
        level_type,
        parameter,
        nx,
        ny,
        values,
        ensemble_member=None,
    ):
        if self.pm.num_members > 1:
            pdtn, dis, cat, num, tosp = self.param_to_id_ens(parameter)
        else:
            pdtn, dis, cat, num, tosp = self.param_to_id(parameter)

        leadtime = validtime - origintime

        grib = ecc.codes_grib_new_from_samples("GRIB2")
        grib = self.set_geometry(grib, nx, ny)

        ecc.codes_set(grib, "tablesVersion", 21)
        ecc.codes_set(grib, "resolutionAndComponentFlags", 56)
        ecc.codes_set(grib, "discipline", dis)
        ecc.codes_set(grib, "parameterCategory", cat)
        ecc.codes_set(grib, "parameterNumber", num)
        ecc.codes_set(grib, "significanceOfReferenceTime", 1)
        ecc.codes_set(grib, "typeOfProcessedData", 2)
        ecc.codes_set(grib, "productionStatusOfProcessedData", 0)
        ecc.codes_set(grib, "productDefinitionTemplateNumber", pdtn)
        ecc.codes_set(grib, "scanningMode", 64)
        ecc.codes_set(grib, "typeOfGeneratingProcess", 2)
        ecc.codes_set(grib, "indicatorOfUnitOfTimeRange", 1)
        ecc.codes_set(
            grib, "typeOfFirstFixedSurface", self.level_type_to_id(level_type)
        )
        ecc.codes_set(grib, "level", level_value)
        ecc.codes_set(grib, "dataDate", int(origintime.strftime("%Y%m%d")))
        ecc.codes_set(grib, "dataTime", int(origintime.strftime("%H%M")))
        ecc.codes_set(grib, "forecastTime", int(leadtime.total_seconds() / 3600))
        ecc.codes_set(grib, "bitsPerValue", 24)
        ecc.codes_set(grib, "packingType", "grid_ccsds")
        ecc.codes_set_values(grib, values)

        if (
            ensemble_member is not None
            and self.pm.num_members > 1
            and pdtn in [1, 2, 11]
        ):
            ecc.codes_set(grib, "typeOfEnsembleForecast", 1)
            ecc.codes_set(grib, "perturbationNumber", ensemble_member + 1)
            ecc.codes_set(grib, "numberOfForecastsInEnsemble", self.pm.num_members)

        if tosp == 1:
            year = int(validtime.strftime("%Y"))
            month = int(validtime.strftime("%m"))
            day = int(validtime.strftime("%d"))
            hour = int(validtime.strftime("%H"))

            ecc.codes_set(grib, "typeOfStatisticalProcessing", tosp)
            ecc.codes_set(grib, "yearOfEndOfOverallTimeInterval", year)
            ecc.codes_set(grib, "monthOfEndOfOverallTimeInterval", month)
            ecc.codes_set(grib, "dayOfEndOfOverallTimeInterval", day)
            ecc.codes_set(grib, "hourOfEndOfOverallTimeInterval", hour)

            # forecastTime is start of time interval
            # hard code to 6h
            lengthOfTimeRange = 6
            ecc.codes_set(
                grib,
                "forecastTime",
                int(leadtime.total_seconds() / 3600) - lengthOfTimeRange,
            )
            ecc.codes_set(grib, "lengthOfTimeRange", lengthOfTimeRange)

        for key, val in self.grib_keys.items():
            ecc.codes_set(grib, key, val)

        ecc.codes_write(grib, fp)
        ecc.codes_release(grib)

    async def finalize(self):
        if self.intermediate is not None:
            # Load data from the intermediate and write to disk
            forecast_reference_times = self.intermediate.get_forecast_reference_times()
            for forecast_reference_time in forecast_reference_times:
                # Arange all ensemble members
                pred = np.zeros(self.pm.shape + [self.pm.num_members], np.float32)
                for m in range(self.pm.num_members):
                    curr = self.intermediate.get_forecast(forecast_reference_time, m)
                    if curr is not None:
                        pred[..., m] = curr

                time = forecast_reference_time.astype("datetime64[s]").astype("int")
                filename = self.get_filename(time)
                lead_times = [
                    forecast_reference_time + lt
                    for lt in self.intermediate.pm.leadtimes
                ]
                self.write(filename, lead_times, pred)

            if self.remove_intermediate:
                self.intermediate.cleanup()

    def get_lower(self, array):
        m = np.min(array)
        return np.floor(m / self.interp_res) * self.interp_res

    def get_upper(self, array):
        m = np.max(array)
        return np.ceil(m / self.interp_res) * self.interp_res
