import gridpp
import numpy as np
import xarray as xr
from bris import utils
from bris.conventions import cf
from bris.conventions.metno import Metno
from bris.outputs import Output
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata
from bris import projections


class Netcdf(Output):
    """Write predictions to NetCDF, using CF-standards and local conventions

    Since ensemble is done data-parallel, we do not have all members available when writing the
    files. If we are producing a single deterministic run, then we can directly write data to file
    as soon as we get it. Otherwise write the data to disk in an intermediate format and then merge
    files on finalize. This comes at a penalty since the data is written to disk twice.

    This output can write three types of outputs:
    1) Gridded regional projected data. This requires field_shape to be set in predict_metadata
    2) Irregular grids interpolated to a lat/lon grid. Use interp_res.
    3) Irregular grids. This means the output only has one location dimension.
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename_pattern: str,
        variables=None,
        interp_res=None,
        latrange=None,
        lonrange=None,
        extra_variables=list(),
        proj4_str=None,
        domain_name=None,
        mask_file=None,
        mask_field=None,
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
            self.extract_variables = [i for i in variables]

        self.intermediate = None
        if self.pm.num_members > 1:
            self.intermediate = Intermediate(predict_metadata, workdir)

        self.variable_list = VariableList(self.extract_variables)

        # Conventions specify the names of variables in the output
        # CF-standard names are added in the standard_name attributes
        self.conventions = Metno()
        self.interp_res = interp_res
        self.latrange = latrange
        self.lonrange = lonrange
        self.mask_file = mask_file

        if domain_name is not None:
            self.proj4_str = projections.get_proj4_str(domain_name)
        else:
            self.proj4_str = proj4_str

        if self._is_masked:
            # If a mask was used during training:
            # Compute 1D->2D index to output 2D arrays by using a mask file
            self.ds_mask = xr.open_dataset(mask_file,drop_variables=['time', 'projection_stere'])
            # TODO check if mask has time dimension before using isel
            mask = self.ds_mask.isel(time=0)[mask_field].values
            ind = np.arange(np.size(mask)).reshape(np.shape(mask))
            # Assume mask==1 where have values
            self.indices_1D_to_2D = ind[mask==1.0] 
            
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
    def _is_masked(self):
        """Was a mask_from_dataset applied during training?"""
        return self.mask_file is not None

    @property
    def _is_gridded(self):
        """Is the output gridded?"""
        return len(self.pm.field_shape) == 2 or self.interp_res is not None

    @property
    def _interpolate(self):
        """Should interpolation to a regular lat/lon grid be performed?"""
        return self.interp_res is not None

    def write(self, filename: str, times: list, pred: np.array):
        """Write prediction to NetCDF
        Args:
            times: List of np.datetime64 objects that this forecast is for
            pred: 4D numpy array with dimensions (leadtimes, points, variables, members)
        """

        coords = dict()

        # Function to easily convert from cf names to conventions
        def c(x):
            return self.conventions.get_name(x)

        # TODO: Seconds or hours for leadtimes?
        times_ut = utils.datetime_to_unixtime(times)
        frt_ut = times_ut[0]
        coords[c("time")] = np.array(times_ut).astype(np.double)

        if self._is_gridded:
            if self._interpolate:
                # Find a bounding-box for interpolation
                min_lat = self.get_lower(self.pm.lats)
                max_lat = self.get_upper(self.pm.lats)
                min_lon = self.get_lower(self.pm.lons)
                max_lon = self.get_upper(self.pm.lons)
                if self.latrange is not None:
                    min_lat, max_lat = self.latrange
                if self.lonrange is not None:
                    min_lon, max_lon = self.lonrange

                y = np.arange(
                    min_lat,
                    max_lat + self.interp_res,
                    self.interp_res,
                )
                x = np.arange(
                    min_lon,
                    max_lon + self.interp_res,
                    self.interp_res,
                )
                x_dim_name = c("longitude")
                y_dim_name = c("latitude")
            else:
                # TODO: Handle self.latrange and self.lonrange
                if None not in [self.latrange, self.lonrange]:
                    print("Warning: latrange/lonrange not handled in gridded fields")

                if self.proj4_str:
                    lats = np.reshape(self.pm.lats, self.pm.field_shape).astype(np.double)
                    lons = np.reshape(self.pm.lons, self.pm.field_shape).astype(np.double)
                    x, y = projections.get_xy(lats, lons, self.proj4_str)
                else:
                    x = np.arange(self.pm.field_shape[1]).astype(np.float32)
                    y = np.arange(self.pm.field_shape[0]).astype(np.float32)
                x_dim_name = c("projection_x_coordinate")
                y_dim_name = c("projection_y_coordinate")
            coords[x_dim_name] = x
            coords[y_dim_name] = y
            spatial_dims = (y_dim_name, x_dim_name)
        else:
            if self._is_masked:
                # Use the template to get the (full) grid
                x    = self.ds_mask.X.values
                y    = self.ds_mask.Y.values
                x_dim_name = c("projection_x_coordinate")
                y_dim_name = c("projection_y_coordinate")
                spatial_dims = (y_dim_name, x_dim_name)
            else:
                y = np.arange(len(self.pm.lats)).astype(np.int32)
                coords["location"] = y
                spatial_dims = ("location",)

        if self.pm.num_members > 1:
            coords[c("realization")] = np.arange(self.pm.num_members).astype(np.int32)

        dims_to_add = self.variable_list.dimensions

        attrs = dict()
        # Add dimensions
        for dimname, (level_type, levels) in dims_to_add.items():
            # Don't need to convert dimnames, since these are already to local convention
            coords[dimname] = np.array(levels).astype(np.float32)
            attrs[dimname] = cf.get_attributes(level_type)

        self.ds = xr.Dataset(coords=coords)

        # Add attributes of coordinates
        for var, var_attrs in attrs.items():
            self.ds[var].attrs = var_attrs

        # Set up other coordinate variables
        self.ds[c("forecast_reference_time")] = ([], frt_ut)

        # Set up grid definitions
        if self._is_gridded:
            if self._interpolate:
                proj_attrs = dict()
                proj_attrs["grid_mapping_name"] = "latitude_longitude"
                proj_attrs["earth_radius"] = 6371000.0
                self.ds["projection"] = ([], 1, proj_attrs)
            else:
                lats = self.pm.grid_lats.astype(np.double)
                lons = self.pm.grid_lons.astype(np.double)
                self.ds[c("latitude")] = (
                    spatial_dims,
                    lats,
                )
                self.ds[c("longitude")] = (
                    spatial_dims,
                    lons,
                )

                if self.pm.altitudes is not None:
                    altitudes = self.pm.grid_altitudes.astype(np.double)
                    self.ds[c("surface_altitude")] = (
                        spatial_dims,
                        altitudes
                    )
                proj_attrs = dict()
                if self.proj4_str is not None:
                    proj_attrs = projections.get_proj_attributes(self.proj4_str)
                    # proj_attrs["grid_mapping_name"] = "lambert_conformal_conic"
                    # proj_attrs["standard_parallel"] = (63.3, 63.3)
                    # proj_attrs["longitude_of_central_meridian"] = 15.0
                    # proj_attrs["latitude_of_projection_origin"] = 63.3
                    # proj_attrs["earth_radius"] = 6371000.0
                self.ds[c("projection")] = ([], 0, proj_attrs)
        else:
            if self._is_masked:
                self.ds[c("latitude")] = (
                    spatial_dims,
                    self.ds_mask.lat.values,
                )
                self.ds[c("longitude")] = (
                    spatial_dims,
                    self.ds_mask.lon.values,
                )

            else: 
                self.ds[c("latitude")] = (
                    spatial_dims,
                    self.pm.lats,
                )
                self.ds[c("longitude")] = (
                    spatial_dims,
                    self.pm.lons,
                )
            if self.pm.altitudes is not None:
                self.ds[c("surface_altitude")] = (
                    spatial_dims,
                    self.pm.altitudes
                )

        for cfname in [
            "forecast_reference_time",
            "time",
            "latitude",
            "longitude",
            "sufrace_altitude",
            "projection_x_coordinate",
            "projection_y_coordinate",
            "realization",
        ]:
            ncname = c(cfname)
            if ncname in self.ds:
                self.ds[ncname].attrs = cf.get_attributes(cfname)

        # Set up all prediction variables
        for variable_index, variable in enumerate(self.pm.variables):
            level_index = self.variable_list.get_level_index(variable)
            ncname = self.variable_list.get_ncname_from_anemoi_name(variable)

            if ncname not in self.ds:
                dim_name = self.variable_list.get_level_dimname(ncname)
                if dim_name is not None:
                    dims = [
                        c("time"),
                        dim_name,
                        *spatial_dims,
                    ]
                    if self._is_gridded or self._is_masked:
                        shape = [len(times), len(self.ds[dim_name]), len(y), len(x)]
                    else:
                        shape = [len(times), len(self.ds[dim_name]), len(y)]
                else:
                    dims = [c("time"), *spatial_dims]
                    if self._is_gridded or self._is_masked:
                        shape = [len(times), len(y), len(x)]
                    else:
                        shape = [len(times), len(y)]

                if self.pm.num_members > 1:
                    dims.insert(len(shape) - 2, c("ensemble_member"))
                    shape.insert(len(shape) - 2, self.pm.num_members)

                ar = np.nan * np.zeros(shape, np.float32)
                self.ds[ncname] = (dims, ar)

            if self._is_gridded or self._is_masked:
                shape = [len(times), len(y), len(x), self.pm.num_members]
            else:
                shape = [len(times), len(y), self.pm.num_members]

            if self._interpolate:
                ipoints = gridpp.Points(self.pm.lats, self.pm.lons)
                yy, xx = np.meshgrid(y, x)
                ogrid = gridpp.Grid(yy.transpose(), xx.transpose())

                curr = pred[..., variable_index, :]
                ar = np.nan * np.zeros(
                    [len(times), len(y), len(x), self.pm.num_members], np.float32
                )
                for i in range(self.pm.num_members):
                    ar[:, :, :, i] = gridpp.nearest(ipoints, ogrid, curr[:, :, i])
            elif self._is_masked: 
                curr = pred[..., variable_index, :]
                ar = np.nan * np.zeros(
                    [len(times), len(y), len(x), self.pm.num_members], np.float32
                )
                # Reconstruct the 2D array (nans where no data)
                ar[:, :, :, :].flat[self.indices_1D_to_2D] = curr[:, :, :]
            else:
                ar = np.reshape(pred[..., variable_index, :], shape)

            if self.pm.num_members > 1:
                # Move ensemble dimension into the middle position
                ar = np.moveaxis(ar, [-1], [1])
            else:
                # Remove ensemble dimension
                ar = ar[..., 0]

            if level_index is not None:
                self.ds[ncname][:, level_index, ...] = ar
            else:
                self.ds[ncname][:] = ar

            # Add variable attributes
            cfname = cf.get_metadata(variable)["cfname"]
            attrs = cf.get_attributes(cfname)
            attrs["grid_mapping"] = "projection"
            attrs["coordinates"] = "latitude longitude"
            self.ds[ncname].attrs = attrs

        utils.create_directory(filename)
        self.ds.to_netcdf(filename)

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
