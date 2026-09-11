import datetime
import time as pytime
from collections.abc import Callable
from functools import cached_property

import gridpp
import netCDF4
import numpy as np
import xarray as xr

import bris.units
from bris import projections, utils
from bris.conventions import anemoi as anemoi_conventions
from bris.conventions import cf
from bris.conventions.metno import Metno
from bris.conventions.variable_list import VariableList
from bris.outputs import Output
from bris.outputs.intermediate import Intermediate
from bris.predict_metadata import PredictMetadata


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
        extra_variables=None,
        accumulated_variables=None,
        proj4_str=None,
        domain_name=None,
        mask_file: str = "",
        mask_field=None,
        global_attributes=None,
        remove_intermediate=True,
        compression=False,
    ):
        """
        Args:
            filename_pattern: Save predictions to this filename after time tokens are expanded
            interp_res: Interpolate to this resolution [degrees] on a lat/lon grid
            variables: If None, predict all variables
            global_attributes (dict): Write these global attributes in the output file
            compression (bool): If true, write compressed output files
        """
        super().__init__(predict_metadata, extra_variables)

        self.filename_pattern = filename_pattern
        if variables is None:
            self.extract_variables = list(predict_metadata.variables)
        else:
            if extra_variables is None:
                extra_variables = []
            self.extract_variables = variables + extra_variables

        if accumulated_variables is None:
            self.accumulated_variables = []
        else:
            self.accumulated_variables = [v + "_acc" for v in accumulated_variables]
            self.extract_variables += self.accumulated_variables

        self.intermediate = None
        if self.pm.num_members > 1:
            self.intermediate = Intermediate(
                predict_metadata,
                workdir,
                extra_variables,
            )
        self.remove_intermediate = remove_intermediate
        self.variable_list = VariableList(self.extract_variables)

        # Conventions specify the names of variables in the output
        # CF-standard names are added in the standard_name attributes
        self.conventions = Metno()
        self.interp_res = interp_res
        self.latrange = latrange
        self.lonrange = lonrange
        self.mask_file = mask_file
        self.compression = compression
        self.global_attributes = (
            global_attributes if global_attributes is not None else {}
        )

        if domain_name is not None:
            self.proj4_str = projections.get_proj4_str(domain_name)
        else:
            self.proj4_str = proj4_str

        if self._is_masked:
            # If a mask was used during training:
            # Compute 1D->2D index to output 2D arrays by using a mask file
            self.ds_mask = xr.open_dataset(mask_file)
            if "time" in self.ds_mask.dims:
                mask = self.ds_mask.isel(time=0)[mask_field].values
            else:
                mask = self.ds_mask[mask_field].values
            self.mask = mask == 1.0

    def _add_forecast(
        self, times: list, ensemble_member: int, pred: np.ndarray
    ) -> None:
        t0 = pytime.perf_counter()
        if self.pm.num_members > 1 and self.intermediate is not None:
            # Cache data with intermediate
            utils.LOGGER.debug(
                "Netcdf._add_forecast calling intermediate.add_forecast for ensemble_member 0."
            )
            self.intermediate.add_forecast(times, ensemble_member, pred)
            return
        assert ensemble_member == 0

        forecast_reference_time = times[0].astype("datetime64[s]").astype("int")
        filename = self.get_filename(forecast_reference_time)

        # Add ensemble dimension to the last
        utils.LOGGER.debug(
            f"Netcdf._add_forecast calling intermediate.add_forecast for ensemble_member {ensemble_member}"
        )
        self.write(filename, times, pred[..., None])
        utils.LOGGER.debug(
            f"Netcdf._add_forecast completed for {filename} in {pytime.perf_counter() - t0:.1f}s"
        )

    def get_filename(self, forecast_reference_time: int) -> str:
        """Get the filename for this forecast reference time"""
        return utils.expand_time_tokens(self.filename_pattern, forecast_reference_time)

    @property
    def _is_masked(self) -> bool:
        """Was a mask_from_dataset applied during training?"""
        return self.mask_file != ""

    @property
    def _is_gridded(self) -> bool:
        """Is the output gridded?"""
        return len(self.pm.field_shape) == 2 or self.interp_res is not None

    @property
    def _interpolate(self) -> bool:
        """Should interpolation to a regular lat/lon grid be performed?"""
        return self.interp_res is not None

    def conv_name(self, x):
        """Function to easily convert from cf names to conventions"""
        return self.conventions.get_name(x)

    def write(
        self,
        filename: str,
        times: list[np.datetime64],
        pred: np.ndarray | Callable[[int], np.ndarray | None],
    ):
        """Write prediction to NetCDF

        The file is written in two steps: the coordinates, grid definition and attributes
        are written with xarray, then the prediction variables are created with netCDF4 and
        filled one ensemble member and one variable at a time. Only one member's fields
        are ever held in memory, so large ensembles can be written.

        Args:
            times: List of np.datetime64 objects that this forecast is for
            pred: 4D numpy array with dimensions (leadtimes, points, variables, members), or
                a callable taking an ensemble member index and returning that member's 3D
                array (leadtimes, points, variables), or None if the member is missing.
        """

        coords = {}
        x: np.ndarray | None = None
        y: np.ndarray | None = None

        t0 = pytime.perf_counter()

        # TODO: Seconds or hours for leadtimes?
        times_ut = utils.datetime_to_unixtime(times)
        frt_ut = times_ut[0]
        coords[self.conv_name("time")] = np.array(times_ut).astype(np.double)

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
                x_dim_name = self.conv_name("longitude")
                y_dim_name = self.conv_name("latitude")
                utils.LOGGER.debug(
                    f"netcdf.write _interpolate in {pytime.perf_counter() - t0:.1f}s"
                )
            else:
                # TODO: Handle self.latrange and self.lonrange
                if None not in [self.latrange, self.lonrange]:
                    utils.LOGGER.warning(
                        "Warning: latrange/lonrange not handled in gridded fields"
                    )

                if self.proj4_str:
                    lats = np.reshape(self.pm.lats, self.pm.field_shape).astype(
                        np.double
                    )
                    lons = np.reshape(self.pm.lons, self.pm.field_shape).astype(
                        np.double
                    )
                    x, y = projections.get_xy(lats, lons, self.proj4_str)
                else:
                    x = np.arange(self.pm.field_shape[1]).astype(np.float32)
                    y = np.arange(self.pm.field_shape[0]).astype(np.float32)
                x_dim_name = self.conv_name("projection_x_coordinate")
                y_dim_name = self.conv_name("projection_y_coordinate")
                utils.LOGGER.debug(
                    f"netcdf.write not _interpolate in {pytime.perf_counter() - t0:.1f}s"
                )
            coords[x_dim_name] = x
            coords[y_dim_name] = y
            spatial_dims = (y_dim_name, x_dim_name)
        else:
            if self._is_masked:
                # Use the template to get the (full) grid
                if hasattr(self.ds_mask, "X") and hasattr(self.ds_mask, "Y"):
                    x = self.ds_mask.X.values
                    y = self.ds_mask.Y.values
                elif hasattr(self.ds_mask, "x") and hasattr(self.ds_mask, "y"):
                    x = self.ds_mask.x.values
                    y = self.ds_mask.y.values
                else:
                    raise AttributeError(
                        "Mask dataset does not contain projected coordinates variables 'x', 'y' or 'X', 'Y'"
                    )

                x_dim_name = self.conv_name("projection_x_coordinate")
                y_dim_name = self.conv_name("projection_y_coordinate")
                coords[x_dim_name] = x
                coords[y_dim_name] = y
                spatial_dims = (y_dim_name, x_dim_name)
                utils.LOGGER.debug(
                    f"netcdf.write _is_masked in {pytime.perf_counter() - t0:.1f}s"
                )
            else:
                y = np.arange(len(self.pm.lats)).astype(np.int32)
                coords["location"] = y
                spatial_dims = ("location",)
                utils.LOGGER.debug(
                    f"netcdf.write else in {pytime.perf_counter() - t0:.1f}s"
                )

        if self.pm.num_members > 1:
            coords[self.conv_name("realization")] = np.arange(
                self.pm.num_members
            ).astype(np.int32)

            utils.LOGGER.debug(
                f"netcdf.write realization in {pytime.perf_counter() - t0:.1f}s"
            )

        dims_to_add = self.variable_list.dimensions

        attrs = {}
        # Add dimensions
        for dimname, (level_type, levels) in dims_to_add.items():
            # Don't need to convert dimnames, since these are already to local convention
            coords[dimname] = np.array(levels).astype(np.float32)
            attrs[dimname] = cf.get_attributes(level_type)
        utils.LOGGER.debug(
            f"netcdf.write Add dimensions in {pytime.perf_counter() - t0:.1f}s"
        )

        self.ds = xr.Dataset(coords=coords)

        # Add attributes of coordinates
        for var, var_attrs in attrs.items():
            self.ds[var].attrs = var_attrs

        # Set up other coordinate variables
        # Forecast reference time should be double, not int64 (otherwise thredds complains)
        self.ds[self.conv_name("forecast_reference_time")] = (
            [],
            frt_ut,
            {},
            {"dtype": "double"},
        )

        # Set up grid definitions
        if self._is_gridded:
            if self._interpolate:
                self._set_coords_gridded_interpolate(spatial_dims, x, y)
            else:
                self._set_coords_gridded_not_interpolated(spatial_dims)

        else:
            if self._is_masked:
                self._not_gridded_masked(spatial_dims, x, y)
            else:
                self._not_gridded_not_masked(spatial_dims)

        self._set_projection_info()
        self._set_attrs()
        self._write_skeleton(filename)

        def get_member(m: int, _pred=pred) -> np.ndarray | None:
            return _pred[..., m]

        if callable(pred):
            get_member = pred
        self._write_prediction_vars(filename, spatial_dims, times, x, y, get_member)

    def _not_gridded_masked(self, spatial_dims: tuple, x, y):
        t0 = pytime.perf_counter()
        if hasattr(self.ds_mask, "lat") and hasattr(self.ds_mask, "lon"):
            lat = self.ds_mask.lat.values
            lon = self.ds_mask.lon.values
        elif hasattr(self.ds_mask, "latitude") and hasattr(self.ds_mask, "longitude"):
            lat = self.ds_mask.latitude.values
            lon = self.ds_mask.longitude.values
        else:
            raise ValueError(
                "Mask dataset does not contain coordinates variables 'lat', 'lon' or 'latitude', 'longitude'"
            )

        self.ds[self.conv_name("latitude")] = (
            spatial_dims,
            lat,
        )
        self.ds[self.conv_name("longitude")] = (
            spatial_dims,
            lon,
        )
        if self.pm.altitudes is not None:
            altitudes_rec = np.nan * np.zeros([len(y), len(x)], np.float32)
            # Reconstruct the 2D array
            altitudes_rec[self.mask] = self.pm.altitudes
            self.ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                altitudes_rec,
            )

        proj_attrs = {}
        if self.proj4_str is not None:
            proj_attrs = projections.get_proj_attributes(self.proj4_str)
        # Projection
        self.ds[self.conv_name("projection")] = ([], 0, proj_attrs, {"dtype": "int32"})
        utils.LOGGER.debug(
            f"netcdf._not_gridded_masked in {pytime.perf_counter() - t0:.1f}s"
        )

    def _not_gridded_not_masked(self, spatial_dims: tuple):
        t0 = pytime.perf_counter()
        self.ds[self.conv_name("latitude")] = (
            spatial_dims,
            self.pm.lats,
        )
        self.ds[self.conv_name("longitude")] = (
            spatial_dims,
            self.pm.lons,
        )
        if self.pm.altitudes is not None:
            self.ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                self.pm.altitudes,
            )
        utils.LOGGER.debug(
            f"netcdf._not_gridded_not_masked in {pytime.perf_counter() - t0:.1f}s"
        )

    def _set_coords_gridded_not_interpolated(self, spatial_dims: tuple) -> None:
        t0 = pytime.perf_counter()
        lats = self.pm.grid_lats.astype(np.double)
        lons = self.pm.grid_lons.astype(np.double)
        self.ds[self.conv_name("latitude")] = (
            spatial_dims,
            lats,
        )
        self.ds[self.conv_name("longitude")] = (
            spatial_dims,
            lons,
        )

        if self.pm.altitudes is not None:
            altitudes = self.pm.grid_altitudes.astype(np.double)
            self.ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                altitudes,
            )
        proj_attrs = {}
        if self.proj4_str is not None:
            proj_attrs = projections.get_proj_attributes(self.proj4_str)
        self.ds[self.conv_name("projection")] = ([], 0, proj_attrs, {"dtype": "int32"})
        utils.LOGGER.debug(
            f"netcdf._set_coords_gridded_not_interpolated in {pytime.perf_counter() - t0:.1f}s"
        )

    def _set_coords_gridded_interpolate(self, spatial_dims: tuple, x, y) -> None:
        """If is gridded and interpolation should be done"""
        proj_attrs = {}
        proj_attrs["grid_mapping_name"] = "latitude_longitude"
        proj_attrs["earth_radius"] = "6371000.0"
        self.ds[self.conv_name("projection")] = ([], 0, proj_attrs, {"dtype": "int32"})

        if self.pm.altitudes is not None:
            ipoints = gridpp.Points(self.pm.lats, self.pm.lons)
            yy, xx = np.meshgrid(y, x)
            ogrid = gridpp.Grid(yy.transpose(), xx.transpose())

            altitudes = gridpp.nearest(ipoints, ogrid, self.pm.altitudes).astype(
                np.double
            )
            self.ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                altitudes,
            )

        utils.LOGGER.debug("netcdf._set_coords_gridded_interpolate")

    def _set_projection_info(self) -> None:
        for cfname in [
            "forecast_reference_time",
            "time",
            "latitude",
            "longitude",
            "surface_altitude",
            "projection_x_coordinate",
            "projection_y_coordinate",
            "realization",
        ]:
            ncname = self.conv_name(cfname)
            if ncname in self.ds:
                self.ds[ncname].attrs = cf.get_attributes(cfname)

                if cfname == "surface_altitude":
                    self.ds[ncname].attrs["grid_mapping"] = "projection"
                    self.ds[ncname].attrs["coordinates"] = "latitude longitude"
        utils.LOGGER.debug("netcdf._set_projection_info")

    @cached_property
    def get_projection_rotation_matrices(self) -> tuple[np.ndarray]:
        """Precompute rotation matrices for field rotation from east/north to projected coordinates"""
        if self.proj4_str is None:
            raise ValueError("No projection defined for field rotation")

        e_x, n_x, e_y, n_y = projections.compute_local_mapping_from_lonlat(
            self.pm.lons, self.pm.lats, self.proj4_str, dist=1.0
        )
        return e_x, n_x, e_y, n_y

    def _rotate_fields(self, sub: np.ndarray, columns: dict[str, int]) -> None:
        """Rotate wind components from east/north to projected coordinates, in place

        Args:
            sub: 3D array (leadtimes, points, columns) for one ensemble member
            columns: anemoi variable name -> column index in sub
        """
        for u_field, v_field in self.conventions.fields_to_rotate:
            if u_field not in columns or v_field not in columns:
                continue
            if (
                u_field not in self.extract_variables
                or v_field not in self.extract_variables
            ):
                continue

            # Cached property so this will be computed only once
            e_x, n_x, e_y, n_y = self.get_projection_rotation_matrices

            u_values = sub[:, :, columns[u_field]]
            v_values = sub[:, :, columns[v_field]]
            x_values = e_x * u_values + n_x * v_values
            y_values = e_y * u_values + n_y * v_values
            sub[:, :, columns[u_field]] = x_values
            sub[:, :, columns[v_field]] = y_values

    def _source_variable(self, variable: str) -> str:
        """The anemoi variable a (possibly accumulated) output variable is computed from"""
        if variable in self.accumulated_variables:
            return variable.removesuffix("_acc")
        return variable

    def _member_columns(self) -> dict[str, int]:
        """Which model variables are needed, and their column in the per-member array"""
        names = []
        for variable in self.extract_variables:
            name = self._source_variable(variable)
            if name not in names:
                names.append(name)
        return {name: i for i, name in enumerate(names)}

    def _extract_columns(
        self, pred_m: np.ndarray, columns: dict[str, int]
    ) -> np.ndarray:
        """Reads the needed variables of one member into a 3D array (leadtimes, points, columns)

        pred_m may be a memory-mapped intermediate file; only the needed variables are read.
        """
        indices = [self.pm.variables.index(name) for name in columns]
        sub = np.asarray(pred_m[:, :, indices], dtype=np.float32)
        assert sub.shape == (self.pm.num_leadtimes, self.pm.num_points, len(indices)), (
            sub.shape
        )
        return sub

    def _prepare_variable(
        self,
        variable: str,
        sub: np.ndarray,
        columns: dict[str, int],
        spatial_shape: tuple,
        interp: tuple | None,
    ) -> np.ndarray:
        """One output variable for one member: on the output grid, accumulated if needed, in
        CF units. Returns an array with dimensions (time, y, x) or (time, location)."""
        curr = sub[:, :, columns[self._source_variable(variable)]]

        if self._interpolate:
            ipoints, ogrid = interp
            ar = np.asarray(gridpp.nearest(ipoints, ogrid, curr), dtype=np.float32)
        elif self._is_masked:
            ar = np.nan * np.zeros((len(curr),) + spatial_shape, np.float32)
            # Reconstruct the 2D array (nans where no data)
            ar[:, self.mask] = curr
        else:
            ar = np.reshape(curr, (len(curr),) + spatial_shape)

        if variable in self.accumulated_variables:
            # Accumulate over lead times
            ar = np.cumsum(np.nan_to_num(ar, nan=0), axis=0)

        # Unit conversion from anemoi to CF
        cfname = cf.get_metadata(variable)["cfname"]
        attrs = cf.get_attributes(cfname)
        from_units = anemoi_conventions.get_units(variable)
        if "units" in attrs:
            to_units = attrs["units"]
            ar, _ = bris.units.convert(ar, from_units, to_units, inplace=False)

        return np.asarray(ar, dtype=np.float32)

    def _variable_attrs(self, variable: str) -> dict:
        cfname = cf.get_metadata(variable)["cfname"]
        attrs = dict(cf.get_attributes(cfname))
        attrs["grid_mapping"] = "projection"
        attrs["coordinates"] = "latitude longitude"
        return attrs

    def _write_skeleton(self, filename: str) -> None:
        """Write coordinates, grid definition and global attributes (no prediction data)"""
        t0 = pytime.perf_counter()
        utils.create_directory(filename)

        utils.LOGGER.debug(f"netcdf._write_skeleton writing to {filename}")
        self.ds.to_netcdf(
            filename,
            mode="w",
            engine="netcdf4",
            unlimited_dims=["time"],
        )
        utils.LOGGER.debug(
            f"netcdf._write_skeleton done in {pytime.perf_counter() - t0:.1f}s"
        )

    def _write_prediction_vars(
        self,
        filename: str,
        spatial_dims: tuple,
        times: list,
        x: np.ndarray | None,
        y: np.ndarray | None,
        get_member: Callable[[int], np.ndarray | None],
    ) -> None:
        """Create the prediction variables in the file and fill them member by member"""
        t0 = pytime.perf_counter()
        ensemble = self.pm.num_members > 1
        time_dim = self.conv_name("time")
        member_dim = self.conv_name("ensemble_member")
        if self._is_gridded or self._is_masked:
            spatial_shape = (len(y), len(x))
        else:
            spatial_shape = (len(y),)

        interp = None
        if self._interpolate:
            ipoints = gridpp.Points(self.pm.lats, self.pm.lons)
            yy, xx = np.meshgrid(y, x)
            ogrid = gridpp.Grid(yy.transpose(), xx.transpose())
            interp = (ipoints, ogrid)

        columns = self._member_columns()

        with netCDF4.Dataset(filename, "a") as nc:
            # Define the variables. Chunks hold one (time, level, member) slice of the
            # field, which is how the data is written.
            for variable in self.extract_variables:
                ncname = self.variable_list.get_ncname_from_anemoi_name(variable)
                if ncname in nc.variables:
                    continue
                dim_name = self.variable_list.get_level_dimname(ncname)
                dims = [time_dim]
                chunks = [1]
                if dim_name is not None:
                    dims.append(dim_name)
                    chunks.append(1)
                if ensemble:
                    dims.append(member_dim)
                    chunks.append(1)
                dims += list(spatial_dims)
                chunks += list(spatial_shape)
                var = nc.createVariable(
                    ncname,
                    "f4",
                    dims,
                    zlib=self.compression,
                    fill_value=np.float32(np.nan),
                    chunksizes=chunks,
                )
                var.setncatts(self._variable_attrs(variable))

            for m in range(self.pm.num_members):
                t1 = pytime.perf_counter()
                pred_m = get_member(m)
                if pred_m is None:
                    utils.LOGGER.warning(
                        f"netcdf: no data for ensemble member {m}, leaving it missing"
                    )
                    continue
                sub = self._extract_columns(pred_m, columns)
                del pred_m
                if self.proj4_str is not None:
                    self._rotate_fields(sub, columns)

                for variable in self.extract_variables:
                    ar = self._prepare_variable(
                        variable, sub, columns, spatial_shape, interp
                    )
                    ncname = self.variable_list.get_ncname_from_anemoi_name(variable)
                    level_index = self.variable_list.get_level_index(variable)
                    index: list = [slice(None)]
                    if level_index is not None:
                        index.append(level_index)
                    if ensemble:
                        index.append(m)
                    nc.variables[ncname][tuple(index)] = ar
                utils.LOGGER.debug(
                    f"netcdf._write_prediction_vars member {m} in {pytime.perf_counter() - t1:.1f}s"
                )
        utils.LOGGER.debug(
            f"netcdf._write_prediction_vars done in {pytime.perf_counter() - t0:.1f}s"
        )

    def _set_attrs(self) -> None:
        """Add global attributes"""
        datestr = datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%d %H:%M:%S +00:00"
        )
        self.ds.attrs["history"] = f"{datestr} Created by bris-inference"
        self.ds.attrs["Convensions"] = "CF-1.6"
        for key, value in self.global_attributes.items():
            self.ds.attrs[key] = value

    def _write_file(self, filename: str) -> None:
        """Write netcdf to disk"""
        t0 = pytime.perf_counter()
        utils.create_directory(filename)

        utils.LOGGER.debug(f"netcdf._write_file writing to {filename}")
        self.ds.to_netcdf(
            filename,
            mode="w",
            engine="netcdf4",
            unlimited_dims=["time"],
            encoding=self.nc_encoding,
        )
        utils.LOGGER.debug(
            f"netcdf._write_file Done in {pytime.perf_counter() - t0:.1f}s"
        )

    def finalize(self):
        t0 = pytime.perf_counter()

        if self.pm.num_members > 1:
            # Load data from the intermediate and write to disk, one member at a time
            forecast_reference_times = self.intermediate.get_forecast_reference_times()
            for forecast_reference_time in forecast_reference_times:
                time = forecast_reference_time.astype("datetime64[s]").astype("int")
                filename = self.get_filename(time)
                lead_times = [
                    forecast_reference_time + lt
                    for lt in self.intermediate.pm.leadtimes
                ]

                def get_member(
                    m: int, frt=forecast_reference_time
                ) -> np.ndarray | None:
                    # Memory-mapped, so only the variables written are read from disk
                    return self.intermediate.get_forecast(frt, m, mmap_mode="r")

                self.write(filename, lead_times, get_member)

            if self.remove_intermediate:
                self.intermediate.cleanup()
        utils.LOGGER.debug(
            f"Netcdf.finalize: {pytime.perf_counter() - t0:.1f}s",
        )

    def get_lower(self, array):
        m = np.min(array)
        return np.floor(m / self.interp_res) * self.interp_res

    def get_upper(self, array):
        m = np.max(array)
        return np.ceil(m / self.interp_res) * self.interp_res
