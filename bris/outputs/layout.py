import datetime
from collections.abc import Callable, Iterator
from functools import cached_property

import gridpp
import numpy as np
import xarray as xr

import bris.units
from bris import projections, utils
from bris.conventions import anemoi as anemoi_conventions
from bris.conventions import cf
from bris.conventions.metno import Metno
from bris.conventions.variable_list import VariableList
from bris.predict_metadata import PredictMetadata


class Layout:
    """How a forecast is laid out in a file, using CF-standards and local conventions

    This is what the Zarr, NetCDF and Grib outputs have in common, independent of the file
    format: the coordinates, the grid definition, the names, dimensions and attributes of the
    variables, and how the predictions of one ensemble member are turned into these variables
    (wind rotation, regridding, accumulation and unit conversion).

    There are three types of layouts:
    1) Gridded regional projected data. This requires field_shape to be set in predict_metadata
    2) Irregular grids interpolated to a lat/lon grid. Use interp_res.
    3) Irregular grids. This means the output only has one location dimension.
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        extract_variables: list[str],
        accumulated_variables: list[str] | None = None,
        interp_res=None,
        latrange=None,
        lonrange=None,
        proj4_str=None,
        mask_file: str = "",
        mask_field=None,
        global_attributes=None,
    ):
        """
        Args:
            predict_metadata: Metadata of the predictions, including any extra variables
            extract_variables: The variables to write, with anemoi names. Accumulated variables
                have the suffix _acc.
            accumulated_variables: Which of extract_variables are accumulated over lead time
            interp_res: Interpolate to this resolution [degrees] on a lat/lon grid
            proj4_str: Projection of the grid, for x/y coordinates and wind rotation
            global_attributes (dict): Global attributes of the file
        """
        self.pm = predict_metadata
        self.extract_variables = extract_variables
        self.accumulated_variables = accumulated_variables or []
        self.variable_list = VariableList(self.extract_variables)

        # Conventions specify the names of variables in the output
        # CF-standard names are added in the standard_name attributes
        self.conventions = Metno()
        self.interp_res = interp_res
        self.latrange = latrange
        self.lonrange = lonrange
        self.mask_file = mask_file
        self.proj4_str = proj4_str
        self.global_attributes = (
            global_attributes if global_attributes is not None else {}
        )

        if self._is_masked:
            # If a mask was used during training:
            # Compute 1D->2D index to output 2D arrays by using a mask file
            self.ds_mask = xr.open_dataset(mask_file)
            if "time" in self.ds_mask.dims:
                mask = self.ds_mask.isel(time=0)[mask_field].values
            else:
                mask = self.ds_mask[mask_field].values
            self.mask = mask == 1.0

    @property
    def _is_masked(self) -> bool:
        """Was a mask_from_dataset applied during training?"""
        return self.mask_file != ""

    @property
    def _is_gridded(self) -> bool:
        """Is the output gridded?"""
        return self.pm.is_gridded or self.interp_res is not None

    @property
    def _interpolate(self) -> bool:
        """Should interpolation to a regular lat/lon grid be performed?"""
        return self.interp_res is not None

    def conv_name(self, x):
        """Function to easily convert from cf names to conventions"""
        return self.conventions.get_name(x)

    @cached_property
    def _grid(self) -> tuple[tuple, np.ndarray | None, np.ndarray]:
        """The spatial dimensions of the variables, and their x and y coordinates (x is None
        for an irregular grid, where y numbers the locations)"""
        x: np.ndarray | None = None
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
            else:
                # The whole grid is written: latrange and lonrange only select the area of an
                # interpolated lat/lon grid
                if self.latrange is not None or self.lonrange is not None:
                    utils.LOGGER.warning(
                        "latrange and lonrange are only used with interp_res: they are "
                        "ignored, and the whole grid is written"
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
            return (y_dim_name, x_dim_name), x, y

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
            return (y_dim_name, x_dim_name), x, y

        y = np.arange(len(self.pm.lats)).astype(np.int32)
        return ("location",), None, y

    @property
    def grid_mapping_name(self) -> str | None:
        """The CF grid mapping name of the grid of the variables, if it has a grid mapping"""
        if self._is_gridded and self._interpolate:
            return "latitude_longitude"
        if (self._is_gridded or self._is_masked) and self.proj4_str:
            attrs = projections.get_proj_attributes(self.proj4_str)
            return attrs.get("grid_mapping_name")
        return None

    @property
    def spatial_dims(self) -> tuple:
        """Names of the spatial dimensions of the variables"""
        return self._grid[0]

    @property
    def num_spatial_dims(self) -> int:
        """Number of spatial dimensions of the variables: 2 for a grid (y, x), 1 for an
        irregular grid (location)"""
        return 2 if self._is_gridded or self._is_masked else 1

    @property
    def spatial_shape(self) -> tuple:
        _, x, y = self._grid
        if x is None:
            return (len(y),)
        return (len(y), len(x))

    def skeleton(self, times: list[np.datetime64]) -> xr.Dataset:
        """Everything in the file but the predictions: the coordinates, the grid definition and
        the attributes

        Args:
            times: List of np.datetime64 objects that this forecast is for
        """
        coords = {}
        spatial_dims, x, y = self._grid

        times_ut = utils.datetime_to_unixtime(times)
        frt_ut = times_ut[0]
        coords[self.conv_name("time")] = np.array(times_ut).astype(np.double)

        if x is None:
            coords["location"] = y
        else:
            y_dim_name, x_dim_name = spatial_dims
            coords[x_dim_name] = x
            coords[y_dim_name] = y

        if self.pm.num_members > 1:
            coords[self.conv_name("realization")] = np.arange(
                self.pm.num_members
            ).astype(np.int32)

        attrs = {}
        # Add dimensions
        for dimname, (level_type, levels) in self.variable_list.dimensions.items():
            # Don't need to convert dimnames, since these are already to local convention
            coords[dimname] = np.array(levels).astype(np.float32)
            attrs[dimname] = cf.get_attributes(level_type)

        ds = xr.Dataset(coords=coords)

        # Add attributes of coordinates
        for var, var_attrs in attrs.items():
            ds[var].attrs = var_attrs

        # Set up other coordinate variables
        # Forecast reference time should be double, not int64 (otherwise thredds complains)
        ds[self.conv_name("forecast_reference_time")] = (
            [],
            frt_ut,
            {},
            {"dtype": "double"},
        )

        # Set up grid definitions
        if self._is_gridded:
            if self._interpolate:
                self._set_coords_gridded_interpolate(ds, spatial_dims, x, y)
            else:
                self._set_coords_gridded_not_interpolated(ds, spatial_dims)
        else:
            if self._is_masked:
                self._not_gridded_masked(ds, spatial_dims, x, y)
            else:
                self._not_gridded_not_masked(ds, spatial_dims)

        self._set_projection_info(ds)
        self._set_attrs(ds)
        return ds

    def _not_gridded_masked(self, ds: xr.Dataset, spatial_dims: tuple, x, y):
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

        ds[self.conv_name("latitude")] = (
            spatial_dims,
            lat,
        )
        ds[self.conv_name("longitude")] = (
            spatial_dims,
            lon,
        )
        if self.pm.altitudes is not None:
            altitudes_rec = np.nan * np.zeros([len(y), len(x)], np.float32)
            # Reconstruct the 2D array
            altitudes_rec[self.mask] = self.pm.altitudes
            ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                altitudes_rec,
            )

        proj_attrs = {}
        if self.proj4_str is not None:
            proj_attrs = projections.get_proj_attributes(self.proj4_str)
        # Projection
        ds[self.conv_name("projection")] = ([], 0, proj_attrs, {"dtype": "int32"})

    def _not_gridded_not_masked(self, ds: xr.Dataset, spatial_dims: tuple):
        ds[self.conv_name("latitude")] = (
            spatial_dims,
            self.pm.lats,
        )
        ds[self.conv_name("longitude")] = (
            spatial_dims,
            self.pm.lons,
        )
        if self.pm.altitudes is not None:
            ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                self.pm.altitudes,
            )

    def _set_coords_gridded_not_interpolated(
        self, ds: xr.Dataset, spatial_dims: tuple
    ) -> None:
        lats = self.pm.grid_lats.astype(np.double)
        lons = self.pm.grid_lons.astype(np.double)
        ds[self.conv_name("latitude")] = (
            spatial_dims,
            lats,
        )
        ds[self.conv_name("longitude")] = (
            spatial_dims,
            lons,
        )

        if self.pm.altitudes is not None:
            altitudes = self.pm.grid_altitudes.astype(np.double)
            ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                altitudes,
            )
        proj_attrs = {}
        if self.proj4_str is not None:
            proj_attrs = projections.get_proj_attributes(self.proj4_str)
        ds[self.conv_name("projection")] = ([], 0, proj_attrs, {"dtype": "int32"})

    def _set_coords_gridded_interpolate(
        self, ds: xr.Dataset, spatial_dims: tuple, x, y
    ) -> None:
        """If is gridded and interpolation should be done"""
        proj_attrs = {}
        proj_attrs["grid_mapping_name"] = "latitude_longitude"
        proj_attrs["earth_radius"] = "6371000.0"
        ds[self.conv_name("projection")] = ([], 0, proj_attrs, {"dtype": "int32"})

        if self.pm.altitudes is not None:
            altitudes = np.asarray(self.pm.altitudes, dtype=np.float32)[
                self._interpolation_index
            ].astype(np.double)
            ds[self.conv_name("surface_altitude")] = (
                spatial_dims,
                altitudes,
            )

    def _set_projection_info(self, ds: xr.Dataset) -> None:
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
            if ncname in ds:
                ds[ncname].attrs = cf.get_attributes(cfname)

                if cfname == "surface_altitude":
                    ds[ncname].attrs["grid_mapping"] = "projection"
                    ds[ncname].attrs["coordinates"] = "latitude longitude"

    def _set_attrs(self, ds: xr.Dataset) -> None:
        """Add global attributes"""
        datestr = datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%d %H:%M:%S +00:00"
        )
        ds.attrs["history"] = f"{datestr} Created by bris-inference"
        ds.attrs["Convensions"] = "CF-1.6"
        for key, value in self.global_attributes.items():
            ds.attrs[key] = value

    @cached_property
    def _interpolation(self) -> tuple:
        """The points and the lat/lon grid that they are interpolated to"""
        _, x, y = self._grid
        ipoints = gridpp.Points(self.pm.lats, self.pm.lons)
        yy, xx = np.meshgrid(y, x)
        ogrid = gridpp.Grid(yy.transpose(), xx.transpose())
        return ipoints, ogrid

    @cached_property
    def _interpolation_index(self) -> np.ndarray:
        """For each point of the lat/lon grid (y, x), the index of the nearest of the points
        that are interpolated

        The search for the nearest points takes much longer than taking their values, and holds
        Python's global interpreter lock, so it is done once and not for every variable. The
        values are those of gridpp.nearest.
        """
        ipoints, ogrid = self._interpolation
        indices = np.arange(self.pm.num_points)
        # gridpp works with single precision, which holds integers up to 2^24 exactly: the
        # indices are passed through in two parts that are small enough
        part = 4096
        high = gridpp.nearest(ipoints, ogrid, (indices // part).astype(np.float32))
        low = gridpp.nearest(ipoints, ogrid, (indices % part).astype(np.float32))
        return np.asarray(high).astype(np.intp) * part + np.asarray(low).astype(np.intp)

    @cached_property
    def get_projection_rotation_matrices(self) -> tuple[np.ndarray]:
        """Precompute rotation matrices for field rotation from east/north to projected coordinates"""
        if self.proj4_str is None:
            raise ValueError("No projection defined for field rotation")

        e_x, n_x, e_y, n_y = projections.compute_local_mapping_from_lonlat(
            self.pm.lons, self.pm.lats, self.proj4_str, dist=1.0
        )
        return e_x, n_x, e_y, n_y

    def _source_variable(self, variable: str) -> str:
        """The anemoi variable a (possibly accumulated) output variable is computed from"""
        if variable in self.accumulated_variables:
            return variable.removesuffix("_acc")
        return variable

    def _values(self, pred: np.ndarray, variable: str) -> np.ndarray:
        """One model variable of the predictions (leadtimes, points), without copying it"""
        assert pred.shape[1] == self.pm.num_points, pred.shape
        return np.asarray(
            pred[:, :, self.pm.variables.index(variable)], dtype=np.float32
        )

    @property
    def _winds_to_rotate(self) -> list[tuple[str, str]]:
        """The pairs of wind components (anemoi names) that are written and rotated"""
        if self.proj4_str is None:
            return []
        return [
            (u_field, v_field)
            for u_field, v_field in self.conventions.fields_to_rotate
            if u_field in self.extract_variables and v_field in self.extract_variables
        ]

    def prepare(self) -> None:
        """Compute what rotated_winds and field_values share between the variables, which is
        otherwise computed when it is first needed. Call it before using these from several
        threads, so that this is done once."""
        _ = self._grid
        if self._interpolate:
            _ = self._interpolation_index
        if self._winds_to_rotate:
            _ = self.get_projection_rotation_matrices

    def rotated_winds(
        self, pred: np.ndarray, mapper: Callable = map
    ) -> dict[str, np.ndarray]:
        """The wind components that are written, rotated from east/north to projected
        coordinates

        Args:
            pred: 3D array (leadtimes, points, variables) for one ensemble member
            mapper: Applies a function to the pairs of wind components, as map does, e.g. the
                map of a pool of threads

        Returns:
            anemoi variable name -> rotated values (leadtimes, points). Empty without a
            projection.
        """
        pairs = self._winds_to_rotate
        if not pairs:
            return {}

        # Cached property so this will be computed only once
        e_x, n_x, e_y, n_y = self.get_projection_rotation_matrices

        def rotate(pair: tuple[str, str]) -> dict[str, np.ndarray]:
            u_field, v_field = pair
            u_values = self._values(pred, u_field)
            v_values = self._values(pred, v_field)
            return {
                u_field: np.asarray(e_x * u_values + n_x * v_values, dtype=np.float32),
                v_field: np.asarray(e_y * u_values + n_y * v_values, dtype=np.float32),
            }

        rotated = {}
        for winds in mapper(rotate, pairs):
            rotated.update(winds)
        return rotated

    def field_values(
        self,
        variable: str,
        pred: np.ndarray,
        rotated: dict[str, np.ndarray],
        accumulated: dict[str, np.ndarray] | None = None,
    ) -> tuple[str, int | None, np.ndarray]:
        """One output variable for one member: on the output grid, accumulated if needed, in
        CF units

        Args:
            variable: One of extract_variables
            pred: 3D array (leadtimes, points, variables) with a block of consecutive lead
                times of one member
            rotated: The rotated wind components of pred (see rotated_winds)
            accumulated: The accumulated variables at the end of the previous block, as left
                here by the call for that block (updated in place). None or empty for the
                first block.

        Returns:
            The name of the variable in the file, its index in the level dimension (None if the
            variable has no level dimension) and its values (time, y, x) or (time, location)
        """
        source = self._source_variable(variable)
        curr = rotated[source] if source in rotated else self._values(pred, source)

        if self._interpolate:
            # The nearest point, as gridpp.nearest
            ar = curr[:, self._interpolation_index]
        elif self._is_masked:
            ar = np.nan * np.zeros((len(curr),) + self.spatial_shape, np.float32)
            # Reconstruct the 2D array (nans where no data)
            ar[:, self.mask] = curr
        else:
            ar = np.reshape(curr, (len(curr),) + self.spatial_shape)

        if variable in self.accumulated_variables:
            # Accumulate over lead times, continuing from the lead times before this block
            ar = np.nan_to_num(ar, nan=0)
            if accumulated is not None and variable in accumulated:
                ar[0] += accumulated[variable]
            ar = np.cumsum(ar, axis=0)
            if accumulated is not None:
                accumulated[variable] = ar[-1].copy()

        # Unit conversion from anemoi to CF
        cfname = cf.get_metadata(variable)["cfname"]
        attrs = cf.get_attributes(cfname)
        from_units = anemoi_conventions.get_units(variable)
        if "units" in attrs:
            to_units = attrs["units"]
            ar, _ = bris.units.convert(ar, from_units, to_units, inplace=False)

        ncname = self.variable_list.get_ncname_from_anemoi_name(variable)
        level_index = self.variable_list.get_level_index(variable)
        return ncname, level_index, np.asarray(ar, dtype=np.float32)

    def _variable_attrs(self, variable: str) -> dict:
        cfname = cf.get_metadata(variable)["cfname"]
        attrs = dict(cf.get_attributes(cfname))
        attrs["grid_mapping"] = "projection"
        attrs["coordinates"] = "latitude longitude"
        return attrs

    @cached_property
    def fields(self) -> dict[str, dict]:
        """The prediction variables in the file: name -> dims, shape and attrs

        The dimensions are time, the level dimension if the variable has one, the ensemble
        member if there are several, then the spatial dimensions. Several anemoi variables can
        be levels of the same variable in the file.
        """
        ensemble = self.pm.num_members > 1
        fields = {}
        for variable in self.extract_variables:
            ncname = self.variable_list.get_ncname_from_anemoi_name(variable)
            if ncname in fields:
                continue
            dims = [self.conv_name("time")]
            shape = [self.pm.num_leadtimes]
            level_dim = self.variable_list.get_level_dimname(ncname)
            if level_dim is not None:
                dims.append(level_dim)
                shape.append(len(self.variable_list.dimensions[level_dim][1]))
            if ensemble:
                dims.append(self.conv_name("ensemble_member"))
                shape.append(self.pm.num_members)
            dims += list(self.spatial_dims)
            shape += list(self.spatial_shape)
            fields[ncname] = {
                "dims": dims,
                "shape": shape,
                "attrs": self._variable_attrs(variable),
            }
        return fields

    def transform(
        self, pred: np.ndarray, accumulated: dict[str, np.ndarray] | None = None
    ) -> Iterator[tuple[str, int | None, np.ndarray]]:
        """Turns the predictions of one ensemble member into the variables in the file

        Args:
            pred: 3D array (leadtimes, points, variables) with a block of consecutive lead
                times of one member
            accumulated: The accumulated variables at the end of the previous block, as left
                here by the call for that block (updated in place). None or empty for the
                first block.

        Yields:
            The name of the variable in the file, its index in the level dimension (None if the
            variable has no level dimension) and its values (time, y, x) or (time, location)
        """
        rotated = self.rotated_winds(pred)
        for variable in self.extract_variables:
            yield self.field_values(variable, pred, rotated, accumulated)

    def get_lower(self, array):
        m = np.min(array)
        return np.floor(m / self.interp_res) * self.interp_res

    def get_upper(self, array):
        m = np.max(array)
        return np.ceil(m / self.interp_res) * self.interp_res
