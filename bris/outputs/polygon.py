"""Forecasts restricted to the union of polygons in geospatial files."""

from functools import cached_property
from pathlib import Path

import numpy as np
import xarray as xr

from bris import utils
from bris.outputs.netcdf import Netcdf
from bris.predict_metadata import PredictMetadata


class Polygon(Netcdf):
    """Keep grid points covered by polygons, with selectable NetCDF or Zarr storage.

    Regular grids retain their full shape, with NaNs outside the polygons. Irregular
    grids retain only selected locations. Selection uses point centres, including
    polygon boundaries; it does not average values over polygon areas.
    """

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename_pattern: str,
        polygon_files: str | list[str],
        format: str = "netcdf",
        spatial_chunk_size: int = 256,
        compression: bool = True,
        **kwargs,
    ):
        """
        Args:
            polygon_files: Feather, GeoParquet or GIS files (e.g. GeoJSON/shapefile).
                Multiple files and features are combined as a union. Each file must
                declare its CRS; forecast coordinates are transformed to that CRS.
            format: "netcdf" (zlib) or "zarr" (Zarr v2, Blosc/Zstd with bitshuffle).
            spatial_chunk_size: Maximum chunk length along each spatial dimension.
                Time, level and ensemble chunks each contain one element.
            compression: Enable lossless compression. Zarr also omits all-NaN chunks.
            **kwargs: Other Netcdf options, including variables, extra_variables,
                accumulated_variables, interp_res, mask_file and projections.
        """
        if format not in ("netcdf", "zarr"):
            raise ValueError("format must be 'netcdf' or 'zarr'")
        if (
            isinstance(spatial_chunk_size, bool)
            or not isinstance(spatial_chunk_size, int)
            or spatial_chunk_size < 1
        ):
            raise ValueError("spatial_chunk_size must be a positive integer")
        if isinstance(polygon_files, (str, Path)):
            polygon_files = [polygon_files]
        if not polygon_files:
            raise ValueError("polygon_files must contain at least one file")

        self.format = format
        self.spatial_chunk_size = spatial_chunk_size
        self._polygons = self._read_polygons(polygon_files)
        self._missing_members = []
        super().__init__(
            predict_metadata,
            workdir,
            filename_pattern,
            compression=compression,
            **kwargs,
        )

    @staticmethod
    def _read_polygons(filenames):
        # Keep these optional imports local so other outputs need no GIS dependencies.
        try:
            import geopandas as gpd
            import shapely
            from pyproj import Transformer
        except ImportError as exc:
            raise ImportError(
                "Polygon output requires pip install 'bris[polygons]'"
            ) from exc

        polygons = []
        for filename in filenames:
            suffix = Path(filename).suffix.lower()
            if suffix in (".feather", ".arrow"):
                frame = gpd.read_feather(filename)
            elif suffix in (".parquet", ".geoparquet"):
                frame = gpd.read_parquet(filename)
            else:
                frame = gpd.read_file(filename)
            if frame.crs is None:
                raise ValueError(f"Polygon file has no CRS: {filename}")
            geometry = frame.geometry.dropna()
            geometry = geometry[~geometry.is_empty]
            if geometry.empty:
                raise ValueError(
                    f"Polygon file has no non-empty geometries: {filename}"
                )
            if not geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                raise ValueError(
                    f"Expected Polygon or MultiPolygon geometries: {filename}"
                )
            if not geometry.is_valid.all():
                raise ValueError(
                    f"Polygon file contains invalid geometries: {filename}"
                )
            union = shapely.union_all(geometry.values)
            shapely.prepare(union)
            transform = Transformer.from_crs("EPSG:4326", frame.crs, always_xy=True)
            polygons.append((union, transform))
        return polygons

    def _select_points(self, lats, lons):
        import shapely

        selected = np.zeros(np.shape(lats), dtype=bool)
        for geometry, transform in self._polygons:
            x, y = transform.transform(lons, lats)
            selected |= shapely.intersects_xy(geometry, x, y)
        return selected

    @property
    def _compact_locations(self):
        return not (self._is_gridded or self._is_masked)

    def _write_file(self, filename):
        # Select on the OUTPUT coordinates, after any interpolation or reconstruction
        # of a training mask, so interpolation cannot leak data outside the polygons.
        latitude = self.ds[self.conv_name("latitude")]
        longitude = self.ds[self.conv_name("longitude")]
        latitude, longitude = xr.broadcast(latitude, longitude)
        polygon_mask = self._select_points(latitude.values, longitude.values)
        if self._is_masked and not self._interpolate:
            polygon_mask &= self.mask
        if not polygon_mask.any():
            raise ValueError("No output grid points are covered by the polygon files")

        prediction_names = {
            self.variable_list.get_ncname_from_anemoi_name(variable)
            for variable in self.extract_variables
        }
        if self._compact_locations:
            self.ds = self.ds.isel(location=np.flatnonzero(polygon_mask))
        else:
            mask = xr.DataArray(polygon_mask, dims=latitude.dims)
            # Mask after the parent has accumulated and converted variables, so
            # its NaN-to-zero accumulation cannot restore data outside polygons.
            for name in prediction_names | {self.conv_name("surface_altitude")}:
                if name in self.ds:
                    self.ds[name] = self.ds[name].where(mask)

        # Accumulation also changes missing ensemble members from NaNs to zeros.
        if self._missing_members:
            for name in prediction_names:
                self.ds[name][
                    {self.conv_name("ensemble_member"): self._missing_members}
                ] = np.nan

        if self.format == "netcdf":
            for name in prediction_names:
                var = self.ds[name]
                self.nc_encoding[name] = {
                    **self.nc_encoding.get(name, {}),
                    "_FillValue": np.float32(np.nan),
                    "chunksizes": self._chunks(var, latitude.dims),
                }
            super()._write_file(filename)
        else:
            utils.create_directory(filename)
            encoding = {}
            for name, var in self.ds.variables.items():
                # Integer coordinates (including member zero) must not be decoded as
                # missing values by xarray's Zarr v2 reader.
                encoding[name] = {
                    **var.encoding,
                    "compressor": self._zarr_compressor,
                    "_FillValue": np.nan if name in prediction_names else None,
                }
                if var.ndim:
                    encoding[name]["chunks"] = self._chunks(var, latitude.dims)
            self.ds.to_zarr(
                filename,
                mode="w",
                encoding=encoding,
                consolidated=True,
                write_empty_chunks=False,
            )

    def _chunks(self, variable, spatial_dims):
        return tuple(
            min(size, self.spatial_chunk_size) if dim in spatial_dims else 1
            for dim, size in zip(variable.dims, variable.shape, strict=True)
        )

    def finalize(self):
        if self.intermediate is None:
            return
        for reference_time in self.intermediate.get_forecast_reference_times():
            timestamp = reference_time.astype("datetime64[s]").astype("int")
            times = [reference_time + lt for lt in self.pm.leadtimes]
            # The older writer assembles the ensemble in memory. The intermediate
            # loader supplies NaNs for missing members, rather than zeros.
            pred = self.intermediate.get_forecast(reference_time)
            self._missing_members = [
                member
                for member in range(self.pm.num_members)
                if not Path(
                    self.intermediate.get_filename(reference_time, member)
                ).exists()
            ]
            self.write(self.get_filename(timestamp), times, pred)
        if self.remove_intermediate:
            self.intermediate.cleanup()

    @cached_property
    def _zarr_compressor(self):
        from numcodecs import Blosc

        if not self.compression:
            return None
        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
