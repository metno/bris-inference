import glob
import hashlib
import json
import os
import re
import time as pytime
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np
import xarray as xr
import zarr

import bris.convert
from bris import projections, utils, zarrstore
from bris.conventions import grib as grib_conventions
from bris.outputs import Output
from bris.outputs.layout import Layout
from bris.predict_metadata import PredictMetadata

# Aim for chunks of about this size when the number of lead times in a chunk is not given
CHUNK_BYTES = 100e6

# The number of threads that write the variables of a member to a store. More than this gains
# little: the threads share the memory bandwidth and the file system.
WRITE_THREADS = 4

# The chunk sizes that can be set for the spatial dimensions (whatever these are named in the
# store, e.g. latitude and longitude for y and x), by the number of spatial dimensions
SPATIAL_CHUNKS = {2: ["y", "x"], 1: ["location"]}


class Zarr(Output):
    """Write predictions to Zarr (version 2), using CF-standards and local conventions

    There is one store per forecast, with the same variables, dimensions and attributes as
    the NetCDF output (see bris.outputs.layout), so that it reads the same with xarray.

    Since ensemble is done data-parallel, the members of a forecast arrive in different
    processes. Each of them writes its member directly into the store: an ensemble member is a
    chunk of its own, so the processes never write the same chunk. The store is complete when
    the last member is written, and nothing is left to do when the predictions end.

    The store can be converted to other formats (see bris.convert), at the end of the
    predictions or afterwards with bris-convert. This is how the NetCDF and Grib outputs are
    written.
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
        compression=False,
        chunks=None,
        convert=None,
        remove_after_convert=False,
    ):
        """
        Args:
            filename_pattern: Save predictions to this store after time tokens are expanded
            interp_res: Interpolate to this resolution [degrees] on a lat/lon grid
            variables: If None, predict all variables
            global_attributes (dict): Write these global attributes in the store
            compression (bool): If true, compress the chunks
            chunks (dict): Chunk sizes. time: the number of lead times in a chunk (default:
                as many as give chunks of about 100 MB). y and x, or location for an irregular
                grid: the number of points in a chunk along the spatial dimensions (default:
                the whole field). A chunk always holds one level and one ensemble member.
            convert (list): Other formats to convert the store to, as in the configuration:
                [{netcdf: {filename_pattern: ..., compression: ...}},
                 {grib: {filename_pattern: ..., grib_keys: ...}}]
            remove_after_convert (bool): Remove the store when it has been converted
        """
        super().__init__(predict_metadata, extra_variables)

        self.filename_pattern = filename_pattern
        self.workdir = workdir
        if variables is None:
            extract_variables = list(predict_metadata.variables)
        else:
            if extra_variables is None:
                extra_variables = []
            extract_variables = variables + extra_variables

        if accumulated_variables is None:
            accumulated_variables = []
        else:
            accumulated_variables = [v + "_acc" for v in accumulated_variables]
            extract_variables += accumulated_variables

        if domain_name is not None:
            proj4_str = projections.get_proj4_str(domain_name)

        self.layout = Layout(
            self.pm,
            extract_variables,
            accumulated_variables,
            interp_res=interp_res,
            latrange=latrange,
            lonrange=lonrange,
            proj4_str=proj4_str,
            mask_file=mask_file,
            mask_field=mask_field,
            global_attributes=global_attributes,
        )
        self.compression = compression
        self.chunks = _plain(chunks) if chunks is not None else {}
        spatial_keys = [key for keys in SPATIAL_CHUNKS.values() for key in keys]
        unknown = [key for key in self.chunks if key not in ["time"] + spatial_keys]
        if unknown:
            raise ValueError(
                f"chunks: unknown dimensions {unknown}; expected some of "
                f"{['time'] + spatial_keys}"
            )
        if any(key in spatial_keys for key in self.chunks):
            allowed = SPATIAL_CHUNKS[self.layout.num_spatial_dims]
            unknown = [key for key in self.chunks if key not in ["time"] + allowed]
            if unknown:
                raise ValueError(
                    f"chunks: {unknown} cannot be set for this output, whose spatial "
                    f"dimensions are {allowed}"
                )
        self.convert_targets = [_plain(target) for target in convert or []]
        for target in self.convert_targets:
            unknown = set(target) - set(bris.convert.WRITERS)
            if unknown or len(target) != 1:
                raise ValueError(
                    f"convert: expected one of {list(bris.convert.WRITERS)}, got {list(target)}"
                )
        if self._converts_to_grib:
            grib_conventions.check_grid(self.layout.grid_mapping_name)
        self.remove_after_convert = remove_after_convert
        if self.remove_after_convert and not self.has_targets:
            raise ValueError(
                "remove_after_convert: the output is not converted to anything (no convert), "
                "so nothing would be left of it"
            )

        # Accumulated variables at the end of the last block written, per store and member
        self._accumulated: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def get_filename(self, forecast_reference_time: int) -> str:
        """Get the filename for this forecast reference time"""
        return utils.expand_time_tokens(self.filename_pattern, forecast_reference_time)

    def get_store(self, forecast_reference_time: int) -> str:
        """Where the store of this forecast reference time is written"""
        return self.get_filename(forecast_reference_time)

    def get_stores(self) -> list[str]:
        """The stores that the processes of this run have written for this output

        The forecasts can be divided between groups of processes, so this is more than the
        stores that this process has written to.
        """
        # get_store with the time tokens as wildcards
        parts = re.split(r"(%.)", self.filename_pattern)
        pattern = "".join(
            ("%" if part == "%%" else "*") if i % 2 else glob.escape(part)
            for i, part in enumerate(parts)
        )
        stores = []
        for path in glob.glob(os.path.abspath(pattern)):
            info = zarrstore.read_info(path)
            if info is None or info.get("run_id") != zarrstore.run_id():
                continue
            # The wildcards also match the stores of an output with a longer name
            # (lam_%Y%m%d.zarr for %Y%m%d.zarr): keep the ones that have this output's name
            frt = info.get("forecast_reference_time")
            if frt is not None and self.get_store(frt) != path:
                continue
            stores.append(path)
        return sorted(stores)

    @property
    def _converts_to_grib(self) -> bool:
        return any("grib" in target for target in self.convert_targets)

    @property
    def has_targets(self) -> bool:
        """Are the stores converted to other formats?"""
        return len(self.convert_targets) > 0

    def get_targets(self, forecast_reference_time: int) -> list[dict]:
        """What the store of this forecast reference time is converted to, as bris.convert
        takes it"""
        targets = []
        for target in self.convert_targets:
            for name, options in target.items():
                options = dict(options)
                filename = utils.expand_time_tokens(
                    options.pop("filename_pattern"), forecast_reference_time
                )
                targets.append({"format": name, "filename": filename, **options})
        return targets

    def _time_chunk(self) -> int:
        if self.chunks.get("time") is not None:
            return max(1, min(int(self.chunks["time"]), self.pm.num_leadtimes))
        field_bytes = 4 * int(np.prod(self._spatial_chunks()))
        largest = max(1, int(CHUNK_BYTES // field_bytes))
        # Chunks of about the same size: the last chunk takes up the space of a whole one
        num_chunks = -(-self.pm.num_leadtimes // largest)
        return -(-self.pm.num_leadtimes // num_chunks)

    def _spatial_chunks(self) -> list[int]:
        """Chunk sizes along the spatial dimensions: as configured, else the whole field"""
        spatial_shape = self.layout.spatial_shape
        return [
            size
            if self.chunks.get(key) is None
            else max(1, min(int(self.chunks[key]), size))
            for key, size in zip(
                SPATIAL_CHUNKS[len(spatial_shape)], spatial_shape, strict=True
            )
        ]

    def _add_forecast(
        self, times: list, ensemble_member: int, pred: np.ndarray
    ) -> None:
        self.write_block(times, ensemble_member, pred)

    def write_block(
        self, times: list, ensemble_member: int, pred: np.ndarray, start: int = 0
    ) -> None:
        """Write a block of consecutive lead times of one ensemble member to the store

        The blocks of a member must be written in order, each by one call. The store is
        converted (if asked for) when a single-member forecast is complete; with several
        members, on finalize.

        Args:
            times: All the times of the forecast (np.datetime64), also outside this block
            pred: 3D array (leadtimes in the block, points, variables), with the extra
                variables (as add_forecast passes it on)
            start: Index into times of the first lead time in pred
        """
        t0 = pytime.perf_counter()
        frt = int(utils.datetime_to_unixtime(times[0]))
        path = self.get_store(frt)
        key = (frt, ensemble_member)
        if start == 0:
            self._ensure_store(path, frt, times, ensemble_member)
            self._accumulated[key] = {}
        root = zarr.open_group(path, mode="r+")
        ensemble = self.pm.num_members > 1
        accumulated = self._accumulated[key]

        def write(variable: str) -> None:
            name, level_index, values = self.layout.field_values(
                variable, pred, rotated, accumulated
            )
            index: list = [slice(start, start + len(values))]
            if level_index is not None:
                index.append(level_index)
            if ensemble:
                index.append(ensemble_member)
            root[name][tuple(index)] = values

        # The variables are prepared and written by a few threads: each variable (and level)
        # has chunks of its own, and most of the work is done without Python's global
        # interpreter lock
        self.layout.prepare()
        with ThreadPoolExecutor(max_workers=get_num_threads()) as pool:
            rotated = self.layout.rotated_winds(pred, pool.map)
            # list: raise what a thread raised
            list(pool.map(write, self.layout.extract_variables))

        if start + len(pred) >= len(times):
            del self._accumulated[key]
            if not ensemble and self.has_targets:
                # Nothing more will be written to the store
                self._convert(path)
        utils.LOGGER.debug(
            f"Zarr.write_block member {ensemble_member} to {path} in {pytime.perf_counter() - t0:.1f}s"
        )

    def _ensure_store(
        self, path: str, frt: int, times: list, ensemble_member: int
    ) -> None:
        """Create the store with its coordinates, attributes and (empty) variables, unless a
        process of this run already has"""
        fields = self.layout.fields
        time_chunk = self._time_chunk()
        num_spatial = len(self.layout.spatial_dims)

        spatial_chunks = self._spatial_chunks()

        def chunks(shape: list) -> list:
            # One level and one ensemble member per chunk: the members are written by
            # different processes, which must not write the same chunk
            lead = len(shape) - num_spatial
            return [time_chunk] + [1] * (lead - 1) + spatial_chunks

        times_ut = [int(t) for t in utils.datetime_to_unixtime(times)]
        layout = {
            k: [v["dims"], v["shape"], chunks(v["shape"])] for k, v in fields.items()
        }
        fingerprint = hashlib.sha256(
            json.dumps([times_ut, layout, bool(self.compression)]).encode()
        ).hexdigest()
        if zarrstore.is_current(path, fingerprint):
            return

        compressor = (
            numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
            if self.compression
            else None
        )
        ds = self.layout.skeleton(times)

        def build(tmp: str) -> None:
            root = zarr.open_group(tmp, mode="w")
            write_skeleton(root, ds, compressor)
            for name, field in fields.items():
                array = root.create_dataset(
                    name,
                    shape=field["shape"],
                    chunks=chunks(field["shape"]),
                    dtype="f4",
                    fill_value=np.float32(np.nan),
                    compressor=compressor,
                )
                array.attrs.update({bris.convert.DIMENSIONS: field["dims"]})
                array.attrs.update(field["attrs"])
            zarr.consolidate_metadata(tmp)

        # Zarr sorts variables and attributes by name. Their order in the layout is kept here,
        # so that a conversion can restore it.
        attribute_order = {"": list(ds.attrs)}
        attribute_order |= {name: list(var.attrs) for name, var in ds.variables.items()}
        attribute_order |= {
            name: list(field["attrs"]) for name, field in fields.items()
        }
        info = {
            "format_version": bris.convert.FORMAT_VERSION,
            "fields": list(fields),
            "spatial_dims": list(self.layout.spatial_dims),
            "variables": list(ds.variables) + list(fields),
            "attribute_order": attribute_order,
            "forecast_reference_time": frt,
            "convert": self.get_targets(frt),
            "remove_after_convert": self.remove_after_convert,
        }
        # A store left by an earlier run is replaced by the process that writes the first
        # member: exactly one of the processes that write to this store
        zarrstore.ensure(path, build, fingerprint, info, replace=ensemble_member == 0)

    def _convert(self, path: str) -> None:
        t0 = pytime.perf_counter()
        filenames = bris.convert.convert(path)
        if filenames:
            utils.LOGGER.debug(
                f"Zarr: converted {path} to {filenames} in {pytime.perf_counter() - t0:.1f}s"
            )

    def finalize(self) -> None:
        """Convert the stores, if asked for. A store that is not converted needs no
        finalizing: it is complete when its last member has been written."""
        t0 = pytime.perf_counter()
        if self.has_targets and self.pm.num_members > 1:
            # Single-member stores were converted when they were written
            for path in self.get_stores():
                self._convert(path)
        self._cleanup()
        utils.LOGGER.debug(
            f"{type(self).__name__}.finalize: {pytime.perf_counter() - t0:.1f}s"
        )

    def _cleanup(self) -> None:
        """Nothing to clean up: the stores are the output"""


def get_num_threads() -> int:
    """The number of threads that write the variables of a member: WRITE_THREADS, or the
    number of CPUs that this process may use if that is less"""
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count() or 1
    return max(1, min(WRITE_THREADS, num_cpus))


def write_skeleton(root: zarr.Group, ds: xr.Dataset, compressor=None) -> None:
    """Write the variables and attributes of an xarray dataset (see Layout.skeleton) to a
    zarr group as they are, in the way xarray reads them (dimension names in the
    _ARRAY_DIMENSIONS attribute)"""
    root.attrs.update(_json_attrs(ds.attrs))
    for name, var in ds.variables.items():
        values = np.asarray(var.values, dtype=var.encoding.get("dtype", var.dtype))
        array = root.create_dataset(
            name,
            data=values,
            shape=values.shape,
            chunks=values.shape if values.ndim > 0 else None,
            fill_value=values.dtype.type(np.nan) if values.dtype.kind == "f" else None,
            compressor=compressor,
        )
        array.attrs.update({bris.convert.DIMENSIONS: list(var.dims)})
        array.attrs.update(_json_attrs(var.attrs))


def _json_attrs(attrs: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray | np.generic) else value
        for key, value in attrs.items()
    }


def _plain(value):
    """A configuration value (e.g. from OmegaConf) as plain dicts and lists, for json"""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain(v) for v in value]
    return value
