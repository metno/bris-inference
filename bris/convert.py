"""Convert the Zarr stores written by bris (see bris.outputs.zarr) to NetCDF or Grib

A store holds one forecast in the layout of the NetCDF output: the same variables, dimensions
and attributes. Converting it to NetCDF is therefore a copy, and converting it to Grib encodes
each field with the metadata found in the store.

The conversion runs when the predictions end (the NetCDF and Grib outputs, and the Zarr
output with `convert`), or later in a job of its own, on the stores of a Zarr output:

    bris-convert out/pred_20230319T00Z.zarr --netcdf out/pred_20230319T00Z.nc
    bris-convert out/pred_*.zarr --netcdf "out/pred_%Y%m%dT%HZ.nc" -w 4

Time tokens in the filenames are expanded with the forecast reference time of each store.

This module must stay cheap to import (no torch).
"""

import contextlib
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from argparse import ArgumentParser
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from multiprocessing import get_context

import netCDF4
import numpy as np
import zarr

from bris import zarrstore
from bris.conventions import grib as grib_conventions

LOGGER = logging.getLogger("bris")

# HDF5 is usually not built thread-safe, and the outputs convert from the writer's threads
_NETCDF_LOCK = threading.Lock()

# The Grib conversion handles at most this many time steps at a time (one temporary file each)
GRIB_TIME_GROUP = 64

# The number of threads that read a store ahead of the thread that writes the file. Reading the
# store takes longer than writing the file; with a few readers the write sets the pace.
READ_THREADS = 4

# The version of the format of a store (major.minor): its variables, dimensions and attributes,
# and its sidecar (bris.zarrstore). Recorded in the sidecar as format_version. A store with a
# later version than this is refused, so raise it with a change that the code reading the
# version before would get wrong (e.g. other dimensions, or another meaning of a key in the
# sidecar), not with an addition that such code can ignore.
FORMAT_VERSION = "0.1"

DIMENSIONS = "_ARRAY_DIMENSIONS"
CONSOLIDATED = (
    ".zmetadata"  # where zarr.consolidate_metadata puts the metadata of a store
)
TIME = "time"
MEMBER = "ensemble_member"


def _version(version) -> tuple[int, int]:
    """A format version (major.minor) as numbers, to compare versions"""
    try:
        major, minor = str(version).split(".")
        return int(major), int(minor)
    except ValueError:
        raise ValueError(
            f"{version!r} is not a format version of a bris store (as {FORMAT_VERSION})"
        ) from None


class Store:
    """A forecast stored by bris.outputs.zarr, opened for reading"""

    def __init__(self, path: str):
        self.path = path
        self.info = zarrstore.read_info(path) or {}
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a Zarr store: no such directory")
        # A store without a version (written before it was recorded, or without a sidecar)
        # is read as the first version
        version = self.info.get("format_version", "0.1")
        if _version(version) > _version(FORMAT_VERSION):
            raise ValueError(
                f"{path} has format version {version}, and this bris reads stores up to "
                f"version {FORMAT_VERSION}: it was written by a later bris. Convert it with "
                "that one."
            )
        if os.path.isfile(os.path.join(path, CONSOLIDATED)):
            self.root = zarr.open_consolidated(path, mode="r")
        else:
            LOGGER.warning(
                f"{path} has no consolidated metadata ({CONSOLIDATED}), which bris writes when "
                "it creates a store. Reading the metadata of each variable instead; the "
                "store may be incomplete or changed by another tool."
            )
            self.root = zarr.open_group(path, mode="r")

    def dims(self, name: str) -> list[str]:
        return list(self.root[name].attrs[DIMENSIONS])

    def attrs(self, name: str | None = None) -> dict:
        """Attributes of a variable (or the global ones), in the order they were given when
        the store was made if that is known (zarr sorts them by name)"""
        node = self.root if name is None else self.root[name]
        attrs = {k: v for k, v in node.attrs.asdict().items() if k != DIMENSIONS}
        order = self.info.get("attribute_order", {}).get(name or "", [])
        keys = [k for k in order if k in attrs] + [k for k in attrs if k not in order]
        return {k: attrs[k] for k in keys}

    @property
    def variables(self) -> list[str]:
        """All variables, in the order they were added when the store was made if that is
        known"""
        names = [name for name, _ in self.root.arrays()]
        order = self.info.get("variables", [])
        return [n for n in order if n in names] + [n for n in names if n not in order]

    @property
    def fields(self) -> list[str]:
        """The variables holding the predictions"""
        if "fields" in self.info:
            return [name for name in self.info["fields"] if name in self.root]
        return [
            name
            for name in self.variables
            if self.root[name].ndim > 1 and self.dims(name)[0] == TIME
        ]

    @property
    def spatial_dims(self) -> list[str]:
        """The spatial dimensions of the fields: (y, x) or (location,)"""
        if "spatial_dims" in self.info:
            return list(self.info["spatial_dims"])
        dims = self.dims("latitude")
        if dims == ["latitude"]:
            return ["latitude", "longitude"]
        return dims

    @property
    def num_members(self) -> int:
        return len(self.root[MEMBER]) if MEMBER in self.root else 1

    @property
    def forecast_reference_time(self) -> float:
        return float(self.root["forecast_reference_time"][()])

    def chunks_written(self, name: str) -> dict[tuple, tuple[int, int]]:
        """For each (level, member) index of a field: how many of its chunks have been
        written, and how many it has

        A chunk that was never written reads as missing values: none of them for an ensemble
        member that never arrived, some of them for a field that a run did not finish.
        """
        array = self.root[name]
        lead = array.ndim - len(self.spatial_dims)
        counts = [
            -(-size // chunk)
            for size, chunk in zip(array.shape, array.chunks, strict=True)
        ]
        total = counts[0] * int(np.prod(counts[lead:]))
        written: Counter = Counter()
        with contextlib.suppress(OSError):
            for entry in os.listdir(os.path.join(self.path, name)):
                if re.fullmatch(r"\d+(\.\d+)*", entry):
                    key = tuple(int(i) for i in entry.split("."))
                    if len(key) == array.ndim:
                        written[key[1:lead]] += 1
        result = {}
        for index in np.ndindex(*array.shape[1:lead]):
            chunk = tuple(
                i // c for i, c in zip(index, array.chunks[1:lead], strict=True)
            )
            result[index] = (written[chunk], total)
        return result

    def missing(self, written: dict | None = None) -> tuple[list[int], list[str]]:
        """What the store lacks (see chunks_written)

        Args:
            written: chunks_written of each field, if already known

        Returns:
            The ensemble members without any data: members that never arrived
            The fields that lack data for other members (or for the only member): what a
                run that was killed while it wrote leaves behind
        """
        if written is None:
            written = {name: self.chunks_written(name) for name in self.fields}
        position = {
            name: self.dims(name).index(MEMBER) - 1
            for name in written
            if MEMBER in self.dims(name)
        }
        absent = set(range(self.num_members)) if position else set()
        for name, indices in written.items():
            for index, (n, _) in indices.items():
                if n > 0 and name in position:
                    absent.discard(index[position[name]])
        gaps = []
        for name, indices in written.items():
            for index, (n, total) in indices.items():
                member = index[position[name]] if name in position else None
                if n < total and member not in absent and name not in gaps:
                    gaps.append(name)
        return sorted(absent), gaps

    def report_missing(self, label: str, filename: str) -> dict:
        """Warn about the data that the store lacks, before it is converted to filename

        Returns:
            chunks_written of each field
        """
        written = {name: self.chunks_written(name) for name in self.fields}
        absent, gaps = self.missing(written)
        for member in absent:
            LOGGER.warning(
                f"{label}: no data for ensemble member {member}, leaving it missing"
            )
        if gaps:
            LOGGER.warning(
                f"{label}: {self.path} has no or incomplete data for {gaps}; the values "
                f"are missing in {filename}"
            )
        return written


def read_ahead(
    tasks: Iterable, read: Callable, num_threads: int = READ_THREADS
) -> Iterator[tuple]:
    """Yield (task, read(task)) for each task, in the order of the tasks

    The reads are done by a few threads, ahead of whoever uses the results: at most
    num_threads results are read or waiting at any time. The tasks are taken from their
    iterable in the calling thread.
    """
    tasks = iter(tasks)
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        pending: deque = deque()

        def refill() -> None:
            while len(pending) < num_threads:
                try:
                    task = next(tasks)
                except StopIteration:
                    return
                pending.append((task, pool.submit(read, task)))

        try:
            refill()
            while pending:
                task, future = pending.popleft()
                yield task, future.result()
                refill()
        finally:
            for _, future in pending:
                future.cancel()


def to_netcdf(path: str, filename: str, compression: bool = False) -> None:
    """Write the forecast in the store at path to a NetCDF file

    The fields are copied a few chunks at a time (see read_ahead), so that the memory use does
    not depend on the size of the forecast.

    Args:
        compression: If true, write compressed fields
    """
    t0 = time.perf_counter()
    store = Store(path)
    fields = store.fields
    num_spatial = len(store.spatial_dims)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with _NETCDF_LOCK, netCDF4.Dataset(filename, "w") as nc:
        nc.setncatts(store.attrs())
        for name in store.variables:
            array = store.root[name]
            dims = store.dims(name)
            for dim, size in zip(dims, array.shape, strict=True):
                if dim not in nc.dimensions:
                    nc.createDimension(dim, None if dim == TIME else size)
            kwargs = {}
            if name in fields:
                # Chunks hold one (time, level, member) slice of the field
                lead = array.ndim - num_spatial
                kwargs = {
                    "zlib": compression,
                    "chunksizes": [1] * lead + list(array.shape[lead:]),
                }
            fill_value = array.dtype.type(np.nan) if array.dtype.kind == "f" else None
            var = nc.createVariable(
                name, array.dtype, dims, fill_value=fill_value, **kwargs
            )
            var.setncatts(store.attrs(name))
            if name not in fields:
                var[...] = array[...]

        written = store.report_missing("netcdf", filename)
        arrays = {name: store.root[name] for name in fields}

        def blocks() -> Iterator[tuple]:
            """The parts of the fields to copy: a chunk in time of one level and member"""
            for name, array in arrays.items():
                num_times, time_chunk = array.shape[0], array.chunks[0]
                for index in np.ndindex(*array.shape[1 : array.ndim - num_spatial]):
                    if written[name][index][0] == 0:
                        # Nothing to copy: the variable keeps its fill value
                        continue
                    for start in range(0, num_times, time_chunk):
                        stop = min(start + time_chunk, num_times)
                        yield name, (slice(start, stop),) + index

        # Only this thread writes to the file (HDF5); the store is read ahead of it
        for (name, block), values in read_ahead(
            blocks(), lambda task: arrays[task[0]][task[1]]
        ):
            nc.variables[name][block] = values
    LOGGER.debug(
        f"convert.to_netcdf {path} -> {filename} in {time.perf_counter() - t0:.1f}s"
    )


def to_grib(path: str, filename: str, grib_keys: dict | None = None) -> None:
    """Write the forecast in the store at path to a Grib file (edition 2)

    Points without a value (NaN) are encoded as missing, with a bitmap.

    Args:
        grib_keys: Set these keys in every message
    """
    import eccodes as ecc

    t0 = time.perf_counter()
    store = Store(path)
    root = store.root
    num_spatial = len(store.spatial_dims)
    num_members = store.num_members

    unknown = [name for name in store.fields if name not in grib_conventions.PARAMETERS]
    if unknown:
        raise ValueError(f"grib: no parameter definition for {unknown}")
    store.report_missing("grib", filename)

    lats, lons = root["latitude"][...], root["longitude"][...]
    if num_spatial == 2:
        spatial_shape = root[store.fields[0]].shape[-2:]
        ny, nx = spatial_shape
    else:
        # only one location dimension
        ny, nx = len(lats), 1

    # set geometry from proj attributes
    proj_attrs = store.attrs("projection") if "projection" in root else {}
    grid_mapping_name = proj_attrs.get("grid_mapping_name")
    grib_conventions.check_grid(grid_mapping_name)
    geometry = {
        "gridDefinitionTemplateNumber": grib_conventions.GRID_TEMPLATES.get(
            grid_mapping_name, 30
        ),
        "latitudeOfSouthernPole": -90000000,
        "longitudeOfSouthernPole": 0,
        "latitudeOfFirstGridPoint": int(lats.flat[0] * 1_000_000),
        "longitudeOfFirstGridPoint": int(lons.flat[0] * 1_000_000),
        "LaD": proj_attrs.get("latitude_of_projection_origin", 63.3) * 1_000_000,
        "LoV": proj_attrs.get("longitude_of_central_meridian", 15.0) * 1_000_000,
    }
    if num_spatial == 2 and proj_attrs.get("grid_mapping_name") not in [
        None,
        "latitude_longitude",
    ]:
        # Dx/Dy in meters
        y_dim, x_dim = store.spatial_dims
        x, y = root[x_dim][...], root[y_dim][...]
        if len(x) > 1 and len(y) > 1 and int(x[1] - x[0]) and int(y[1] - y[0]):
            geometry["DxInMetres"] = int(x[1] - x[0])
            geometry["DyInMetres"] = int(y[1] - y[0])
    geometry |= {
        "Nx": nx,
        "Ny": ny,
        "Latin1": 63300000,
        "Latin2": 63300000,
        "shapeOfTheEarth": 6,
    }
    if grid_mapping_name == "latitude_longitude" and num_spatial == 2:
        # A regular grid of latitudes (increasing, as the scanning mode says) and longitudes
        geometry = {
            "gridDefinitionTemplateNumber": grib_conventions.GRID_TEMPLATES[
                grid_mapping_name
            ],
            "Ni": nx,
            "Nj": ny,
            "latitudeOfFirstGridPointInDegrees": float(lats[0]),
            "longitudeOfFirstGridPointInDegrees": float(lons[0]) % 360,
            "latitudeOfLastGridPointInDegrees": float(lats[-1]),
            "longitudeOfLastGridPointInDegrees": float(lons[-1]) % 360,
            "iDirectionIncrementInDegrees": float(lons[1] - lons[0]) if nx > 1 else 0,
            "jDirectionIncrementInDegrees": float(lats[1] - lats[0]) if ny > 1 else 0,
            "shapeOfTheEarth": 6,
        }

    origintime = datetime.fromtimestamp(store.forecast_reference_time, UTC)
    validtimes = [datetime.fromtimestamp(t, UTC) for t in root[TIME][...]]

    # One (level index, level value, level type) per level of each field
    levels = {}
    for name in store.fields:
        dims = store.dims(name)
        num_levels = root[name].ndim - num_spatial - 1 - (MEMBER in dims)
        if num_levels == 1:
            level_type = root[dims[1]].attrs.get("standard_name", "height")
            levels[name] = [
                ((i,), int(value), level_type)
                for i, value in enumerate(root[dims[1]][...])
            ]
        else:
            levels[name] = [((), 0, "surface")]

    # The messages of a time step are in the order of items
    items = [
        (name, level_index, level_value, level_type, member)
        for name in store.fields
        for level_index, level_value, level_type in levels[name]
        for member in range(num_members)
    ]

    def write_message(
        file_handle, values, validtime, name, level_value, level_type, member
    ):
        dis, cat, num, tosp, pdtn, pdtn_ens = grib_conventions.PARAMETERS[name]
        if num_members > 1:
            pdtn = pdtn_ens
        leadtime = validtime - origintime

        grib = ecc.codes_grib_new_from_samples("GRIB2")
        for key, value in geometry.items():
            ecc.codes_set(grib, key, value)

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
            grib,
            "typeOfFirstFixedSurface",
            grib_conventions.LEVEL_TYPES.get(level_type, 103),
        )
        ecc.codes_set(grib, "level", level_value)
        ecc.codes_set(grib, "dataDate", int(origintime.strftime("%Y%m%d")))
        ecc.codes_set(grib, "dataTime", int(origintime.strftime("%H%M")))
        ecc.codes_set(grib, "forecastTime", int(leadtime.total_seconds() / 3600))
        ecc.codes_set(grib, "bitsPerValue", 24)
        ecc.codes_set(grib, "packingType", "grid_ccsds")
        values = np.asarray(values, dtype=np.float64).ravel()
        missing = np.isnan(values)
        if missing.any():
            # Points without a value (e.g. a diagnostic variable at lead
            # time 0, or an ensemble member that never arrived) are marked
            # as missing in a bitmap
            ecc.codes_set(grib, "bitmapPresent", 1)
            ecc.codes_set(grib, "missingValue", grib_conventions.MISSING_VALUE)
            values = np.where(missing, grib_conventions.MISSING_VALUE, values)
        ecc.codes_set_values(grib, values)

        if num_members > 1 and pdtn in [1, 2, 11]:
            ecc.codes_set(grib, "typeOfEnsembleForecast", 1)
            ecc.codes_set(grib, "perturbationNumber", member + 1)
            ecc.codes_set(grib, "numberOfForecastsInEnsemble", num_members)

        if tosp == 1:
            ecc.codes_set(grib, "typeOfStatisticalProcessing", tosp)
            ecc.codes_set(grib, "yearOfEndOfOverallTimeInterval", validtime.year)
            ecc.codes_set(grib, "monthOfEndOfOverallTimeInterval", validtime.month)
            ecc.codes_set(grib, "dayOfEndOfOverallTimeInterval", validtime.day)
            ecc.codes_set(grib, "hourOfEndOfOverallTimeInterval", validtime.hour)

            # forecastTime is start of time interval
            # hard code to 6h
            length_of_time_range = 6
            ecc.codes_set(
                grib,
                "forecastTime",
                int(leadtime.total_seconds() / 3600) - length_of_time_range,
            )
            ecc.codes_set(grib, "lengthOfTimeRange", length_of_time_range)

        for key, value in (grib_keys or {}).items():
            ecc.codes_set(grib, key, value)

        ecc.codes_write(grib, file_handle)
        ecc.codes_release(grib)

    # The file has the messages of the first time step, then those of the next one, while a
    # chunk of the store has several time steps of one field. The time steps are therefore
    # taken a group at a time: each field (level, member) of the group is read once, and
    # its messages are spread over one temporary file per time step, which are put together
    # at the end of the group. This keeps one field of the group in memory, not all of them.
    num_times = len(validtimes)
    time_chunks = [root[name].chunks[0] for name in store.fields]
    group = max(1, min(time_chunks + [GRIB_TIME_GROUP]))
    folder = os.path.dirname(os.path.abspath(filename))
    os.makedirs(folder, exist_ok=True)
    with open(filename, "wb") as file_handle:
        for start in range(0, num_times, group):
            stop = min(start + group, num_times)
            with contextlib.ExitStack() as stack:
                # One time step is written straight to the file
                parts = [file_handle]
                if stop - start > 1:
                    parts = [
                        stack.enter_context(tempfile.TemporaryFile(dir=folder))
                        for _ in range(start, stop)
                    ]
                for name, level_index, level_value, level_type, member in items:
                    index = level_index + ((member,) if num_members > 1 else ())
                    block = root[name][(slice(start, stop),) + index]
                    for i, part in enumerate(parts):
                        write_message(
                            part,
                            block[i],
                            validtimes[start + i],
                            name,
                            level_value,
                            level_type,
                            member,
                        )
                if stop - start > 1:
                    for part in parts:
                        part.seek(0)
                        shutil.copyfileobj(part, file_handle, 16 * 1024 * 1024)
    LOGGER.debug(
        f"convert.to_grib {path} -> {filename} in {time.perf_counter() - t0:.1f}s"
    )


WRITERS = {"netcdf": to_netcdf, "grib": to_grib}


def convert(
    path: str, targets: list[dict] | None = None, remove: bool | None = None
) -> list[str]:
    """Convert the store at path

    Args:
        targets: What to convert to: dicts with format (netcdf or grib), filename and the
            options of that format's writer. Default: what the configuration of the run
            asked for (recorded in the store).
        remove: Remove the store afterwards? Default: as the configuration of the run asked
            for (recorded in the store). A store that was not converted to anything is kept,
            and so is a store with fields written in part (what a run that was killed left
            behind; see Store.missing).

    Returns:
        The files written
    """
    info = zarrstore.read_info(path) or {}
    if targets is None:
        targets = info.get("convert", [])
    if remove is None:
        remove = info.get("remove_after_convert", False)

    filenames = []
    for target in targets:
        options = {k: v for k, v in target.items() if k not in ["format", "filename"]}
        WRITERS[target["format"]](path, target["filename"], **options)
        filenames.append(target["filename"])
    if remove and filenames:
        _, gaps = Store(path).missing()
        if gaps:
            LOGGER.warning(
                f"{path} lacks data for {gaps} and is kept; remove it when it is no "
                "longer needed"
            )
        else:
            zarrstore.remove(path)
    return filenames


def expand_time_tokens(filename: str, unixtime: float) -> str:
    """Expand time tokens in a filename and return absolute path."""
    return os.path.abspath(time.strftime(filename, time.gmtime(unixtime)))


def _convert_job(job: tuple) -> list[str]:
    return convert(*job)


def main(arg_list: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Convert Zarr stores written by bris to NetCDF or Grib. Without --netcdf "
        "or --grib, each store is converted to what the configuration of its run asked for."
    )
    parser.add_argument("stores", nargs="+", help="Zarr stores, one forecast in each")
    parser.add_argument(
        "--netcdf",
        metavar="FILENAME",
        help="Write this NetCDF file. Time tokens (e.g. %%Y%%m%%dT%%HZ) are expanded with "
        "the forecast reference time of the store",
    )
    parser.add_argument("--grib", metavar="FILENAME", help="Write this Grib file")
    parser.add_argument(
        "--compression", action="store_true", help="Compress the NetCDF file"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--remove_zarr",
        dest="remove",
        action="store_const",
        const=True,
        default=None,
        help="Remove each store after converting it",
    )
    group.add_argument(
        "--keep_zarr",
        dest="remove",
        action="store_const",
        const=False,
        help="Keep the stores (default: as the configuration of the run asked for)",
    )
    parser.add_argument(
        "-w",
        type=int,
        dest="workers",
        default=1,
        help="Number of stores to convert at the same time",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(arg_list)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    jobs = []
    for path in args.stores:
        path = path.rstrip("/")
        targets = None
        if args.netcdf or args.grib:
            frt = Store(path).forecast_reference_time
            targets = []
            if args.netcdf:
                targets.append(
                    {
                        "format": "netcdf",
                        "filename": expand_time_tokens(args.netcdf, frt),
                        "compression": args.compression,
                    }
                )
            if args.grib:
                targets.append(
                    {"format": "grib", "filename": expand_time_tokens(args.grib, frt)}
                )
        elif not (zarrstore.read_info(path) or {}).get("convert"):
            parser.error(
                f"{path}: the run did not ask for a conversion of this store; use --netcdf "
                "or --grib"
            )
        jobs.append((path, targets, args.remove))

    filenames = [
        target["filename"] for _, targets, _ in jobs for target in targets or []
    ]
    if len(filenames) != len(set(filenames)):
        parser.error(
            "Several stores would be written to the same file; use time tokens in the filename"
        )

    t0 = time.perf_counter()
    if args.workers > 1 and len(jobs) > 1:
        # A new interpreter per worker: HDF5 does not survive a fork
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)), mp_context=get_context("spawn")
        ) as pool:
            results = list(pool.map(_convert_job, jobs))
    else:
        results = [_convert_job(job) for job in jobs]
    for (path, _, _), written in zip(jobs, results, strict=True):
        LOGGER.info(f"{path} -> {', '.join(written)}")
    LOGGER.info(f"Converted {len(jobs)} store(s) in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
