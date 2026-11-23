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

import logging
import os
import sys
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
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

DIMENSIONS = "_ARRAY_DIMENSIONS"
CONSOLIDATED = (
    ".zmetadata"  # where zarr.consolidate_metadata puts the metadata of a store
)
TIME = "time"
MEMBER = "ensemble_member"


class Store:
    """A forecast stored by bris.outputs.zarr, opened for reading"""

    def __init__(self, path: str):
        self.path = path
        self.info = zarrstore.read_info(path) or {}
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is not a Zarr store: no such directory")
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

    def has_data(self, name: str, index: tuple) -> bool:
        """Has anything been written to the field at this (level, member) index? False for
        an ensemble member that never arrived."""
        array = self.root[name]
        lead = array.ndim - len(self.spatial_dims)
        if any(c != 1 for c in array.chunks[1:lead]):
            return True
        coords = (0,) + tuple(index) + (0,) * len(self.spatial_dims)
        try:
            return array._chunk_key(coords) in array.chunk_store
        except AttributeError:
            return True


def to_netcdf(path: str, filename: str, compression: bool = False) -> None:
    """Write the forecast in the store at path to a NetCDF file

    The fields are copied one chunk at a time, so that the memory use does not depend on the
    size of the forecast.

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

        missing_members = set()
        for name in fields:
            array = store.root[name]
            var = nc.variables[name]
            dims = store.dims(name)
            num_times, time_chunk = array.shape[0], array.chunks[0]
            for index in np.ndindex(*array.shape[1 : array.ndim - num_spatial]):
                if not store.has_data(name, index):
                    if MEMBER in dims:
                        missing_members.add(index[dims.index(MEMBER) - 1])
                    continue
                for start in range(0, num_times, time_chunk):
                    block = (slice(start, min(start + time_chunk, num_times)),) + index
                    var[block] = array[block]
        for member in sorted(missing_members):
            LOGGER.warning(
                f"netcdf: no data for ensemble member {member}, leaving it missing"
            )
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

    lats, lons = root["latitude"][...], root["longitude"][...]
    if num_spatial == 2:
        spatial_shape = root[store.fields[0]].shape[-2:]
        ny, nx = spatial_shape
    else:
        # only one location dimension
        ny, nx = len(lats), 1

    # set geometry from proj attributes
    proj_attrs = store.attrs("projection") if "projection" in root else {}
    geometry = {
        "gridDefinitionTemplateNumber": grib_conventions.GRID_TEMPLATES.get(
            proj_attrs.get("grid_mapping_name"), 30
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

    # The fields are read one chunk at a time and kept until the next chunk in time is needed
    cache: dict = {}

    def get_values(name: str, time_index: int, index: tuple) -> np.ndarray:
        array = root[name]
        time_chunk = array.chunks[0]
        start = time_index - time_index % time_chunk
        if cache.get("start") != start:
            cache.clear()
            cache["start"] = start
        key = (name, index)
        if key not in cache:
            cache[key] = array[(slice(start, start + time_chunk),) + index]
        return cache[key][time_index - start]

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "wb") as file_handle:
        for time_index, validtime in enumerate(validtimes):
            leadtime = validtime - origintime
            for name in store.fields:
                dis, cat, num, tosp, pdtn, pdtn_ens = grib_conventions.PARAMETERS[name]
                if num_members > 1:
                    pdtn = pdtn_ens
                for level_index, level_value, level_type in levels[name]:
                    for member in range(num_members):
                        index = level_index + ((member,) if num_members > 1 else ())
                        values = get_values(name, time_index, index)

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
                        ecc.codes_set(
                            grib, "dataDate", int(origintime.strftime("%Y%m%d"))
                        )
                        ecc.codes_set(
                            grib, "dataTime", int(origintime.strftime("%H%M"))
                        )
                        ecc.codes_set(
                            grib, "forecastTime", int(leadtime.total_seconds() / 3600)
                        )
                        ecc.codes_set(grib, "bitsPerValue", 24)
                        ecc.codes_set(grib, "packingType", "grid_ccsds")
                        values = np.asarray(values, dtype=np.float64).ravel()
                        missing = np.isnan(values)
                        if missing.any():
                            # Points without a value (e.g. a diagnostic variable at lead
                            # time 0, or an ensemble member that never arrived) are marked
                            # as missing in a bitmap
                            ecc.codes_set(grib, "bitmapPresent", 1)
                            ecc.codes_set(
                                grib, "missingValue", grib_conventions.MISSING_VALUE
                            )
                            values = np.where(
                                missing, grib_conventions.MISSING_VALUE, values
                            )
                        ecc.codes_set_values(grib, values)

                        if num_members > 1 and pdtn in [1, 2, 11]:
                            ecc.codes_set(grib, "typeOfEnsembleForecast", 1)
                            ecc.codes_set(grib, "perturbationNumber", member + 1)
                            ecc.codes_set(
                                grib, "numberOfForecastsInEnsemble", num_members
                            )

                        if tosp == 1:
                            ecc.codes_set(grib, "typeOfStatisticalProcessing", tosp)
                            ecc.codes_set(
                                grib, "yearOfEndOfOverallTimeInterval", validtime.year
                            )
                            ecc.codes_set(
                                grib, "monthOfEndOfOverallTimeInterval", validtime.month
                            )
                            ecc.codes_set(
                                grib, "dayOfEndOfOverallTimeInterval", validtime.day
                            )
                            ecc.codes_set(
                                grib, "hourOfEndOfOverallTimeInterval", validtime.hour
                            )

                            # forecastTime is start of time interval
                            # hard code to 6h
                            length_of_time_range = 6
                            ecc.codes_set(
                                grib,
                                "forecastTime",
                                int(leadtime.total_seconds() / 3600)
                                - length_of_time_range,
                            )
                            ecc.codes_set(
                                grib, "lengthOfTimeRange", length_of_time_range
                            )

                        for key, value in (grib_keys or {}).items():
                            ecc.codes_set(grib, key, value)

                        ecc.codes_write(grib, file_handle)
                        ecc.codes_release(grib)
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
            for (recorded in the store).

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
    if remove:
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
