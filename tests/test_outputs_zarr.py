import logging
import multiprocessing
import os
import re

import numpy as np
import pytest
import xarray as xr
import zarr

import bris.convert
import bris.outputs.zarr
from bris import outputs, zarrstore
from bris.outputs.grib import Grib
from bris.outputs.netcdf import Netcdf
from bris.outputs.zarr import Zarr
from bris.predict_metadata import PredictMetadata

FRT = 1672552800
VARIABLES = ["u_800", "u_600", "v_800", "v_600", "2t", "10u", "10v", "tp", "msl"]


def get_pm(num_members=1, num_leadtimes=4, gridded=True):
    """A 3 x 4 grid around Oslo, or the same points as an irregular grid"""
    lats, lons = np.meshgrid(
        np.linspace(59, 60, 3), np.linspace(10, 11.5, 4), indexing="ij"
    )
    altitudes = np.arange(12, dtype=np.float32)
    leadtimes = np.arange(0, 3600 * num_leadtimes, 3600)
    field_shape = [3, 4] if gridded else (12,)
    return PredictMetadata(
        list(VARIABLES),
        lats.flatten(),
        lons.flatten(),
        altitudes,
        leadtimes,
        num_members,
        field_shape,
    )


def get_pred(pm, seed=0):
    """Predictions (leadtimes, points, variables, members), with a missing value in tp"""
    rng = np.random.default_rng(seed)
    pred = rng.random(pm.shape + [pm.num_members]).astype(np.float32)
    pred[0, 0, pm.variables.index("tp")] = np.nan
    return pred


def get_times(pm, frt=FRT):
    return (frt + pm.leadtimes).astype("datetime64[s]")


def add_members(output, pm, pred, members=None, frt=FRT):
    for member in range(pm.num_members) if members is None else members:
        output.add_forecast(get_times(pm, frt), member, pred[..., member])


def open_zarr(path):
    """The store as xarray reads it (xr.open_zarr(path, decode_times=False)), but without
    xarray's zarr backend, which newer versions of xarray only have for zarr-python 3"""
    root = zarr.open_consolidated(str(path), mode="r")
    variables = {}
    for name, array in root.arrays():
        attrs = array.attrs.asdict()
        dims = attrs.pop("_ARRAY_DIMENSIONS")
        variables[name] = xr.Variable(dims, array[...], attrs)
    ds = xr.Dataset(variables, attrs=root.attrs.asdict())
    return xr.decode_cf(ds, decode_times=False)


def test_xarray(tmp_path):
    """xarray reads the store, when it supports the installed version of zarr-python"""
    pm = get_pm(num_members=2)
    output = Zarr(pm, "", str(tmp_path / "test.zarr"), domain_name="meps")
    add_members(output, pm, get_pred(pm))
    try:
        ds = xr.open_zarr(tmp_path / "test.zarr", decode_times=False).load()
    except TypeError as e:
        pytest.skip(
            f"xarray {xr.__version__} cannot read with zarr {zarr.__version__}: {e}"
        )
    assert_same(ds, open_zarr(tmp_path / "test.zarr"))


def open_netcdf(path):
    with xr.open_dataset(path, decode_times=False) as ds:
        return ds.load()


def assert_same(a: xr.Dataset, b: xr.Dataset):
    """Same variables, dimensions, types, attributes and values (the creation time aside)"""
    a.attrs.pop("history")
    b.attrs.pop("history")
    xr.testing.assert_identical(a, b)
    for name in a.variables:
        assert a[name].dtype == b[name].dtype, name
        assert a[name].dims == b[name].dims, name


OPTIONS = {
    "all": {},
    "projected": {
        "domain_name": "meps",
        "variables": ["2t", "10u", "10v", "u_800", "v_800", "u_600", "v_600", "tp"],
        "accumulated_variables": ["tp"],
        "extra_variables": ["ws"],
        "global_attributes": {"creator": "met.no", "version": 2},
    },
    "interpolated": {"interp_res": 0.25, "variables": ["2t", "tp"]},
}


@pytest.mark.parametrize("num_members", [1, 3])
@pytest.mark.parametrize("gridded", [True, False])
@pytest.mark.parametrize("options", ["all", "projected", "interpolated"])
def test_same_as_netcdf(tmp_path, num_members, gridded, options):
    """The store reads the same with xarray as the NetCDF file of the same forecast"""
    if options == "projected" and not gridded:
        return
    pm = get_pm(num_members, gridded=gridded)
    pred = get_pred(pm)
    kwargs = OPTIONS[options]

    zarr_output = Zarr(
        pm, str(tmp_path / "work1"), str(tmp_path / "%Y%m%dT%HZ.zarr"), **kwargs
    )
    netcdf_output = Netcdf(
        pm, str(tmp_path / "work2"), str(tmp_path / "%Y%m%dT%HZ.nc"), **kwargs
    )
    for output in [zarr_output, netcdf_output]:
        add_members(output, pm, pred)

    # The store is complete without finalize
    ds = open_zarr(tmp_path / "20230101T06Z.zarr")
    netcdf_output.finalize()
    assert_same(ds, open_netcdf(tmp_path / "20230101T06Z.nc"))
    assert not os.path.exists(tmp_path / "work2")

    zarr_output.finalize()
    assert sorted(os.listdir(tmp_path)) == ["20230101T06Z.nc", "20230101T06Z.zarr"]


def test_layout(tmp_path, monkeypatch):
    """Dimensions, chunks (one level and one member per chunk) and missing members"""
    pm = get_pm(num_members=3)
    pred = get_pred(pm)
    output = Zarr(
        pm,
        str(tmp_path / "work"),
        str(tmp_path / "%Y%m%dT%HZ.zarr"),
        variables=["2t", "u_800", "u_600"],
        chunks={"time": 3},
    )
    add_members(output, pm, pred, members=[2, 0])
    output.finalize()

    root = zarr.open_group(str(tmp_path / "20230101T06Z.zarr"), mode="r")
    assert root.store.getsize() > 0
    array = root["x_wind_pl"]
    assert array.attrs["_ARRAY_DIMENSIONS"] == [
        "time",
        "pressure",
        "ensemble_member",
        "y",
        "x",
    ]
    assert array.shape == (4, 2, 3, 3, 4)
    assert array.chunks == (3, 1, 1, 3, 4)
    assert array.compressor is None
    assert list(root["pressure"][:]) == [600, 800]

    u_800 = pred[:, :, pm.variables.index("u_800"), :].reshape(4, 3, 4, 3)
    assert np.array_equal(array[:, 1, 0], u_800[..., 0])
    assert np.array_equal(array[:, 1, 2], u_800[..., 2])
    assert np.all(np.isnan(array[:, :, 1]))

    # Default: all lead times in a chunk, when that is less than about 100 MB
    output = Zarr(
        pm, "", str(tmp_path / "default.zarr"), variables=["2t"], compression=True
    )
    add_members(output, pm, pred)
    array = zarr.open_group(str(tmp_path / "default.zarr"), mode="r")[
        "air_temperature_2m"
    ]
    assert array.chunks == (4, 1, 1, 3, 4)
    assert array.compressor is not None

    # ... and otherwise chunks of about the same size
    pm = get_pm(num_leadtimes=21)
    output = Zarr(pm, "", "", variables=["2t"])
    for chunk_bytes, expected in [(48 * 15, 11), (48 * 10, 7), (48 * 21, 21), (1, 1)]:
        monkeypatch.setattr(bris.outputs.zarr, "CHUNK_BYTES", chunk_bytes)
        assert output._time_chunk() == expected


@pytest.mark.parametrize("num_members", [1, 2])
@pytest.mark.parametrize(
    "gridded,options,chunks,expected",
    [
        (True, "projected", {"time": 3, "y": 2, "x": 3}, (3, 2, 3)),
        # Only one dimension set, and more than there is
        (True, "all", {"x": 100, "y": 1}, (4, 1, 4)),
        # latitude and longitude of the interpolated grid
        (False, "interpolated", {"time": 1, "y": 2, "x": 2}, (1, 2, 2)),
        (False, "all", {"location": 5}, (4, 5)),
    ],
)
def test_spatial_chunks(tmp_path, num_members, gridded, options, chunks, expected):
    """The fields can be chunked along the spatial dimensions, with the same content"""
    pm = get_pm(num_members, gridded=gridded)
    pred = get_pred(pm)
    kwargs = OPTIONS[options]

    chunked = Zarr(pm, "", str(tmp_path / "chunked.zarr"), chunks=chunks, **kwargs)
    whole = Zarr(pm, "", str(tmp_path / "whole.zarr"), **kwargs)
    netcdf = Netcdf(pm, str(tmp_path / "work"), str(tmp_path / "netcdf.nc"), **kwargs)
    for output in [chunked, whole, netcdf]:
        # The members take turns, as they would in different processes
        add_members(output, pm, pred, members=reversed(range(num_members)))
        output.finalize()

    root = zarr.open_group(str(tmp_path / "chunked.zarr"), mode="r")
    array = root["air_temperature_2m"]
    num_lead = array.ndim - len(expected) + 1
    assert array.chunks == (expected[0],) + (1,) * (num_lead - 1) + expected[1:]
    # The coordinates are small and not chunked
    assert root["latitude"].chunks == root["latitude"].shape

    assert_same(
        open_zarr(tmp_path / "chunked.zarr"), open_zarr(tmp_path / "whole.zarr")
    )
    assert_same(
        open_zarr(tmp_path / "chunked.zarr"), open_netcdf(tmp_path / "netcdf.nc")
    )

    # ... and it converts to the same NetCDF file
    bris.convert.to_netcdf(
        str(tmp_path / "chunked.zarr"), str(tmp_path / "converted.nc")
    )
    assert_same(
        open_netcdf(tmp_path / "converted.nc"), open_netcdf(tmp_path / "netcdf.nc")
    )


def test_chunks_of_other_layout(tmp_path):
    """Chunk sizes for dimensions that the output does not have are refused"""
    with pytest.raises(ValueError, match="location"):
        Zarr(get_pm(gridded=True), "", "x.zarr", chunks={"location": 4})
    with pytest.raises(ValueError, match="x"):
        Zarr(get_pm(gridded=False), "", "x.zarr", chunks={"x": 4})
    with pytest.raises(ValueError, match="member"):
        Zarr(get_pm(), "", "x.zarr", chunks={"member": 2})
    # An irregular grid interpolated to lat/lon is gridded
    Zarr(get_pm(gridded=False), "", "x.zarr", interp_res=0.25, chunks={"x": 4})


@pytest.mark.parametrize("options", ["projected", "interpolated"])
def test_threads(tmp_path, monkeypatch, options):
    """The variables of a member are written by several threads, with the same result as by
    one, and what goes wrong in a thread is raised"""
    pm = get_pm(num_members=2)
    pred = get_pred(pm)
    for num_threads in [1, 4]:
        monkeypatch.setattr(bris.outputs.zarr, "WRITE_THREADS", num_threads)
        assert bris.outputs.zarr.get_num_threads() in [1, num_threads]
        output = Zarr(pm, "", str(tmp_path / f"{num_threads}.zarr"), **OPTIONS[options])
        add_members(output, pm, pred)
    assert_same(open_zarr(tmp_path / "1.zarr"), open_zarr(tmp_path / "4.zarr"))

    output = Zarr(pm, "", str(tmp_path / "fails.zarr"), **OPTIONS[options])
    with pytest.raises(IndexError):
        # Predictions without most of the variables
        output.write_block(get_times(pm), 0, pred[:, :, :2, 0])


def test_blocks(tmp_path):
    """Writing a member in blocks of lead times gives the same as writing it at once, also
    for the variables accumulated over lead time"""
    pm = get_pm(num_members=2, num_leadtimes=7)
    pred = get_pred(pm)
    times = get_times(pm)
    kwargs = {
        "variables": ["2t", "tp"],
        "accumulated_variables": ["tp"],
        "chunks": {"time": 2},
    }

    whole = Zarr(pm, "", str(tmp_path / "whole.zarr"), **kwargs)
    blocks = Zarr(pm, "", str(tmp_path / "blocks.zarr"), **kwargs)
    for member in range(pm.num_members):
        whole.add_forecast(times, member, pred[..., member])
    for start, end in [(0, 2), (2, 6), (6, 7)]:
        # The members take turns, as they would in different processes
        for member in range(pm.num_members):
            blocks.write_block(times, member, pred[start:end, ..., member], start=start)

    a, b = open_zarr(tmp_path / "whole.zarr"), open_zarr(tmp_path / "blocks.zarr")
    assert np.all(np.isfinite(a["precipitation_amount_acc"].values))
    assert_same(a, b)


def _write_member(path, member, num_members, barrier):
    os.environ["BRIS_RUN_ID"] = "test_processes"
    pm = get_pm(num_members)
    output = Zarr(pm, "", path, domain_name="meps")
    barrier.wait()
    for frt in [FRT, FRT + 6 * 3600]:
        output.add_forecast(get_times(pm, frt), member, get_pred(pm, frt)[..., member])


def test_processes(tmp_path):
    """The members are written by different processes at the same time, into stores that
    the first of them creates. A store left by an earlier run is replaced."""
    num_members = 4
    pm = get_pm(num_members)
    path = str(tmp_path / "%Y%m%dT%HZ.zarr")
    stale = tmp_path / "20230101T06Z.zarr"
    os.makedirs(stale / "air_temperature_2m")
    zarrstore.write_info(str(stale), {"run_id": "an earlier run"})

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(num_members)
    processes = [
        context.Process(target=_write_member, args=(path, member, num_members, barrier))
        for member in range(num_members)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0

    assert sorted(os.listdir(tmp_path)) == ["20230101T06Z.zarr", "20230101T12Z.zarr"]
    for frt, name in [
        (FRT, "20230101T06Z.zarr"),
        (FRT + 6 * 3600, "20230101T12Z.zarr"),
    ]:
        ds = open_zarr(tmp_path / name)
        index = pm.variables.index("2t")
        expected = get_pred(pm, frt)[:, :, index, :].reshape(4, 3, 4, num_members)
        values = ds["air_temperature_2m"].values[:, 0]
        assert np.array_equal(np.moveaxis(values, 1, -1), expected)


def test_convert(tmp_path):
    """The store is converted to the formats asked for: when it is written for one member,
    on finalize for an ensemble"""
    for num_members in [1, 2]:
        pm = get_pm(num_members)
        pred = get_pred(pm)[..., [m for m in range(num_members)]]
        pred = np.nan_to_num(pred)  # grib has no missing values here
        folder = tmp_path / str(num_members)
        output = outputs.instantiate(
            "zarr",
            pm,
            str(folder / "work"),
            {
                "filename_pattern": str(folder / "%Y%m%dT%HZ.zarr"),
                "variables": ["2t", "10u", "10v"],
                "domain_name": "meps",
                "convert": [
                    {
                        "netcdf": {
                            "filename_pattern": str(folder / "%Y%m%dT%HZ.nc"),
                            "compression": True,
                        }
                    },
                    {"grib": {"filename_pattern": str(folder / "%Y%m%dT%HZ.grib")}},
                ],
            },
        )
        add_members(output, pm, pred)
        converted = sorted(os.listdir(folder)) == [
            "20230101T06Z.grib",
            "20230101T06Z.nc",
            "20230101T06Z.zarr",
        ]
        assert converted == (num_members == 1)
        output.finalize()
        assert sorted(os.listdir(folder)) == [
            "20230101T06Z.grib",
            "20230101T06Z.nc",
            "20230101T06Z.zarr",
        ]
        assert_same(
            open_zarr(folder / "20230101T06Z.zarr"),
            open_netcdf(folder / "20230101T06Z.nc"),
        )
        with xr.open_dataset(
            folder / "20230101T06Z.grib",
            backend_kwargs={"filter_by_keys": {"shortName": "2t"}},
            decode_timedelta=True,
        ) as ds:
            assert ds["t2m"].size == 4 * 12 * num_members

    with pytest.raises(ValueError):
        Zarr(pm, "", "x.zarr", convert=[{"csv": {"filename_pattern": "x.csv"}}])


@pytest.mark.parametrize("kind", ["netcdf", "zarr"])
def test_forecasts_in_groups(tmp_path, monkeypatch, kind):
    """The forecasts can be divided between groups of processes. The main process converts
    the stores of all of them on finalize, also the ones it did not write to, and the stores
    of an earlier run are replaced whichever group writes to them."""
    monkeypatch.setenv("BRIS_RUN_ID", "earlier")
    pm = get_pm(num_members=2)
    pred = get_pred(pm)
    frts = [FRT, FRT + 6 * 3600]

    def get_output():
        if kind == "netcdf":
            return Netcdf(pm, str(tmp_path / "work"), str(tmp_path / "%Y%m%dT%HZ.nc"))
        return Zarr(
            pm,
            "",
            str(tmp_path / "[x]_%Y%m%dT%HZ.zarr"),
            convert=[{"netcdf": {"filename_pattern": str(tmp_path / "%Y%m%dT%HZ.nc")}}],
        )

    # An earlier run leaves its stores behind
    earlier = get_output()
    for frt in frts:
        add_members(earlier, pm, pred * 0, frt=frt)

    monkeypatch.setenv("BRIS_RUN_ID", "this")
    # One output object per process: the first group has the main process
    groups = [get_output(), get_output()]
    for output, frt in zip(groups, frts, strict=True):
        add_members(output, pm, pred, frt=frt)
    groups[0].finalize()

    for name in ["20230101T06Z.nc", "20230101T12Z.nc"]:
        ds = open_netcdf(tmp_path / name)
        expected = pred[:, :, pm.variables.index("2t"), :].reshape(4, 3, 4, 2)
        values = np.moveaxis(ds["air_temperature_2m"].values[:, 0], 1, -1)
        assert np.array_equal(values, expected)
    if kind == "netcdf":
        assert not os.path.exists(tmp_path / "work")
    else:
        assert len(groups[0].get_stores()) == 2


def test_remove_after_convert(tmp_path):
    pm = get_pm()
    output = Zarr(
        pm,
        "",
        str(tmp_path / "%Y%m%dT%HZ.zarr"),
        convert=[{"netcdf": {"filename_pattern": str(tmp_path / "%Y%m%dT%HZ.nc")}}],
        remove_after_convert=True,
    )
    add_members(output, pm, get_pred(pm))
    output.finalize()
    assert os.listdir(tmp_path) == ["20230101T06Z.nc"]


def test_store_that_is_not_converted_is_kept(tmp_path):
    """Removing the store is for after a conversion: without one the store is the output"""
    pm = get_pm()
    pattern = str(tmp_path / "%Y%m%dT%HZ.zarr")
    with pytest.raises(ValueError, match="remove_after_convert"):
        Zarr(pm, "", pattern, remove_after_convert=True)

    output = Zarr(pm, "", pattern)
    add_members(output, pm, get_pred(pm))
    output.finalize()
    store = str(tmp_path / "20230101T06Z.zarr")
    assert os.path.isdir(store)

    assert bris.convert.convert(store, targets=[], remove=True) == []
    assert os.path.isdir(store)
    # ... also when the store itself says that it is to be removed
    zarrstore.write_info(
        store, {**zarrstore.read_info(store), "remove_after_convert": True}
    )
    assert bris.convert.convert(store) == []
    assert os.path.isdir(store)


def test_store_of_an_earlier_bris_with_the_same_run_id(tmp_path):
    """A store that an earlier bris left under the same run identifier (e.g. in the same
    interactive job step) is not taken for this one's when it was to be converted to something
    else: the forecast would end up in the file that the earlier one asked for"""
    pm = get_pm(num_members=2)
    work = str(tmp_path / "work")
    earlier = Netcdf(
        pm, work, str(tmp_path / "earlier_%Y%m%dT%HZ.nc"), remove_intermediate=False
    )
    add_members(earlier, pm, get_pred(pm, seed=0))
    # ... and it never got to finalize

    pred = get_pred(pm, seed=1)
    output = Netcdf(pm, work, str(tmp_path / "this_%Y%m%dT%HZ.nc"))
    add_members(output, pm, pred)
    output.finalize()
    assert os.listdir(tmp_path) == ["this_20230101T06Z.nc"]
    ds = open_netcdf(tmp_path / "this_20230101T06Z.nc")
    # (time, points, member) -> (time, member, y, x)
    expected = pred[:, :, pm.variables.index("msl"), :].transpose(0, 2, 1)
    np.testing.assert_array_equal(
        ds["air_pressure_at_sea_level"].values.squeeze(),
        expected.reshape(4, 2, 3, 4),
    )

    # The same for a store that was not to be converted, and other global attributes
    pattern = str(tmp_path / "%Y%m%dT%HZ.zarr")
    add_members(Zarr(pm, "", pattern, global_attributes={"source": "a"}), pm, pred)
    convert = [{"netcdf": {"filename_pattern": str(tmp_path / "%Y%m%dT%HZ.nc")}}]
    output = Zarr(pm, "", pattern, convert=convert, global_attributes={"source": "b"})
    add_members(output, pm, pred)
    output.finalize()
    assert os.path.exists(tmp_path / "20230101T06Z.nc")
    assert open_zarr(tmp_path / "20230101T06Z.zarr").attrs["source"] == "b"


def test_format_version(tmp_path):
    """A store records the version of its format, and a store of a later version is refused
    instead of being read as something it is not"""
    pm = get_pm()
    output = Zarr(pm, "", str(tmp_path / "%Y%m%dT%HZ.zarr"))
    add_members(output, pm, get_pred(pm))
    output.finalize()
    store = str(tmp_path / "20230101T06Z.zarr")
    info = zarrstore.read_info(store)
    assert info["format_version"] == bris.convert.FORMAT_VERSION

    def to_netcdf(name):
        target = {"format": "netcdf", "filename": str(tmp_path / name)}
        return bris.convert.convert(store, [target])

    assert info["format_version"] == "0.1"

    # 0.10 is later than 0.9: the parts are numbers
    for later in ["0.2", "0.10", "1.0"]:
        zarrstore.write_info(store, {**info, "format_version": later})
        with pytest.raises(ValueError, match=f"format version {later}"):
            to_netcdf("later.nc")
    zarrstore.write_info(store, {**info, "format_version": "latest"})
    with pytest.raises(ValueError, match="not a format version"):
        to_netcdf("later.nc")
    assert not os.path.exists(tmp_path / "later.nc")

    # A store from before the version was recorded has the first version
    del info["format_version"]
    zarrstore.write_info(store, info)
    assert to_netcdf("before.nc") == [str(tmp_path / "before.nc")]


@pytest.mark.parametrize("num_members", [1, 3])
@pytest.mark.parametrize("gridded", [True, False])
@pytest.mark.parametrize("options", ["all", "projected", "interpolated"])
def test_written_store_is_valid(tmp_path, num_members, gridded, options):
    """What the output writes is what the conversion takes for a store of bris: the two are
    separate programs, and this is what they agree on"""
    if options == "projected" and not gridded:
        return
    pm = get_pm(num_members, gridded=gridded)
    output = Zarr(pm, "", str(tmp_path / "%Y%m%dT%HZ.zarr"), **OPTIONS[options])
    # The store is valid from the first member on
    add_members(output, pm, get_pred(pm), members=[0])
    store = bris.convert.Store(str(tmp_path / "20230101T06Z.zarr"))
    assert store.problems() == []
    assert store.fields


def _drop_dimension_names(root, info):
    del root["air_temperature_2m"].attrs["_ARRAY_DIMENSIONS"]


def _drop_reference_time(root, info):
    del root["forecast_reference_time"]


def _swap_dimensions(root, info):
    dims = root["air_temperature_2m"].attrs["_ARRAY_DIMENSIONS"]
    root["air_temperature_2m"].attrs["_ARRAY_DIMENSIONS"] = dims[:1] + dims[:0:-1]


def _other_length(root, info):
    array = root.zeros("other", shape=(7,), dtype="f4")
    array.attrs["_ARRAY_DIMENSIONS"] = ["time"]


def _double_precision(root, info):
    field = root["air_temperature_2m"]
    array = root.zeros("double", shape=field.shape, dtype="f8")
    array.attrs["_ARRAY_DIMENSIONS"] = field.attrs["_ARRAY_DIMENSIONS"]
    info["fields"].append("double")


def _drop_members(root, info):
    del root["ensemble_member"]


def _field_not_in_store(root, info):
    info["fields"].append("air_pressure_at_sea_level_2")


def _unknown_conversion(root, info):
    info["convert"] = [{"format": "feather", "filename": "a.feather"}]


@pytest.mark.parametrize(
    "change,problem",
    [
        (_drop_dimension_names, "air_temperature_2m lacks the names of its 5 dim"),
        (_drop_reference_time, "no variable forecast_reference_time"),
        (_swap_dimensions, "air_temperature_2m has the dimensions"),
        (_other_length, "dimension time has length"),
        (_double_precision, "double is float64, not float32"),
        (_drop_members, "values of the dimension ensemble_member"),
        (_field_not_in_store, "air_pressure_at_sea_level_2.* are not in the store"),
        (_unknown_conversion, "feather.* is not a conversion"),
    ],
)
def test_invalid_store(tmp_path, change, problem):
    """A store that is not as bris writes it is refused with what is wrong with it, before
    anything is converted"""
    pm = get_pm(num_members=2)
    output = Zarr(pm, "", str(tmp_path / "%Y%m%dT%HZ.zarr"))
    add_members(output, pm, get_pred(pm))
    output.finalize()
    store = str(tmp_path / "20230101T06Z.zarr")

    info = zarrstore.read_info(store)
    change(zarr.open_group(store, mode="r+"), info)
    zarr.consolidate_metadata(store)
    zarrstore.write_info(store, info)

    target = {"format": "netcdf", "filename": str(tmp_path / "a.nc")}
    with pytest.raises(ValueError, match="not a forecast as bris stores it") as error:
        bris.convert.convert(store, [target])
    assert re.search(problem, str(error.value))
    assert not os.path.exists(tmp_path / "a.nc")


def test_convert_recorded_targets(tmp_path):
    """Without --netcdf or --grib, bris-convert converts a store to what its output asked for
    (here: the stores of a netcdf output that was not finalized), with the same result"""
    pm = get_pm(num_members=2)
    pred = get_pred(pm)
    kwargs = {"domain_name": "meps", "accumulated_variables": ["tp"]}
    frts = [FRT, FRT + 6 * 3600]

    complete = Netcdf(
        pm, str(tmp_path / "work1"), str(tmp_path / "complete_%Y%m%dT%HZ.nc"), **kwargs
    )
    interrupted = Netcdf(
        pm, str(tmp_path / "work2"), str(tmp_path / "later_%Y%m%dT%HZ.nc"), **kwargs
    )
    for output in [complete, interrupted]:
        for frt in frts:
            add_members(output, pm, pred, frt=frt)
    complete.finalize()

    stores = sorted(
        str(tmp_path / "work2" / name) for name in os.listdir(tmp_path / "work2")
    )
    assert len(stores) == 2
    assert not os.path.exists(tmp_path / "work1")
    assert not os.path.exists(tmp_path / "later_20230101T06Z.nc")

    assert bris.convert.main(stores) == 0
    assert os.listdir(tmp_path / "work2") == []
    for name in ["20230101T06Z.nc", "20230101T12Z.nc"]:
        assert_same(
            open_netcdf(tmp_path / f"complete_{name}"),
            open_netcdf(tmp_path / f"later_{name}"),
        )


def test_convert_command(tmp_path):
    """bris-convert with explicit files, for stores that the run did not ask to convert"""
    pm = get_pm(num_members=2)
    pred = np.nan_to_num(get_pred(pm))
    output = Zarr(pm, "", str(tmp_path / "%Y%m%dT%HZ.zarr"), variables=["2t", "msl"])
    for frt in [FRT, FRT + 6 * 3600]:
        add_members(output, pm, pred, frt=frt)
    stores = [str(tmp_path / "20230101T06Z.zarr"), str(tmp_path / "20230101T12Z.zarr")]

    # Nothing to do without a target
    with pytest.raises(SystemExit):
        bris.convert.main(stores)
    # ... and two stores cannot go to the same file
    with pytest.raises(SystemExit):
        bris.convert.main(stores + ["--netcdf", str(tmp_path / "same.nc")])

    pattern = str(tmp_path / "out" / "fc_%Y%m%dT%HZ")
    args = ["--netcdf", pattern + ".nc", "--grib", pattern + ".grib", "-w", "2"]
    assert bris.convert.main(stores + args) == 0
    assert sorted(os.listdir(tmp_path / "out")) == [
        "fc_20230101T06Z.grib",
        "fc_20230101T06Z.nc",
        "fc_20230101T12Z.grib",
        "fc_20230101T12Z.nc",
    ]
    assert_same(
        open_zarr(stores[1]), open_netcdf(tmp_path / "out" / "fc_20230101T12Z.nc")
    )
    assert zarrstore.read_info(stores[0]) is not None

    assert (
        bris.convert.main([stores[0], "--netcdf", pattern + "_b.nc", "--remove_zarr"])
        == 0
    )
    assert not os.path.exists(stores[0])
    assert os.path.exists(stores[1])


def test_convert_without_consolidated_metadata(tmp_path, caplog):
    """A store that lost its consolidated metadata still converts, with a warning"""
    pm = get_pm()
    output = Zarr(pm, "", str(tmp_path / "%Y%m%dT%HZ.zarr"), variables=["2t", "msl"])
    add_members(output, pm, get_pred(pm))
    store = str(tmp_path / "20230101T06Z.zarr")

    with caplog.at_level(logging.WARNING, logger="bris"):
        bris.convert.to_netcdf(store, str(tmp_path / "with.nc"))
    assert "consolidated metadata" not in caplog.text

    os.remove(os.path.join(store, ".zmetadata"))
    with caplog.at_level(logging.WARNING, logger="bris"):
        bris.convert.to_netcdf(store, str(tmp_path / "without.nc"))
    assert "no consolidated metadata" in caplog.text
    assert_same(open_netcdf(tmp_path / "with.nc"), open_netcdf(tmp_path / "without.nc"))

    # Consolidated metadata that cannot be read is an error, not a reason to read the rest
    with open(os.path.join(store, ".zmetadata"), "w") as file:
        file.write('{"zarr_consolidated_format": 1}')
    with pytest.raises(KeyError):
        bris.convert.Store(store)

    with pytest.raises(FileNotFoundError, match="no such directory"):
        bris.convert.Store(str(tmp_path / "missing.zarr"))


def test_read_ahead():
    """The results come in the order of the tasks, a few are read ahead, and an error in a
    read reaches the caller"""
    started = []

    def read(task):
        started.append(task)
        return task * 10

    results = bris.convert.read_ahead(iter(range(20)), read, num_threads=3)
    assert next(results) == (0, 0)
    assert len(started) <= 4  # the three ahead, and one more when the first was taken
    assert list(results) == [(i, i * 10) for i in range(1, 20)]
    assert list(bris.convert.read_ahead([], read)) == []

    def fail(task):
        if task == 2:
            raise OSError("no chunk")
        return task

    with pytest.raises(OSError, match="no chunk"):
        list(bris.convert.read_ahead(range(5), fail))


@pytest.mark.parametrize("num_members", [1, 2])
def test_grib_from_chunks_in_time(tmp_path, monkeypatch, num_members):
    """The Grib file does not depend on how the store is chunked in time, nor on how many
    time steps the conversion handles at a time"""
    pm = get_pm(num_members)
    pred = np.nan_to_num(get_pred(pm))[..., list(range(num_members))]
    files = []
    for time_chunk, group in [(1, 64), (3, 64), (4, 64), (4, 3)]:
        folder = tmp_path / f"{time_chunk}_{group}"
        output = Zarr(
            pm,
            "",
            str(folder / "%Y%m%dT%HZ.zarr"),
            variables=["2t", "10u", "10v"],
            domain_name="meps",
            chunks={"time": time_chunk},
        )
        add_members(output, pm, pred)
        monkeypatch.setattr(bris.convert, "GRIB_TIME_GROUP", group)
        bris.convert.to_grib(
            str(folder / "20230101T06Z.zarr"), str(folder / "out.grib")
        )
        # No temporary files are left
        assert sorted(os.listdir(folder)) == ["20230101T06Z.zarr", "out.grib"]
        files.append((folder / "out.grib").read_bytes())
    assert len(files[0]) > 0
    assert all(content == files[0] for content in files[1:])


def test_grib_of_a_latlon_grid(tmp_path):
    """A store on a regular lat/lon grid is converted to Grib with that grid"""
    import eccodes as ecc

    pm = get_pm(gridded=False)
    output = Zarr(
        pm,
        "",
        str(tmp_path / "%Y%m%dT%HZ.zarr"),
        variables=["2t"],
        interp_res=0.5,
        convert=[{"grib": {"filename_pattern": str(tmp_path / "%Y%m%dT%HZ.grib")}}],
    )
    add_members(output, pm, np.nan_to_num(get_pred(pm)))
    output.finalize()

    ds = open_zarr(tmp_path / "20230101T06Z.zarr")
    lats, lons = np.meshgrid(
        ds["latitude"].values, ds["longitude"].values, indexing="ij"
    )
    with open(tmp_path / "20230101T06Z.grib", "rb") as file:
        grib = ecc.codes_grib_new_from_file(file)
        assert ecc.codes_get(grib, "gridType") == "regular_ll"
        grib_lats = ecc.codes_get_array(grib, "latitudes")
        grib_lons = ecc.codes_get_array(grib, "longitudes")
        values = ecc.codes_get_values(grib)
        ecc.codes_release(grib)
    np.testing.assert_allclose(grib_lats, lats.ravel(), atol=1e-6)
    np.testing.assert_allclose(grib_lons % 360, lons.ravel() % 360, atol=1e-6)
    expected = ds["air_temperature_2m"].values[0, 0].ravel()
    np.testing.assert_allclose(values, expected, atol=1e-4)


def test_grib_refuses_grids_it_cannot_describe(tmp_path):
    """... when the output is created, not after the forecast has been made"""
    polar = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-30 +R=6371000"
    pm = get_pm()
    pattern = str(tmp_path / "%Y%m%dT%HZ")
    targets = [{"grib": {"filename_pattern": pattern + ".grib"}}]
    with pytest.raises(ValueError, match="polar_stereographic"):
        Zarr(pm, "", pattern + ".zarr", proj4_str=polar, convert=targets)
    with pytest.raises(ValueError, match="polar_stereographic"):
        Grib(pm, str(tmp_path / "work"), pattern + ".grib", proj4_str=polar)

    # The conversion of a store that was written without the target refuses too
    output = Zarr(pm, "", pattern + ".zarr", proj4_str=polar, variables=["2t"])
    add_members(output, pm, get_pred(pm))
    with pytest.raises(ValueError, match="polar_stereographic"):
        bris.convert.to_grib(
            str(tmp_path / "20230101T06Z.zarr"), str(tmp_path / "out.grib")
        )
    assert not os.path.exists(tmp_path / "out.grib")


def test_grib_refuses_variables_it_cannot_describe(tmp_path):
    """... when the output is created as well"""
    pm = get_pm()
    pattern = str(tmp_path / "%Y%m%dT%HZ")
    targets = [{"grib": {"filename_pattern": pattern + ".grib"}}]
    kwargs = {"variables": ["2t", "tp"], "accumulated_variables": ["tp"]}
    with pytest.raises(ValueError, match="precipitation_amount_acc"):
        Zarr(pm, "", pattern + ".zarr", convert=targets, **kwargs)
    # The other formats take them
    netcdf = [{"netcdf": {"filename_pattern": pattern + ".nc"}}]
    output = Zarr(pm, "", pattern + ".zarr", convert=netcdf, **kwargs)

    # The conversion of a store that was written without the target refuses too
    add_members(output, pm, get_pred(pm))
    with pytest.raises(ValueError, match="precipitation_amount_acc"):
        bris.convert.to_grib(
            str(tmp_path / "20230101T06Z.zarr"), str(tmp_path / "out.grib")
        )
    assert not os.path.exists(tmp_path / "out.grib")


def test_incomplete_store(tmp_path, caplog):
    """What a run that was killed leaves: a member that never arrived and a field that was
    written in part. The file is written, with warnings, and because of the field written in
    part the store is kept."""
    pm = get_pm(num_members=3)
    pred = get_pred(pm)
    output = Netcdf(
        pm,
        str(tmp_path / "work"),
        str(tmp_path / "%Y%m%dT%HZ.nc"),
        variables=["2t", "msl"],
    )
    add_members(output, pm, pred, members=[0, 2])
    (path,) = output.get_stores()
    store = bris.convert.Store(path)
    assert store.missing() == ([1], [])
    assert store.chunks_written("air_temperature_2m") == {
        (0, 0): (4, 4),
        (0, 1): (0, 4),
        (0, 2): (4, 4),
    }

    # Member 2 lost the last two time steps of one field
    name = "air_pressure_at_sea_level"
    for time_index in [2, 3]:
        os.remove(os.path.join(path, name, f"{time_index}.0.2.0.0"))
    assert store.chunks_written(name)[(0, 2)] == (2, 4)
    assert store.missing() == ([1], [name])

    with caplog.at_level(logging.WARNING, logger="bris"):
        output.finalize()
    assert "no data for ensemble member 1" in caplog.text
    assert "no data for ensemble member 0" not in caplog.text
    assert f"incomplete data for ['{name}']" in caplog.text
    assert "is kept" in caplog.text
    assert os.path.isdir(path)

    ds = open_netcdf(tmp_path / "20230101T06Z.nc")
    msl = ds[name].values[:, 0]
    expected = pred[:, :, pm.variables.index("msl"), :].reshape(4, 3, 4, 3)
    assert np.array_equal(msl[:, 0], expected[..., 0])
    assert np.all(np.isnan(msl[:, 1]))
    assert np.array_equal(msl[:2, 2], expected[:2, ..., 2])
    assert np.all(np.isnan(msl[2:, 2]))

    # A complete store is removed as before
    output = Netcdf(pm, str(tmp_path / "work2"), str(tmp_path / "b_%Y%m%dT%HZ.nc"))
    add_members(output, pm, pred)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bris"):
        output.finalize()
    assert caplog.text == ""
    assert not os.path.exists(tmp_path / "work2")


def test_stores_of_outputs_with_similar_names(tmp_path):
    """An output only takes the stores with its own name for its own"""
    pm = get_pm(num_members=2)
    pred = get_pred(pm)
    names = ["%Y%m%dT%HZ.zarr", "lam_%Y%m%dT%HZ.zarr", "%Y%m%dT%HZ_lam.zarr"]
    outputs = [Zarr(pm, "", str(tmp_path / name), variables=["2t"]) for name in names]
    for output in outputs:
        for frt in [FRT, FRT + 6 * 3600]:
            add_members(output, pm, pred, frt=frt)
    for output, name in zip(outputs, names, strict=True):
        stores = [os.path.basename(path) for path in output.get_stores()]
        assert stores == [
            name.replace("%Y%m%dT%HZ", time)
            for time in ["20230101T06Z", "20230101T12Z"]
        ]
