import multiprocessing
import os

import numpy as np
import pytest
import xarray as xr
import zarr

import bris.convert
import bris.outputs.zarr
from bris import outputs, zarrstore
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
    os.environ["SLURM_PROCID"] = str(member)
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


@pytest.mark.parametrize("num_members", [1, 2])
def test_convert_later(tmp_path, num_members):
    """With convert_at_end False the stores are left for bris-convert, which writes the same
    file as the run would have"""
    pm = get_pm(num_members)
    pred = get_pred(pm)
    kwargs = {"domain_name": "meps", "accumulated_variables": ["tp"]}
    frts = [FRT, FRT + 6 * 3600]

    direct = Netcdf(
        pm, str(tmp_path / "work1"), str(tmp_path / "direct_%Y%m%dT%HZ.nc"), **kwargs
    )
    later = Netcdf(
        pm,
        str(tmp_path / "work2"),
        str(tmp_path / "later_%Y%m%dT%HZ.nc"),
        convert_at_end=False,
        **kwargs,
    )
    for output in [direct, later]:
        for frt in frts:
            add_members(output, pm, pred, frt=frt)
        output.finalize()

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
            open_netcdf(tmp_path / f"direct_{name}"),
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
        bris.convert.main([stores[0], "--netcdf", pattern + "_b.nc", "--remove"]) == 0
    )
    assert not os.path.exists(stores[0])
    assert os.path.exists(stores[1])
