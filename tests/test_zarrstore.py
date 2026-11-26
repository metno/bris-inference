import os

import pytest
import zarr

from bris import zarrstore


def build(value):
    def _build(tmp):
        zarr.open_group(tmp, mode="w").attrs["value"] = value

    return _build


def value(path):
    return zarr.open_group(path, mode="r").attrs["value"]


def test_ensure_creates_once(tmp_path):
    path = str(tmp_path / "sub" / "store.zarr")
    zarrstore.ensure(path, build(1), "a", info={"convert": []})
    assert zarrstore.is_current(path, "a")
    assert zarrstore.read_info(path)["convert"] == []

    # The store of this run is kept, also by a rank that is not the main rank
    zarrstore.ensure(path, build(2), "a")
    assert value(path) == 1

    # No temporary directories are left behind
    assert os.listdir(tmp_path / "sub") == ["store.zarr"]


def test_ensure_replaces_other_runs(tmp_path, monkeypatch):
    path = str(tmp_path / "store.zarr")
    monkeypatch.setenv("BRIS_RUN_ID", "run1")
    zarrstore.ensure(path, build(1), "a")

    # Another layout
    zarrstore.ensure(path, build(2), "b")
    assert value(path) == 2

    # Another run
    monkeypatch.setenv("BRIS_RUN_ID", "run2")
    assert not zarrstore.is_current(path, "b")
    zarrstore.ensure(path, build(3), "b")
    assert value(path) == 3

    # An empty directory
    os.makedirs(tmp_path / "dir.zarr")
    zarrstore.ensure(str(tmp_path / "dir.zarr"), build(4))
    assert value(str(tmp_path / "dir.zarr")) == 4
    assert sorted(os.listdir(tmp_path)) == ["dir.zarr", "store.zarr"]


@pytest.mark.parametrize("replace", [True, False])
def test_ensure_keeps_what_bris_did_not_write(tmp_path, replace):
    """A directory with other data, a store of another tool and a file are not replaced, by
    any rank and without waiting"""
    os.makedirs(tmp_path / "results" / "sub")
    zarr.open_group(str(tmp_path / "other.zarr"), mode="w").attrs["value"] = 0
    (tmp_path / "file.zarr").write_text("data")

    for name in ["results", "other.zarr", "file.zarr"]:
        assert zarrstore.is_foreign(str(tmp_path / name))
        with pytest.raises(FileExistsError, match="not written by bris"):
            zarrstore.ensure(str(tmp_path / name), build(1), replace=replace)

    assert os.path.isdir(tmp_path / "results" / "sub")
    assert value(str(tmp_path / "other.zarr")) == 0
    assert (tmp_path / "file.zarr").read_text() == "data"
    assert sorted(os.listdir(tmp_path)) == ["file.zarr", "other.zarr", "results"]

    assert not zarrstore.is_foreign(str(tmp_path / "missing.zarr"))
    zarrstore.ensure(str(tmp_path / "store.zarr"), build(1))
    assert not zarrstore.is_foreign(str(tmp_path / "store.zarr"))


def test_only_one_process_replaces(tmp_path, monkeypatch):
    path = str(tmp_path / "store.zarr")
    monkeypatch.setenv("BRIS_RUN_ID", "run1")
    zarrstore.ensure(path, build(1))

    # A process that is not the one to replace waits for it
    monkeypatch.setenv("BRIS_RUN_ID", "run2")
    assert not zarrstore.is_from_this_run(path)
    with pytest.raises(TimeoutError):
        zarrstore.ensure(path, build(2), replace=False, timeout=1)
    assert value(path) == 1

    # ... but any process creates a store that is not there
    zarrstore.ensure(str(tmp_path / "new.zarr"), build(2), replace=False)
    assert value(str(tmp_path / "new.zarr")) == 2

    # The one that replaces need not be the main rank of the run
    monkeypatch.setenv("SLURM_PROCID", "3")
    zarrstore.ensure(path, build(3), replace=True)
    assert value(path) == 3
    assert zarrstore.is_from_this_run(path)


def test_store_created_meanwhile_is_kept(tmp_path, monkeypatch):
    """A store that another rank creates while the replacing rank looks at the path is this
    run's store, not a stale one"""
    path = str(tmp_path / "store.zarr")
    read_info = zarrstore.read_info
    landed = []

    def read_info_then_other_rank(store):
        info = read_info(store)
        if not landed:
            # The other rank creates the store right after this look at the sidecar
            landed.append(store)
            zarrstore.ensure(path, build(1), "a", replace=False)
        return info

    monkeypatch.setattr(zarrstore, "read_info", read_info_then_other_rank)
    zarrstore.ensure(path, build(2), "a", replace=True)
    assert landed == [path]
    assert value(path) == 1
    assert os.listdir(tmp_path) == ["store.zarr"]


def test_run_id(monkeypatch):
    for name in ["BRIS_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID"]:
        monkeypatch.delenv(name, raising=False)
    process_id = zarrstore.run_id()
    monkeypatch.setenv("SLURM_JOB_ID", "12")
    assert zarrstore.run_id() == process_id
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    assert zarrstore.run_id() == "12.0"
    monkeypatch.setenv("BRIS_RUN_ID", "x")
    assert zarrstore.run_id() == "x"
