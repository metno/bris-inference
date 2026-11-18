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

    # Something that is not a store
    os.makedirs(tmp_path / "dir.zarr" / "sub")
    zarrstore.ensure(str(tmp_path / "dir.zarr"), build(4))
    assert value(str(tmp_path / "dir.zarr")) == 4
    assert sorted(os.listdir(tmp_path)) == ["dir.zarr", "store.zarr"]


def test_only_main_rank_replaces(tmp_path, monkeypatch):
    path = str(tmp_path / "store.zarr")
    monkeypatch.setenv("BRIS_RUN_ID", "run1")
    zarrstore.ensure(path, build(1))

    monkeypatch.setenv("BRIS_RUN_ID", "run2")
    monkeypatch.setenv("SLURM_PROCID", "1")
    assert not zarrstore.is_main_rank()
    with pytest.raises(TimeoutError):
        zarrstore.ensure(path, build(2), timeout=1)
    assert value(path) == 1

    # ... but any rank creates a store that is not there
    zarrstore.ensure(str(tmp_path / "new.zarr"), build(2))
    assert value(str(tmp_path / "new.zarr")) == 2


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
