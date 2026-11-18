"""Zarr (version 2) stores that the ranks of a run write into together

A store is a directory. The ranks (one process per ensemble member) write different chunks of
the same arrays, which needs no locking as long as the metadata exists before the first chunk
is written and no two ranks write the same chunk. `ensure` takes care of the first part: the
store is built in a temporary directory and renamed into place, so that it appears complete or
not at all, whichever rank gets there first.

Next to the zarr metadata each store has a small sidecar file (SIDECAR) with what bris needs to
know about it: which run made it and how it should be converted to other formats (see
bris.convert). It is kept out of the zarr attributes so that these hold the same metadata as a
NetCDF file of the same forecast.

This module must stay cheap to import (no torch), since bris.convert uses it.
"""

import contextlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable

LOGGER = logging.getLogger("bris")

SIDECAR = ".bris.json"

# Identifies a run that is not launched by srun (a single process)
_PROCESS_RUN_ID = uuid.uuid4().hex


def run_id() -> str:
    """An identifier shared by all ranks of this run, and by no other run

    BRIS_RUN_ID if set, else the job and step id of a run launched by srun, else an identifier
    of this process.
    """
    if "BRIS_RUN_ID" in os.environ:
        return os.environ["BRIS_RUN_ID"]
    if "SLURM_JOB_ID" in os.environ and "SLURM_STEP_ID" in os.environ:
        return f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"
    return _PROCESS_RUN_ID


def is_main_rank() -> bool:
    """Is this the rank that finalizes the outputs (the only rank outside a distributed run)?"""
    return os.environ.get("SLURM_PROCID", "0") == "0"


def read_info(path: str) -> dict | None:
    """The sidecar of the store at path, or None if there is no store with a sidecar"""
    try:
        with open(os.path.join(path, SIDECAR), encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, NotADirectoryError):
        return None


def write_info(path: str, info: dict) -> None:
    """Write the sidecar of the store at path (atomically)"""
    tmp = os.path.join(path, f"{SIDECAR}.{uuid.uuid4().hex}")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=1)
    os.replace(tmp, os.path.join(path, SIDECAR))


def remove(path: str) -> None:
    """Remove the store at path, if it exists"""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def is_current(path: str, fingerprint: str | None = None) -> bool:
    """Is there a store of this run, with this layout, at path?"""
    info = read_info(path)
    return (
        info is not None
        and info.get("run_id") == run_id()
        and info.get("fingerprint") == fingerprint
    )


def ensure(
    path: str,
    build: Callable[[str], None],
    fingerprint: str | None = None,
    info: dict | None = None,
    timeout: float = 3600.0,
) -> None:
    """Make sure that the store of this run exists at path, creating it if needed

    Safe to call from all ranks at the same time: exactly one of them creates the store.

    Something else at path (a store of an earlier run, or of this run with another layout) is
    replaced. Only the main rank does that; the other ranks wait for it, since a rank cannot
    otherwise tell a store that is about to be replaced from one that it should write into.

    Args:
        path: Where the store should be
        build: Creates the store, with the metadata of all its arrays, in the directory it is
            passed (which does not exist yet)
        fingerprint: Describes the layout of the store. A store with another fingerprint is
            not this run's store, even when made by a run with the same identifier (e.g. two
            runs in the same interactive job step).
        info: Stored in the sidecar, with the run identifier and the fingerprint
        timeout: How long a rank that is not the main rank waits for the main rank to replace
            what is at path [s]
    """
    identity = {"run_id": run_id(), "fingerprint": fingerprint}
    t0 = time.perf_counter()
    warned = False
    while True:
        if is_current(path, fingerprint):
            return

        if os.path.lexists(path):
            if not is_main_rank():
                if not warned:
                    LOGGER.info(
                        f"Waiting for the main rank to replace {path}, left by another run"
                    )
                    warned = True
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(
                        f"{path} exists and is not a store of this run, and the main rank did "
                        f"not replace it within {timeout:.0f} s"
                    )
                time.sleep(0.5)
                continue
            LOGGER.info(f"Replacing {path}, left by another run")
            trash = f"{path}.stale.{uuid.uuid4().hex}"
            os.rename(path, trash)
            remove(trash)
            continue

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            build(tmp)
            write_info(tmp, {**(info or {}), **identity})
            try:
                os.rename(tmp, path)
            except OSError:
                # Another rank got there first: the loop then finds that rank's store
                if not os.path.lexists(path):
                    raise
        finally:
            if os.path.lexists(tmp):
                remove(tmp)
