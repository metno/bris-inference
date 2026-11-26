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


def is_from_this_run(path: str) -> bool:
    """Is there a store made by a process of this run at path?"""
    info = read_info(path)
    return info is not None and info.get("run_id") == run_id()


def is_current(path: str, fingerprint: str | None = None) -> bool:
    """Is there a store of this run, with this layout, at path?"""
    info = read_info(path)
    return (
        info is not None
        and info.get("run_id") == run_id()
        and info.get("fingerprint") == fingerprint
    )


def is_foreign(path: str) -> bool:
    """Is there something at path that bris did not write, other than an empty directory?

    That is anything without a sidecar. Such content does not change during a run, while a
    store can be replaced at any moment, between the looks at its sidecar and at its path:
    what is found must therefore be found twice.
    """

    def look() -> bool:
        if read_info(path) is not None or not os.path.lexists(path):
            return False
        try:
            is_directory = os.path.isdir(path) and not os.path.islink(path)
            return not (is_directory and not os.listdir(path))
        except OSError:
            return False

    return look() and look()


def ensure(
    path: str,
    build: Callable[[str], None],
    fingerprint: str | None = None,
    info: dict | None = None,
    replace: bool = True,
    timeout: float = 600.0,
) -> None:
    """Make sure that the store of this run exists at path, creating it if needed

    Safe to call from all ranks at the same time: exactly one of them creates the store.

    Another store written by bris at path (by an earlier run, or by this run with another
    layout) is replaced, and so is an empty directory. Exactly one of the processes that write
    to the store must do that (replace), and the others wait for it, since a process cannot
    otherwise tell a store that is about to be replaced from one that it should write into.

    Anything else at path is left alone: the path comes from the configuration, and may by
    mistake be a directory with other data.

    Args:
        path: Where the store should be
        build: Creates the store, with the metadata of all its arrays, in the directory it is
            passed (which does not exist yet)
        fingerprint: Describes the layout of the store. A store with another fingerprint is
            not this run's store, even when made by a run with the same identifier (e.g. two
            runs in the same interactive job step).
        info: Stored in the sidecar, with the run identifier and the fingerprint
        replace: Is this the process that replaces what another run left at path? True for
            exactly one of the processes that write to the store (e.g. the one that writes
            the first ensemble member).
        timeout: How long a process that does not replace waits for the one that does [s]

    Raises:
        FileExistsError: There is something at path that bris did not write
    """
    identity = {"run_id": run_id(), "fingerprint": fingerprint}
    t0 = time.perf_counter()
    warned = False
    while True:
        # Look at the path before reading the sidecar. A store appears complete, with its
        # sidecar, and only the replacing process removes one, so what was already there and
        # is not this run's store is stale. In the other order, a store that another rank
        # creates between the two looks would be taken for a stale one and removed.
        exists = os.path.lexists(path)
        if is_current(path, fingerprint):
            return

        if exists:
            if is_foreign(path):
                raise FileExistsError(
                    f"{path} exists and was not written by bris (it has no {SIDECAR}), so "
                    "it is not replaced. Remove it, or write to another path (is the file "
                    "name missing from the filename pattern?)."
                )
            if not replace:
                if not warned:
                    LOGGER.info(
                        f"Waiting for {path}, left by another run, to be replaced"
                    )
                    warned = True
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(
                        f"{path} exists and is not the store that this process should write "
                        f"to, and it was not replaced within {timeout:.0f} s. Do two outputs "
                        "write to the same path, or is the path the same for several "
                        "forecasts (no time tokens)?"
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
