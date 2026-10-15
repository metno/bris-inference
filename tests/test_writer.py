import os
import tempfile
import time
from concurrent.futures import Future
from queue import Queue

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bris.writer import CustomWriter, reduce_contributions


class DummyOutput:
    def __init__(self, calls):
        print("DummyOutput init")
        self.calls = calls

    def add_forecast(self, times, ensemble_member, pred):
        print("DummyOutput add_forecast")
        time.sleep(0.01)
        self.calls.put((times, ensemble_member, pred.copy()))


@pytest.fixture
def prediction():
    return {
        "times": [np.datetime64("2024-01-01T00:00")],
        "ensemble_member": 0,
        "group_rank": 0,
        "pred": {"data": np.ones((1, 2, 3))},  # shape: (batch, grid, var)
    }


def test_custom_writer_async(prediction):
    """Test that DummyOutput is being called, and with the correct arguments for background writing. Calls must be a list shared between threads."""
    calls = Queue()

    dummy_output = DummyOutput(calls)
    output_dict = [
        {
            "decoder_name": "data",
            "start_gridpoint": 0,
            "end_gridpoint": 2,
            "outputs": [dummy_output],
        }
    ]
    thread_list: list[Future] = []
    writer = CustomWriter(output_dict, thread_list)
    writer.write_on_batch_end(None, None, prediction, None, None, 0, 0)
    assert len(thread_list) == 1
    for thread in thread_list:
        print("thread", thread)
        thread.result()

    print("calls", calls)

    assert calls.qsize() == 1
    times, member, pred = calls.get()
    assert member == 0
    np.testing.assert_array_equal(pred, np.ones((2, 3)))


class DummyReducedOutput:
    """Output using the ensemble-reduction protocol"""

    reduce_across_members = True
    ensemble_reductions = {"sum": "sum", "max": "max"}

    def __init__(self, calls):
        self.calls = calls

    def member_contribution(self, times, ensemble_member, pred):
        return {"sum": pred, "max": pred}

    def add_forecast(self, times, ensemble_member, pred):
        raise AssertionError("add_forecast should not be called for reduced outputs")

    def add_reduced_forecast(self, times, contributions, num_members):
        self.calls.put((times, contributions, num_members))


@pytest.mark.parametrize("background", [True, False])
def test_custom_writer_reduced_output(prediction, background):
    """Without torch.distributed, a reduced output gets its own contribution unchanged"""
    calls = Queue()
    output = DummyReducedOutput(calls)
    output_dict = [
        {
            "decoder_name": "data",
            "start_gridpoint": 0,
            "end_gridpoint": 2,
            "outputs": [output],
        }
    ]
    thread_list = [] if background else None
    writer = CustomWriter(output_dict, thread_list)
    writer.write_on_batch_end(None, None, prediction, None, None, 0, 0)
    if background:
        assert len(thread_list) == 1
        for thread in thread_list:
            thread.result()

    assert calls.qsize() == 1
    times, contributions, num_members = calls.get()
    assert num_members == 1
    assert set(contributions.keys()) == {"sum", "max"}
    np.testing.assert_array_equal(contributions["sum"], np.ones((2, 3)))


def test_reduce_contributions_no_group():
    contributions = {"sum": np.ones(3), "min": np.zeros(3)}
    reduced, num_members, is_root = reduce_contributions(
        contributions, {"sum": "sum", "min": "min"}, None
    )
    assert num_members == 1
    assert is_root
    assert reduced is contributions


def _reduce_worker(rank, world_size, init_file, result_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        group = dist.new_group(list(range(world_size)))
        contributions = {
            "sum": np.full((2, 3), rank + 1, dtype=np.float64),
            "min": np.full((2, 3), rank + 1, dtype=np.float64),
            "max": np.full((2, 3), rank + 1, dtype=np.float64),
        }
        reduced, num_members, is_root = reduce_contributions(
            contributions, {"sum": "sum", "min": "min", "max": "max"}, group, "cpu"
        )
        np.save(f"{result_dir}/num_members_{rank}.npy", num_members)
        np.save(f"{result_dir}/is_root_{rank}.npy", is_root)
        if reduced is not None:
            for name, value in reduced.items():
                np.save(f"{result_dir}/{name}_{rank}.npy", value)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed not available")
def test_reduce_contributions_gloo():
    """Reduce across 3 CPU processes: only the root gets the result"""
    world_size = 3
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "init")
        mp.spawn(
            _reduce_worker,
            args=(world_size, init_file, temp_dir),
            nprocs=world_size,
            join=True,
        )
        for rank in range(world_size):
            assert int(np.load(f"{temp_dir}/num_members_{rank}.npy")) == world_size
            assert bool(np.load(f"{temp_dir}/is_root_{rank}.npy")) == (rank == 0)
            if rank > 0:
                assert not os.path.exists(f"{temp_dir}/sum_{rank}.npy")
        np.testing.assert_array_equal(
            np.load(f"{temp_dir}/sum_0.npy"), np.full((2, 3), 1 + 2 + 3)
        )
        np.testing.assert_array_equal(np.load(f"{temp_dir}/min_0.npy"), np.ones((2, 3)))
        np.testing.assert_array_equal(
            np.load(f"{temp_dir}/max_0.npy"), np.full((2, 3), 3)
        )


if __name__ == "__main__":
    _ = pytest.main([__file__])
