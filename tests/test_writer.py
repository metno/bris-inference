import time
from concurrent.futures import Future
from queue import Queue

import numpy as np
import pytest

from bris.__main__ import use_background_write
from bris.writer import CustomWriter


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


def test_custom_writer_sync(prediction):
    """background_write: False -> process_list is None and add_forecast is called in the calling thread
    (write_on_batch_end used to fail on len(None) before writing anything)."""
    calls = Queue()
    output_dict = [
        {
            "decoder_name": "data",
            "start_gridpoint": 0,
            "end_gridpoint": 2,
            "outputs": [DummyOutput(calls), DummyOutput(calls)],
        }
    ]
    writer = CustomWriter(output_dict, None)
    writer.write_on_batch_end(None, None, prediction, None, None, 0, 0)
    assert calls.qsize() == 2
    writer.on_predict_end(None, None)


def _decoder_outputs(n_outputs: int):
    return [
        {
            "decoder_name": "data",
            "start_gridpoint": 0,
            "end_gridpoint": 2,
            "outputs": [object()] * n_outputs,
        }
    ]


def test_use_background_write_default():
    # one member and several outputs on a decoder: synchronous (concurrent netcdf writes segfault)
    assert use_background_write({}, _decoder_outputs(2), num_members=1) is False
    # one output, or several members (intermediate store, files assembled at finalize): background
    assert use_background_write({}, _decoder_outputs(1), num_members=1) is True
    assert use_background_write({}, _decoder_outputs(2), num_members=2) is True
    # an explicit setting wins
    assert (
        use_background_write(
            {"background_write": True}, _decoder_outputs(2), num_members=1
        )
        is True
    )
    assert (
        use_background_write(
            {"background_write": False}, _decoder_outputs(1), num_members=4
        )
        is False
    )


if __name__ == "__main__":
    _ = pytest.main([__file__])
