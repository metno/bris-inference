import time
from concurrent.futures import Future
from queue import Queue

import numpy as np
import pytest

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
        "pred": [np.ones((1, 2, 3))],  # shape: (batch, grid, var)
    }

def test_custom_writer_async(prediction):
    """Test that DummyOutput is being called, and with the correct arguments for background writing. Calls must be a list shared between threads."""
    calls = Queue()

    dummy_output = DummyOutput(calls)
    output_dict = [
        {
            "decoder_index": 0,
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


if __name__ == "__main__":
    _ = pytest.main([__file__])
