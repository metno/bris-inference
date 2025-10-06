import pytest
import numpy as np
import time
import multiprocessing

from bris.writer import CustomWriter


class DummyOutput:
    def __init__(self, calls):
        self.calls = calls

    def add_forecast(self, times, ensemble_member, pred):
        time.sleep(0.01)
        self.calls.append((times, ensemble_member, pred.copy()))


@pytest.fixture
def prediction():
    return {
        "times": [np.datetime64("2024-01-01T00:00")],
        "ensemble_member": 0,
        "group_rank": 0,
        "pred": [np.ones((1, 2, 3))],  # shape: (batch, grid, var)
    }


def test_custom_writer_sync(prediction):
    """Test that DummyOutput is being called, and with the correct arguments for non-background writing."""
    calls = []
    dummy_output = DummyOutput(calls)
    output_dict = [{
        "decoder_index": 0,
        "start_gridpoint": 0,
        "end_gridpoint": 2,
        "outputs": [dummy_output],
    }]
    current_batch_no = multiprocessing.Value('i', -1)
    writer = CustomWriter(output_dict, current_batch_no, None)
    writer.write_on_batch_end(None, None, prediction, None, None, 0, 0)
    assert len(calls) == 1

    times, member, pred = calls[0]
    assert member == 0
    np.testing.assert_array_equal(pred, np.ones((2, 3)))


def test_custom_writer_async(prediction):
    """Test that DummyOutput is being called, and with the correct arguments for background writing. Calls must be a list shared between processes here."""
    manager = multiprocessing.Manager()
    calls = manager.list()

    dummy_output = DummyOutput(calls)
    output_dict = [{
        "decoder_index": 0,
        "start_gridpoint": 0,
        "end_gridpoint": 2,
        "outputs": [dummy_output],
    }]
    current_batch_no = multiprocessing.Value('i', -1)
    process_list = []
    writer = CustomWriter(output_dict, current_batch_no, process_list)
    writer.write_on_batch_end(None, None, prediction, None, None, 0, 0)
    assert len(process_list) == 1
    for p in process_list:
        p.join()

    print("calls", calls)

    assert len(calls) == 1
    times, member, pred = calls[0]
    assert member == 0
    np.testing.assert_array_equal(pred, np.ones((2, 3)))


if __name__ == "__main__":
    _ = pytest.main([__file__])
