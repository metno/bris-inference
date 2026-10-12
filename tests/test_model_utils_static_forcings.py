import numpy as np
import pytest
import torch

from bris.model.model_utils import get_model_static_forcings, static_forcing_time_index


class _Index:
    """Minimal stand-in for DataIndex.input with a name_to_index mapping."""

    def __init__(self, names):
        self.name_to_index = {n: i for i, n in enumerate(names)}


class _DataIndex:
    def __init__(self, names):
        self.input = _Index(names)


def test_get_model_static_forcings_from_batched_state():
    names = ["2t", "lsm", "z", "cos_latitude", "insolation", "urban"]
    internal_data = _DataIndex(names)
    # (batch, ensemble, points, variables): the static fields differ per point and per batch member
    data = torch.zeros((2, 3, 2, len(names)))
    data[..., 1] = torch.tensor([0.0, 1.0])
    data[0, ..., 2] = torch.tensor([100.0, 200.0])
    data[1, ..., 2] = torch.tensor([150.0, 250.0])
    data[..., 3] = torch.tensor([0.5, 0.3])
    data[..., 4] = torch.tensor([7.0, 7.0])  # dynamic forcing, must not be picked up
    data[..., 5] = torch.tensor([0.0, 0.9])

    selection = ["cos_latitude", "insolation", "lsm", "z", "urban"]
    sf = get_model_static_forcings(selection, data, internal_data)

    # every non-dynamic forcing of the selection, taken from the state as it is
    assert set(sf) == {"cos_latitude", "lsm", "z", "urban"}
    assert torch.equal(sf["cos_latitude"][1, 2], torch.tensor([0.5, 0.3]))
    assert torch.equal(sf["urban"][0, 0], torch.tensor([0.0, 0.9]))
    # field forcings keep the leading (batch, ensemble) dimensions of the state
    assert sf["z"].shape == (2, 3, 2) and sf["lsm"].shape == (2, 3, 2)
    assert torch.equal(sf["z"][0, 0], torch.tensor([100.0, 200.0]))
    assert torch.equal(sf["z"][1, 2], torch.tensor([150.0, 250.0]))
    assert torch.equal(sf["lsm"][1, 1], torch.tensor([0.0, 1.0]))
    assert sf["z"].dtype == torch.float32 and sf["z"].is_contiguous()

    # a (points, variables) state gives (points,) forcings
    sf1 = get_model_static_forcings(["z"], data[0, 0], internal_data)
    assert sf1["z"].shape == (2,)

    assert get_model_static_forcings(None, data, internal_data) is None


def test_static_forcing_time_index():
    dates = np.array(
        ["2020-01-01T00", "2020-01-01T06", "2020-01-01T12"], dtype="datetime64[s]"
    )
    # exact match
    assert static_forcing_time_index(dates, np.datetime64("2020-01-01T06")) == 1
    # between two dates: the latest date before the time
    assert static_forcing_time_index(dates, np.datetime64("2020-01-01T09:30")) == 1
    # after the last date
    assert static_forcing_time_index(dates, np.datetime64("2021-06-01T00")) == 2
    # before the first date: the first time step
    assert static_forcing_time_index(dates, np.datetime64("2019-12-31T18")) == 0
    # other datetime64 units and a plain string work too
    assert (
        static_forcing_time_index(dates.astype("datetime64[ns]"), "2020-01-01T12") == 2
    )


@pytest.mark.parametrize("unit", ["s", "ns"])
def test_static_forcing_time_index_units(unit):
    dates = np.arange(
        np.datetime64("2022-01-01"), np.datetime64("2022-01-03"), np.timedelta64(1, "D")
    ).astype(f"datetime64[{unit}]")
    assert static_forcing_time_index(dates, np.datetime64("2022-01-02T12")) == 1
