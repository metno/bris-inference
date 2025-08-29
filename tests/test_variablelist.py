import pytest

from bris.conventions.metno import Metno
from bris.conventions.variable_list import VariableList


def test_initialization():
    """Test initialization of VariableList."""

    vl = VariableList(anemoi_names=[])
    assert isinstance(vl.conventions, Metno)


def test_dimensions_property():
    """Test the dimensions property."""
    vl = VariableList(anemoi_names=["2t"])
    dimensions = vl.dimensions
    assert isinstance(dimensions, dict)
    assert "height" in dimensions["height"][0]
    assert dimensions["height"][1] == [2]
    assert vl._ncname_to_level_dim == {"air_temperature_2m": "height"}


def test_get_level_index():
    vl = VariableList(anemoi_names=["2t", "10u", "10v"])
    level = vl.get_level_index(anemoi_name="2t")
    assert level == 0


def test_get_ncname_from_anemoi_name():
    vl = VariableList(anemoi_names=["2t"])
    ncname = vl.get_ncname_from_anemoi_name("2t")
    assert ncname == "air_temperature_2m"
