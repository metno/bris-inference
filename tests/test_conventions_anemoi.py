import pytest
from bris.conventions.anemoi import get_units

def test_get_units_with_known_variable():
    variable_name = "2t"
    expected_units = "K"  # Kelvin
    assert get_units(variable_name) == expected_units


def test_get_units_with_unknown_variable():
    # Test with an unknown variable that does not have units
    variable_name = "unknown_variable"
    assert get_units(variable_name) is None

def test_get_units_with_override():
    # Test with a variable that has an override in the function
    variable_name = "tp"
    expected_units = "Mg/m^2"
    assert get_units(variable_name) == expected_units

def test_get_units_with_missing_attributes():
    # Test with a variable that has a CF name but no "units" attribute
    variable_name = "no_units_variable"
    assert get_units(variable_name) is None
