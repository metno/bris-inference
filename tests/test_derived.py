import numpy as np
import pytest

from bris import derived


def test_required_inputs():
    assert derived.get_required_inputs("ws") == ["10u", "10v"]
    assert derived.get_required_inputs("wz_500") == ["w_500", "t_500", "q_500"]
    assert derived.get_required_inputs("wz_50") == ["w_50", "t_50", "q_50"]
    assert derived.get_required_inputs("2t") is None
    assert derived.get_required_inputs("w_500") is None
    assert derived.is_derived("wz_850")
    assert not derived.is_derived("t_850")

    with pytest.raises(ValueError):
        derived.get_required_inputs("wz_abc")


def test_compute_ws():
    fields = {"10u": np.array([3.0, 0.0]), "10v": np.array([4.0, -2.0])}
    ws = derived.compute("ws", fields.__getitem__)
    np.testing.assert_allclose(ws, [5.0, 2.0])


def test_compute_wz():
    # Rising air: omega < 0 gives a positive (upward) velocity.
    # At 500 hPa, T=250 K, dry air: rho = p/(R T) = 50000/(287*250) = 0.6969 kg/m^3,
    # so omega = -1 Pa/s -> wz = 1/(rho g) = 0.1463 m/s. Moist air is lighter -> larger wz.
    fields = {
        "w_500": np.array([-1.0, 2.0, -1.0]),
        "t_500": np.array([250.0, 250.0, 250.0]),
        "q_500": np.array([0.0, 0.0, 0.001]),
    }
    wz = derived.compute("wz_500", fields.__getitem__)
    np.testing.assert_allclose(wz, [0.146329, -0.292659, 0.146419], rtol=1e-5)

    # Same omega is a larger geometric velocity where the air is thinner
    fields = {
        "w_50": np.array([-1.0]),
        "t_50": np.array([250.0]),
        "q_50": np.array([0.0]),
    }
    assert derived.compute("wz_50", fields.__getitem__)[0] == pytest.approx(
        1.463293, rel=1e-5
    )

    with pytest.raises(ValueError):
        derived.compute("unknown", fields.__getitem__)


if __name__ == "__main__":
    test_required_inputs()
    test_compute_ws()
    test_compute_wz()
