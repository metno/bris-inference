import pytest

import bris.checkpoint
import bris.inspect


def test_clean_version_name():
    """Very basic test of removing +... from version name"""
    assert bris.inspect.clean_version_name("2.6.0+cu124") == "2.6.0"


def test_get_required_variables():
    """Check required variables for test-checkpoint match expected list.

       Multiencdec checkpoint requires this specific anemoi-models version:
       https://github.com/metno/anemoi-models/tree/feature/netatmo
    """

    # Simple checkpoint
    expected_simple = [
        "10u",
        "10v",
        "2d",
        "2t",
        "lsm",
        "msl",
        "q_100",
        "q_1000",
        "q_150",
        "q_200",
        "q_250",
        "q_300",
        "q_400",
        "q_50",
        "q_500",
        "q_600",
        "q_700",
        "q_850",
        "sin_longitude",
        "slor",
        "sp",
        "t_100",
        "t_1000",
        "t_150",
        "t_200",
        "t_250",
        "t_300",
        "t_400",
        "t_50",
        "t_500",
        "t_600",
        "t_700",
        "t_850",
        "t_925",
        "tcw",
        "u_100",
        "u_1000",
        "u_150",
        "u_200",
        "u_250",
        "u_300",
        "u_400",
        "u_50",
        "u_500",
        "u_600",
        "u_700",
        "u_850",
        "u_925",
        "v_100",
        "v_1000",
        "v_150",
        "v_200",
        "v_250",
        "v_300",
        "v_400",
        "v_50",
        "v_500",
        "v_600",
        "v_700",
        "v_850",
        "v_925",
        "w_100",
        "w_1000",
        "w_150",
        "w_200",
        "w_250",
        "w_300",
        "w_400",
        "w_50",
        "w_500",
        "w_600",
        "w_700",
        "w_925",
        "z",
        "z_100",
        "z_1000",
        "z_150",
        "z_200",
        "z_250",
        "z_300",
        "z_400",
        "z_50",
        "z_500",
        "z_600",
        "z_700",
        "cos_latitude",
        "cos_longitude",
        "q_925",
        "sdor",
        "sin_latitude",
        "skt",
        "w_850",
    ]
    checkpoint_simple = bris.checkpoint.Checkpoint("tests/files/checkpoint.ckpt")
    required_simple = bris.inspect.get_required_variables(checkpoint_simple)[0]
    for var in expected_simple:
        assert var in required_simple, (
            f"Variable {var} not returned by get_required_variables() for test-checkpoint {checkpoint_simple}."
        )

    # Multiencdec checkpoint
    expected_multi = {
        0: ["10u", "10v", "z"],
        1: [
            "skt",
            "msl",
            "cos_longitude",
            "cos_latitude",
            "sin_longitude",
            "sin_latitude",
        ],
    }
    checkpoint_multi = bris.checkpoint.Checkpoint("tests/files/multiencdec.ckpt")
    required_multi = bris.inspect.get_required_variables(checkpoint_multi)
    for dataset, required_variables in required_multi.items():
        for expected_variable in expected_multi[dataset]:
            assert expected_variable in required_variables, (
                f"Expected variable {expected_variable} not in test-checkpoint {checkpoint_multi}."
            )


def test_check_module_versions():
    """This depends on the current venv, so just test it doesn't crash"""
    checkpoint = bris.checkpoint.Checkpoint("tests/files/checkpoint.ckpt")
    _bad = bris.inspect.check_module_versions(checkpoint)

    # assert "fsspec==2025.2.0" in bad


def test_inspect():
    """This depends on the current venv, so just test it doesn't crash"""
    _status = bris.inspect.inspect(checkpoint_path="tests/files/checkpoint.ckpt")


if __name__ == "__main__":
    test_clean_version_name()
    test_get_required_variables()
    test_check_module_versions()
    # test_inspect()
