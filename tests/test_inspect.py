import pytest

import bris.checkpoint
import bris.inspect


def test_clean_version_name():
    """Very basic test of removing +... from version name"""
    assert bris.inspect.clean_version_name("2.6.0+cu124") == "2.6.0"


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
    test_check_module_versions()
