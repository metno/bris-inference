import pytest

import bris.inspect
import bris.checkpoint

def test_clean_version_name():
    assert bris.inspect.clean_version_name("2.6.0+cu124") == "2.6.0"


def test_check_module_versions():
    checkpoint = bris.checkpoint.Checkpoint("tests/files/checkpoint.ckpt")
    bad = bris.inspect.check_module_versions(checkpoint)

    print(bad)
    assert "fsspec==2025.2.0" in bad


if __name__ == "__main__":
    test_clean_version_name()
    test_check_module_versions()