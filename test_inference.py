# Attempt to replace tox test inference_CI: bris --config tox_test_inference.yaml

import pytest
import bris.__main__ as b

def test_inference():
    b.main(arg_list=["--config", "../../../config/tox_test_py_inference.yaml"])
