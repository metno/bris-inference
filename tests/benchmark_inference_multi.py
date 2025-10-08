# Attempt to replace tox test inference_CI: bris --config tox_test_inference.yaml

import pytest

import bris.__main__ as b


def test_inference(benchmark):
    benchmark(b.main, arg_list=["--config", "./tox_test_inference.yaml"])
