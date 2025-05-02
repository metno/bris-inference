import os
import time

import bris.utils


def test_expand_time_tokens():
    unixtime = time.time()
    t = bris.utils.expand_time_tokens(filename="/test_file%Y-%m-%d", unixtime=unixtime)
    assert time.strftime("test_file%Y-%m-%d") in t, f"Time tokens not found in {t}."


def test_get_base_seed():
    # Set up test vars
    os.environ["AIFS_BASE_SEED"] = "1234"

    seed = bris.utils.get_base_seed(env_var_list=("AIFS_BASE_SEED", "SLURM_JOB_ID"))

    assert isinstance(seed, int)
    assert seed > 1000


def test_validate():
    filenames = ["working_example.yaml"]
    for filename in filenames:
        full_filename = (
            os.path.dirname(os.path.abspath(__file__)) + "/../config/" + filename
        )
        bris.utils.validate(full_filename, raise_on_error=True)


def test_parse_args():
    # Test the parse_args function
    args = bris.utils.parse_args(["--config", "test_config.yaml"])
    assert args["config"] == "test_config.yaml"

    # Test the parse_args function with an invalid argument
    try:
        bris.utils.parse_args(["--invalid_arg"])
    except SystemExit:
        pass
    else:
        assert False, "parse_args did not raise SystemExit for invalid argument"


def test_create_config():
    # Test the create_config function
    config = bris.utils.create_config("config/tox_test_inference.yaml", {})
    assert config is not None
    assert hasattr(config, "checkpoints")
    assert hasattr(config, "start_date")
    assert hasattr(config, "end_date")
    assert config["start_date"] == "2022-01-01T00:00:00"

    # Test the create_config function with an argument override
    config = bris.utils.create_config("config/tox_test_inference.yaml", {"start_date": "2022-01-01T12:34:56"})
    assert config is not None
    assert hasattr(config, "start_date")
    assert config["start_date"] == "2022-01-01T12:34:56"
