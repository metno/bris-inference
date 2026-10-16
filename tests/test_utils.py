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
    filenames = ["ensemble_stretched-grid.yaml", "interpolator_template.yaml"]
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
        raise AssertionError("parse_args did not raise SystemExit for invalid argument")


def test_create_config():
    # Test the create_config function
    config = bris.utils.create_config("config/tox_test_inference.yaml", {})
    assert config is not None
    assert hasattr(config, "checkpoints")
    assert hasattr(config, "start_date")
    assert hasattr(config, "end_date")
    assert config["start_date"] == "2022-01-01T00:00:00"

    # Test the create_config function with an argument override
    config = bris.utils.create_config(
        "config/tox_test_inference.yaml", {"start_date": "2022-01-01T12:34:56"}
    )
    assert config is not None
    assert hasattr(config, "start_date")
    assert config["start_date"] == "2022-01-01T12:34:56"


def test_shift_date():
    assert (
        bris.utils.shift_date("2022-01-01T00:00:00", 6 * 3600) == "2022-01-01T06:00:00"
    )
    assert (
        bris.utils.shift_date("2022-01-01T00:00:00", -6 * 3600) == "2021-12-31T18:00:00"
    )
    assert bris.utils.shift_date("2022-01-01T00:00:00", 0) == "2022-01-01T00:00:00"


def test_get_interpolator_boundary_timestep_seconds():
    from types import SimpleNamespace

    with_task = SimpleNamespace(
        config=SimpleNamespace(task=SimpleNamespace(input_timestep="6h"))
    )
    assert bris.utils.get_interpolator_boundary_timestep_seconds(with_task) == 21600
    # the configured value wins
    assert (
        bris.utils.get_interpolator_boundary_timestep_seconds(with_task, 3600) == 3600
    )

    # explicit input times are in units of the training data's time step ...
    explicit_times = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(explicit_times=SimpleNamespace(input=[0, 6])),
            data=SimpleNamespace(timestep="1h", frequency="1h"),
        )
    )
    assert (
        bris.utils.get_interpolator_boundary_timestep_seconds(explicit_times) == 21600
    )
    # ... or of its frequency when there is no time step
    explicit_times.config.data = SimpleNamespace(frequency="3h")
    assert (
        bris.utils.get_interpolator_boundary_timestep_seconds(explicit_times) == 64800
    )
    assert (
        bris.utils.get_interpolator_boundary_timestep_seconds(explicit_times, 21600)
        == 21600
    )
    import pytest

    no_times = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace()))
    with pytest.raises(ValueError, match="boundary_timestep_seconds"):
        bris.utils.get_interpolator_boundary_timestep_seconds(no_times)


def _checkpoint(**config):
    from types import SimpleNamespace

    def ns(d):
        return SimpleNamespace(
            **{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()}
        )

    checkpoint = ns({"config": config})
    checkpoint.multistep = config.get("training", {}).get("multistep_input", 2)
    return checkpoint


FORECASTER = _checkpoint(data={"timestep": "6h"}, training={"multistep_input": 2})
INTERPOLATOR = _checkpoint(
    training={"explicit_times": {"input": [0, 6], "target": [1, 2, 3, 4, 5, 6]}}
)
TASK_INTERPOLATOR = _checkpoint(
    task={
        "input_timestep": "6h",
        "output_timestep": "1h",
        "output_left_boundary": False,
        "output_right_boundary": False,
    }
)


def test_get_interpolator_boundary_states():
    assert bris.utils.get_interpolator_boundary_states(INTERPOLATOR, 5) == 6
    assert bris.utils.get_interpolator_boundary_states(TASK_INTERPOLATOR, 1) == 2


def test_get_run_window_forecaster():
    from omegaconf import OmegaConf

    config = OmegaConf.create({"checkpoints": {"forecaster": {"leadtimes": 3}}})
    window = bris.utils.get_run_window(config, {"forecaster": FORECASTER})
    assert window.checkpoint is FORECASTER
    assert (window.timestep, window.timestep_seconds) == ("6h", 21600)
    assert (window.multistep, window.rollout) == (2, 0)
    assert list(window.leadtimes) == [0, 21600, 43200]
    assert config.checkpoints.forecaster.timestep_seconds == 21600


def test_get_run_window_forecaster_and_interpolator():
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "checkpoints": {
                "forecaster": {"leadtimes": 3},
                "interpolator": {"leadtimes": 1},
            }
        }
    )
    window = bris.utils.get_run_window(
        config, {"forecaster": FORECASTER, "interpolator": INTERPOLATOR}
    )
    assert (window.multistep, window.rollout) == (2, 0)
    assert config.checkpoints.interpolator.timestep_seconds == 3600
    assert list(window.leadtimes) == [h * 3600 for h in range(7)] + [43200]


def test_get_run_window_interpolator_only():
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "checkpoints": {
                "interpolator": {"leadtimes": 2, "boundary_timestep_seconds": 21600}
            }
        }
    )
    window = bris.utils.get_run_window(config, {"interpolator": INTERPOLATOR})
    assert window.checkpoint is INTERPOLATOR
    assert (window.timestep, window.timestep_seconds) == ("21600s", 21600)
    # one state at the initialisation time and two boundary states after it
    assert (window.multistep, window.rollout) == (1, 2)
    assert config.checkpoints.interpolator.timestep_seconds == 3600
    assert list(window.leadtimes) == [h * 3600 for h in range(13)]

    # the boundary time step defaults to the checkpoint's input time step
    config = OmegaConf.create({"checkpoints": {"interpolator": {"leadtimes": 2}}})
    window = bris.utils.get_run_window(config, {"interpolator": TASK_INTERPOLATOR})
    assert config.checkpoints.interpolator.boundary_timestep_seconds == 21600
    assert list(window.leadtimes) == [h * 3600 for h in range(13)]


def test_get_run_window_without_checkpoints():
    import pytest
    from omegaconf import OmegaConf

    with pytest.raises(ValueError, match="forecaster and/or an interpolator"):
        bris.utils.get_run_window(OmegaConf.create({"checkpoints": {}}), {})


def test_set_dataset_dates():
    from omegaconf import OmegaConf

    forecaster = bris.utils.RunWindow(None, "6h", 21600, 2, 0, None)
    config = OmegaConf.create({"end_date": "2022-01-02T00:00:00"})
    bris.utils.set_dataset_dates(config, forecaster)
    assert (config.start_date, config.end_date) == (
        "2022-01-01T18:00:00",
        "2022-01-02T00:00:00",
    )

    config = OmegaConf.create(
        {"start_date": "2022-01-01T00:00:00", "end_date": "2022-01-02T00:00:00"}
    )
    bris.utils.set_dataset_dates(config, forecaster)
    assert (config.start_date, config.end_date) == (
        "2021-12-31T18:00:00",
        "2022-01-02T00:00:00",
    )

    interpolator_only = bris.utils.RunWindow(None, "21600s", 21600, 1, 2, None)
    config = OmegaConf.create(
        {"start_date": "2022-01-01T00:00:00", "end_date": "2022-01-02T00:00:00"}
    )
    bris.utils.set_dataset_dates(config, interpolator_only)
    assert (config.start_date, config.end_date) == (
        "2022-01-01T00:00:00",
        "2022-01-02T12:00:00",
    )
