"""The datasets of a model that a run uses: all of them (a model with several encoders/decoders),
or those in the checkpoint's ``datasets`` option (a model trained on several domains)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from anemoi.utils.config import DotDict
from test_model_inputs import (
    DATASET_VARIABLES,
    DIAGNOSTIC,
    FORCING,
    NAMES,
    _DataIndex,
    _ModelIndex,
)

from bris.model.brispredictor import BrisPredictor
from bris.model.interpolator import Interpolator
from bris.model.interpolator_only import InterpolatorOnly
from bris.model.model_utils import check_run_datasets, select_dataset_names
from bris.model.runner import ModelRunner

POINTS = 3
T0 = np.datetime64("2022-06-21T00")
H = np.timedelta64(1, "h")
HARDWARE = {"num_gpus_per_model": 1, "num_gpus_per_node": 1, "num_nodes": 1}


class _Checkpoint:
    """Stand-in for the Checkpoint of a model with several datasets. Like the anemoi model, its
    forward takes any of its datasets, as long as they share a hidden mesh."""

    def __init__(self, dataset2hidden: dict[str, str]):
        names = list(dataset2hidden)
        model = torch.nn.Module()
        model.model = SimpleNamespace(dataset_names=names)
        model.pre_processors = dict.fromkeys(names, lambda x, in_place=False: x)
        model.post_processors = dict.fromkeys(names, lambda x, in_place=False: x)
        self.model = model
        self.data_indices = {
            name: SimpleNamespace(
                data=_DataIndex(NAMES, FORCING, DIAGNOSTIC),
                model=_ModelIndex(NAMES, FORCING, DIAGNOSTIC),
            )
            for name in names
        }
        self.multistep = 2
        self.config = SimpleNamespace(training=SimpleNamespace(multistep_input=2))
        self.metadata = DotDict(
            {"provenance_training": {"module_versions": {"anemoi.training": "0"}}}
        )
        self.calls: list[list[str]] = []
        num_outputs = len(self.data_indices[names[0]].model.output.name_to_index)

        def forward(x, model_comm_group=None, fcstep=None, **kwargs):
            hidden = {dataset2hidden[name] for name in x}
            if len(hidden) != 1:
                raise ValueError(
                    "All datasets in a batch must use the same hidden mesh"
                )
            self.calls.append(list(x))
            return {
                name: torch.full(
                    (x[name].shape[0], x[name].shape[2], POINTS, num_outputs), 100.0
                )
                for name in x
            }

        model.forward = forward


def multi_domain():
    return _Checkpoint({"domain_1": "hidden_1", "domain_2": "hidden_2"})


def multi_encoder():
    return _Checkpoint({"global": "hidden", "regional": "hidden"})


def datamodule(names):
    return SimpleNamespace(
        dataset_names=list(names),
        data_readers={
            name: SimpleNamespace(variables=DATASET_VARIABLES) for name in names
        },
        latitudes={name: np.full(POINTS, 60.0) for name in names},
        longitudes={name: np.full(POINTS, 10.0) for name in names},
    )


def predictor(checkpoint, data_names, datasets=None):
    config = {"timestep_seconds": 21600, "leadtimes": 3}
    if datasets is not None:
        config["datasets"] = datasets
    model = BrisPredictor(
        checkpoints={"forecaster": checkpoint},
        datamodule=datamodule(data_names),
        checkpoints_config={"forecaster": config},
        required_variables=dict.fromkeys(data_names, ["2t"]),
        hardware_config=HARDWARE,
        num_members_in_parallel=1,
    )
    model.member_id = 0  # set by the strategy in a run
    return model


def test_select_dataset_names():
    checkpoint = _Checkpoint({"a": "h", "b": "h", "c": "h"})
    assert select_dataset_names(checkpoint) == ["a", "b", "c"]
    assert select_dataset_names(checkpoint, None) == ["a", "b", "c"]
    assert select_dataset_names(checkpoint, "b") == ["b"]
    # In the order of the model, whatever the order in the config
    assert select_dataset_names(checkpoint, ["c", "a"]) == ["a", "c"]
    with pytest.raises(ValueError, match="not datasets of the model"):
        select_dataset_names(checkpoint, ["a", "d"])
    with pytest.raises(ValueError, match="at least one"):
        select_dataset_names(checkpoint, [])


def test_check_run_datasets():
    check_run_datasets("P", "forecaster", ["a", "b"], ["b", "a"])
    for data_names in [["a"], ["a", "b", "c"], ["a", "c"]]:
        with pytest.raises(ValueError, match="checkpoints.forecaster.datasets"):
            check_run_datasets("P", "forecaster", ["a", "b"], data_names)


def test_model_runner_datasets():
    """The runner only knows the datasets it runs on."""
    runner = ModelRunner(multi_domain(), timestep=6 * H, datasets=["domain_2"])
    assert runner.dataset_names == ["domain_2"]
    assert list(runner.data_index) == ["domain_2"]
    assert list(runner.model_index) == ["domain_2"]

    runner = ModelRunner(multi_domain(), timestep=6 * H)
    assert runner.dataset_names == ["domain_1", "domain_2"]


def test_multi_encoder_requires_all_datasets():
    """Without the option the input data must hold all the datasets of the model."""
    model = predictor(multi_encoder(), ["regional", "global"])
    assert model.model.dataset_names == ["global", "regional"]

    with pytest.raises(ValueError, match="do not match"):
        predictor(multi_encoder(), ["global"])
    with pytest.raises(ValueError, match="do not match"):
        predictor(multi_encoder(), ["global", "regional", "other"])


def test_multi_domain_requires_the_option():
    with pytest.raises(ValueError, match="checkpoints.forecaster.datasets"):
        predictor(multi_domain(), ["domain_1"])


def test_multi_domain_checks_the_option():
    # A dataset the model does not have
    with pytest.raises(ValueError, match="not datasets of the model"):
        predictor(multi_domain(), ["domain_3"], datasets=["domain_3"])
    # Input data of another domain than the one asked for
    with pytest.raises(ValueError, match="do not match"):
        predictor(multi_domain(), ["domain_2"], datasets=["domain_1"])
    # More input data than asked for
    with pytest.raises(ValueError, match="do not match"):
        predictor(multi_domain(), ["domain_1", "domain_2"], datasets=["domain_1"])


@pytest.mark.parametrize("datasets", [["domain_2"], "domain_2"])
def test_multi_domain_forecast(datasets):
    """A model trained on two domains runs on one of them: only that dataset reaches the model."""
    checkpoint = multi_domain()
    model = predictor(checkpoint, ["domain_2"], datasets=datasets)
    assert model.model.dataset_names == ["domain_2"]
    assert list(model.model.input_builders) == ["domain_2"]

    batch = {"domain_2": torch.ones((1, 2, 1, POINTS, len(DATASET_VARIABLES)))}
    out = model.predict_step((batch, [str(T0)]), 0)

    assert checkpoint.calls == [["domain_2"], ["domain_2"]]
    assert list(out["pred"]) == ["domain_2"]
    assert [(t - T0) / H for t in out["times"]] == [0, 6, 12]
    values = out["pred"]["domain_2"][0, :, 0, 0]
    assert list(values) == [1, 100, 100]


def test_multi_domain_two_domains_fail_in_the_model():
    """Bris does not judge which of the model's datasets go together; the model does."""
    checkpoint = multi_domain()
    names = ["domain_1", "domain_2"]
    model = predictor(checkpoint, names)

    batch = {
        name: torch.ones((1, 2, 1, POINTS, len(DATASET_VARIABLES))) for name in names
    }
    with pytest.raises(ValueError, match="same hidden mesh"):
        model.predict_step((batch, [str(T0)]), 0)


@pytest.mark.parametrize(
    "predictor_class, checkpoints",
    [
        (Interpolator, ["forecaster", "interpolator"]),
        (InterpolatorOnly, ["interpolator"]),
    ],
)
def test_interpolators_name_the_option(predictor_class, checkpoints):
    """The interpolators run on one dataset: a model with several needs the option."""
    config = {"timestep_seconds": 3600, "leadtimes": 2}
    with pytest.raises(AssertionError, match=r"checkpoints\.\w+\.datasets"):
        predictor_class(
            checkpoints={name: _interpolating(multi_domain()) for name in checkpoints},
            datamodule=datamodule(["domain_1"]),
            checkpoints_config=dict.fromkeys(checkpoints, config),
            required_variables={"domain_1": ["2t"]},
            hardware_config=HARDWARE,
            num_members_in_parallel=1,
        )


def _interpolating(checkpoint):
    """Give a checkpoint the time geometry of an interpolator (boundaries 0 and 6, outputs 1..5)."""
    checkpoint.config.training.explicit_times = SimpleNamespace(
        input=[0, 6], target=[1, 2, 3, 4, 5]
    )
    checkpoint.config.data = SimpleNamespace(timestep="1h", frequency="1h")
    return checkpoint
