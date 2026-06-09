from typing import Any

import numpy as np
import torch

from bris.utils import LOGGER
from bris.utils import get_base_seed
from bris.utils import timedelta64_from_timestep

from .basepredictor import BasePredictor


class SparseForecasterPredictor(BasePredictor):
    """BRIS predictor for sparse forecaster checkpoints."""

    def __init__(
        self,
        *args: Any,
        checkpoints: dict[str, Any],
        datamodule: Any,
        checkpoints_config: dict,
        required_variables: dict[str, list[str]],
        release_cache: bool = False,
        ensemble_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoints=checkpoints, **kwargs)

        checkpoint = checkpoints["forecaster"]
        self.model = checkpoint.model
        self.metadata = checkpoint.metadata
        self.data_indices = checkpoint.data_indices
        self.timestep = timedelta64_from_timestep(self.metadata.config.data.timestep)
        self.dataset_names = list(datamodule.dataset_names)
        self.forecast_length = int(checkpoints_config["forecaster"]["leadtimes"])
        self.required_variables = {
            str(dataset_name): list(variables)
            for dataset_name, variables in required_variables.items()
        }
        self.dataset_variables = {
            dataset_name: list(datamodule.data_readers[dataset_name].variables)
            for dataset_name in self.dataset_names
        }
        self.dataset_input_offsets = {
            dataset_name: [int(offset) for offset in datamodule.ds_predict.dataset_input_offsets[dataset_name]]
            for dataset_name in self.dataset_names
        }
        self.output_dataset_names = set(datamodule.ds_predict.output_dataset_names)
        self.forecast_reference_offset = int(datamodule.ds_predict.forecast_reference_offset)
        self.output_pairs_by_dataset = {
            dataset_name: sorted(
                (
                    (str(name), int(index))
                    for name, index in self.data_indices[dataset_name].model.output.name_to_index.items()
                ),
                key=lambda item: item[1],
            )
            for dataset_name in self.dataset_names
        }
        self.output_names_by_dataset = {
            dataset_name: [name for name, _ in pairs]
            for dataset_name, pairs in self.output_pairs_by_dataset.items()
        }
        self.model_output_dataset_names = {
            dataset_name
            for dataset_name, pairs in self.output_pairs_by_dataset.items()
            if pairs
        }

        n_step_output = getattr(self.model, "n_step_output", None)
        if n_step_output is None:
            n_step_output = getattr(getattr(self.model, "model", None), "n_step_output", None)
        if n_step_output is None:
            raise ValueError("Could not infer n_step_output from sparse forecaster checkpoint.")
        self.step_size = int(n_step_output)

        self.model.eval()
        self.release_cache = release_cache
        self.ensemble_seed = (
            int(ensemble_seed) if ensemble_seed is not None else get_base_seed()
        )
        self.batch_info: dict[np.datetime64, int] = {}

        LOGGER.info(
            "SparseForecasterPredictor: forecast_length=%s step_size=%s output_datasets=%s input_offsets=%s",
            self.forecast_length,
            self.step_size,
            sorted(self.output_dataset_names),
            self.dataset_input_offsets,
        )

    @classmethod
    def variable_index(cls, names: list[str], name: str) -> int:
        candidates = [name]
        if name == "rr":
            candidates.append("lwe_precipitation_rate")
        if name == "lwe_precipitation_rate":
            candidates.append("rr")

        for candidate in candidates:
            if candidate in names:
                return names.index(candidate)
        raise ValueError(f"Could not find variable '{name}' in {names}")

    def set_static_forcings(self, datareader) -> None:
        self.static_forcings = {}

    def forward(self, x: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        fcstep = int(kwargs.get("fcstep", 0))
        return self.predict_sparse_rollout_step(x, fcstep=fcstep)

    def predict_sparse_rollout_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        fcstep: int,
    ) -> dict[str, torch.Tensor]:
        pre_processors = getattr(self.model, "pre_processors", None)
        post_processors = getattr(self.model, "post_processors", None)
        inner_model = getattr(self.model, "model", None)
        if pre_processors is None or post_processors is None or inner_model is None:
            raise RuntimeError(
                "Sparse forecaster inference requires model.pre_processors, post_processors, and model.model."
            )

        x = {}
        for dataset_name, tensor in batch.items():
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(2)
            if tensor.ndim != 5:
                raise ValueError(
                    f"The {dataset_name} input tensor has an incorrect shape: "
                    f"expected 4 or 5 dimensions, got {tensor.shape}."
                )
            x[dataset_name] = pre_processors[dataset_name](tensor, in_place=False)

        y_hat = inner_model.forward(x, fcstep=fcstep, model_comm_group=self.model_comm_group)
        if not isinstance(y_hat, dict):
            raise TypeError(f"Expected sparse forecaster output dict, got {type(y_hat)!r}.")

        for dataset_name, prediction in list(y_hat.items()):
            if dataset_name in self.model_output_dataset_names:
                y_hat[dataset_name] = post_processors[dataset_name](prediction, in_place=False)
        return y_hat

    def prediction_as_time_ensemble_cell_variable(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim == 5:
            prediction = prediction[0]
        if prediction.ndim == 4:
            return prediction
        if prediction.ndim == 3:
            return prediction.unsqueeze(1)
        raise ValueError(f"Unexpected sparse prediction shape: {tuple(prediction.shape)}")

    def initial_output_frame(self, dataset_name: str, window: torch.Tensor) -> torch.Tensor:
        source_names = self.dataset_variables[dataset_name]
        output_names = self.required_variables[dataset_name]
        offset_index = self.dataset_input_offsets[dataset_name].index(self.forecast_reference_offset)
        frame = window[offset_index, 0]
        columns = [frame[:, self.variable_index(source_names, name)] for name in output_names]
        return torch.stack(columns, dim=-1)

    def ensemble_member_for_forecast(self, forecast_reference_time: np.datetime64) -> int:
        member_id = getattr(self, "member_id", 0)
        occurrence = self.batch_info.get(forecast_reference_time, 0)
        return member_id + self.num_members_in_parallel * occurrence

    def seed_for_forecast_member(
        self,
        forecast_reference_time: np.datetime64,
        ensemble_member: int,
    ) -> int:
        epoch = np.datetime64("1970-01-01T00:00:00", "m")
        forecast_minutes = int(
            (forecast_reference_time.astype("datetime64[m]") - epoch) / np.timedelta64(1, "m")
        )
        return int((self.ensemble_seed + forecast_minutes * 1000 + int(ensemble_member)) % (2**63 - 1))

    def advance_input_predict(
        self,
        current: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        time: np.datetime64,
    ) -> dict[str, torch.Tensor]:
        elapsed_steps = int((np.datetime64(time, "ns") - self.forecast_reference_time) / self.timestep)
        pred_window_start = self.forecast_reference_offset + self.steps_done + 1
        next_current = {}

        for dataset_name in self.dataset_names:
            data_index = self.data_indices[dataset_name]
            input_name_to_index = {
                str(name): int(index)
                for name, index in data_index.data.input.name_to_index.items()
            }
            prediction = y_pred.get(dataset_name)
            available_frames = dict(self.source_frames[dataset_name])
            for offset, frame in zip(self.current_offsets[dataset_name], current[dataset_name], strict=True):
                available_frames[int(offset)] = frame

            if prediction is not None:
                for produced_idx in range(self.produced_steps):
                    target_offset = pred_window_start + produced_idx
                    candidates = [available for available in sorted(available_frames) if available <= target_offset]
                    if not candidates:
                        raise ValueError(
                            f"Dataset '{dataset_name}' has no sparse frame at or before relative time {target_offset}."
                        )
                    frame = available_frames[int(candidates[-1])].clone()
                    predicted_frame = prediction[produced_idx, 0]
                    for variable_name, output_index in self.output_pairs_by_dataset[dataset_name]:
                        input_index = input_name_to_index.get(variable_name)
                        if input_index is not None:
                            frame[:, input_index] = predicted_frame[:, output_index]
                    available_frames[target_offset] = frame

            next_frames = []

            requested_times = [
                int(offset + elapsed_steps)
                for offset in self.dataset_input_offsets[dataset_name]
            ]
            for requested_time in requested_times:
                candidates = [available for available in sorted(available_frames) if available <= requested_time]
                if not candidates:
                    raise ValueError(
                        f"Dataset '{dataset_name}' has no sparse frame at or before relative time {requested_time}."
                    )

                next_frames.append(available_frames[int(candidates[-1])].clone())

            self.current_offsets[dataset_name] = requested_times
            self.source_frames[dataset_name].update(available_frames)
            next_current[dataset_name] = torch.stack(next_frames, dim=0)

        return next_current

    @torch.inference_mode()
    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        batch, time_stamp = batch
        forecast_reference_time = np.datetime64(time_stamp[0])
        self.forecast_reference_time = np.datetime64(forecast_reference_time, "ns")
        ensemble_member = self.ensemble_member_for_forecast(self.forecast_reference_time)
        self.batch_info[self.forecast_reference_time] = self.batch_info.get(self.forecast_reference_time, 0) + 1
        seed = self.seed_for_forecast_member(self.forecast_reference_time, ensemble_member)
        times = [
            forecast_reference_time + step * self.timestep
            for step in range(self.forecast_length)
        ]
        LOGGER.info(
            "SparseForecasterPredictor batch=%s forecast_reference_time=%s ensemble_member=%s seed=%s",
            batch_idx,
            forecast_reference_time,
            ensemble_member,
            seed,
        )

        current = {}
        self.source_frames = {}
        self.current_offsets = {}
        outputs = {}
        for dataset_name in self.dataset_names:
            window = batch[dataset_name][0, :, 0, :, :]
            offsets = self.dataset_input_offsets[dataset_name]
            self.source_frames[dataset_name] = {
                int(offset): window[offset_index].clone()
                for offset_index, offset in enumerate(offsets)
            }
            self.current_offsets[dataset_name] = list(offsets)
            current[dataset_name] = torch.stack(
                [self.source_frames[dataset_name][int(offset)] for offset in offsets],
                dim=0,
            )

            if dataset_name in self.required_variables:
                outputs[dataset_name] = torch.empty(
                    (1, self.forecast_length, window.shape[-2], len(self.required_variables[dataset_name])),
                    dtype=window.dtype,
                    device="cpu",
                )
                outputs[dataset_name][0, 0] = self.initial_output_frame(dataset_name, batch[dataset_name][0]).cpu()

        self.steps_done = 0
        self.rollout_iter = 0
        horizon_steps = self.forecast_length - 1
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            while self.steps_done < horizon_steps:
                model_batch = {
                    dataset_name: dataset_current.unsqueeze(0)
                    for dataset_name, dataset_current in current.items()
                }
                raw_prediction = self(model_batch, fcstep=self.rollout_iter)

                y_pred = {}
                self.produced_steps = None
                for dataset_name, prediction in raw_prediction.items():
                    prediction = self.prediction_as_time_ensemble_cell_variable(prediction)
                    y_pred[dataset_name] = prediction
                    take = min(horizon_steps - self.steps_done, prediction.shape[0])
                    self.produced_steps = take if self.produced_steps is None else min(self.produced_steps, take)

                    if dataset_name in outputs:
                        selected = torch.stack(
                            [
                                prediction[
                                    :take,
                                    0,
                                    :,
                                    self.variable_index(self.output_names_by_dataset[dataset_name], name),
                                ]
                                for name in self.required_variables[dataset_name]
                            ],
                            dim=-1,
                        )
                        outputs[dataset_name][0, self.steps_done + 1 : self.steps_done + 1 + take] = selected.cpu()

                if self.produced_steps is None or self.produced_steps < 1:
                    raise RuntimeError("Sparse forecaster produced no prediction frames.")

                next_time = forecast_reference_time + (self.steps_done + self.produced_steps) * self.timestep
                current = self.advance_input_predict(current, y_pred, next_time)
                self.steps_done += self.produced_steps
                self.rollout_iter += 1
                if self.release_cache:
                    torch.cuda.empty_cache()

        return {
            "pred": {dataset_name: prediction.to(torch.float32).numpy() for dataset_name, prediction in outputs.items()},
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": ensemble_member,
        }
