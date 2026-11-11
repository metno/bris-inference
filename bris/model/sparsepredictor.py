from typing import Any

import numpy as np
import torch

from bris.utils import LOGGER
from bris.utils import get_base_seed
from bris.utils import get_model_timestep
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
        self.timestep = timedelta64_from_timestep(get_model_timestep(checkpoint))
        self.dataset_names = list(datamodule.dataset_names)
        self.forecast_length = int(checkpoints_config["forecaster"]["leadtimes"])
        self.required_variables = {
            str(dataset_name): list(variables)
            for dataset_name, variables in required_variables.items()
        }
        self.requested_variables = {name: list(variables) for name, variables in self.required_variables.items()}
        self.lightning_motion_head = getattr(self.model, "lightning_motion_head", None)
        self.apply_lightning_motion = (
            self.lightning_motion_head is not None
            and self.lightning_motion_head.output_variable
            in self.required_variables.get(self.lightning_motion_head.lightning_dataset, [])
        )
        self.motion_output = (
            (self.lightning_motion_head.lightning_dataset, self.lightning_motion_head.output_variable)
            if self.apply_lightning_motion else None
        )
        if self.apply_lightning_motion:
            head = self.lightning_motion_head
            variables = self.required_variables[head.lightning_dataset]
            if head.lightning_variable not in variables:
                variables.append(head.lightning_variable)
            head.set_grid(
                radar_latitudes=datamodule.latitudes[head.radar_dataset],
                radar_longitudes=datamodule.longitudes[head.radar_dataset],
                lightning_latitudes=datamodule.latitudes[head.lightning_dataset],
                lightning_longitudes=datamodule.longitudes[head.lightning_dataset],
            )
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
        if self.apply_lightning_motion:
            for name, count in ((head.radar_dataset, 7), (head.lightning_dataset, 5)):
                expected = list(range(self.forecast_reference_offset - count + 1, self.forecast_reference_offset + 1))
                if self.dataset_input_offsets[name][-count:] != expected:
                    raise ValueError(f"Lightning motion requires {count} consecutive history frames ending at issue for {name}")
            if self.timestep != np.timedelta64(head.timestep_minutes, "m"):
                raise ValueError("Lightning motion timestep differs from the forecast timestep")
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
        if isinstance(n_step_output, dict):
            output_steps = {
                int(n_step_output[dataset_name])
                for dataset_name in self.model_output_dataset_names
                if int(n_step_output[dataset_name]) > 0
            }
            if len(output_steps) != 1:
                raise ValueError(
                    "Sparse forecaster output datasets must have one common positive output length, "
                    f"got {n_step_output}."
                )
            n_step_output = output_steps.pop()
        self.step_size = int(n_step_output)

        self.precipitation_motion_head = getattr(self.model, "precipitation_motion_head", None)
        rain_head = self.precipitation_motion_head
        self.apply_precipitation_motion = (
            rain_head is not None
            and any(
                name == rain_head.radar_variable or {name, rain_head.radar_variable} == {"rr", "lwe_precipitation_rate"}
                for name in self.required_variables.get(rain_head.radar_dataset, [])
            )
        )
        if self.apply_precipitation_motion:
            expected = list(range(self.forecast_reference_offset - 3, self.forecast_reference_offset + 1))
            if self.dataset_input_offsets[rain_head.radar_dataset][-4:] != expected:
                raise ValueError("Precipitation motion requires four consecutive radar frames ending at issue")
            if self.step_size < rain_head.minimum_forecast_steps:
                raise ValueError("Precipitation motion reference must fit in the first output block")
            if self.timestep != np.timedelta64(rain_head.timestep_minutes, "m"):
                raise ValueError("Precipitation motion timestep differs from the forecast timestep")

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

        model_comm_group = self.model_comm_group
        if model_comm_group is not None and model_comm_group.size() == 1:
            model_comm_group = None
        y_hat = inner_model.forward(x, fcstep=fcstep, model_comm_group=model_comm_group)
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
        columns = []
        for name in output_names:
            if (dataset_name, name) == self.motion_output:
                columns.append(torch.full_like(frame[:, 0], torch.nan))
            else:
                columns.append(frame[:, self.variable_index(source_names, name)])
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
        lightning_steps = (
            self.lightning_motion_head.max_lead_minutes // self.lightning_motion_head.timestep_minutes
            if self.apply_lightning_motion else 0
        )
        required_steps = max(lightning_steps, self.step_size if self.apply_precipitation_motion else 0)
        model_horizon = max(horizon_steps, required_steps) if horizon_steps else 0
        rain_context, lightning_context = [], []
        reference_rain = None
        noise_processes = getattr(self.model.model, "absolute_lead_noise_process", {})
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            if self.apply_precipitation_motion and horizon_steps and ensemble_member != 0:
                # Imputers replace these masks; preserve their state across the control forecast.
                imputer_state = [
                    (module, name, getattr(module, name))
                    for module in self.model.pre_processors.modules()
                    for name in ("nan_locations", "loss_mask_training")
                    if hasattr(module, name)
                ]
                torch.manual_seed(self.seed_for_forecast_member(self.forecast_reference_time, 0))
                for process in noise_processes.values():
                    process._noise_generator = None
                    process._noise_generator_device = None
                control = self({name: value.unsqueeze(0) for name, value in current.items()}, fcstep=0)
                head = self.precipitation_motion_head
                values = self.prediction_as_time_ensemble_cell_variable(control[head.radar_dataset])
                column = self.variable_index(self.output_names_by_dataset[head.radar_dataset], head.radar_variable)
                reference_rain = values[:, 0, :, column].float().cpu()
                for module, name, value in imputer_state:
                    setattr(module, name, value)
                del control, values, imputer_state
            torch.manual_seed(seed)
            if self.precipitation_motion_head is not None:
                # Private decoder noise must follow the forecast member seed as well.
                for process in noise_processes.values():
                    process._noise_generator = None
                    process._noise_generator_device = None
            while self.steps_done < model_horizon:
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
                    take = min(model_horizon - self.steps_done, prediction.shape[0])
                    self.produced_steps = take if self.produced_steps is None else min(self.produced_steps, take)

                    if self.apply_precipitation_motion and dataset_name == self.precipitation_motion_head.radar_dataset:
                        head = self.precipitation_motion_head
                        column = self.variable_index(self.output_names_by_dataset[dataset_name], head.radar_variable)
                        rain_context.append(prediction[:take, :1, :, column].float().cpu())
                    if self.apply_lightning_motion and self.steps_done < lightning_steps and dataset_name == self.lightning_motion_head.lightning_dataset:
                        head = self.lightning_motion_head
                        column = self.variable_index(self.output_names_by_dataset[dataset_name], head.lightning_variable)
                        lightning_context.append(prediction[:take, :1, :, column].float().cpu())

                    if dataset_name in outputs:
                        output_take = min(take, max(horizon_steps - self.steps_done, 0))
                        for column, name in enumerate(self.required_variables[dataset_name]):
                            target = outputs[dataset_name][0, self.steps_done + 1 : self.steps_done + 1 + output_take, :, column]
                            if (dataset_name, name) == self.motion_output:
                                target.fill_(torch.nan)
                            else:
                                index = self.variable_index(self.output_names_by_dataset[dataset_name], name)
                                target.copy_(prediction[:output_take, 0, :, index].cpu())

                if self.produced_steps is None or self.produced_steps < 1:
                    raise RuntimeError("Sparse forecaster produced no prediction frames.")

                next_time = forecast_reference_time + (self.steps_done + self.produced_steps) * self.timestep
                current = self.advance_input_predict(current, y_pred, next_time)
                self.steps_done += self.produced_steps
                self.rollout_iter += 1
                if self.release_cache:
                    torch.cuda.empty_cache()

        if self.apply_precipitation_motion and horizon_steps:
            head = self.precipitation_motion_head
            column = self.variable_index(self.dataset_variables[head.radar_dataset], head.radar_variable)
            radar_history = batch[head.radar_dataset][0, -4:, 0, :, column].float().cpu()
            raw_rain = torch.cat(rain_context, dim=0)
            rain_context.clear()
            if reference_rain is None:
                reference_rain = raw_rain[:self.step_size, 0]
            corrected = head(raw_rain, radar_history,
                np.arange(1, len(raw_rain) + 1) * head.timestep_minutes,
                reference_prediction=reference_rain)
            column = self.variable_index(self.required_variables[head.radar_dataset], head.radar_variable)
            outputs[head.radar_dataset][0, 1:, :, column] = corrected[:horizon_steps, 0]
            del raw_rain, corrected, reference_rain

        if self.apply_lightning_motion and horizon_steps:
            head = self.lightning_motion_head
            variables = self.required_variables[head.lightning_dataset]
            prediction = outputs[head.lightning_dataset]
            raw = torch.cat(lightning_context, dim=0)[:lightning_steps]
            lightning_context.clear()
            radar_channel = self.variable_index(self.dataset_variables[head.radar_dataset], head.radar_variable)
            lightning_channel = self.variable_index(self.dataset_variables[head.lightning_dataset], head.lightning_variable)
            scores = head(
                raw,
                batch[head.radar_dataset][0, -7:, 0, :, radar_channel],
                batch[head.lightning_dataset][0, -5:, 0, :, lightning_channel],
                np.arange(1, len(raw) + 1) * head.timestep_minutes,
            )
            count = min(horizon_steps, len(scores))
            prediction[0, 1:count + 1, :, variables.index(head.output_variable)] = scores[:count, 0]
        if self.apply_lightning_motion:
            head = self.lightning_motion_head
            variables = self.required_variables[head.lightning_dataset]
            prediction = outputs[head.lightning_dataset]
            requested = self.requested_variables[head.lightning_dataset]
            if requested != variables:
                outputs[head.lightning_dataset] = prediction[..., [variables.index(name) for name in requested]]

        return {
            "pred": {dataset_name: prediction.to(torch.float32).numpy() for dataset_name, prediction in outputs.items()},
            "times": times,
            "group_rank": self.model_comm_group_rank,
            "ensemble_member": ensemble_member,
        }
