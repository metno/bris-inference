from collections.abc import Iterator
from functools import cached_property
from typing import Any

import numpy as np
import torch
from anemoi.utils.dates import frequency_to_seconds
from einops import rearrange
from torch.utils.data import IterableDataset

from bris.utils import LOGGER


class SparseZarrDataset(IterableDataset):
    """Sparse multi-source zarr sampling for forecaster inference.

    The yielded timestamp is the forecast reference time. Input frames are read
    at configured offsets before that timestamp, with non-output datasets sampled
    from the last available time.
    """

    def __init__(
        self,
        data_readers: dict[str, Any],
        grid_indices: dict[str, Any],
        *,
        dataset_input_offsets: dict[str, list[int]],
        output_dataset_names: list[str],
        forecast_start_date: str | None = None,
        forecast_end_date: str | None = None,
        forecast_reference_frequency: str | None = None,
        timeincrement: int = 1,
        num_members_in_sequence: int = 1,
        **_,
    ) -> None:
        self.data = data_readers
        self.dataset_names = list(data_readers.keys())
        self.grid_indices = grid_indices
        self.dataset_input_offsets = {
            str(name): [int(offset) for offset in offsets]
            for name, offsets in dataset_input_offsets.items()
        }
        self.output_dataset_names = set(str(name) for name in output_dataset_names)
        self.frequency = np.timedelta64(5 * int(timeincrement), "m")
        self.num_members_in_sequence = int(num_members_in_sequence)
        self.forecast_start_date = None
        self.forecast_end_date = None
        if forecast_start_date is not None:
            self.forecast_start_date = np.datetime64(forecast_start_date, "ns")
        if forecast_end_date is not None:
            self.forecast_end_date = np.datetime64(forecast_end_date, "ns")
        self.forecast_reference_frequency = None
        if forecast_reference_frequency is not None:
            self.forecast_reference_frequency = np.timedelta64(
                frequency_to_seconds(forecast_reference_frequency), "s"
            )
        self.forecast_reference_offset = max(
            max(self.dataset_input_offsets[name])
            for name in self.output_dataset_names
        )

        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0
        self.reader_group_rank = 0
        self.reader_group_size = 1
        self.ens_comm_group_id = 0
        self.ens_comm_num_groups = 1
        self.chunk_index_range: np.ndarray | None = None

        missing = sorted(set(self.dataset_names) - set(self.dataset_input_offsets))
        if missing:
            raise ValueError(f"Missing sparse input offsets for datasets: {missing}")

        LOGGER.info(
            "SparseZarrDataset: datasets=%s output_datasets=%s forecast_reference_offset=%s input_offsets=%s",
            self.dataset_names,
            sorted(self.output_dataset_names),
            self.forecast_reference_offset,
            self.dataset_input_offsets,
        )
        if self.forecast_start_date is not None or self.forecast_end_date is not None:
            LOGGER.info(
                "SparseZarrDataset: requested forecast_reference_time range %s to %s",
                self.forecast_start_date,
                self.forecast_end_date,
            )
        if self.forecast_reference_frequency is not None:
            LOGGER.info(
                "SparseZarrDataset: forecast_reference_frequency=%s",
                self.forecast_reference_frequency,
            )

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        base_dataset_name = sorted(self.output_dataset_names)[0]
        base_dates = np.asarray(self.data[base_dataset_name].dates, dtype="datetime64[ns]")
        dataset_dates = {
            dataset_name: np.asarray(dataset.dates, dtype="datetime64[ns]")
            for dataset_name, dataset in self.data.items()
        }

        valid_indices = []
        for base_index, base_date in enumerate(base_dates):
            forecast_reference_time = np.datetime64(
                base_date + self.forecast_reference_offset * self.frequency,
                "ns",
            )
            if self.forecast_start_date is not None and forecast_reference_time < self.forecast_start_date:
                continue
            if self.forecast_end_date is not None and forecast_reference_time > self.forecast_end_date:
                continue
            if self.forecast_reference_frequency is not None:
                anchor_time = self.forecast_start_date
                if anchor_time is None:
                    anchor_time = np.datetime64("1970-01-01T00:00:00", "ns")
                if (forecast_reference_time - anchor_time) % self.forecast_reference_frequency != np.timedelta64(0, "ns"):
                    continue

            all_datasets_available = True
            for dataset_name, offsets in self.dataset_input_offsets.items():
                dates = dataset_dates[dataset_name]
                for offset in offsets:
                    requested_time = np.datetime64(base_date + offset * self.frequency, "ns")
                    if dataset_name in self.output_dataset_names:
                        sample_index = np.searchsorted(dates, requested_time)
                        if sample_index >= len(dates) or dates[sample_index] != requested_time:
                            all_datasets_available = False
                            break
                    else:
                        sample_index = np.searchsorted(dates, requested_time, side="right") - 1
                        if sample_index < 0:
                            all_datasets_available = False
                            break
                if not all_datasets_available:
                    break

            if all_datasets_available:
                valid_indices.append(base_index)

        valid_indices = np.asarray(valid_indices, dtype=np.uint32)
        LOGGER.info(
            "SparseZarrDataset: found %s valid base dates from %s output dates",
            len(valid_indices),
            len(base_dates),
        )
        if len(valid_indices) > 0:
            first_base_date = base_dates[int(valid_indices[0])]
            last_base_date = base_dates[int(valid_indices[-1])]
            first_forecast_time = np.datetime64(first_base_date + self.forecast_reference_offset * self.frequency, "ns")
            last_forecast_time = np.datetime64(last_base_date + self.forecast_reference_offset * self.frequency, "ns")
            LOGGER.info(
                "SparseZarrDataset: forecast_reference_time range %s to %s",
                first_forecast_time,
                last_forecast_time,
            )
        return valid_indices

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_num_groups = ens_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        shard_size = len(self.valid_date_indices) // self.ens_comm_num_groups
        if shard_size < 1:
            raise RuntimeError("No sparse inference samples available for this worker.")

        shard_start = self.ens_comm_group_id * shard_size
        shard_end = (self.ens_comm_group_id + 1) * shard_size
        worker_size = max(1, (shard_end - shard_start) // n_workers)
        low = shard_start + worker_id * worker_size
        high = min(shard_start + (worker_id + 1) * worker_size, shard_end)
        self.chunk_index_range = np.tile(np.arange(low, high, dtype=np.uint32), self.num_members_in_sequence)

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], str]]:
        if self.chunk_index_range is None:
            self.chunk_index_range = np.tile(
                np.arange(len(self.valid_date_indices), dtype=np.uint32),
                self.num_members_in_sequence,
            )

        base_dataset_name = sorted(self.output_dataset_names)[0]
        base_dates = np.asarray(self.data[base_dataset_name].dates, dtype="datetime64[ns]")
        dataset_dates = {
            dataset_name: np.asarray(dataset.dates, dtype="datetime64[ns]")
            for dataset_name, dataset in self.data.items()
        }

        for valid_position in self.chunk_index_range:
            base_index = int(self.valid_date_indices[int(valid_position)])
            base_date = np.datetime64(base_dates[base_index], "ns")
            forecast_reference_time = np.datetime64(
                base_date + self.forecast_reference_offset * self.frequency,
                "ns",
            )
            LOGGER.info(
                "SparseZarrDataset: reading base_date=%s forecast_reference_time=%s",
                base_date,
                forecast_reference_time,
            )

            batch = {}
            for dataset_name, dataset in self.data.items():
                frames = []
                dates = dataset_dates[dataset_name]
                grid_shard_indices = self.grid_indices[dataset_name].get_shard_indices(self.reader_group_rank)

                for offset in self.dataset_input_offsets[dataset_name]:
                    requested_time = np.datetime64(base_date + offset * self.frequency, "ns")
                    if dataset_name in self.output_dataset_names:
                        sample_index = int(np.searchsorted(dates, requested_time))
                    else:
                        sample_index = int(np.searchsorted(dates, requested_time, side="right") - 1)

                    frame = dataset[sample_index : sample_index + 1, :, :, :]
                    frames.append(frame[..., grid_shard_indices])

                window = np.concatenate(frames, axis=0)
                batch[dataset_name] = torch.from_numpy(
                    rearrange(window, "time variable ensemble cell -> time ensemble cell variable")
                )

            yield batch, str(forecast_reference_time)
