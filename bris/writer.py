import os
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.trainer.trainer import Trainer

from .utils import LOGGER

REDUCE_OPS = {"sum": "SUM", "min": "MIN", "max": "MAX"}


def reduce_contributions(
    contributions: dict[str, np.ndarray],
    reductions: dict[str, str],
    group,
    device=None,
) -> tuple[dict[str, np.ndarray] | None, int, bool]:
    """Reduces the contributions from all ranks in group onto the root rank of the group.

    All ranks in the group must call this, in the same order for all outputs, since it performs
    collective operations.

    Args:
        contributions: name -> array from this rank
        reductions: name -> reduction operation ("sum", "min" or "max")
        group: torch.distributed process group to reduce over (or None, when running without
            torch.distributed, in which case the contributions are returned unchanged)
        device: Put tensors on this device before reducing. Required for the nccl backend. If
            None, the device is chosen based on the backend of the group.

    Returns:
        reduced: name -> reduced array on the root rank, None on the other ranks
        num_members: how many ranks (i.e. ensemble members) were reduced
        is_root: whether this rank is the root of the group
    """
    if group is None or not dist.is_available() or not dist.is_initialized():
        return contributions, 1, True

    world_size = dist.get_world_size(group)
    if world_size < 1:
        # This rank is not part of the group, so it has nothing to contribute
        return None, 0, False
    if world_size == 1:
        return contributions, 1, True

    root = dist.get_global_rank(group, 0)
    is_root = dist.get_rank() == root

    if device is None:
        device = "cuda" if dist.get_backend(group) == "nccl" else "cpu"

    reduced = {}
    for name in sorted(contributions.keys()):
        op = getattr(dist.ReduceOp, REDUCE_OPS[reductions[name]])
        tensor = torch.from_numpy(np.ascontiguousarray(contributions[name])).to(device)
        dist.reduce(tensor, dst=root, op=op, group=group)
        if is_root:
            reduced[name] = tensor.cpu().numpy()
        del tensor

    return (reduced if is_root else None), world_size, is_root


class CustomWriter(BasePredictionWriter):
    """This class is used in a callback to the trainer to write data to output."""

    def __init__(
        self,
        outputs: list[dict],
        process_list: list[Future] | None,
        write_interval: str = "batch",
        max_processes: int = os.cpu_count(),
    ) -> None:
        """
        Args:
            outputs (dict): Dict of domain-name to dict, where dict has "start", "end", and
                "outputs", where "outputs" is a list of Output objects that the writer will call.

            process_list (list): reference to empty list to add new process objects to, so the
                caller can keep track of background writer processs spawned by this function. Caller
                must run .join() on each process in list to wait for them to finish.

            write_interval (str): Only "batch" is supported.

            max_processes (int): Max background writing processes. Don't set <1.
        """
        super().__init__(write_interval)

        self.outputs = outputs
        self.process_list = process_list
        if max_processes < 1:
            max_processes = 1
        self.pool = ThreadPoolExecutor(max_workers=max_processes)
        LOGGER.debug(f"CustomWriter max_processes set to {max_processes}")

    def write_on_batch_end(
        self,
        trainer: Trainer,  # Not used
        pl_module: LightningModule,  # Not used
        prediction,
        batch_indices: Sequence[int] | None,  # Not used
        batch,  # Not used
        batch_idx: int,
        dataloader_idx: int,  # Not used
    ) -> None:
        """
        Args:
            prediction: This comes from predict_step in forecaster
        """

        # Wait for processes from the previous batch to finish
        LOGGER.debug(f"CustomWriter process_list contains {self.process_list}")
        while self.process_list is not None and len(self.process_list) > 0:
            LOGGER.debug(
                "CustomWriter waiting for previous process to complete before writing new data."
            )
            process = self.process_list.pop()
            process.result()
            LOGGER.debug("CustomWriter previous process completed.")

        times = prediction["times"]
        ensemble_member = prediction["ensemble_member"]
        if prediction["group_rank"] == 0:
            for output_dict in self.outputs:
                pred = prediction["pred"][output_dict["decoder_name"]]
                assert pred.shape[0] == 1, "Batchsize (per dataparallel) should be 1"
                pred = np.squeeze(pred, axis=0)
                pred = pred[
                    ...,
                    output_dict["start_gridpoint"] : output_dict["end_gridpoint"],
                    :,
                ]

                for output in output_dict["outputs"]:
                    if getattr(output, "reduce_across_members", False):
                        self._add_reduced_forecast(
                            output, pl_module, times, ensemble_member, pred, batch_idx
                        )
                    elif self.process_list is not None:
                        self.process_list.append(
                            self.pool.submit(
                                output.add_forecast, times, ensemble_member, pred
                            )
                        )
                        LOGGER.debug(
                            f"CustomWriter starting async add_forecast for member <{ensemble_member}>, times {times} for writing, batch_idx {batch_idx}."
                        )
                    else:
                        output.add_forecast(times, ensemble_member, pred)
                        LOGGER.debug(
                            f"CustomWriter added forecast for member <{ensemble_member}>, times {times} for writing, batch_idx {batch_idx}."
                        )

    def _add_reduced_forecast(
        self,
        output,
        pl_module: LightningModule | None,
        times: list,
        ensemble_member: int,
        pred: np.ndarray,
        batch_idx: int,
    ) -> None:
        """Reduces this member's contribution to output across the ensemble members that run in
        parallel, and registers the result with the output on the ensemble-root rank.

        The contribution and the reduction are done synchronously in the calling thread, since
        the reduction is a collective operation that must be issued in the same order on all
        ranks. Only the (cheap) storing of the reduced result is done in the background.
        """
        contributions = output.member_contribution(times, ensemble_member, pred)

        group = getattr(pl_module, "ens_output_comm_group", None)
        device = getattr(pl_module, "device", None)
        reduced, num_members, is_root = reduce_contributions(
            contributions, output.ensemble_reductions, group, device
        )
        LOGGER.debug(
            f"CustomWriter reduced member <{ensemble_member}> contribution across {num_members} members, times {times}, batch_idx {batch_idx}."
        )
        if not is_root:
            return

        if self.process_list is not None:
            self.process_list.append(
                self.pool.submit(
                    output.add_reduced_forecast, times, reduced, num_members
                )
            )
        else:
            output.add_reduced_forecast(times, reduced, num_members)

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # Not used
    ) -> None:
        """Called when prediction ends."""
        LOGGER.debug(
            "CustomWriter on_predict_end called, waiting for all processes to finish."
        )
        # Wait for all processes to finish
        if self.process_list is not None:
            while len(self.process_list) > 0:
                process = self.process_list.pop()
                process.result()
        LOGGER.debug("CustomWriter all processes finished.")
        self.pool.shutdown(wait=True)
