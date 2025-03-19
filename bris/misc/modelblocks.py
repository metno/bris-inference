import numpy as np
import torch
import einops
import pytorch_lightning as pl

from pathlib import Path
from typing import Optional
from torch.distributed.distributed_c10d import ProcessGroup

from bris.checkpoint import Checkpoint
from bris.misc.shapes import get_shape_shards
from bris.model import get_variable_indices
from bris.data.datamodule import DataModule


class ModelBlocks(pl.LightningModule):
    def __init__(self, checkpoint: Checkpoint, which_block: str) -> None:
        super().__init__()

        assert isinstance(which_block, str)
        assert which_block in [
            "encoder",
            "processor",
            "decoder",
        ], f"Expecting which_block param to be encoder, processor or decoder. Got {which_block}"

        self.which_block = which_block
        torch.set_float32_matmul_precision("high")

        self.model_comm_group = None
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1

        self.model = checkpoint.model
        self._graph_name_data = "data"
        self._graph_name_hidden = "hidden"

        self.data_indices = self.model.data_indices

        if hasattr(self.data_indices, "internal_model") and hasattr(
            self.data_indices, "internal_data"
        ):
            self.internal_model = self.data_indices.internal_model
            self.internal_data = self.data_indices.internal_data
        else:
            self.internal_model = self.data_indices.model
            self.internal_data = self.data_indices.data

        self._encoder = self.model.model.encoder
        self._processor = self.model.model.processor
        self._decoder = self.model.model.decoder

    def encoder(
        self, x: torch.Tensor, model_comm_group: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self._preproces_data(batch=x, model_comm_group=model_comm_group)
        return self._encoder(
            (self.x_data_latent, self.x_hidden_latent),
            batch_size=self.batch_size,
            shard_shapes=(self.shard_shapes_data, self.shard_shapes_hidden),
            model_comm_group=model_comm_group,
        )

    def processor(
        self,
        x_latent: torch.Tensor,
        with_skip_connection: Optional[bool] = True,
        model_comm_group: Optional[int] = None,
    ) -> torch.Tensor:

        x_latent_proc = self._processor(
            x_latent,
            batch_size=self.batch_size,
            shard_shapes=self.shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        if with_skip_connection:
            x_latent_proc = x_latent_proc + x_latent

        return x_latent_proc

    def decoder(
        self,
        x_latent_proc: torch.Tensor,
        x_data_latent: torch.Tensor,
        model_comm_group: Optional[int] = None,
        abstract=False,
    ) -> torch.Tensor:
        # TODO: write logic
        x_out = self._decoder(
            (x_latent_proc, x_data_latent),
            batch_size=self.batch_size,
            shard_shapes=(self.shard_shapes_hidden, self.shard_shapes_data),
            model_comm_group=model_comm_group,
        )

        if abstract:
            return x_out
        else:
            x_out = (
                einops.rearrange(
                    x_out,
                    "(batch ensemble grid) vars -> batch ensemble grid vars",
                    batch=self.batch_size,
                    ensemble=self.ensemble_size,
                )
                .to(dtype=x.dtype)
                .clone()
            )

            # residual connection (just for the prognostic variables)
            x_out[..., self._internal_output_idx] += x[
                :, -1, :, :, self._internal_input_idx
            ]

            for bounding in self.boundings:
                # bounding performed in the order specified in the config file
                x_out = bounding(x_out)

    def _preproces_data(
        self, batch: torch.Tensor, model_comm_group: Optional[int] = None
    ) -> None:
        """
        hidden class method, preproccesses the data and enables the
        data as attributes that can be accessed.


        args:
            batch : input data

        """
        # batch = batch.to(self.device)
        batch = self.model.pre_processors(batch, in_place=True)
        x = batch[..., self.internal_data.input.full]

        self.batch_size = x.shape[0]
        self.ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        self.x_data_latent = torch.cat(
            (
                einops.rearrange(
                    x,
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                self.model.model.node_attributes(
                    self._graph_name_data, batch_size=self.batch_size
                ),
            ),
            dim=-1,  # feature dimension
        )

        self.x_hidden_latent = self.model.model.node_attributes(
            self._graph_name_hidden, batch_size=self.batch_size
        )

        self.shard_shapes_data = get_shape_shards(
            self.x_data_latent, 0, model_comm_group
        )
        self.shard_shapes_hidden = get_shape_shards(
            self.x_hidden_latent, 0, model_comm_group
        )

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int = None,
        model_comm_group_rank: int = None,
        model_comm_num_groups: int = None,
        model_comm_group_size: int = None,
    ) -> None:
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.which_block == "encoder":
            return self.encoder(x, self.model_comm_group)

        if self.which_block == "processor":
            x_data_latent, x_latent = self.encoder(
                x=x, model_comm_group=self.model_comm_group
            )
            return self.processor(
                x_latent=x_latent, model_comm_group=self.model_comm_group
            )

        if self.which_block == "decoder":
            x_data_latent, x_latent = self.encoder(x, self.model_comm_group)
            x_latent_proc = self.processor(
                x_latent,
                model_comm_group=self.model_comm_group,
                with_skip_connection=True,
            )
            return self.decoder(
                x_latent_proc=x_latent_proc,
                x_data_latent=x_data_latent,
                model_comm_group=self.model_comm_group,
            )

    @torch.inference_mode
    def predict_step(
        self, batch: tuple[torch.Tensor, np.datetime64], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)
