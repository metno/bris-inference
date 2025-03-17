import torch
import einops
from typing import Optional

from bris.checkpoint import Checkpoint
from bris.misc.shapes import get_shape_shards


class ModelBlocks(Checkpoint):
    def __init__(self, path):
        super().__init__(path)
        # sketch works

        """
        example from terminal:
        
        >>> from bris.checkpoint import Checkpoint
        >>> checkpoint = Checkpoint(path)
        >>> proc = checkpoint._model_instance.model.processor
        >>> proc.state_dict()["proc.0.blocks.7.lin_value.bias"]
        tensor([ 0.0225, -0.0109,  0.0125,  ..., -0.0232,  0.0181,  0.0023])
        >>> # this is the state_dict. Checking if this is not random. Fetching model params
        >>> checkpoint._get_copy_model_params["model.processor.proc.0.blocks.7.lin_value.bias"]
        Parameter containing:
        tensor([ 0.0225, -0.0109,  0.0125,  ..., -0.0232,  0.0181,  0.0023],
            requires_grad=True)
        >>> checkpoint._get_copy_model_params["model.processor.proc.0.blocks.7.lin_value.bias"] ==proc.state_dict()["proc.0.blocks.7.lin_value.bias"]
        tensor([True, True, True,  ..., True, True, True])
        """

        ######### DO NOT REMOVE :) #########
        # This is not the end product. this class
        # will be fully changed in future and be more generic
        # this will include, multi-domain, normal-enc-proc-dec
        # ensemble, multi-enc-dec, etc..

        # fetch from checkpoint class
        self._model = self._model_instance.model
        self.graph_data = self.graph

        # if torch.cuda.is_available():
        self._encoder = self._model.encoder.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._processor = self._model.encoder.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._decoder = self._model.encoder.to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # self._node_attributes = self._model.node#NamedNodesAttributes(0, self._graph_data)

    def _preproces_data(
        self, x: torch.Tensor, model_comm_group: Optional[int] = None
    ) -> None:
        """
        hidden class method, preproccesses the data and enables the
        data as attributes that can be accessed.

        """
        self.batch_size = x.shape[0]
        self.ensemble_size = x.shape[2]

        # add data positional info (lat/lon)
        self.x_data_latent = torch.cat(
            (
                einops.rearrange(
                    x,
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                self._model.node_attributes(
                    self._graph_name_data, batch_size=self.batch_size
                ),
            ),
            dim=-1,  # feature dimension
        )

        self.x_hidden_latent = self._model.node_attributes(
            self._graph_name_hidden, batch_size=self.batch_size
        )

        self.shard_shapes_data = get_shape_shards(
            self.x_data_latent, 0, model_comm_group
        )
        self.shard_shapes_hidden = get_shape_shards(
            self.x_hidden_latent, 0, model_comm_group
        )

    def encoder(
        self, x: torch.Tensor, model_comm_group: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self._preproces_data(x=x)

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
        # TODO: write logic
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
        x: torch.Tensor,
        model_comm_group: Optional[int] = None,
        abstract=False,
    ) -> torch.Tensor:
        # TODO: write logic
        x_out = self._decoder(
            (x, self.x_data_latent),
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
