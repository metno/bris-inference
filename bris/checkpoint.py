import logging
import os
from copy import deepcopy
from functools import cached_property

import torch
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.config import DotDict
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)

try:
    from anemoi.models.data_indices.collection import IndexCollection
except ImportError:
    LOGGER.error(
        "\nAnemoi-models package missing. Install a version compatible with the checkpoint. <https://pypi.org/project/anemoi-models/>\n"
    )


class TrainingConfig(DotDict):
    data: DotDict
    dataloader: DotDict
    diagnostics: DotDict
    hardware: DotDict
    graph: DotDict
    model: DotDict
    training: DotDict


class Metadata(DotDict):
    config: TrainingConfig
    version: str
    seed: int
    run_id: str
    dataset: DotDict
    data_indices: DotDict
    provenance_training: DotDict
    timestamp: str
    uuid: str
    model: DotDict
    tracker: DotDict
    training: DotDict
    supporting_arrays_paths: DotDict


class Checkpoint:
    """This class makes accessible various information stored in Anemoi checkpoints."""

    def __init__(self, path: str, graph: str | None = None):
        assert os.path.exists(path), f"The given checkpoint {path} does not exist!"

        self.path = path
        self._model_instance = self._load_model()
        if graph:
            LOGGER.info("Updating graph to the one provided in config")
            self.update_graph(graph)

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def _metadata(self) -> Metadata:
        """
        Metadata of the model. This includes everything as in:
        -> data_indices (data, model (and internal) indices)
        -> dataset information (may vary from anemoi-datasets version used)
        -> runid
        -> model_summary, tracker, etc..

        args:
            None
        return
            metadata in DotDict format.

        Examples usage:
            metadata.data_indices.data.input (gives indices from data inputs)
            metadata.runid (the hash given when the model was trained)
        """
        try:
            return DotDict(load_metadata(self.path))
        except ValueError as e:
            LOGGER.warning(
                "Could not load and peek into the checkpoint metadata. Raising an expection"
            )
            raise e

    @property
    def config(self) -> TrainingConfig:
        """
        The configuriation used during model
        training.
        """
        return self._metadata.config

    @property
    def version(self) -> str:
        """
        Model version
        """
        return self._metadata.version

    @property
    def multistep(self) -> int:
        """
        Fetches multistep from metadata
        """
        if hasattr(self._metadata.config.training, "multistep"):
            return self._metadata.config.training.multistep
        if hasattr(self._metadata.config.training, "multistep_input"):
            return self._metadata.config.training.multistep_input
        raise RuntimeError("Cannot find multistep")

    @property
    def model(self) -> torch.nn.Module:
        return self._model_instance

    def _load_model(self) -> torch.nn.Module:
        """
        Loads a given model instance. This instance
        includes both the model interface and its
        corresponding model weights.
        """
        try:
            inst = torch.load(self.path, map_location="cpu", weights_only=False)
        except AttributeError as e:
            if str(e.args[0]).startswith("Can't get attribute"):
                raise RuntimeError(
                    "You most likely have a version of anemoi-models that is "
                    "not compatible with the checkpoint. Use bris-inspect to "
                    "check module versions."
                ) from e
            raise e
        if not torch.cuda.is_available():
            self._apply_triton_cpu_fallback(inst)
        self._clear_sharding_caches(inst)
        return inst

    # Attributes anemoi-models processor blocks use to cache per-rank halo exchange
    # metadata when the model is sharded across GPUs. They are plain attributes
    # (not registered buffers), so torch.load restores them on CPU and .to(device)
    # does not move them.
    _SHARDING_CACHE_ATTRS = (
        "_cached_halo_info",
        "_cached_partition",
        "_cached_halo_cache_specs",
    )

    def _clear_sharding_caches(self, model: torch.nn.Module) -> None:
        """Drop halo/partition caches that were pickled into the checkpoint.

        When a model is trained with model sharding (num_gpus_per_model > 1), the
        GraphTransformer processor blocks cache halo exchange metadata (edge indices,
        send indices, ...) as plain attributes. Saving the model with torch.save()
        pickles these caches, and torch.load(map_location="cpu") restores them on
        CPU. Because they are not registered buffers, moving the model to the GPU
        leaves them behind. If the inference sharding layout matches the training
        layout, the cache key matches and the CPU tensors are fed to the Triton
        attention kernel, which fails with
        "Pointer argument cannot be accessed from Triton (cpu tensor?)".

        The caches are pure derived state, so dropping them is safe: they are
        rebuilt on the correct device on the first forward pass.
        """
        cleared = 0
        for module in model.modules():
            for attr in self._SHARDING_CACHE_ATTRS:
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, None)
                    cleared += 1

        if cleared:
            LOGGER.info(
                "Cleared %d cached sharding attribute(s) restored from the checkpoint; "
                "they will be rebuilt on the inference device.",
                cleared,
            )

    def _apply_triton_cpu_fallback(self, model: torch.nn.Module) -> None:
        """Replace Triton graph attention with the PyG backend when running on CPU.

        anemoi-models checks is_triton_available() at model construction time and falls
        back to the PyG backend automatically. However, when a model is loaded from a
        checkpoint via torch.load() (weights_only=False), __init__ is not called —
        pickle restores __dict__ directly — so the Triton function reference is
        preserved even when no GPU is available. This method applies the same fallback
        after loading.

        GraphTransformerConv has no trainable parameters, so the swap is safe.
        """
        try:
            from anemoi.models.layers.block import GraphTransformerBaseBlock
            from anemoi.models.layers.conv import GraphTransformerConv
        except ImportError:
            LOGGER.warning(
                "Could not import anemoi.models layers to apply Triton->PyG CPU fallback."
            )
            return

        patched = 0
        for module in model.modules():
            if (
                isinstance(module, GraphTransformerBaseBlock)
                and module.graph_attention_backend == "triton"
            ):
                module.graph_attention_backend = "pyg"
                module.conv = GraphTransformerConv(
                    out_channels=module.out_channels_conv
                )
                patched += 1

        if patched:
            LOGGER.warning(
                "Checkpoint was saved with the Triton graph attention backend but no GPU "
                "is available. Fell back to the PyG backend for %d block(s).",
                patched,
            )

    @property
    def graph(self) -> HeteroData:
        """
        The graph used during model training.
        This is fetched from the model instance of the
        checkpoint.

        args:
            None

        return:
            HeteroData graph object
        """
        return (
            self._model_instance.graph_data
            if hasattr(self._model_instance, "graph_data")
            else None
        )

    # @property
    # def _get_copy_model_params(self) -> dict:
    #     """
    #     Caches the model's state in CPU memory.

    #     This cache includes only the model's weights
    #     and their corresponding layer names. It does not include the
    #     optimizer state. Note that this specifically refers to
    #     model.named_parameters() and not model.state_dict().

    #     A deep copy of the model state is performed
    #     to ensure the integrity of the cached data,
    #     even if the user decides to update
    #     the internal graph of the model later.

    #     Args:
    #         None
    #     Return
    #         torch dict containing the state of the model.
    #         Keys: name of the layer
    #         Value: The state for a given layer
    #     """

    #     _model_params = self._model_instance.named_parameters()
    #     return deepcopy(dict(_model_params))

    def update_graph(self, path: str | None = None) -> HeteroData:
        """
        Replaces existing graph object within model instance.
        The new graph is either provided as an torch file or
        generated on the fly with AnemoiGraphs (future implementation)

        args:
            Optional[str] path: path to graph

        return
            HeteroData graph object
        """

        external_graph = torch.load(path, map_location="cpu", weights_only=False)
        LOGGER.info("Loaded external graph from path")

        state_dict = deepcopy(self._model_instance.state_dict())

        self._model_instance.graph_data = external_graph
        self._model_instance.config = self.config

        self._model_instance._build_model()

        new_state_dict = self._model_instance.state_dict()

        for key in new_state_dict:
            if key in state_dict and state_dict[key].shape != new_state_dict[key].shape:
                # These are parameters like data_latlon, which are different now because of the graph
                pass
            else:
                # Overwrite with the old parameters
                new_state_dict[key] = state_dict[key]

        LOGGER.info(
            "Successfully built model with external graph and reassigning model weights!"
        )
        self._model_instance.load_state_dict(new_state_dict)
        return self._model_instance.graph_data

    @cached_property
    def data_indices(self) -> dict[str, IndexCollection]:
        _data_indices = self._model_instance.data_indices
        if isinstance(_data_indices, IndexCollection):  # Backwards compatibility
            return {"data": _data_indices}
        else:
            return _data_indices
