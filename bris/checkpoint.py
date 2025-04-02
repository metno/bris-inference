import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Optional

import torch
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.checkpoints import load_metadata
from anemoi.utils.config import DotDict
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


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
    """This class makes accessible various information stored in Anemoi checkpoints"""

    def __init__(self, path: str, graph: Optional[str] = None):
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
        return inst

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

    @property
    def _get_copy_model_params(self) -> dict:
        """
        Caches the model's state in CPU memory.

        This cache includes only the model's weights
        and their corresponding layer names. It does not include the
        optimizer state. Note that this specifically refers to
        model.named_parameters() and not model.state_dict().

        A deep copy of the model state is performed
        to ensure the integrity of the cached data,
        even if the user decides to update
        the internal graph of the model later.

        Args:
            None
        Return
            torch dict containing the state of the model.
            Keys: name of the layer
            Value: The state for a given layer
        """

        _model_params = self._model_instance.named_parameters()
        return deepcopy(dict(_model_params))

    def update_graph(self, path: Optional[str] = None) -> HeteroData:
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

    @property
    def name_to_index(self) -> tuple[dict[str, int], ...]:
        """
        Mapping between name and their corresponding variable index.
        Returns a tuple. If the model is a multiencoder-decoder model
        the tuple will contain two dicts, one for each decoder. If not
        the tuple will contain a single dict.
        """
        _data_indices = self._model_instance.data_indices
        if isinstance(_data_indices, (tuple, list)) and len(_data_indices) >= 2:
            return tuple(
                _data_indices[k].name_to_index for k in range(len(_data_indices))
            )

        return (_data_indices.name_to_index,)

    @property
    def index_to_name(self) -> tuple[dict[int, str], ...]:
        """
        Mapping between index and their corresponding variable name.
        Returns a tuple. If the model is a multiencoder-decoder model
        the tuple will contain two dicts, one for each decoder. If not
        the tuple will contain a single dict.
        """
        _data_indices = self._model_instance.data_indices
        if isinstance(_data_indices, (tuple, list)) and len(_data_indices) >= 2:
            return tuple(
                {
                    index: var
                    for (var, index) in self.name_to_index[decoder_index].items()
                }
                for decoder_index in range(len(self.name_to_index))
            )
        return ({index: name for name, index in _data_indices.name_to_index.items()},)

    def _make_indices_mapping(self, indices_from, indices_to):
        """
        Creates a mapping for a given model and data output
        or model input and data input, and vice versa.

        args:
            indices_from (dict)
            indices_to (dict)
        return
            a mapping between indices_from and indices_to
        """
        assert len(indices_from) == len(indices_to)
        return dict(zip(indices_from, indices_to))

    @property
    def model_output_index_to_name(self) -> tuple[dict[int, str], ...]:
        """
        A mapping from model output to data output. This
        dict returns index and name pairs according to model.output.full to
        data.output.full

        args:
            None
        return:
            tuple of dicts where the tuple index represents decoder index. Each
            tuple dicts contains a dict with a mapping between model.output.full and
            data.output.full
        Example:
        -> if the parameter skt has index 27 in name_to_index or data.output.full it will
        have index 17 in model.output.full. This dict will yield that index 17 is skt
        """
        if (
            isinstance(self._metadata.data_indices, (tuple, list))
            and len(self._metadata.data_indices) >= 2
        ):
            mapping = {
                grid_index: self._make_indices_mapping(
                    self._metadata.data_indices[grid_index].model.output.full,
                    self._metadata.data_indices[grid_index].data.output.full,
                )
                for grid_index in range(len(self._metadata.data_indices))
            }
            return tuple(
                {name: self.index_to_name[k][index] for (name, index) in v.items()}
                for (k, v) in mapping.items()
            )

        mapping = self._make_indices_mapping(
            self._metadata.data_indices.model.output.full,
            self._metadata.data_indices.data.output.full,
        )
        return ({k: self._metadata.dataset.variables[v] for k, v in mapping.items()},)

    @property
    def model_output_name_to_index(self) -> tuple[dict[str, int], ...]:
        """
        A mapping from model output to data output. This
        dict returns name and index pairs according to model.output.full to
        data.output.full

        args:
            None
        return:
            Identical tuple[dict] from model_output_index_to_name but key
            value pairs are switched.
        """
        if (
            isinstance(self._metadata.data_indices, (tuple, list))
            and len(self._metadata.data_indices) >= 2
        ):
            return tuple(
                {name: index for (index, name) in v.items()}
                for (k, v) in enumerate(self.model_output_index_to_name)
            )

        return (
            {name: index for index, name in self.model_output_index_to_name[0].items()},
        )

    @cached_property
    def data_indices(self) -> tuple[IndexCollection, ...]:
        """
        Wrapper for model.data_indices. Returns a tuple of dict and None or two dicts.
        """

        # If Multiencdec checkpoint
        if isinstance(self._model_instance.data_indices, (tuple, list)):
            return tuple(self._model_instance.data_indices)

        # If simple checkpoint
        return (self._model_instance.data_indices,)
