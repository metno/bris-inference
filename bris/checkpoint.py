import os
import logging
from functools import cached_property
from typing import Any, Optional

import torch
from anemoi.utils.config import DotDict
from anemoi.utils.checkpoints import load_metadata
from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class Checkpoint:
    """This class makes accessible various information stored in Anemoi checkpoints"""

    AIFS_BASE_SEED = None
    UPDATE_GRAPH = False

    def __init__(self, path: str):
        assert os.path.exists(path), f"The given checkpoint does not exist!"

        self.path = path

    @cached_property
    def _metadata(self) -> dict:
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
        except Exception as e:
            LOGGER.warning(
                "Could not load and peek into the checkpoint metadata. Raising an expection"
            )
            raise e

    @cached_property
    def config(self) -> dict:
        """
        The configuriation used during model
        training.
        """
        return self._metadata.config

    @cached_property
    def version(self) -> str:
        """
        Model version
        """
        return self._metadata.version

    @cached_property
    def multistep(self) -> int:
        """
        Fetches multistep from metadata
        """
        if hasattr(self._metadata.config.training, "multistep"):
            return self._metadata.config.training.multistep
        elif hasattr(self._metadata.config.training, "multistep_input"):
            return self._metadata.config.training.multistep_input
        else:
            raise RuntimeError("Cannot find multistep")

    @cached_property
    def _model_instance(self) -> Any:
        """
        Loads a given model instance. This instance
        includes both the model interface and its
        corresponding model weights.
        """
        try:
            inst = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as e:
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

        # TODO: add check which checks the keys within the graph
        # the model weights have names tied to f.ex stretched grid or grid.
        # if the model is trained with keys named grid and we force new graph with keys
        # stretched grid, the model instance will complain
        # (not 100% sure but i think i have experienced this)

        if self.UPDATE_GRAPH:
            raise RuntimeError(
                "Graph has already been updated. Mutliple updates is not allowed"
            )
        else:
            if path:
                assert os.path.exists(
                    path
                ), f"Cannot locate graph file. Got path: {path}"
                external_graph = torch.load(path, map_location="cpu")
                LOGGER.info("Loaded external graph from path")
                try:
                    self._model_instance.graph_data = external_graph
                    self.UPDATE_GRAPH = True
                    LOGGER.info(
                        "Successfully changed internal graph with external graph!"
                    )
                    return self._model_instance.graph_data

                except Exception as e:
                    raise e  # RuntimeError("Failed to update the graph.") from e
            else:
                # future implementation
                # _graph = anemoi.graphs.create() <-- skeleton
                # self._model_instance.graph_data = _graph <- update graph obj within inst
                # return _graph <- return graph
                raise NotImplementedError

    @cached_property
    def base_seed(self) -> int:
        """
        Fetchs the original base seed used during training.
        If not

        """
        os.environ["AIFS_BASE_SEED"] = f"{self._metadata.seed}"

        self.AIFS_BASE_SEED = os.get(os.environ("AIFS_BASE_SEED"), None)
        if self.AIFS_BASE_SEED:
            # LOGGER.info(f"AIFS_BASE_SEED set to: {self.AIFS_BASE_SEED}")
            return self.AIFS_BASE_SEED

        self.AIFS_BASE_SEED = 1234
        # LOGGER.info(f"Could not find AIFS_BASE_SEED. Setting to a random number {self.AIFS_BASE_SEED}")
        return self.AIFS_BASE_SEED

    @cached_property
    def name_to_index(self) -> dict:
        """
        Mapping between name and their corresponding variable index
        """
        _data_indices = self._model_instance.data_indices
        if isinstance(_data_indices, (tuple, list)) and len(_data_indices) >= 2:
            return {
                f"grid{k}": _data_indices[k].name_to_index
                for k in range(len(_data_indices))
            }

        return (
            self._model_instance.data_indices.name_to_index
        )  # {name : index for index,name in enumerate(self._metadata.dataset.variables)}

    @cached_property
    def index_to_name(self) -> dict:
        """
        Mapping between index and their corresponding variable name
        """
        _data_indices = self._model_instance.data_indices
        if isinstance(_data_indices, (tuple, list)) and len(_data_indices) >= 2:
            return {
                k: {index: var for var, index in self.name_to_index[k].items()}
                for k in self.name_to_index
            }

        return  # {index : name for index,name in enumerate(self._metadata.dataset.variables)}

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
        return {i: j for i, j in zip(indices_from, indices_to)}

    @cached_property
    def model_output_index_to_name(self) -> dict:
        """
        A mapping from model output to data output. This
        dict returns index and name pairs according to model.output.full to
        data.output.full

        args:
            None
        return:
            dict index and name pairs wrt to model.output.full to
            data.output.full
        """
        if (
            isinstance(self._metadata.data_indices, (tuple, list))
            and len(self._metadata.data_indices) >= 2
        ):
            mapping = {
                f"grid{grid_index}": self._make_indices_mapping(
                    self._metadata.data_indices[grid_index].model.output.full,
                    self._metadata.data_indices[grid_index].data.output.full,
                )
                for grid_index in range(len(self._metadata.data_indices))
            }

            return {
                k: {name: self.index_to_name[k][index] for name, index in v.items()}
                for k, v in mapping.items()
            }

        mapping = self._make_indices_mapping(
            self._metadata.data_indices.model.output.full,
            self._metadata.data_indices.data.output.full,
        )
        return {k: self._metadata.dataset.variables[v] for k, v in mapping.items()}

    @cached_property
    def model_output_name_to_index(self) -> dict:
        """
        A mapping from model output to data output. This
        dict returns name and index pairs according to model.output.full to
        data.output.full

        args:
            None
        return:
            dict name and index pairs wrt to model.output.full to
            data.output.full
        """
        if (
            isinstance(self._metadata.data_indices, (tuple, list))
            and len(self._metadata.data_indices) >= 2
        ):
            return {
                k: {name: index for index, name in v.items()}
                for k, v in self.model_output_index_to_name.items()
            }

        return {name: index for index, name in self.model_output_index_to_name.items()}
