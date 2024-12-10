import os
from functools import cached_property
from typing import Any

from anemoi.utils.config import DotDict
from anemoi.utils.checkpoints import load_metadata

from .forecaster import BrisForecaster
from .datamodule import DataModule
from .indices import DataIndices
from .indices import ModelIndices


class Checkpoint:
    """This class makes accessible various information stored in Anemoi checkpoints"""
    AIFS_BASE_SEED = None

    def __init__(self, path: str):
        assert os.path.exists(path), f"The given checkpoint does not exist!"

        self.path = path

    @cached_property
    def _metadata(self) -> dict:
        try:
            return DotDict(load_metadata(self.path))
        except Exception as e:
            LOGGER.warning(
                "Could not load and peek into the checkpoint metadata. Raising an expection"
            )
            raise e

    @cached_property
    def base_seed(self) -> int:
        os.environ["AIFS_BASE_SEED"] = f"{self._metadata.seed}"

        self.AIFS_BASE_SEED = os.get(os.environ("AIFS_BASE_SEED"), None)
        if self.AIFS_BASE_SEED:
            # LOGGER.info(f"AIFS_BASE_SEED set to: {self.AIFS_BASE_SEED}")
            return self.AIFS_BASE_SEED

        self.AIFS_BASE_SEED = 1234
        # LOGGER.info(f"Could not find AIFS_BASE_SEED. Setting to a random number {self.AIFS_BASE_SEED}")
        return self.AIFS_BASE_SEED

    @cached_property
    def graph(self) -> dict:
        return {}

    @cached_property
    def config(self) -> dict:
        return {}

    def update_graph(self, dataset: DataModule):
        """Call this to change the graph"""
        pass

    def write(self, filename: str):
        """Writes the checkpoint to file (with updated graph)"""
        pass

    # Can't cache this, if we can update graph
    @property
    def model(self) -> BrisForecaster:
        """Provide the model instance."""
        select_vars = self.inf_conf.select.vars
        select_indices = np.array(
            [self.indices["name_to_index"][var] for var in select_vars]
        )
        select_indices_truth = np.array(
            [self.data_reader.name_to_index[var] for var in select_vars]
        )

        kwargs = {
            "statistics": self.datamodule.statistics,
            "data_indices": self.datamodule.data_indices,
            "graph_data": self.graph,
            "metadata": self.metadata,
            "config": self.config,
            "select_indices": select_indices,
            "select_indices_truth": select_indices_truth,
            "grid_points_range": (
                self.dset_properties[self.verif_grids[0]]["grid_points_range"]
                if len(self.verif_grids) == 1
                else (0, sum(self.data_reader.grids))
            ),
        }
        # LOGGER.info("Restoring only model weights from %s", self.last_checkpoint)

        # return GraphForecaster.load_from_checkpoint(self.last_checkpoint, **kwargs)
        model = GraphPredictor(**kwargs)
        ckpt = torch.load(self.last_checkpoint, "cpu")
        for name, param in model.named_parameters():
            param.data = ckpt["state_dict"][name].data
        return model

    @cached_property
    def data_indices(self) -> DataIndices:
        # TODO:
        return self._metadata.data_indices.data

    @cached_property
    def model_indices(self) -> ModelIndices:
        # TODO:
        return self._metadata.data_indices.model

    @cached_property
    def num_gridpoints(self):
        return self._metadata.dataset.shape[-1]

    @cached_property
    def num_gridpoints_lam(self):
        return self._metadata.dataset.specific.forward.forward.forward.datasets[
            0
        ].shape[-1]

    @cached_property
    def num_gridpoints_global(self):
        return self._metadata.dataset.specific.forward.forward.forward.datasets[
            1
        ].shape[-1]

    @cached_property
    def num_features(self):
        return len(self.model_indices.input.full)

    @cached_property
    def config(self) -> dict:
        return self._metadata.config

    @cached_property
    def graph_config(self) -> dict:
        return self.config.graphs

    @cached_property
    def name_to_index(self) -> dict:
        return {
            name: index for index, name in enumerate(self._metadata.dataset.variables)
        }

    @cached_property
    def index_to_name(self) -> dict:
        return {
            index: name for index, name in enumerate(self._metadata.dataset.variables)
        }

    def _make_indices_mapping(self, indices_from, indices_to):
        assert len(indices_from) == len(indices_to)
        return {i: j for i, j in zip(indices_from, indices_to)}

    @cached_property
    def model_output_index_to_name(self) -> dict:
        """Return the mapping between output tensor index and variable name"""
        mapping = self._make_indices_mapping(
            self.model_indices.output.full,
            self.data_indices.output.full,
        )
        return {k: self._metadata.dataset.variables[v] for k, v in mapping.items()}

    @cached_property
    def model_output_name_to_index(self) -> dict:
        return {name: index for index, name in self.model_output_index_to_name.items()}

    @cached_property
    def lam_yx_dimensions(self) -> tuple:
        return self._metadata.dataset.specific.forward.forward.forward.datasets[
            0
        ].forward.forward.attrs.field_shape

    @cached_property
    def era_to_meps(self) -> tuple:
        return self._metadata.dataset.specific.forward.forward.forward.datasets[
            0
        ].forward.forward.attrs.era_to_meps_mapping

    @cached_property
    def LAM_latlon(self) -> tuple | list:
        return self._metadata.dataset.specific.forward.forward.forward.datasets[0]

    @cached_property
    def latitudes(self) -> Any:
        raise NotImplementedError
