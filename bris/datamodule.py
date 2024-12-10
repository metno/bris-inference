from anemoi.training.data.datamodule import AnemoiDatasetsDataModule


class DataModule(AnemoiDatasetsDataModule):
    def __init__(self, graph, config):
        """DataModule instance and DataSets.

        It reads the spatial indices from the graph object. These spatial indices
        correspond to the order (and data points) that will be read from each data
        source. If not specified, the dataset will be read full in the order that it is
        stored.
        """
        self.graph = graph
        self.config = config

        spatial_mask = {}
        for mesh_name, mesh in self.graph.items():
            if (
                isinstance(mesh_name, str)
                and mesh_name != self.config.graphs.hidden_mesh.name
            ):
                spatial_mask[mesh_name] = mesh.get("dataset_idx", None)

        # Note: Only 1 dataset supported for now. If more datasets are specified, the first one will be used.
        super.__init__(
            self.config,
            spatial_index=spatial_mask[self.config.graphs.encoders[0]["src_mesh"]],
            predict_datareader=self.data_reader,
        )
        self.config.data.num_features = len(datamodule.ds_train.data.variables)

    @property
    def grids(self):
        """Returns a diction of grids and their grid point ranges"""
        return {"global": {"start": 1, "end": 2}, "meps": {"start": 3, "end": 4}}
