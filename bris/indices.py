from functools import cached_property

from checkpoint import Checkpoint


class DataIndices:
    def __init__(self, ckpt: Checkpoint):
        self.ckpt = ckpt

    @cached_property
    def _data_indices_input(self) -> dict:
        return self.ckpt.data_indices.input

    @cached_property
    def data_indices_full(self) -> dict:
        return self.ckpt._data_indices_input.full

    @cached_property
    def data_indices_forcing(self) -> dict:
        return self.ckpt._data_indices_input.forcing

    @cached_property
    def data_indices_diagnostic(self) -> dict:
        return self.ckpt._data_indices_input.diagnostic

    @cached_property
    def data_indices_prognostic(self) -> dict:
        return self.ckpt._data_indices_input.prognostic


class ModelIndices:
    def __init__(self, ckpt: Checkpoint):
        self.ckpt = ckpt

    @cached_property
    def _model_indices_model(self) -> dict:
        return self.ckpt.model_indices.input

    @cached_property
    def model_indices_full(self) -> dict:
        return self.ckpt._model_indices_input.full

    @cached_property
    def model_indices_diagnostic(self) -> dict:
        return self.ckpt._model_indices_input.diagnostic

    @cached_property
    def model_indices_prognostic(self) -> dict:
        return self.ckpt._model_indices_input.prognostic

    @cached_property
    def model_indices_forcing(self) -> dict:
        return self.ckpt._model_indices_input.forcing

    @cached_property
    def model_indices_diagnostic(self) -> dict:
        return self.ckpt._model_indices_input.diagnostic

    @cached_property
    def model_indices_prognostic(self) -> dict:
        return self.ckpt._model_indices_input.prognostic
