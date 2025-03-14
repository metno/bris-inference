import torch

from bris.checkpoint import Checkpoint


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
        self.model = self._model_instance.model

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write logic
        return self.model.encoder

    def processor(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write logic
        return self.model.processor

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write logic
        return self.model.decoder
