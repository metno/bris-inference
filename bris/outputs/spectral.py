import numpy as np

from abc import abstractmethod

from bris.outputs import Output
from bris.predict_metadata import PredictMetadata
from bris.outputs.intermediate import IntermediateSpatial


class SpectralSkillSpread():
    """Calculates skill and spread on different length scales using the Discrete Cosine Transform (DCT)"""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
        obs_dataset: str,
        remove_intermediate: bool = True,
    ) -> None:
        extra_variables = []
        if variable not in predict_metadata.variables:
            extra_variables += [variable]
        
        self.pm = predict_metadata
        self.variable = variable
        self.filename = filename
        shape = (predict_metadata.num_points)
        self.intermediate = IntermediateSpatial(
            predict_metadata=predict_metadata,
            workdir=workdir,
            metric_shape = shape,
            extra_variables=extra_variables,
        )
        self.remove_intermediate = remove_intermediate
        

    @abstractmethod
    def transform(data: np.ndarray): ...

    def _add_forecast(
        self, times: list, ensemble_member: int, pred: np.ndarray
    ) -> None:
        if self.variable == "ws":
            Ix = self.pm.variables.index("10u")
            Iy = self.pm.variables.index("10v")
            pred = np.sqrt(pred[..., [Ix]] ** 2 + pred[..., [Iy]] ** 2)
        else:
            pred = pred[..., [self.pm.variables.index(self.variable)]]
        
        self.intermediate._add_forecast(times, ensemble_member, pred)

    def finalize(self) -> None:
        """ Calculate skill spread and write to file"""
        frts = self.intermediate.get_forecast_reference_times()
        for frt in frts:
            pred = np.zeros((self.pm.num_leadtimes, self.pm.num_points, self.pm.num_members))
            for member in range(self.pm.num_members):
                pred[..., member] = self.intermediate.get_forecast(frt, member)
            
            pred_transformed = self.transform(pred)


        