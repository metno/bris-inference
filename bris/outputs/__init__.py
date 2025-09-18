import copy
import time
from typing import Optional

import numpy as np

from bris import sources
from bris.predict_metadata import PredictMetadata
from bris.utils import LOGGER
from bris.utils import datetime_to_unixtime, expr_to_var, safe_eval_expr


def instantiate(name: str, predict_metadata: PredictMetadata, workdir: str, init_args):
    """Creates an object of type name with config

    Args:
        predict_metadata: Contains metadata about the bathc the output will recive
        init_args: Arguments to pass to Output constructor
    """
    if name == "verif":
        # Parse obs sources
        obs_sources = []

        # Convert to dict, since overriding obs_sources doesn't seem to work with OmegaConf
        args = {**init_args}
        for source in init_args["obs_sources"]:
            for source_name, opts in source.items():
                obs_sources += [sources.instantiate(source_name, opts)]
        args["obs_sources"] = obs_sources
        return Verif(predict_metadata, workdir, **args)

    if name == "netcdf":
        return Netcdf(predict_metadata, workdir, **init_args)

    if name == "grib":
        return Grib(predict_metadata, workdir, **init_args)

    if name == "powerspectrum_global":
        return SHPowerSpectrum(predict_metadata, workdir, **init_args)

    if name == "powerspectrum_gridded":
        return DCTPowerSpectrum(predict_metadata, workdir, **init_args)

    raise ValueError(f"Invalid output: {name}")


def get_required_variables(name, init_args):
    """What variables does this output require? Return None if it will process all variables
    provided
    """

    if name in ["netcdf", "grib"]:
        if "variables" in init_args:
            variables = init_args["variables"]
            if "extra_variables" in init_args:
                for var_name in init_args["extra_variables"]:
                    extr_vars, _, _ = expr_to_var(var_name)
                    variables.extend(extr_vars)
            variables = sorted(list(set(variables)))
            return variables
        return [None]

    if name in ["verif", "powerspectrum_gridded", "powerspectrum_global"]:
        extr_vars, _, _ = expr_to_var(init_args["variable"])
        return extr_vars

    raise ValueError(f"Invalid output: {name}")


class Output:
    """This class writes output for a specific part of the domain"""

    def __init__(
        self, predict_metadata: PredictMetadata, extra_variables: Optional[list] = None
    ):
        """Creates an object of type name with config

        Args:
            predict_metadata: Contains metadata about the batch the output will recieve
        """
        if extra_variables is None:
            extra_variables = []

        predict_metadata = copy.deepcopy(predict_metadata)
        for name in extra_variables:
            if name not in predict_metadata.variables:
                predict_metadata.variables += [name]

        self.pm = predict_metadata
        self.extra_variables = extra_variables

    def add_forecast(self, times: list, ensemble_member: int, pred: np.ndarray):
        """Registers a forecast from a single ensemble member in the output

        Args:
            times: List of np.datetime64 objects
            ensemble_member: Which ensemble member is this?
            pred: 3D numpy array with dimensions (leadtime, location, variable)
        """

        # Append extra variables to prediction
        for name in self.extra_variables:
            if name not in self.pm.variables:
                clean_name = name.replace("[", "").replace("]", "").replace("*", "")
                self.pm.variables.append(clean_name)

        # only do this once. For multiple members, intermediate calls this several times
        t0 = time.perf_counter()
        if pred.shape[2] != len(self.pm.variables):
            # Append extra variables to prediction
            extra_pred = []
            for name in self.extra_variables:
                extr_vars, name, success = expr_to_var(name)
                assert success, "Variables could not be extracted from expression"

                variables_dict = {}
                for v in extr_vars:
                    idx = self.pm.variables.index(v)
                    variables_dict[v] = pred[..., idx]
                extra_pred += [safe_eval_expr(name, variables_dict)[..., None]]

            pred = np.concatenate([pred] + extra_pred, axis=2)
        LOGGER.debug(
            f"outputs.add_forecast Calculate ws in {time.perf_counter() - t0:.1f}s"
        )

        assert pred.shape[0] == self.pm.num_leadtimes
        assert pred.shape[1] == len(self.pm.lats)
        assert pred.shape[2] == len(self.pm.variables), (
            pred.shape[2],
            len(self.pm.variables),
        )
        assert ensemble_member >= 0
        assert ensemble_member < self.pm.num_members

        t1 = time.perf_counter()
        self._add_forecast(times, ensemble_member, pred)
        LOGGER.debug(
            f"outputs.add_forecast called _add_forecast in {time.perf_counter() - t1:.1f}s"
        )

    def _add_forecast(self, times: list, ensemble_member: int, pred: np.ndarray):
        """Subclasses should implement this"""
        raise NotImplementedError()

    def finalize(self):
        """Finalizes the output. This gets called after all add_forecast calls are done. Subclasses
        can override this, if necessary."""

    def reshape_pred(self, pred):
        """Reshape predictor matrix from points to x,y based on field_shape

        Args:
            pred: 3D numpy array with dimensions (leadtime, location, variable)

        Returns:
            4D numy array with dimensions (leadtime, y, x, variable)
        """
        assert self.pm.field_shape is not None

        T, _L, V = pred.shape
        shape = [T, self.pm.field_shape[0], self.pm.field_shape[1], V]
        pred = np.reshape(pred, shape)
        return pred

    def flatten_pred(self, pred):
        """Reshape predictor matrix on 2D points back to the original 1D set of points

        Args:
            pred: 4D numy array with dimensions (leadtime, y, x, variable)

        Returns:
            3D numpy array with dimensions (leadtime, location, variable)
        """
        assert len(pred.shape) == 4
        T, _Y, _X, V = pred.shape

        shape = [T, self.pm.field_shape[0] * self.pm.field_shape[1], V]
        pred = np.reshape(pred, shape)
        return pred


from .grib import Grib
from .intermediate import Intermediate
from .netcdf import Netcdf
from .spatial import DCTPowerSpectrum, SHPowerSpectrum
from .verif import Verif
