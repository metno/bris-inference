from abc import abstractmethod
from functools import cached_property
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from pyshtools.expand import SHGLQ, SHExpandGLQ

from bris.outputs import Output
from bris import projections, utils
from bris.conventions import cf
from bris.predict_metadata import PredictMetadata
from bris.outputs.intermediate import IntermediateSpatial
from bris.conventions import cf

class Spatial(Output):
    """Metrics that require spatial averaging.

    Power spectrum (wavelet spectrum)
    Sharpness https://github.com/ai2es/sharpness/blob/main/src/sharpness/metrics.py
    """
    # Input: Full prediction of one ensemble member
    # Output: Spatial average metric as a function of time, leadtime, and ens member?

    def __init__(
        self, 
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str = None,
    ):
        extra_variables = []
        if variable not in predict_metadata.variables:
            extra_variables += [variable]

        super().__init__(predict_metadata, extra_variables)
        shape = self.get_metric_shape()
        self.intermediate = IntermediateSpatial(
            predict_metadata, workdir, shape, extra_variables=extra_variables
        )
        self.variable = variable
        self.metric_name = self.get_metric_name()
        self.metric_shape = self.get_metric_shape()
        self.filename = filename


    @abstractmethod
    def calculate_metric(self, prediction: np.ndarray) -> np.ndarray: ...
    """Calculate the metric from a single ensemble member

    Args:
        prediction: np.ndarray with shape (leadtime, location, variable)

    Returns:
        metric: np.ndarray with dimensions given by get_metric_shape()
    """

    @abstractmethod
    def get_metric_shape(self, **kwargs) -> tuple: ...
    """Shape of the metric output from a single leadtime and ensemble member"""

    @abstractmethod
    def get_metric_name(self) -> str: ...
    """Name of the metric variable in the output dataset"""

    @abstractmethod
    def get_extra_dimensions(self) -> dict: ...
    """Returns {name: values} of the metric specific dimensions"""

    def _add_forecast(
        self, 
        times: list,
        ensemble_member: int, 
        pred: np.ndarray
    ) -> None:
        #Called by Output.add_forecast
        """Registers a forecast from a single ensemble member in the output"""
        
        metric = self.calculate_metric(pred)
        self.intermediate._add_forecast(times, ensemble_member, metric)

    def finalize(self) -> None:
        """Writes output to file"""
        coords = {}
        coords["time"] = (["time"], [], cf.get_attributes("time"))
        coords["leadtime"] = (
            ["leadtime"],
            self.intermediate.pm.leadtimes.astype(np.float32) / 3600,
            {"units": "hour"}
        )
        for name, values in self.get_extra_dimensions().items():
            coords[name] = ([name], values, cf.get_attributes(name))
        if self.pm.num_members > 1:
            coords["ensemble_member"] = (
                ["ensemble_member"],
                np.arange(self.pm.num_members),
            )

        self.ds = xr.Dataset(coords=coords)
        frts = self.intermediate.get_forecast_reference_times()
        self.ds["time"] = utils.datetime_to_unixtime(frts).astype(np.double)

        data_shape = (len(frts),) + (self.intermediate.pm.num_leadtimes,) + self.metric_shape
        dims = ["time", "leadtime"] + list(self.get_extra_dimensions().keys())
        if self.pm.num_members > 1:
            data_shape += (self.pm.num_members,)
            dims += ["ensemble_member"]

        data = np.full(data_shape, np.nan, dtype=np.float32)

        for i, frt in enumerate(frts):
            curr = self.intermediate.get_forecast(frt)
            data[i,...] = curr

        self.ds[self.metric_name] = (dims, data)

        utils.create_directory(self.filename)
        self.ds.to_netcdf(self.filename, mode="w", engine="netcdf4")

    def get_projected_coords(self, proj4_str) -> tuple:
        x, y = projections.get_xy_1D(self.pm.lats, self.pm.lons, proj4_str)
        return x, y


class SHPowerSpectrum(Spatial):
    """Calculates the spherical harmonic power spectrum of a variable"""

    def __init__(
        self, 
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
    ):
        super().__init__(predict_metadata, workdir, filename, variable)
        assert not self.pm.is_gridded, "SHPowerSpectrum is meant to be used for global ungridded data"

    def get_metric_name(self) -> str:
        return f"sh_power_spectrum_{self.variable}"

    def get_metric_shape(self) -> tuple:
        x_regular, y_regular = self.get_regular_projected_coords
        nl = ( len(x_regular) + 1 ) / 2
        return (nl,)

    def get_extra_dimensions(self) -> dict:
        nl = self.metric_shape[0]
        return {"wavenumber": np.arange(1, nl + 1)}

    @cached_property
    def get_regular_projected_coords(self) -> tuple:
        """Get regular x and y coordinates in equidistant cylindrical projection
        
        Grid spacing is determined by the minimum spacing in the original coordinates
        """
        proj4_str = projections.get_proj4_str("equidistant_cylindrical")
        x, y = self.get_projected_coords(proj4_str)

        delta_y = np.min(np.diff(y))
        min_delta_y = np.min(np.abs(delta_y[delta_y != 0]))

        n_y = int(np.floor(abs(y.max() - y.min()) / min_delta_y))
        n_x = (n_y - 1) * 2 + 1 
        y_regular = np.linspace(y.min(), y.max(), n_y)
        x_regular = np.linspace(x.min(), x.max(), n_x)
        return x_regular, y_regular

    def calculate_metric(self, pred: np.ndarray) -> np.ndarray:
        """Calculate the wavenumber spherical harmonic power spectrum of the variable"""

        var_index = self.pm.variables.index(self.variable)
        leadtimes = self.pm.num_leadtimes
        metric = np.full((leadtimes,) + self.metric_shape, np.nan, dtype=np.float32)

        proj4_str = projections.get_proj4_str("equidistant_cylindrical") #TODO: fix so this isn't called both in get_regular_projected_coords and here
        x, y = self.get_projected_coords(proj4_str) 
        x_reg, y_reg = self.get_regular_projected_coords
        xx_reg, yy_reg = np.meshgrid(x_reg, y_reg)

        for lt in range(leadtimes):
            field = pred[lt, :, var_index]
            
            #Consider using linear / cubic - requires some tuning of the regular grid since we get NaNs at the border. 
            field_reg = griddata(
                (x, y), field, (xx_reg, yy_reg), method="nearest", fill_value=np.nan
            )
            nanmask = np.isnan(field_reg)
            if nanmask.any():
                print("Warning: SHPowerSpectrum - missing values in regular grid, replacing with zeros")
                field_reg[nanmask] = 0.0
            
            # Compute spherical harmonic coefficients and power spectrum
            lmax = len(x_reg) - 1
            zero_w = SHGLQ(lmax)
            coeffs_field = SHExpandGLQ(field_reg, w=zero_w[1], zero=zero_w[0])
            coeff_amp = coeffs_field[0,:,:] ** 2 + coeffs_field[1,:,:] ** 2
            power_spectrum = np.sum(coeff_amp, axis=0)

            metric[lt, :] = power_spectrum

        return metric
                



            








        