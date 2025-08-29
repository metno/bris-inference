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

    @cached_property
    def get_latlons(self) -> tuple:
        return self.pm.lats, self.pm.lons


class SHPowerSpectrum(Spatial):
    """Calculates the spherical harmonic power spectrum of a variable"""

    def __init__(
        self, 
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
        delta_degrees: float = None, 
    ):  
        self.delta_degrees = delta_degrees
        super().__init__(predict_metadata, workdir, filename, variable)
        assert not self.pm.is_gridded, "SHPowerSpectrum is meant to be used for global ungridded data"


    def get_metric_name(self) -> str:
        return f"sh_power_spectrum_{self.variable}"

    def get_metric_shape(self) -> tuple:
        lats_reg_grid, _ = self.get_grid_reg_latlons
        return (lats_reg_grid.shape[0],)

    def get_extra_dimensions(self) -> dict:
        nl = self.metric_shape[0]
        return {"wavenumber": np.arange(1, nl + 1)}
    
    @cached_property
    def get_grid_reg_latlons(self) -> tuple:
        lats = self.pm.lats
        lons = self.pm.lons
        
        if self.delta_degrees is not None:
            delta = self.delta_degrees
        else:
            lat_min = abs(np.diff(lats))
            delta = np.min(abs(lat_min[lat_min != 0]))

        n_lats = int(np.floor((lats.max() - lats.min()) / delta))
        n_lons = (n_lats - 1) * 2 + 1
        lats_regular = np.linspace(lats.min(), lats.max(), n_lats)
        lons_regular = np.linspace(lons.min(), lons.max(), n_lons)
        lons_reg_grid, lats_reg_grid = np.meshgrid(lons_regular, lats_regular)
        return lats_reg_grid, lons_reg_grid

    def calculate_metric(self, pred: np.ndarray) -> np.ndarray:
        """Calculate the wavenumber spherical harmonic power spectrum of the variable"""

        var_index = self.pm.variables.index(self.variable)
        leadtimes = self.pm.num_leadtimes
        metric = np.full((leadtimes,) + self.metric_shape, np.nan, dtype=np.float32)

        lats, lons = self.get_latlons
        lons_reg_grid, lats_reg_grid = self.get_grid_reg_latlons
        for lt in range(leadtimes):
            field = pred[lt, :, var_index]
            
            field_reg = griddata(
                (lons, lats), field, (lons_reg_grid, lats_reg_grid), method="nearest", fill_value=np.nan
            )
            nanmask = np.isnan(field_reg)
            if nanmask.any():
                print("Warning: SHPowerSpectrum - missing values in regular grid, replacing with zeros")
                field_reg[nanmask] = 0.0
            
            # Compute spherical harmonic coefficients and power spectrum
            lmax = lats_reg_grid.shape[0] - 1
            zero_w = SHGLQ(lmax)
            coeffs_field = SHExpandGLQ(field_reg, w=zero_w[1], zero=zero_w[0])
            coeff_amp = coeffs_field[0,:,:] ** 2 + coeffs_field[1,:,:] ** 2
            power_spectrum = np.sum(coeff_amp, axis=0)

            metric[lt, :] = power_spectrum

        return metric

                



            








        