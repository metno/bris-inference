import datetime

from abc import abstractmethod
from functools import cached_property
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.fft import dctn
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

        frts = self.intermediate.get_forecast_reference_times()
        times_unix = utils.datetime_to_unixtime(frts).astype(np.double)
        coords["time"] = (["time"], times_unix, cf.get_attributes("time"))
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

        data_shape = (len(frts),) + (self.intermediate.pm.num_leadtimes,) + self.metric_shape + (self.pm.num_members,)
        dims = ["time", "leadtime"] + list(self.get_extra_dimensions().keys())

        data = np.full(data_shape, np.nan, dtype=np.float32)

        for i, frt in enumerate(frts):
            curr = self.intermediate.get_forecast(frt)
            data[i,...] = curr

        if self.pm.num_members > 1:
            dims += ["ensemble_member"]
        else:
            data = data.squeeze(-1)

        self.ds[self.metric_name] = (dims, data)

        datestr = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S +00:00"
        )
        self.ds.attrs["history"] = f"{datestr} Created by bris-inference"
        self.ds.attrs["Convensions"] = "CF-1.6"

        utils.create_directory(self.filename)
        self.ds.to_netcdf(self.filename, mode="w", engine="netcdf4")

    @cached_property
    def get_latlons(self) -> tuple:
        return self.pm.lats, self.pm.lons


class SHPowerSpectrum(Spatial):
    """Calculates the isotropic power spectrum of a variables for global grids using the Spherical Harmonics fourier transform"""

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
        return (lats_reg_grid.shape[0] - 1,)

    def get_extra_dimensions(self) -> dict:
        l_max = self.metric_shape[0]
        return {"l": np.arange(1, l_max + 1)}
    
    @cached_property
    def get_grid_reg_latlons(self) -> tuple:
        "Create a regular lat-lon grid based on the data resolution or input resolution if delta_degrees is given"
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
        return lons_reg_grid, lats_reg_grid
     

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
            power_spectrum = np.sum(coeff_amp, axis=1)

            metric[lt, ...] = power_spectrum[1:]

        return metric


class PowerSpectrum(Spatial):
    """Calculates the isotropic power spectrum of a variables for regular projected grids using the Discrete Cosine Transform"""

    def __init__(
        self,
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
        resolution: float = None,
        proj4_str: str = None,
        domain_name: str = None,
        n_bins: int = None,
    ):
        self.resolution = resolution
        self.n_bins = n_bins
        if domain_name is not None:
            self.proj4_str = projections.get_proj4_str(domain_name)
        else:
            self.proj4_str = proj4_str
        assert self.proj4_str is not None, "Either domain_name or proj4_str must be provided"
        super().__init__(predict_metadata, workdir, filename, variable)
        assert self.pm.is_gridded, "PowerSpectrum is meant to be used for gridded data"


    def get_metric_name(self) -> str:
        return f"power_spectrum_{self.variable}"
    
    def get_metric_shape(self) -> tuple:
        _, k_bins, _ = self.get_bins
        return (k_bins.shape[0],)

    @cached_property
    def get_bins(self):
        """ Calculates wavenumbers, bins and bin-edges used in the CDT calculation. """
        nx, ny = self.pm.field_shape
        lats, lons = self.get_latlons
        x, y = projections.get_xy(
            lats.reshape(nx, ny),
            lons.reshape(nx, ny),
            self.proj4_str
        )

        dx = np.mean(np.diff(x))
        dy = np.mean(np.diff(y))
        assert np.allclose(np.diff(x), dx, atol=1.0), "Non-uniform grid spacing in x-direction"
        assert np.allclose(np.diff(y), dy, atol=1.0), "Non-uniform grid spacing in y-direction"

        kx = np.pi * np.arange(nx) / (nx*dx)
        ky = np.pi * np.arange(ny) / (ny*dy)

        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k = np.sqrt(KX**2 + KY**2)
        k_max = k.max()
        if self.n_bins is not None:
            n_bins = self.n_bins
        else:
            n_bins = min(nx, ny) // 2
        
        k_edges = np.linspace(0.0, k_max, n_bins + 1)
        k_bins = 0.5 * (k_edges[1:] + k_edges[:-1])
        k = k.flatten()
        return k_edges, k_bins, k

    def get_extra_dimensions(self) -> dict:
        _, k_bin, _ = self.get_bins
        return {"k": k_bin}

    def calculate_metric(self, pred: np.ndarray) -> np.ndarray:
        "Calculate the isotropic power spectrum"

        var_index = self.pm.variables.index(self.variable)
        leadtimes = self.pm.num_leadtimes
        metric = np.full((leadtimes,) + self.metric_shape, np.nan, dtype=np.float32)
        nx, ny = self.pm.field_shape

        k_edges, k_bins, k = self.get_bins
        n_bins = k_bins.shape[0]
        digitized = np.digitize(k.flatten(), k_edges)

        for lt in range(leadtimes):
            field = pred[lt, :, var_index].reshape(nx, ny)

            P = np.abs(dctn(field, type=2, norm='ortho'))**2

            E_k = np.array([P.flatten()[digitized==i].mean() if np.any(digitized==i) else 0 for i in range(1, n_bins + 1)])
            metric[lt, :] = E_k

        return metric







        

                



            








        