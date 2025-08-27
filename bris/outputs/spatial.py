from abc import abstractmethod
import numpy as np
import xarray as xr
import gridpp

from bris.outputs import Output
from bris import utils
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

    @abstractmethod
    def get_metric_shape(self) -> tuple: ...
    '''Shape of the metric output from a single leadtime and ensemble member'''

    @abstractmethod
    def get_metric_name(self) -> str: ...
    '''Name of the metric variable in the output dataset'''

    @abstractmethod
    def get_extra_dimensions(self) -> list: ...
    '''Returns {name: values} of the metric specific dimensions'''

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
    
    def get_lower(self, array):
        m = np.min(array)
        return np.floor(m / self.interp_res) * self.interp_res

    def get_upper(self, array):
        m = np.max(array)
        return np.ceil(m / self.interp_res) * self.interp_res

class PowerSpectrum(Spatial):
    """Calculates the power spectrum of a variable"""
    
    def __init__(
        self, 
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str,
        n_bins: int = 10,
        k_min: float = 1/128,
        k_max: float = 1/2,
        interp_res: float = None,
    ):
        self.n_bins = n_bins
        self.k_min = k_min
        self.k_max = k_max
        self.interp_res = interp_res
        super().__init__(predict_metadata, workdir, filename, variable)

    def get_metric_name(self) -> str:
        return f"power_spectrum_{self.variable}"

    def get_metric_shape(self) -> tuple:
        return (self.n_bins,)

    def get_extra_dimensions(self) -> list:
        k_bins = np.logspace(np.log10(self.k_min), np.log10(self.k_max), self.n_bins)
        return {"k": k_bins}

    def calculate_metric(self, prediction: np.ndarray) -> np.ndarray:
        #TODO: Results look weird, something wrong with the implementation?
        from scipy.fft import fft2, fftshift

        # prediction is (leadtime, points, variables)
        var_index = self.intermediate.pm.variables.index(self.variable)
        leadtimes = prediction.shape[0]
        metric = np.full((leadtimes, self.n_bins), np.nan, dtype=np.float32)

        for lt in range(leadtimes):
            if self.pm.is_gridded:
                field = prediction[lt,:,var_index].reshape(
                    (self.pm.field_shape)
                )
                nx = self.pm.field_shape[1]
                ny = self.pm.field_shape[0]
            else:
                assert self.interp_res is not None, "Need interp_res for ungridded data"
                # Should move most of this so it only happens once
                min_lat = self.get_lower(self.pm.lats)
                max_lat = self.get_upper(self.pm.lats)
                min_lon = self.get_lower(self.pm.lons)
                max_lon = self.get_upper(self.pm.lons)
                lons_regular = np.arange(min_lon, max_lon + self.interp_res, self.interp_res)
                lats_regular = np.arange(min_lat, max_lat + self.interp_res, self.interp_res)

                yy, xx = np.meshgrid(lats_regular, lons_regular)
                ipoints = gridpp.Points(self.pm.lats, self.pm.lons)
                ogrid = gridpp.Grid(yy.transpose(), xx.transpose())
                field = gridpp.nearest(ipoints, ogrid, prediction[lt,:,var_index])
                nx = len(lons_regular)
                ny = len(lats_regular)
                

            fft_field = fftshift(fft2(field))
            psd2d = np.abs(fft_field)**2

            ky = np.fft.fftfreq(ny)[:, None]
            kx = np.fft.fftfreq(nx)[None, :]
            k_radius = np.sqrt(kx**2 + ky**2)
            k_radius = fftshift(k_radius)
            k_bins = self.get_extra_dimensions()["k"]
            bin_indices = np.digitize(k_radius.flat, k_bins) - 1

            for b in range(self.n_bins):
                bin_mask = bin_indices == b
                if np.any(bin_mask):
                    metric[lt, b] = np.mean(psd2d.flat[bin_mask])
                else:
                    metric[lt, b] = np.nan  # No data in this bin               
        return metric

'''
class WaveletSpectrum(Spatial):
    """Calculates the wavelet power spectrum of a variable"""

    def __init__(
        self, 
        predict_metadata: PredictMetadata,
        workdir: str,
        filename: str,
        variable: str = "z",
        wavelet: str = "haar",
        n_scales: int = 5,
    ):
        self.wavelet = wavelet
        self.n_scales = n_scales
        super().__init__(predict_metadata, workdir, filename, variable)

    def get_metric_name(self) -> str:
        return f"power_spectrum_{self.variable}"

    def get_metric_shape(self) -> tuple:
        return (self.n_scales,)

    def get_extra_dimensions(self) -> list:
        scales = np.arange(1, self.n_scales + 1)
        return {"scale": scales}

    def calculate_metric(self, prediction: np.ndarray) -> np.ndarray:
        from pywt import wavedec2

        # prediction is (leadtime, points, variables)
        var_index = self.intermediate.pm.variables.index(self.variable)
        leadtimes = prediction.shape[0]
        metric = np.full((leadtimes, self.n_scales), np.nan, dtype=np.float32)

        for lt in range(leadtimes):
            field = prediction[lt,:,var_index].reshape(
                (self.intermediate.pm.ny, self.intermediate.pm.nx)
            )
            coeffs = wavedec2(field, wavelet=self.wavelet, level=self.n_scales)
            for scale in range(self.n_scales):
                cH, cV, cD = coeffs[scale + 1]
                power = (
                    np.mean(cH**2) + np.mean(cV**2) + np.mean(cD**2)
                ) / 3.0
                metric[lt, scale] = power

        return metric
'''        

            








        