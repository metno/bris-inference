import numpy as np
from functools import cached_property
from abc import abstractmethod
from scipy.fft import dctn
import xarray as xr
import datetime

from bris import projections, utils
from bris.conventions import cf
from bris.outputs import Output
from bris.predict_metadata import PredictMetadata
from bris.outputs.intermediate import IntermediateSpatial
from anemoi.datasets import open_dataset


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
        n_bins: int = None,
        proj4_str: str | None = None,
        domain_name: str | None = None,
    ) -> None:
        extra_variables = []
        if variable not in predict_metadata.variables:
            extra_variables += [variable]
        
        self.pm = predict_metadata
        if domain_name is not None:
            self.proj4_str = projections.get_proj4_str(domain_name)
        else:
            self.proj4_str = proj4_str
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
        self.n_bins = n_bins
        assert self.pm.is_gridded, "SpectralSkillSpread only works for gridded data"
        self.obs_dataset = open_dataset(obs_dataset)

    def get_observations_for_valid_times(self, frt: np.datetime64) -> np.ndarray:
        valid_times = [frt + np.timedelta64(lt, "s") for lt in self.pm.leadtimes]
        time_indices = [np.argmin(np.abs(self.obs_dataset.dates - t)) for t in valid_times]
        if self.variable == "ws":
            u_index = self.obs_dataset.name_to_index["10u"]
            v_index = self.obs_dataset.name_to_index["10v"]
            u = self.obs_dataset[time_indices, u_index, 0, ...]
            v = self.obs_dataset[time_indices, v_index, 0, ...]
            return np.sqrt(u**2 + v**2)

        var_index = self.obs_dataset.name_to_index[self.variable]
        #Debug
        print("valid_times: ", valid_times)
        print("time_indices: ", time_indices)
        return self.obs_dataset[time_indices, var_index, 0, ...]

    def add_forecast(
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
        nx, ny = self.pm.field_shape

        _skill = np.zeros((self.pm.num_leadtimes, nx, ny))
        spread = np.zeros((self.pm.num_leadtimes, nx, ny))
        
        frts = self.intermediate.get_forecast_reference_times()
        N_t = len(frts)
        N = self.pm.num_members

        for frt in frts:
            pred = np.zeros((self.pm.num_leadtimes, self.pm.num_points, self.pm.num_members))
            for member in range(self.pm.num_members):
                pred[..., member] = self.intermediate.get_forecast(frt, member).squeeze(-1)
            
            # Calculate skill
            obs = self.get_observations_for_valid_times(frt) #(time, latlon)
            
            obs = obs.reshape(obs.shape[0], nx, ny) #(lt, x, y)
            pred = pred.reshape(pred.shape[0], nx, ny, pred.shape[2]) #(lt, x, y, ens)

            obs_k = dctn(obs, axes=(1,2), type=2, norm="ortho") #time, kx, ky
            pred_k = dctn(pred, axes=(1,2), type=2, norm="ortho") #time, kx, ky, ens

            ens_mean = pred_k.mean(axis=3) / N
            
            _skill += (ens_mean - obs_k)**2 / N_t
            spread += np.sqrt( ( (pred_k - ens_mean[..., None])**2).sum(axis=3) / (N-1) ) / N_t

        skill = np.sqrt(_skill)

        k_edges, k_bins, k = self.get_bins
        n_bins = k_bins.shape[0]
        digitized = np.digitize(k.flatten(), k_edges)

        skill_k = np.full((self.pm.num_leadtimes, n_bins), np.nan)
        spread_k = np.full((self.pm.num_leadtimes, n_bins), np.nan)

        for lt in range(self.pm.num_leadtimes):
            skill_k[lt] = np.array(
                [
                    skill[lt].flatten()[digitized == i].mean() if np.any(digitized == i) else 0 
                    for i in range(1, n_bins + 1)
                ]
            )
            spread_k[lt] = np.array(
                [
                    spread[lt].flatten()[digitized == i].mean() if np.any(digitized == i) else 0 
                    for i in range(1, n_bins + 1)
                ]
            )

        self.write(spread_k, skill_k, k_bins)
        if self.remove_intermediate:
            self.intermediate.cleanup()

    def write(self, spread_k: np.ndarray, skill_k: np.ndarray, k_bin: np.ndarray) -> None:
        """Writes output to file"""
        coords = {}

        coords["leadtime"] = (
            ["leadtime"],
            self.intermediate.pm.leadtimes.astype(np.float32) / 3600,
            {"units": "hour"},
        )
        coords["k"] = (
            ["k"],
            k_bin,
            cf.get_attributes("k")
        )
        #TODO: convert from k to wavelength and add wavelength as a coordinate

        ds = xr.Dataset(coords=coords)

        dims = ["leadtime", "k"]
        ds["skill"] = (dims, skill_k)
        ds["spread"] = (dims, spread_k)

        datestr = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S +00:00"
        )
        ds.attrs["history"] = f"{datestr} Created by bris-inference"
        ds.attrs["Conventions"] = "CF-1.6"

        utils.create_directory(self.filename)
        ds.to_netcdf(self.filename, mode="w", engine="netcdf4")

    @cached_property
    def get_bins(self):
        """Calculates wavenumbers, bins and bin-edges used in the CDT calculation."""
        nx, ny = self.pm.field_shape
        lats, lons = self.get_latlons
        x, y = projections.get_xy(
            lats.reshape(nx, ny), lons.reshape(nx, ny), self.proj4_str
        )

        dx = np.mean(np.diff(x))
        dy = np.mean(np.diff(y))
        assert np.allclose(np.diff(x), dx, atol=1.0), (
            "Non-uniform grid spacing in x-direction"
        )
        assert np.allclose(np.diff(y), dy, atol=1.0), (
            "Non-uniform grid spacing in y-direction"
        )

        kx = np.pi * np.arange(nx) / (nx * dx)
        ky = np.pi * np.arange(ny) / (ny * dy)

        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k = np.sqrt(KX**2 + KY**2)
        k_max = k.max()

        n_bins = self.n_bins if self.n_bins is not None else min(nx, ny) // 2

        k_edges = np.linspace(0.0, k_max, n_bins + 1)
        k_bins = 0.5 * (k_edges[1:] + k_edges[:-1])
        return k_edges, k_bins, k
    
    @cached_property
    def get_latlons(self) -> tuple:
        return self.pm.lats, self.pm.lons
