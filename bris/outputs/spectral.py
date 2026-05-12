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


class SpectralGridded:
    """Calculates quantities in spectral space using the Discrete Cosine Transform (DCT).
    - Error variance: Variance of the error between the ensemble members and the analysis in spectral space.
    - Spread variance: Variance of the ensemble members around the ensemble mean in spectral space.
    - Spectrum of the forecast and observation: Average power in spectral space for the forecast and observation.

    Writes to a file as a function of time, leadtime, wavenumber and ensemblemember (when relevant).
    """

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
        shape = predict_metadata.num_points
        self.intermediate = IntermediateSpatial(
            predict_metadata=predict_metadata,
            workdir=workdir,
            metric_shape=shape,
            extra_variables=extra_variables,
        )
        self.remove_intermediate = remove_intermediate
        self.n_bins = n_bins
        assert self.pm.is_gridded, "SpectralSkillSpread only works for gridded data"
        self.obs_dataset = open_dataset(obs_dataset)

    def get_observations_for_valid_times(self, frt: np.datetime64) -> np.ndarray:
        """Retrieves observations for the valid times corresponding to the forecast reference time (frt) and leadtimes."""

        valid_times = [frt + np.timedelta64(lt, "s") for lt in self.pm.leadtimes]
        time_indices = [
            np.argmin(np.abs(self.obs_dataset.dates - t)) for t in valid_times
        ]
        if self.variable == "ws":
            u_index = self.obs_dataset.name_to_index["10u"]
            v_index = self.obs_dataset.name_to_index["10v"]
            u = self.obs_dataset[time_indices, u_index, 0, ...]
            v = self.obs_dataset[time_indices, v_index, 0, ...]
            return np.sqrt(u**2 + v**2)

        var_index = self.obs_dataset.name_to_index[self.variable]
        return self.obs_dataset[time_indices, var_index, 0, ...]

    def add_forecast(self, times: list, ensemble_member: int, pred: np.ndarray) -> None:
        """Adds forecast to intermediate storage. If variable is "ws", calculates wind speed from u and v components."""
        if self.variable == "ws":
            Ix = self.pm.variables.index("10u")
            Iy = self.pm.variables.index("10v")
            pred = np.sqrt(pred[..., [Ix]] ** 2 + pred[..., [Iy]] ** 2)
        else:
            pred = pred[..., [self.pm.variables.index(self.variable)]]

        self.intermediate._add_forecast(times, ensemble_member, pred)

    def finalize(self) -> None:
        """Calculates skill, spread and spectra in spectral space and writes to file."""
        nx, ny = self.pm.field_shape

        _skill = np.zeros((self.pm.num_leadtimes, nx, ny))
        spread = np.zeros((self.pm.num_leadtimes, nx, ny))

        frts = self.intermediate.get_forecast_reference_times()
        print("frts:", frts)
        N_t = len(frts)
        N = self.pm.num_members
        print("N_t:", N_t, "N:", N)

        k_edges, k_bins, k = self.get_bins
        n_bins = k_bins.shape[0]
        digitized = np.digitize(k.flatten(), k_edges)

        spread_variance = np.full((len(frts), self.pm.num_leadtimes, n_bins), np.nan)
        error_variance = np.full((len(frts), self.pm.num_leadtimes, n_bins), np.nan)
        spectrum_forecast = np.full(
            (len(frts), self.pm.num_leadtimes, n_bins, N), np.nan
        )
        spectrum_observation = np.full(
            (len(frts), self.pm.num_leadtimes, n_bins), np.nan
        )
        for i, frt in enumerate(frts):
            pred = np.zeros(
                (self.pm.num_leadtimes, self.pm.num_points, self.pm.num_members)
            )
            for member in range(self.pm.num_members):
                pred[..., member] = self.intermediate.get_forecast(frt, member).squeeze(
                    -1
                )

            # Calculate skill
            obs = self.get_observations_for_valid_times(frt)  # (time, latlon)

            obs = obs.reshape(obs.shape[0], nx, ny)  # (lt, x, y)
            pred = pred.reshape(pred.shape[0], nx, ny, pred.shape[2])  # (lt, x, y, ens)

            obs_k = dctn(obs, axes=(1, 2), type=2, norm="ortho")  # time, kx, ky
            pred_k = dctn(pred, axes=(1, 2), type=2, norm="ortho")  # time, kx, ky, ens

            P_obs = np.abs(obs_k) ** 2
            P_pred = np.abs(pred_k) ** 2

            ens_mean = pred_k.mean(axis=3)

            spread_variance_k = ((pred_k - ens_mean[..., None]) ** 2).sum(axis=3) / (
                N - 1
            )  # time, kx, ky
            error_variance_k = (ens_mean - obs_k) ** 2  # time, kx, ky

            # Bin average over k space
            for lt in range(self.pm.num_leadtimes):
                spread_variance[i, lt, :] = np.array(
                    [
                        spread_variance_k[lt].flatten()[digitized == j].mean()
                        if np.any(digitized == j)
                        else np.nan
                        for j in range(1, n_bins + 1)
                    ]
                )
                error_variance[i, lt, :] = np.array(
                    [
                        error_variance_k[lt].flatten()[digitized == j].mean()
                        if np.any(digitized == j)
                        else np.nan
                        for j in range(1, n_bins + 1)
                    ]
                )
                spectrum_forecast[i, lt, :, :] = np.array(
                    [
                        P_pred[lt].reshape(-1, N)[digitized == j, :].mean(axis=0)
                        if np.any(digitized == j)
                        else np.full(N, np.nan)
                        for j in range(1, n_bins + 1)
                    ]
                )
                spectrum_observation[i, lt, :] = np.array(
                    [
                        P_obs[lt].flatten()[digitized == j].mean()
                        if np.any(digitized == j)
                        else np.nan
                        for j in range(1, n_bins + 1)
                    ]
                )

        self.write(
            spread_variance,
            error_variance,
            spectrum_forecast,
            spectrum_observation,
            k_bins,
        )
        if self.remove_intermediate:
            self.intermediate.cleanup()

    def write(
        self,
        spread_variance: np.ndarray,
        error_variance: np.ndarray,
        spectrum_forecast: np.ndarray,
        spectrum_observation: np.ndarray,
        k_bin: np.ndarray,
    ) -> None:
        """Writes output to file"""
        coords = {}
        frts = self.intermediate.get_forecast_reference_times()
        times_unix = utils.datetime_to_unixtime(frts).astype(np.double)
        coords["time"] = (["time"], times_unix, cf.get_attributes("time"))

        coords["leadtime"] = (
            ["leadtime"],
            self.intermediate.pm.leadtimes.astype(np.float32) / 3600,
            {"units": "hour"},
        )
        coords["k"] = (["k"], k_bin, cf.get_attributes("k"))
        coords["wavelength"] = (
            ["k"],
            2 * np.pi / k_bin,
            cf.get_attributes("wavelength"),
        )
        coords["ensemble_member"] = (
            ["ensemble_member"],
            np.arange(self.pm.num_members),
            cf.get_attributes("ensemble_member"),
        )

        ds = xr.Dataset(coords=coords)

        dims = ["time", "leadtime", "k"]
        ds["error_variance"] = (dims, error_variance)
        ds["spread_variance"] = (dims, spread_variance)
        ds["spectrum_forecast"] = (dims + ["ensemble_member"], spectrum_forecast)
        ds["spectrum_target"] = (dims, spectrum_observation)

        datestr = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S +00:00"
        )
        ds.attrs["history"] = f"{datestr} Created by bris-inference"
        ds.attrs["Conventions"] = "CF-1.6"
        ds.attrs["standard_name"] = cf.get_metadata(self.variable)["cfname"]

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
