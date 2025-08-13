from typing import Any, Callable, Optional
from jax._src.dtypes import dtype
from numpy.typing import NDArray
from torch.types import Tensor
from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import Prior
import numpy as np
import torch
from scipy.stats import poisson as sp_poisson
from abc import ABC, abstractmethod
import healpy as hp


class LikelihoodBasedModel(ABC):
    def __init__(self, prior: Prior) -> None:
        super().__init__()
        self.prior = prior

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    def prior_transform(self, uniform_deviates: NDArray) -> NDArray:
        uniform_deviates_tens = torch.as_tensor(uniform_deviates)
        return self.prior.transform(uniform_deviates_tens).numpy()

    def log_prior_likelihood(self, theta: NDArray) -> NDArray:
        theta_tens = torch.as_tensor(theta)
        return self.prior.log_prob(theta_tens).numpy()

    def _interface_with_model(self,
            model: Callable[..., NDArray], 
            Theta: NDArray,
            hyper_parameters: dict[str, Any]
    ) -> NDArray: 
        mapping = {
            kwarg: Theta[..., i] for i, kwarg in enumerate(
                self.prior.simulator_kwargs
            )
        } 
        mapping.update(hyper_parameters)
        return model(**mapping)

    @abstractmethod
    def log_likelihood(self, data: NDArray, theta: NDArray) -> NDArray:
        pass

class DipolePoisson(LikelihoodBasedModel):
    def __init__(
            self, 
            prior: Prior, 
            nside: int = 64, 
            mask_map: Optional[NDArray[np.bool_]] = None
    ) -> None:
        super().__init__(prior)
        self.map_model = SimpleDipoleMap(nside)
        self.npix = hp.nside2npix(nside)

        if mask_map is None:
            self.mask_map = np.ones(self.npix, dtype=np.bool_)
        else:
            self.mask_map = mask_map

        self._ndim = 4
    
    @property
    def ndim(self) -> int:
        return self._ndim

    def log_likelihood(self, data: NDArray, theta: NDArray) -> NDArray:
        dipole_signal = self._interface_with_model(
            self.map_model.generate_dipole,
            theta,
            hyper_parameters={'make_poisson_draws': False}
        )
        return np.sum(
            sp_poisson.logpmf(
                k=data[self.mask_map], 
                mu=dipole_signal[:, self.mask_map]
            ),
            axis=1
        )

class CustomModel(LikelihoodBasedModel):
    def __init__(
            self,
            prior: Prior,
            log_likelihood: Callable[[Tensor, Tensor], Tensor]
    ) -> None:
        super().__init__(prior)
        self._log_likelihood_callable = log_likelihood
    
    @property
    def ndim(self) -> int:
        return self.prior.ndim

    def log_likelihood(self, data: NDArray, theta: NDArray) -> NDArray:
        return self._log_likelihood_callable(
            torch.as_tensor(theta, dtype=torch.float32), 
            torch.as_tensor(data, dtype=torch.float32)
        ).numpy()

# class DipolePoisson(LikelihoodBasedInferer):
#     def __init__(self,
#             density_map: np.ndarray[float],
#             amplitude_range: list[float] = [0, 0.1],
#             mean_count_range: list[float] = [0,100.0],
#             longitude_range: list[float] = [0, 2*np.pi],
#             latitude_range: list[float] = [0, np.pi],
#             device: str = 'cpu'
#     ) -> None:
#         super().__init__()
#         self.density_map = density_map
#         self.npix = len(density_map)
#         self.nside = hp.npix2nside(self.npix)
#         self.ndim = 4
#         self.device = device
#         self.sky_map = SkyMap(self.nside, device=self.device)
#         self.mean_count_range = mean_count_range
#         self.amplitude_range = amplitude_range
#         self.longitude_range = longitude_range
#         self.latitude_range = latitude_range
#
#     def prior_transform(self, uTheta):
#         u_mean_count, u_amplitude, u_longitude, u_colatitude = uTheta
#         mean_count = sample_unif(u_mean_count, self.mean_count_range)
#         amplitude = sample_unif(u_amplitude, self.amplitude_range)
#         longitude = sample_unif(u_longitude, self.longitude_range)
#         latitude = sample_polar(u_colatitude, self.latitude_range)
#         return mean_count, amplitude, longitude, latitude
#
#     def log_prior_likelihood(self, Theta):
#         mean_count, amplitude, longitude, latitude = Theta
#         conditions = [
#             self.mean_count_range[0] < mean_count < self.mean_count_range[1],
#             self.amplitude_range[0]  < amplitude  < self.amplitude_range[1],
#             self.longitude_range[0]  < longitude  < self.longitude_range[1],
#             self.latitude_range[0]   < latitude   < self.latitude_range[1]
#         ]
#         if not all(conditions):
#             return -np.inf
#         else:
#             P_mean_count = np.log( unif_pdf (  self.mean_count_range ) )
#             P_amplitude  = np.log( unif_pdf (  self.amplitude_range ) )
#             P_longitude  = np.log( unif_pdf (  self.longitude_range ) )
#             P_colatitude = np.log( polar_pdf(  latitude, self.latitude_range ) )
#             return P_mean_count + P_amplitude + P_longitude + P_colatitude
#
#     def log_likelihood(self, Theta):
#         signal = self.sky_map.dipole_signal(
#             torch.as_tensor(Theta).to(torch.float32)
#         )
#         return np.sum(
#             sp_poisson.logpmf(k=self.density_map, mu=signal)
#         )
