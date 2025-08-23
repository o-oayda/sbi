from typing import Any, Callable, Optional
from numpy.typing import NDArray
from torch.types import Tensor
from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import Prior
import numpy as np
import torch
from scipy.stats import poisson as sp_poisson
from abc import ABC, abstractmethod
import healpy as hp
from jax import numpy as jnp


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

class CustomModelJax(LikelihoodBasedModel):
    def __init__(
            self,
            prior: Prior,
            log_likelihood: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ) -> None:
        super().__init__(prior)
        self._log_likelihood_callable = log_likelihood
    
    @property
    def ndim(self) -> int:
        return self.prior.ndim

    def log_likelihood(self, data: NDArray, theta: NDArray) -> NDArray:
        # this is shit
        return np.asarray(
            self._log_likelihood_callable(
                jnp.asarray(theta, dtype=jnp.float32), 
                jnp.asarray(data, dtype=jnp.float32)
            )
        )
