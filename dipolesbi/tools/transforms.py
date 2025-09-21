from typing import Literal, cast
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
import numpy as np
from dipolesbi.tools.utils import PytreeAdapter, jax_sph2cart
from abc import ABC, abstractmethod
from jax import numpy as jnp


class InvertibleDataTransform(ABC):
    def __init__(self) -> None:
        pass
    
    def __call__(self, data: NDArray) -> tuple[NDArray, NDArray]:
        return self.forward_and_log_det(data)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def forward_and_log_det(self, data: NDArray) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def inverse_and_log_det(self, transformed_data: NDArray) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def compute_mean_and_std(self, data: NDArray) -> None:
        pass

class InvertibleThetaTransformJax(ABC):
    def __init__(self, pytree_adapter: PytreeAdapter) -> None:
        self._theta_mean = None
        self._theta_std = None
        self.adapter = pytree_adapter
    
    def __call__(
            self, 
            theta: dict[str, jnp.ndarray] | jnp.ndarray,
            **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward_and_log_det(theta, **kwargs)

    @property
    def theta_mean(self) -> jnp.ndarray | None:
        return self._theta_mean

    @property
    def theta_std(self) -> jnp.ndarray | None:
        return self._theta_std

    def stats_are_none(self) -> bool:
        if (self.theta_mean is None) and (self.theta_std is None):
            return True
        else:
            return False

    def clear(self) -> None:
        self._theta_mean = None
        self._theta_std = None

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def forward_and_log_det(
            self, 
            theta: dict[str, jnp.ndarray] | jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def inverse_and_log_det(
            self, 
            transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # we don't want it to return a dict in case it's a layer in the flow
        pass

    @abstractmethod
    def compute_mean_and_std(self, theta: dict[str, jnp.ndarray]) -> None:
        pass

class BlankTransform(InvertibleDataTransform):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return 'BlankTransform(No transform to the data.)'

    def forward_and_log_det(self, data: NDArray) -> tuple[NDArray, NDArray]:
        n_batches = data.shape[0]
        return data, np.zeros_like(n_batches)

    def inverse_and_log_det(self, transformed_data: NDArray) -> tuple[NDArray, NDArray]:
        n_batches = transformed_data.shape[0]
        return transformed_data, np.zeros_like(n_batches)

    def clear(self) -> None:
        pass

    def compute_mean_and_std(self, data: NDArray) -> None:
        return 

class ZScore(InvertibleDataTransform):
    def __init__(self) -> None:
        self.mu = None
        self.sigma = None

    def __repr__(self) -> str:
        return (
            f"Zscore("
            f"mu={self.mu}, "
            f"sigma={self.sigma}"
            f")"
        )

    def forward_and_log_det(self, data: NDArray) -> tuple[NDArray, NDArray]:
        '''
        Do a batchwise mean per data dimension.
        '''
        n_batches = data.shape[0]
        if (self.mu is None) or (self.sigma is None):
            self.mu = np.nanmean(data, axis=0)
            self.sigma = np.nanstd(data, axis=0)

        log_det_jac = - np.log(self.sigma).sum() * np.ones(n_batches)
        z = (data - self.mu) / self.sigma
        return z, log_det_jac

    def inverse_and_log_det(self, transformed_data: NDArray) -> tuple[NDArray, NDArray]:
        data = transformed_data * self.sigma + self.mu
        n_batches = data.shape[0]
        log_det_jac = np.log(self.sigma).sum() * np.ones(n_batches) # type: ignore
        return data, log_det_jac

    def clear(self) -> None:
        self.mu = None
        self.sigma = None

class DipoleThetaTransform(InvertibleThetaTransformJax):
    def __init__(self, adapter: PytreeAdapter, method: Literal['cartesian', 'zscore']):
        super().__init__(adapter)

        if method == 'cartesian':
            self._forward_and_log_det = self._forward_and_log_det_cartesian
            self._inverse_and_log_det = self._inverse_and_log_det_cartesian
        elif method == 'zscore':
            self._forward_and_log_det = self._forward_and_log_det_zscore
            self._inverse_and_log_det = self._inverse_and_log_det_zscore
        else:
            raise Exception(f'{method} not recognised.')

        self.method = method
        # TODO: refactor later to be part of the adapter
        self.str2idx = {
            'mean_density': 0,
            'observer_speed': 1,
            'dipole_longitude': 2,
            'dipole_latitude': 3
        }

    def __repr__(self) -> str:
        return (
            'DipoleThetaTransform('
            f'method={self.method}, '
            f'theta_mean={self.theta_mean}, '
            f'theta_std={self.theta_std}, '
            ')'
        )

    def compute_mean_and_std(self, theta: dict[str, jnp.ndarray]) -> None:
        assert len(theta.keys()) == 4

        if not self.stats_are_none():
            raise Exception('Stats should be empty before making new ones.')

        mean_nbar = jnp.nanmean(theta['mean_density'])
        std_nbar = np.nanstd(theta['mean_density'])
        mean_v = np.nanmean(theta['observer_speed'])
        std_v = np.nanstd(theta['observer_speed'])

        self._theta_mean = jnp.asarray([mean_nbar, mean_v, 0, 0])
        self._theta_std = jnp.asarray([std_nbar, std_v, 1, 1])

    def forward_and_log_det(
            self, 
            theta: dict[str, jnp.ndarray] | jnp.ndarray,
            **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._forward_and_log_det(theta, **kwargs)

    def inverse_and_log_det(
            self,
            transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._inverse_and_log_det(transformed_theta)

    @property
    def theta_mean(self) -> jnp.ndarray | None:
        return self._theta_mean

    @property
    def theta_std(self) -> jnp.ndarray | None:
        return self._theta_std

    def _forward_and_log_det_cartesian(
        self,
        theta: dict[str, jnp.ndarray] | jnp.ndarray,
        in_ns: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None

        if type(theta) is dict:
            theta = self.adapter.to_array(theta)

        theta = cast(jnp.ndarray, theta)
        if in_ns: assert theta[self.str2idx['mean_density']].shape == ()

        abslogdet = jnp.zeros_like(theta[self.str2idx['dipole_latitude']])

        lon = jnp.deg2rad(theta[self.str2idx['dipole_longitude']])
        abslogdet += jnp.log(jnp.pi) - np.log(180)

        colat = jnp.pi / 2 - jnp.deg2rad(theta[self.str2idx['dipole_latitude']])
        abslogdet += jnp.log(jnp.pi) - np.log(180)

        x, y, z = jax_sph2cart(lon, colat)
        abslogdet += jnp.sin(colat)

        t_transformed = jnp.stack(
            [
                (
                    theta[self.str2idx['mean_density']] - self.theta_mean[0]
                ) / self.theta_std[0],
                (
                    theta[self.str2idx['observer_speed']] - self.theta_mean[1]
                ) / self.theta_std[1],
                x, y, z
            ],
            axis=1 if not in_ns else 0
        )
        return t_transformed, abslogdet

    def _inverse_and_log_det_cartesian(
        self,
        transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None

        mean_density_norm = transformed_theta[:, 0]
        observer_speed_norm = transformed_theta[:, 1]
        x = transformed_theta[:, 2]
        y = transformed_theta[:, 3]
        z = transformed_theta[:, 4]

        # Denormalize
        mean_density = mean_density_norm * self.theta_std[0] + self.theta_mean[0]
        observer_speed = observer_speed_norm * self.theta_std[1] + self.theta_mean[1]

        # Cartesian to spherical
        colat = jnp.arccos(jnp.clip(z, -1.0, 1.0))
        lon = jnp.arctan2(y, x)

        # Convert to degrees
        dipole_longitude = jnp.rad2deg(lon)
        dipole_latitude = jnp.rad2deg(jnp.pi / 2 - colat)

        # Compute abslogdet (inverse of forward)
        n_batches = transformed_theta.shape[0]
        abslogdet = jnp.zeros(n_batches)
        abslogdet -= np.log(np.pi) - np.log(180)
        abslogdet -= np.log(np.pi) - np.log(180)
        abslogdet -= jnp.sin(colat)

        theta = jnp.stack(
            [mean_density, observer_speed, dipole_longitude, dipole_latitude],
            axis=1
        )
        return theta, abslogdet

    def _forward_and_log_det_zscore(
        self,
        theta: dict[str, jnp.ndarray] | jnp.ndarray
    ) -> tuple[jnp.ndarray,  jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None

        # does this jank bullshit slow it down? please check
        if type(theta) is dict:
            theta = self.adapter.to_array(theta)
        theta = cast(jnp.ndarray, theta)
        theta = jnp.atleast_2d(theta)

        abslogdet_scalar = 0

        lon = jnp.deg2rad(theta[self.str2idx['dipole_longitude']])
        abslogdet_scalar += jnp.log(jnp.pi) - np.log(180)

        colat = jnp.pi / 2 - jnp.deg2rad(theta[self.str2idx['dipole_latitude']])
        abslogdet_scalar += jnp.log(jnp.pi) - np.log(180)

        mean_density_norm = (
            theta[self.str2idx['mean_density']] - self.theta_mean[0]
        ) / self.theta_std[0]
        abslogdet_scalar += - jnp.log(self.theta_std[0])

        observer_speed_norm = (
            theta[self.str2idx['observer_speed']] - self.theta_mean[1]
        ) / self.theta_std[1]
        abslogdet_scalar += - jnp.log(self.theta_std[1])

        t_transformed = jnp.stack(
            [mean_density_norm, observer_speed_norm, lon, colat],
            axis=1
        )
        abslogdet_per_batch = jnp.full((t_transformed.shape[0],), abslogdet_scalar)
        return t_transformed, abslogdet_per_batch

    def _inverse_and_log_det_zscore(
        self,
        transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None
        transformed_theta = jnp.atleast_2d(transformed_theta)

        mean_density_norm = transformed_theta[:, 0]
        observer_speed_norm = transformed_theta[:, 1]
        lon = transformed_theta[:, 2]
        colat = transformed_theta[:, 3]

        mean_density = mean_density_norm * self.theta_std[0] + self.theta_mean[0]
        observer_speed = observer_speed_norm * self.theta_std[1] + self.theta_mean[1]
        dipole_longitude = jnp.rad2deg(lon)
        dipole_latitude = jnp.rad2deg(jnp.pi / 2 - colat)

        abslogdet_scalar = 0
        abslogdet_scalar -= jnp.log(jnp.pi) - np.log(180)
        abslogdet_scalar -= jnp.log(jnp.pi) - np.log(180)
        abslogdet_scalar += jnp.log(self.theta_std[0])
        abslogdet_scalar += jnp.log(self.theta_std[1])

        theta = jnp.stack(
            [mean_density, observer_speed, dipole_longitude, dipole_latitude],
            axis=1
        )
        abslogdet_per_batch = jnp.full((theta.shape[0],), abslogdet_scalar)
        return theta, abslogdet_per_batch

class MapDataset(Dataset):
    def __init__(self, D_all: torch.Tensor):
        """
        Interface for pytorch.

        D_all: (N, Npix) torch tensor in NEST order.
        """
        self.D = D_all
    def __len__(self): return self.D.shape[0]
    def __getitem__(self, i): return self.D[i]

