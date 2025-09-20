from typing import Literal
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
import numpy as np
from dipolesbi.tools.utils import jax_sph2cart
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
    def __init__(self) -> None:
        self._theta_mean = None
        self._theta_std = None
    
    def __call__(
            self, 
            theta: dict[str, jnp.ndarray],
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
            theta: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pass

    @abstractmethod
    def inverse_and_log_det(
            self, 
            transformed_theta: jnp.ndarray
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
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
    def __init__(self, method: Literal['cartesian']):
        super().__init__()

        if method == 'cartesian':
            self._forward_and_log_det = self._forward_and_log_det_cartesian
            self._inverse_and_log_det = self._inverse_and_log_det_cartesian
        else:
            raise NotImplementedError(f'{method}')

        self.method = method

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
            theta: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._forward_and_log_det(theta)

    def inverse_and_log_det(
            self,
            transformed_theta: jnp.ndarray
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        return self._inverse_and_log_det(transformed_theta)

    @property
    def theta_mean(self) -> jnp.ndarray | None:
        return self._theta_mean

    @property
    def theta_std(self) -> jnp.ndarray | None:
        return self._theta_std

    def _forward_and_log_det_cartesian(
        self,
        theta: dict[str, jnp.ndarray],
        in_ns: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None

        n_batches = theta['dipole_latitude'].shape[0]
        abslogdet = jnp.zeros(n_batches)

        lon = jnp.deg2rad(theta['dipole_longitude'])
        abslogdet += np.log(np.pi) - np.log(180)

        colat = jnp.pi / 2 - jnp.deg2rad(theta['dipole_latitude'])
        abslogdet += np.log(np.pi) - np.log(180)

        x, y, z = jax_sph2cart(lon, colat)
        abslogdet += np.sin(colat)

        t_transformed = jnp.stack(
            [
                (theta['mean_density'] - self.theta_mean[0]) / self.theta_std[0],
                (theta['observer_speed'] - self.theta_mean[1]) / self.theta_std[1],
                x, y, z
            ],
            axis=1 if not in_ns else 0
        )
        return t_transformed, abslogdet

    def _inverse_and_log_det_cartesian(
        self,
        transformed_theta: jnp.ndarray
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
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

        theta = {
            'mean_density': mean_density,
            'observer_speed': observer_speed,
            'dipole_longitude': dipole_longitude,
            'dipole_latitude': dipole_latitude,
        }
        return theta, abslogdet

class MapDataset(Dataset):
    def __init__(self, D_all: torch.Tensor):
        """
        Interface for pytorch.

        D_all: (N, Npix) torch tensor in NEST order.
        """
        self.D = D_all
    def __len__(self): return self.D.shape[0]
    def __getitem__(self, i): return self.D[i]

