from typing import Literal, Optional, cast
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
import numpy as np
from dipolesbi.scripts.bijectors import LatitudeBijector, UniformIntervalSigmoid
from dipolesbi.tools.priors_jax import DipolePriorJax
from dipolesbi.tools.utils import PytreeAdapter, jax_sph2cart
from abc import ABC, abstractmethod
import jax
from jax import numpy as jnp


class InvertibleDataTransform(ABC):
    def __init__(self) -> None:
        pass
    
    def __call__(
            self, 
            data: NDArray, 
            mask: Optional[NDArray] = None
    ) -> tuple[tuple[NDArray, NDArray], NDArray]: # (z, mask), logdet
        return self.forward_and_log_det(data, mask)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def forward_and_log_det(
            self, 
            data: NDArray, 
            mask: Optional[NDArray] = None
    ) -> tuple[tuple[NDArray, NDArray], NDArray]:
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
    def __init__(self, prior: DipolePriorJax) -> None:
        self._theta_mean = None
        self._theta_std = None
        self.prior = prior
        self._adapter = self.prior.get_adapter()
    
    def __call__(
            self, 
            theta: dict[str, jnp.ndarray] | jnp.ndarray,
            **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward_and_log_det(theta, **kwargs)

    @property
    def adapter(self) -> PytreeAdapter:
        return self._adapter

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
    def __init__(self, method: Literal['batchwise', 'global'] = 'batchwise') -> None:
        self.mu = None
        self.sigma = None
        self.method = method

        if self.method == 'batchwise':
            self._compute_mean_and_std = self._compute_mean_and_std_batchwise
        elif self.method == 'global':
            self._compute_mean_and_std = self._compute_mean_and_std_global
        else:
            raise Exception(f'Method {method} not recognised.')

    def __repr__(self) -> str:
        return (
            f"Zscore("
            f"mu={self.mu}, "
            f"sigma={self.sigma}, "
            f"method={self.method}"
            f")"
        )

    def compute_mean_and_std(self, data: NDArray) -> None:
        return self._compute_mean_and_std(data)

    def _compute_mean_and_std_batchwise(self, data: NDArray) -> None:
        self.mu = np.nanmean(data, axis=0)
        self.sigma = np.nanstd(data, axis=0)

    def _compute_mean_and_std_global(self, data: NDArray, min_std=1e-14) -> None:
        self.mu = np.nanmean(data)
        std = np.nanstd(data, axis=1)
        std[std < min_std] = min_std
        std = std.mean()
        self.sigma = std

    def forward_and_log_det(self, data: NDArray) -> tuple[NDArray, NDArray]:
        '''
        Do a batchwise mean per data dimension.
        '''
        n_batches = data.shape[0]
        assert self.sigma is not None
        assert self.mu is not None

        log_det_jac = - np.log(self.sigma).sum() * np.ones(n_batches)
        if np.ndim(self.sigma) == 0:
            # times by pixel count if we don't do a batchwise per-pixel zscore
            log_det_jac *= data.shape[1]
        z = (data - self.mu) / self.sigma
        return z, log_det_jac

    def inverse_and_log_det(self, transformed_data: NDArray) -> tuple[NDArray, NDArray]:
        n_batches = transformed_data.shape[0]
        assert self.sigma is not None
        assert self.mu is not None

        data = transformed_data * self.sigma + self.mu
        log_det_jac = np.log(self.sigma).sum() * np.ones(n_batches) # type: ignore
        if np.ndim(self.sigma) == 0:
            log_det_jac *= data.shape[1]
        return data, log_det_jac

    def clear(self) -> None:
        self.mu = None
        self.sigma = None

class DipoleBijectorWrapper(InvertibleThetaTransformJax):
    def __init__(self, prior: DipolePriorJax) -> None:
        super().__init__(prior)
        self.bijectors = {
            'mean_density': UniformIntervalSigmoid(
                self.prior.low_ranges[0], 
                self.prior.high_ranges[0]
            ),
            'observer_speed': UniformIntervalSigmoid(
                self.prior.low_ranges[1], 
                self.prior.high_ranges[1]
            ),
            'dipole_longitude': UniformIntervalSigmoid(
                self.prior.low_ranges[2], 
                self.prior.high_ranges[2]
            ),
            'dipole_latitude': LatitudeBijector() # assume -90, 90 for now
        }

    def __repr__(self) -> str:
        return 'DipoleBijectorWrapper()'

    def compute_mean_and_std(self, theta: dict[str, jnp.ndarray]) -> None:
        pass

    def forward_and_log_det(
            self, 
            theta: dict[str, jnp.ndarray] | jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if type(theta) is dict:
            leaves = jax.tree.leaves(theta)
            if any(leaf.ndim == 0 for leaf in leaves):
                theta = self.adapter.ravel(theta)
            else:
                theta = self.adapter.to_array(theta)
        theta = cast(jnp.ndarray, theta)

        if theta.ndim == 1:
            theta = theta[None, :]
        batches = theta.shape[0]

        unconstrained_vars = []
        logdet = jnp.zeros(batches)

        for key in self.adapter.keys:
            bijector = self.bijectors[key]
            var, ld = bijector.inverse_and_log_det(
                self.adapter.flat_view(theta, key)
            )
            logdet += ld
            unconstrained_vars.append(var)

        t_transformed = jnp.stack(unconstrained_vars, axis=-1)

        return t_transformed.squeeze(), logdet.squeeze()

    def inverse_and_log_det(
            self, 
            transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        if transformed_theta.ndim == 1:
            transformed_theta = transformed_theta[None, :]

        batches = transformed_theta.shape[0]
    
        constrained_vars = []
        logdet = jnp.zeros(batches)

        for key in self.adapter.keys:
            bijector = self.bijectors[key]
            var, ld = bijector.forward_and_log_det(
                self.adapter.flat_view(transformed_theta, key)
            )
            logdet += ld
            constrained_vars.append(var)

        theta = jnp.stack(constrained_vars, axis=-1)

        return theta, logdet

class DipoleThetaTransform(InvertibleThetaTransformJax):
    def __init__(
            self,
            prior: DipolePriorJax,
            method: Literal['cartesian', 'zscore'],
            wrap_longitude: bool = False,
            reflect_latitude: bool = False
    ):
        super().__init__(prior)

        if method == 'cartesian':
            self._forward_and_log_det = self._forward_and_log_det_cartesian
            self._inverse_and_log_det = self._inverse_and_log_det_cartesian
        elif method == 'zscore':
            self._forward_and_log_det = self._forward_and_log_det_zscore
            self._inverse_and_log_det = self._inverse_and_log_det_zscore
        else:
            raise Exception(f'{method} not recognised.')

        self.method = method
        self.wrap_longitude = wrap_longitude
        self.reflect_latitude = reflect_latitude

    def __repr__(self) -> str:
        return (
            'DipoleThetaTransform('
            f'method={self.method}, '
            f'theta_mean={self.theta_mean}, '
            f'theta_std={self.theta_std}, '
            ')'
        )
    
    def _reflect_latitude(self, latitude_deg: jnp.ndarray) -> jnp.ndarray:
        theta_lat = jnp.pi / 2 - jnp.deg2rad(latitude_deg)
        theta_mod = jnp.mod(theta_lat, 2 * jnp.pi)
        theta_lat = jnp.where(theta_mod > jnp.pi, 2 * jnp.pi - theta_mod, theta_mod)
        latitude_deg = jnp.rad2deg(jnp.pi / 2 - theta_lat)
        return latitude_deg

    def _wrap_longitude(self, longitude_deg: jnp.ndarray) -> jnp.ndarray:
        return jnp.mod(longitude_deg, 360.0)

    def compute_mean_and_std(self, theta: dict[str, jnp.ndarray]) -> None:
        assert len(theta.keys()) == 4

        if not self.stats_are_none():
            raise Exception('Stats should be empty before making new ones.')

        means = []
        stds = []
        for key in self.adapter.keys:
            if key == 'mean_density':
                means.append(jnp.nanmean(theta[key]))
                stds.append(jnp.nanstd(theta[key]))
            elif key == 'observer_speed':
                means.append(jnp.nanmean(theta[key]))
                stds.append(jnp.nanstd(theta[key]))
            elif key in ('dipole_longitude', 'dipole_latitude'):
                means.append(jnp.array(0.0))
                stds.append(jnp.array(1.0))
            else:
                raise KeyError(f'Unexpected theta key {key} in adapter.')

        self._theta_mean = jnp.asarray(means)
        self._theta_std = jnp.asarray(stds)

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
            leaves = jax.tree.leaves(theta)
            if any(leaf.ndim == 0 for leaf in leaves):
                theta = self.adapter.ravel(theta) # get shape (ndim,)
            else:
                theta = self.adapter.to_array(theta) # get shape (batches, ndim)

        theta = cast(jnp.ndarray, theta)
        theta = jnp.asarray(theta)

        squeezed_input = False
        if theta.ndim == 1:
            theta = theta[None, :] # make sure at least 2D, (ndim,) -> (1, ndim)
            squeezed_input = True

        lon = jnp.deg2rad(self.adapter.flat_view(theta, 'dipole_longitude'))
        colat = jnp.pi / 2 - jnp.deg2rad(self.adapter.flat_view(theta, 'dipole_latitude'))

        x, y, z = jax_sph2cart(lon, colat)

        mean_density_norm = (
            self.adapter.flat_view(theta, 'mean_density') - self.theta_mean[0]
        ) / self.theta_std[0]
        observer_speed_norm = (
            self.adapter.flat_view(theta, 'observer_speed') - self.theta_mean[1]
        ) / self.theta_std[1]

        t_transformed = jnp.stack(
            [mean_density_norm, observer_speed_norm, x, y, z],
            axis=-1
        )

        abslogdet = jnp.zeros(theta.shape[0], dtype=theta.dtype)
        abslogdet += jnp.log(jnp.pi) - np.log(180)
        abslogdet += jnp.log(jnp.pi) - np.log(180)
        abslogdet += jnp.sin(colat)

        # make sure to output as (ndim,) if this was passed in
        if squeezed_input or in_ns:
            t_transformed = jnp.squeeze(t_transformed, axis=0)
            abslogdet = jnp.squeeze(abslogdet, axis=0)

        return t_transformed, abslogdet

    # TODO: we probably need to integrate the adapter here; this could be a liability
    def _inverse_and_log_det_cartesian(
        self,
        transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None

        transformed_theta = jnp.asarray(transformed_theta)

        squeezed_input = False
        if transformed_theta.ndim == 1:
            transformed_theta = transformed_theta[None, :]
            squeezed_input = True

        mean_density_norm = transformed_theta[..., 0]
        observer_speed_norm = transformed_theta[..., 1]
        x = transformed_theta[..., 2]
        y = transformed_theta[..., 3]
        z = transformed_theta[..., 4]

        mean_density = mean_density_norm * self.theta_std[0] + self.theta_mean[0]
        observer_speed = observer_speed_norm * self.theta_std[1] + self.theta_mean[1]

        colat = jnp.arccos(jnp.clip(z, -1.0, 1.0))
        lon = jnp.arctan2(y, x)

        dipole_longitude = jnp.rad2deg(lon)
        if self.wrap_longitude:
            dipole_longitude = self._wrap_longitude(dipole_longitude)

        dipole_latitude = jnp.rad2deg(jnp.pi / 2 - colat)
        if self.reflect_latitude:
            dipole_latitude = self._reflect_latitude(dipole_latitude)

        abslogdet = jnp.zeros(transformed_theta.shape[0], dtype=transformed_theta.dtype)
        abslogdet -= jnp.log(jnp.pi) - np.log(180)
        abslogdet -= jnp.log(jnp.pi) - np.log(180)
        abslogdet -= jnp.sin(colat)

        theta = jnp.stack(
            [mean_density, observer_speed, dipole_longitude, dipole_latitude],
            axis=-1
        )

        if squeezed_input:
            theta = jnp.squeeze(theta, axis=0)
            abslogdet = jnp.squeeze(abslogdet, axis=0)

        return theta, abslogdet

    def _forward_and_log_det_zscore(
        self,
        theta: dict[str, jnp.ndarray] | jnp.ndarray
    ) -> tuple[jnp.ndarray,  jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None

        # does this jank bullshit slow it down? please check
        if type(theta) is dict:
            leaves = jax.tree.leaves(theta)
            if any(leaf.ndim == 0 for leaf in leaves):
                theta = self.adapter.ravel(theta)
            else:
                theta = self.adapter.to_array(theta)
        theta = cast(jnp.ndarray, theta)
        theta = jnp.asarray(theta)

        squeezed_input = False
        if theta.ndim == 1:
            theta = theta[None, :]
            squeezed_input = True

        lon = jnp.deg2rad(self.adapter.flat_view(theta, 'dipole_longitude'))
        colat = jnp.pi / 2 - jnp.deg2rad(self.adapter.flat_view(theta, 'dipole_latitude'))

        mean_density_norm = (
            self.adapter.flat_view(theta, 'mean_density')
            - self.theta_mean[self.adapter.key_index('mean_density')]
        ) / self.theta_std[self.adapter.key_index('mean_density')]
        observer_speed_norm = (
            self.adapter.flat_view(theta, 'observer_speed')
            - self.theta_mean[self.adapter.key_index('observer_speed')]
        ) / self.theta_std[self.adapter.key_index('observer_speed')]

        components = []
        for key in self.adapter.keys:
            if key == 'mean_density':
                components.append(mean_density_norm)
            elif key == 'observer_speed':
                components.append(observer_speed_norm)
            elif key == 'dipole_longitude':
                components.append(lon)
            elif key == 'dipole_latitude':
                components.append(colat)
            else:
                raise KeyError(f'Unexpected theta key {key} in adapter.')

        t_transformed = jnp.stack(components, axis=-1)

        idx_mean = self.adapter.key_index('mean_density')
        idx_speed = self.adapter.key_index('observer_speed')

        abslogdet_scalar = (
              (jnp.log(jnp.pi) - np.log(180))
            + (jnp.log(jnp.pi) - np.log(180))
            - jnp.log(self.theta_std[idx_mean])
            - jnp.log(self.theta_std[idx_speed])
        )
        abslogdet_per_batch = jnp.full(
            (t_transformed.shape[0],), 
            abslogdet_scalar, 
            dtype=theta.dtype
        )

        if squeezed_input:
            t_transformed = jnp.squeeze(t_transformed, axis=0)
            abslogdet_per_batch = jnp.squeeze(abslogdet_per_batch, axis=0)

        return t_transformed, abslogdet_per_batch

    def _inverse_and_log_det_zscore(
        self,
        transformed_theta: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.theta_mean is not None
        assert self.theta_std is not None
        transformed_theta = jnp.asarray(transformed_theta)

        squeezed_input = False
        if transformed_theta.ndim == 1:
            transformed_theta = transformed_theta[None, :]
            squeezed_input = True

        idx_mean = self.adapter.key_index('mean_density')
        idx_speed = self.adapter.key_index('observer_speed')
        idx_lon = self.adapter.key_index('dipole_longitude')
        idx_lat = self.adapter.key_index('dipole_latitude')

        mean_density_norm = transformed_theta[..., idx_mean]
        observer_speed_norm = transformed_theta[..., idx_speed]
        lon = transformed_theta[..., idx_lon]
        colat = transformed_theta[..., idx_lat]

        mean_density = (
            mean_density_norm * self.theta_std[idx_mean]
            + self.theta_mean[idx_mean]
        )
        observer_speed = (
            observer_speed_norm * self.theta_std[idx_speed]
            + self.theta_mean[idx_speed]
        )

        dipole_longitude = jnp.rad2deg(lon)
        if self.wrap_longitude:
            dipole_longitude = self._wrap_longitude(dipole_longitude)

        dipole_latitude = jnp.rad2deg(jnp.pi / 2 - colat)
        if self.reflect_latitude:
            dipole_latitude = self._reflect_latitude(dipole_latitude)

        abslogdet_scalar = (
            - (jnp.log(jnp.pi) - np.log(180))
            - (jnp.log(jnp.pi) - np.log(180))
            + jnp.log(self.theta_std[idx_mean])
            + jnp.log(self.theta_std[idx_speed])
        )

        components = []
        for key in self.adapter.keys:
            if key == 'mean_density':
                components.append(mean_density)
            elif key == 'observer_speed':
                components.append(observer_speed)
            elif key == 'dipole_longitude':
                components.append(dipole_longitude)
            elif key == 'dipole_latitude':
                components.append(dipole_latitude)
            else:
                raise KeyError(f'Unexpected theta key {key} in adapter.')

        theta = jnp.stack(components, axis=-1)
        abslogdet_per_batch = jnp.full((theta.shape[0],), abslogdet_scalar, dtype=theta.dtype)

        if squeezed_input:
            theta = jnp.squeeze(theta, axis=0)
            abslogdet_per_batch = jnp.squeeze(abslogdet_per_batch, axis=0)

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
