from typing import Callable, Optional, Tuple, cast, Literal
from blackjax.types import Array
from healpy import isnsideok, nside2npix
from numpy.core.multiarray import ndarray
from numpy.typing import NDArray
import torch
from torch import Tensor, device
from nflows.transforms.base import Transform
from torch.utils.data import Dataset, DataLoader, random_split
import math
import numpy as np
from dipolesbi.tools.healpix_helpers import build_funnel_steps, split_off_details
from dipolesbi.tools.utils import jax_sph2cart, np_sph2cart_unitsphere, softplus_pos
import torch.nn.functional as F
from abc import ABC, abstractmethod
import healpy as hp
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

class LogAffineTransform(Transform):
    def __init__(self, mu: Tensor, sigma: Tensor):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x, context=None) -> Tuple[Tensor, Tensor]: # type: ignore
        z = (torch.log1p(x) - self.mu) / self.sigma
        logabsdet = (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return z, logabsdet

    def inverse(self, z, context=None) -> Tuple[Tensor, Tensor]: # type: ignore
        x = torch.expm1(z * self.sigma + self.mu)
        logabsdet = -(-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return (-torch.log1p(x) - torch.log(self.sigma)).sum(dim=-1)

class AnscombeTransform(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, x, context=None): # type: ignore
        z = 2 * torch.sqrt(x + 0.375)
        logabsdet = self.log_abs_det_jacobian(x, z)
        return z, logabsdet

    def inverse(self, z, context=None): # type: ignore
        x = z**2 / 4 - 0.375
        logabsdet = -self.log_abs_det_jacobian(x, z)
        return x, logabsdet

    def log_abs_det_jacobian(self, x, z):
        return - 0.5 * torch.log(x + 0.375).sum(dim=-1)

class LearnableSOn(torch.nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.A = torch.nn.Parameter(0.01 * torch.randn(n,n))

    def forward(self):
        S = self.A - self.A.T # subtracting transpose gives skew-sym. (any nxn)
        return torch.matrix_exp(S) # Q ∈ SO(4)

class Unimodular:
    def __init__(self, device: str = 'cpu') -> None:
        self.U = torch.tensor(
            [[1,1,1,1],
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0]],
            dtype=torch.float32,
            device=device
        )

        self.Uinv = torch.tensor(
            [[ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1],
            [ 1,-1,-1,-1]], 
            dtype=torch.float32,
            device=device
        )
        
        assert (self.U @ self.Uinv == torch.eye(4)).all()

    def to(self, device: str | device) -> None:
        self.U = self.U.to(device)
        self.Uinv = self.Uinv.to(device)

class IntShear4(torch.nn.Module):
    def __init__(self, pairs, k_max=3):
        """
        pairs: list of (i,j) with i != j, e.g. [(1,0),(2,1),(3,2),(2,0)]
        k_max: cap on |k| to keep moves mild (and easy to learn)
        """
        super().__init__()
        self.pairs = pairs
        self.k_raw = torch.nn.Parameter(torch.zeros(len(pairs)))  # real params
        self.k_max = float(k_max)
        self.k_int_buffer = None

    def _hysteresis_round(self, x: Tensor, prev_int=None, margin=0.1):
        '''
        Try to mitigate nll curve spikes during training due to integer flipping.
        '''
        # x: real value; prev_int: last integer used (or None)
        x_det = x.detach()
        base = torch.round(x_det)
        if prev_int is None:
            return (x - x_det) + base  # standard STE at start
        # Keep the previous integer unless we exceed a margin beyond the boundary
        delta = x_det - prev_int
        update = torch.where(delta > 0.5 + margin, prev_int + 1,
                 torch.where(delta < -0.5 - margin, prev_int - 1, prev_int))
        return (x - x_det) + update

    def _ints(self):
        # tanh-cap + straight-through round to get small integers
        k_real = self.k_max * torch.tanh(self.k_raw)
        k_int = self._hysteresis_round(k_real, prev_int=self.k_int_buffer, margin=0.2)
        self.k_int_buffer = k_int.detach()
        # k_int  = (k_real - k_real.detach()) + torch.round(k_real.detach())
        return k_int

    def matrix(self, dtype=torch.float32, device=None):
        k_int = self._ints()
        M = torch.eye(4, dtype=dtype, device=device)
        for (idx, (i,j)) in enumerate(self.pairs):
            S = torch.eye(4, dtype=dtype, device=device)
            S[i, j] = S[i, j] + k_int[idx]
            M = S @ M   # left-multiply: apply in listed order
        return M  # det = 1

    def inv_matrix(self, dtype=torch.float32, device=None):
        k_int = self._ints()
        Minv = torch.eye(4, dtype=dtype, device=device)
        # inverse: reverse order, negate k
        for (idx, (i,j)) in reversed(list(enumerate(self.pairs))):
            S = torch.eye(4, dtype=dtype, device=device)
            S[i, j] = S[i, j] - k_int[idx]
            Minv = S @ Minv
        return Minv

class LearnableNormaliser(torch.nn.Module):
    def __init__(self, init_eps: float = 0.001) -> None:
        super().__init__()
        self.log_kappa = torch.nn.Parameter(torch.zeros(1))
        self.log_eps = torch.nn.Parameter(torch.log(torch.as_tensor(init_eps)))
        self.beta0 = torch.nn.Parameter(torch.zeros(1))

    def forward(
            self, 
            coarse_coefficients: Tensor, 
            detail_coefficients: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        kappa = torch.exp(self.log_kappa)
        eps = torch.exp(self.log_eps)
        a_pos = F.softplus(coarse_coefficients)
        s = torch.sqrt(eps + kappa * a_pos)
        m = self.beta0 + torch.ones_like(coarse_coefficients) # (n_batches, n_pix)

        a_norm = coarse_coefficients
        d_norm = (detail_coefficients - m.unsqueeze(-1)) / s.unsqueeze(-1) # (n_batches, n_pix, n_coefficients)

        n_detail_coefficients = detail_coefficients.shape[-1]
        logdet = - n_detail_coefficients * torch.sum(torch.log(s + 1e-12), dim=1) # (n_batches,)

        return a_norm, d_norm, logdet

class HaarWaveletTransform(InvertibleDataTransform):
    def __init__(
            self, 
            first_nside: int, 
            last_nside: int = 1,
            post_normalise: bool = False,
            matrix_type: Literal['hadamard', 'sparse_average'] = 'hadamard',
            normalise_details: bool = True,
            n_chunks: int = 1,
            xp=np,
            dtype=None
    ) -> None:
        # Array backend (NumPy by default; can be jax.numpy)
        self.xp = xp
        # Promote to consistent precision (float64) unless explicitly provided
        # this is ESSENTIAL for the high_lambda unit test to pass in the jax
        # backend version of this class
        self.dtype = dtype if dtype is not None else getattr(self.xp, 'float64', float)

        # Hard guardrail: if using JAX backend, ensure x64 is enabled and effective
        try:
            is_jax_backend = (self.xp is jnp)
        except Exception:
            is_jax_backend = False

        if is_jax_backend:
            try:
                from jax import config as jax_config  # type: ignore
                x64_flag = False
                try:
                    x64_flag = bool(jax_config.read("jax_enable_x64"))
                except Exception:
                    # Fallback: probe dtype behavior directly
                    pass

                probe = jnp.asarray(0.0, dtype=jnp.float64)
                effective_float64 = (probe.dtype == jnp.float64)
                if not (x64_flag and effective_float64):
                    raise RuntimeError(
                        "HaarWaveletTransformJax requires JAX 64-bit. "
                        "Enable it via env var JAX_ENABLE_X64=true before importing jax, "
                        "or call jax.config.update('jax_enable_x64', True) at process start."
                    )
            except Exception:
                # If the check itself fails, raise with context
                raise

        self.matrix_type = matrix_type
        self.H = 0.5 * self.xp.asarray(
            [[1., 1. , 1. ,  1.],
             [1., 1. , -1., -1.],
             [1., -1., 1. , -1.],
             [1., -1., -1.,  1.]]
        , dtype=self.dtype)
        self.H_inv = self.xp.linalg.inv(self.H)

        self.A = self.xp.asarray(
            [[1., 1., 1., 1.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]
        , dtype=self.dtype)
        self.A_inv = self.xp.linalg.inv(self.A)

        if matrix_type == 'hadamard':
            self.Q = self.H
            self.Q_inv = self.H_inv
        elif matrix_type == 'sparse_average':
            self.Q = self.A.T
            self.Q_inv = self.A_inv.T
        else:
            raise Exception(f'Unrecognised matrix type {self.matrix_type}.')

        self.mu_at_level: list[list[NDArray]] = []
        self.std_at_level: list[list[NDArray]] = []

        self.first_nside = first_nside
        self.last_nside = last_nside

        nside = first_nside
        self.npix_per_level = []

        # stop before nside = 1: out map has nside=2 resolution coarse and details
        # i.e. 12 coarse, 12 x 3 details, total of 48
        while nside > 1:
            self.npix_per_level.append(hp.nside2npix(nside))
            nside //= 2

        self.first_npix = self.npix_per_level[0]
        self.last_npix = 12 * last_nside ** 2

        # Use math for integer-safe levels computation (backend-agnostic)
        self.n_levels = int(math.log2(first_nside) - math.log2(last_nside))
        self.downscale_factors = self.n_levels * [4]

        # 3 detail coefficients; ell levels per coefficient
        # at every level ell, we have nside(ell) total pixels
        # 3/4 are details, the other 1/4 coarse
        self.mu_at_level_post = self._make_post_dict()
        self.std_at_level_post = self._make_post_dict()

        self.empty_norm_stats_flag: bool = True
        self.post_normalise: bool = post_normalise
        self.normalise_details: bool = normalise_details
        self.n_chunks = n_chunks
        self.blocks = self._build_surjective_blocks()

    def _make_post_dict(self) -> dict:
        xp = self.xp
        return {
            'coarse': xp.nan * xp.empty(self.npix_per_level[-1] // 4, dtype=self.dtype),
            'detail': {i:
                {
                    j: xp.nan * xp.empty(self.npix_per_level[::-1][j] // 4, dtype=self.dtype)
                    for j in range(self.n_levels)
                }
            for i in range(3)
            }
        }

    def _post_dict_is_empty(self, check_detail: bool = True, check_coarse: bool = True) -> bool:
        """
        Checks that all arrays in the mu_post_dict contain only np.nan values.
        """
        # Check 'coarse' array
        if check_coarse:
            if not self.xp.isnan(self.mu_at_level_post['coarse']).all():
                return False

            if not self.xp.isnan(self.std_at_level_post['coarse']).all():
                return False

        if check_detail:
            for i in self.mu_at_level_post['detail']:
                for j in self.mu_at_level_post['detail'][i]:
                    arr = self.mu_at_level_post['detail'][i][j]
                    if not self.xp.isnan(arr).all():
                        return False

            for i in self.std_at_level_post['detail']:
                for j in self.std_at_level_post['detail'][i]:
                    arr = self.std_at_level_post['detail'][i][j]
                    if not self.xp.isnan(arr).all():
                        return False
        return True

    def __repr__(self) -> str:
        backend_name = getattr(self.xp, '__name__', str(self.xp))
        return (
            f"HaarWaveletTransform("
            f"first_nside={self.first_nside}, "
            f"last_nside={self.last_nside}, "
            f"first_npix={self.first_npix}, "
            f"last_npix={self.last_npix}, "
            f"n_levels={self.n_levels}, "
            f"downscale_factors={self.downscale_factors}, "
            f"mu_at_level={self.mu_at_level}, "
            f"std_at_level={self.std_at_level}, "
            f"post_normalise={self.post_normalise}, "
            f"backend={backend_name}"
            f")"
        )

    @property
    def parents_at_levels(self):
        P_levels = []
        cur = self.first_npix
        for dscale_factor in self.downscale_factors:
            P_levels.append(cur // dscale_factor)
            cur //= dscale_factor
        return P_levels

    def make_unnormalise_details_func(
            self, 
            level: int | Literal['all']
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        # in mu_at_level_post, level_idx 0 corresponds to the finest level
        # being the first one thrown away in the heirarchical SSNLE
        # note details are interleaved
        # [d1_0, d2_0, d3_0,  d1_1, d2_1, d3_1, ..., d1_{P-1}, d2_{P-1}, d3_{P-1}]

        xp = self.xp
        if level == 'all':
            L = self.n_levels  # = log2(first_nside) - log2(last_nside)
            sizes = []
            mu_levels = []
            std_levels = []
            for lvl_coarse_to_fine in range(L):
                idx = L - 1 - lvl_coarse_to_fine

                mu1 = xp.asarray(self.mu_at_level_post['detail'][0][idx])
                mu2 = xp.asarray(self.mu_at_level_post['detail'][1][idx])
                mu3 = xp.asarray(self.mu_at_level_post['detail'][2][idx])

                sd1 = xp.asarray(self.std_at_level_post['detail'][0][idx])
                sd2 = xp.asarray(self.std_at_level_post['detail'][1][idx])
                sd3 = xp.asarray(self.std_at_level_post['detail'][2][idx])

                P = int(mu1.shape[0])
                sizes.append(P)

                mu_levels.append(xp.stack([mu1, mu2, mu3], axis=-1))   # (P, 3)
                std_levels.append(xp.stack([sd1, sd2, sd3], axis=-1))  # (P, 3)

            def _unnormalise_all(z):
                # z: (B, sum_l 3*P_l), concatenated coarse→fine, interleaved within each level
                assert z.ndim == 2, f"Expected (B, N), got {z.shape}"
                B, N = z.shape

                out_parts = []
                offset = 0
                for P, mu, sd in zip(sizes, mu_levels, std_levels):
                    span = 3 * P
                    block = z[:, offset:offset + span]
                    assert block.shape[1] == span, "Details length mismatch vs stored stats"

                    # Interleaved → (B, P, 3)
                    blk3 = block.reshape(B, P, 3)
                    # Unnormalise per-pixel, per-channel
                    x3 = blk3 * sd[None, :, :] + mu[None, :, :]

                    out_parts.append(xp.reshape(x3, (B, span)))
                    offset += span

                assert offset == N, f"Consumed {offset} coeffs but z had {N}"
                return xp.concatenate(out_parts, axis=1)

            return _unnormalise_all

        else:
            mu1, mu2, mu3 = (self.mu_at_level_post['detail'][i][level] for i in range(3))
            std1, std2, std3 = (self.std_at_level_post['detail'][i][level] for i in range(3))
            n_detail1s = mu1.shape[0]

            def _unnormalise(z: jnp.ndarray):
                assert z.ndim == 2
                assert 3 * n_detail1s == z.shape[-1]

                B, threeP = z.shape
                P = threeP // 3

                # reshape to (B, P, 3): channels last
                z3 = z.reshape(B, P, 3)

                mu  = xp.stack([mu1,  mu2,  mu3],  axis=-1)  # (P, 3)
                std = xp.stack([std1, std2, std3], axis=-1)  # (P, 3)

                x3 = z3 * std[None, :, :] + mu[None, :, :]
                x_final = x3.reshape(B, threeP)

                assert x_final.shape == z.shape, (
                    f'Shapes do not match, ({x_final.shape} vs. {z.shape}).'
                )

                return x_final
            
            return _unnormalise
            
    def clear(self) -> None:
        self.mu_at_level = []
        self.std_at_level = []
        self.mu_at_level_post = self._make_post_dict()
        self.std_at_level_post = self._make_post_dict()
        self.empty_norm_stats_flag = True

    def forward_and_log_det(self, data: NDArray) -> tuple[NDArray, NDArray]:
        # Ensure backend array; remember original dtype for optional cast-back
        orig_dtype = getattr(data, 'dtype', None)
        data = self.xp.asarray(data, dtype=self.dtype)
        z, logdet = self._cycle_healpix_tree(data)
        try:
            if orig_dtype == getattr(self.xp, 'float32'):
                z = z.astype(self.xp.float32)
                logdet = logdet.astype(self.xp.float32)
        except Exception:
            pass
        return z, logdet

    def inverse_and_log_det(self, transformed_data: NDArray) -> tuple[NDArray, NDArray]:
        orig_dtype = getattr(transformed_data, 'dtype', None)
        transformed_data = self.xp.asarray(transformed_data, dtype=self.dtype)
        x, logdet = self._reverse_cycle_healpix_tree(transformed_data)
        try:
            if orig_dtype == getattr(self.xp, 'float32'):
                x = x.astype(self.xp.float32)
                logdet = logdet.astype(self.xp.float32)
        except Exception:
            pass
        return x, logdet

    def _build_surjective_blocks(self) -> list[tuple[int, int]]:
        steps = split_off_details(self.first_nside, self.last_nside)
        return steps

    # TODO: ok you need to verify this function
    def _cycle_healpix_tree(
            self, 
            y: NDArray
    ) -> tuple[NDArray, NDArray]:
        batches, npix_fine = y.shape
        cur_npix = npix_fine
        coefficients_fine2coarse: list[NDArray] = [] 
        a_list: list[NDArray] = []
        logdets_at_each_level: list[NDArray] = []
        post_norm_logdet: NDArray = self.xp.zeros(batches)
        
        if self.post_normalise and self.empty_norm_stats_flag:
            assert batches > 1, 'More than 1 batch needed to define std.'
        
        logdet = self.xp.zeros(batches)
        downstream_coefficients = y

        for lvl, factor in enumerate(self.downscale_factors):
            per_level_logdet = self.xp.zeros(())

            P = cur_npix // factor
            child_pixels = downstream_coefficients.reshape(batches, P, factor)

            z = self._forward_matrix_product(child_pixels) # (n_batches, n_pix // 4, 4)

            # Compute stats per channel
            coarse_coeffs = z[..., 0]
            d1 = z[..., 1]
            d2 = z[..., 2]
            d3 = z[..., 3]

            cur_mus = []
            cur_stds = []
            chans = [coarse_coeffs, d1, d2, d3]
            if self.empty_norm_stats_flag:
                for coef in chans:
                    cur_mus.append(coef.mean())
                    cur_stds.append(coef.std())
                mu_vec = self.xp.stack(cur_mus, axis=-1)      # (4,)
                sigma_vec = self.xp.stack(cur_stds, axis=-1)  # (4,)
            else:
                mu_vec = self.xp.stack([self.mu_at_level[lvl][i] for i in range(4)], axis=-1)
                sigma_vec = self.xp.stack([self.std_at_level[lvl][i] for i in range(4)], axis=-1)

            # Normalise all channels if required, without in-place writes
            if not self.post_normalise:
                z = (z - mu_vec) / sigma_vec

            # Log-det accumulates over channels
            per_level_logdet += - P * self.xp.log(sigma_vec).sum()

            if self.empty_norm_stats_flag:
                self.mu_at_level.append(cur_mus)
                self.std_at_level.append(cur_stds)

            logdets_at_each_level.append(per_level_logdet)
            logdet += per_level_logdet

            coefficients_fine2coarse.append(z)
            a_list.append(z[..., 0])
            downstream_coefficients = z[..., 0]
            cur_npix = P

        a_coarse = downstream_coefficients # (batches, out_npix)

        if self.post_normalise:
            if self.empty_norm_stats_flag:
                mean_coarse = a_coarse.mean(axis=0) # across the batch axis
                std_coarse = a_coarse.std(axis=0)
            else:
                mean_coarse = self.mu_at_level_post['coarse']
                std_coarse = self.std_at_level_post['coarse']

            a_coarse = ( a_coarse - mean_coarse ) / std_coarse

            post_norm_logdet += -self.xp.log(std_coarse).sum(axis=-1) # sum over pixels

            if self.empty_norm_stats_flag:
                assert self._post_dict_is_empty()

                self.mu_at_level_post['coarse'] = mean_coarse
                self.std_at_level_post['coarse'] = std_coarse

        # Build z: keep a_coarse, then details from coarse->fine
        # (drop the a's there)
        z_parts = [a_coarse.reshape(batches, -1)]  # list[ (B, out_npix) ]
        lvl_idx = self.n_levels - 1

        if self.empty_norm_stats_flag:
            assert self._post_dict_is_empty(check_coarse=False)

        for lvl, y in enumerate(reversed(coefficients_fine2coarse)): # now coarse->fine
            
            if self.post_normalise:
                if self.empty_norm_stats_flag:
                    mean_d1 = y[..., 1].mean(axis=0)
                    mean_d2 = y[..., 2].mean(axis=0)
                    mean_d3 = y[..., 3].mean(axis=0)
                    std_d1 = y[..., 1].std(axis=0)
                    std_d2 = y[..., 2].std(axis=0)
                    std_d3 = y[..., 3].std(axis=0)
                else:
                    mean_d1 = self.mu_at_level_post['detail'][0][lvl_idx - lvl]
                    mean_d2 = self.mu_at_level_post['detail'][1][lvl_idx - lvl]
                    mean_d3 = self.mu_at_level_post['detail'][2][lvl_idx - lvl]

                    std_d1 = self.std_at_level_post['detail'][0][lvl_idx - lvl]
                    std_d2 = self.std_at_level_post['detail'][1][lvl_idx - lvl]
                    std_d3 = self.std_at_level_post['detail'][2][lvl_idx - lvl]
                
                if self.normalise_details:
                    means = self.xp.stack([mean_d1, mean_d2, mean_d3], axis=-1)  # (P,3)
                    stds  = self.xp.stack([std_d1,  std_d2,  std_d3],  axis=-1)  # (P,3)
                    y_details = (y[..., 1:] - means[None, :, :]) / stds[None, :, :]
                    y = self.xp.concatenate([y[..., :1], y_details], axis=-1)

                    post_norm_logdet += - self.xp.log(std_d1).sum(axis=-1)
                    post_norm_logdet += - self.xp.log(std_d2).sum(axis=-1)
                    post_norm_logdet += - self.xp.log(std_d3).sum(axis=-1)

                if self.empty_norm_stats_flag:
                    self.mu_at_level_post['detail'][0][lvl_idx - lvl] = mean_d1
                    self.mu_at_level_post['detail'][1][lvl_idx - lvl] = mean_d2
                    self.mu_at_level_post['detail'][2][lvl_idx - lvl] = mean_d3

                    self.std_at_level_post['detail'][0][lvl_idx - lvl] = std_d1
                    self.std_at_level_post['detail'][1][lvl_idx - lvl] = std_d2
                    self.std_at_level_post['detail'][2][lvl_idx - lvl] = std_d3

            z_parts.append(y[..., 1:].reshape(batches, -1))
        z = self.xp.concatenate(z_parts, axis=1)

        self.empty_norm_stats_flag = False
        if self.post_normalise:
            self.logdet = post_norm_logdet
            return z, post_norm_logdet
        else:
            self.logdet = logdet
            return z, logdet

    # TODO: this one too
    def _reverse_cycle_healpix_tree(self,
        transformed_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batches, npix = transformed_data.shape
        P_levels = self.parents_at_levels

        # z = [a_coarse | d_coarse ... d_fine ]
        offset = self.last_npix
        a = transformed_data[:, :offset] # (B, out_npix)

        if self.post_normalise:
            a = a * self.std_at_level_post['coarse'] + self.mu_at_level_post['coarse']

        # iterate from high resolution to low resolution
        details_coarse2fine = []
        for npix, dscale_factor in zip(P_levels[::-1], self.downscale_factors[::-1]):
            dsize = npix * (dscale_factor - 1) # detail coefficients always 1 less 
            d = transformed_data[:, offset:offset+dsize].reshape(
                batches, npix, dscale_factor - 1
            )
            offset += dsize
            details_coarse2fine.append(d)

        # reconstruct coarse->fine
        upstream_data = a

        # reconstruct coarse to fine
        for lvl, d_out in enumerate(details_coarse2fine):
            # y_out is what forward produced after coupling: [a | d_out]
            a_here = upstream_data[..., None]
            a_here = cast(Array, a_here)
            y_pre = self.xp.concatenate([a_here, d_out], axis=-1) # (B, P_l, n_coefficients)

            lvl_idx = self.n_levels - 1
            if not self.post_normalise:
                mu_vec = self.xp.stack([
                    self.mu_at_level[-1 - lvl][i] for i in range(4)
                ], axis=-1)
                sigma_vec = self.xp.stack([
                    self.std_at_level[-1 - lvl][i] for i in range(4)
                ], axis=-1)
                y_pre = y_pre * sigma_vec + mu_vec
            else:
                if self.normalise_details:
                    means = self.xp.stack([
                        self.mu_at_level_post['detail'][0][lvl_idx - lvl],
                        self.mu_at_level_post['detail'][1][lvl_idx - lvl],
                        self.mu_at_level_post['detail'][2][lvl_idx - lvl],
                    ], axis=-1)  # (P,3)
                    stds = self.xp.stack([
                        self.std_at_level_post['detail'][0][lvl_idx - lvl],
                        self.std_at_level_post['detail'][1][lvl_idx - lvl],
                        self.std_at_level_post['detail'][2][lvl_idx - lvl],
                    ], axis=-1)  # (P,3)
                    y_details = y_pre[..., 1:] * stds[None, :, :] + means[None, :, :]
                    y_pre = self.xp.concatenate([y_pre[..., :1], y_details], axis=-1)

            data = self._inverse_matrix_product(y_pre)

            # upsample to next finer 'a'
            upstream_data = data.reshape(batches, -1)

        x_rec = upstream_data
        inverse_logdet = -self.logdet # hack slightly and assume forward has been called
        return x_rec, inverse_logdet

    def _forward_matrix_product(self, v: NDArray) -> NDArray:
        return v @ self.Q

    def _inverse_matrix_product(self, z: NDArray) -> NDArray:
        return z @ self.Q_inv

# Convenience subclass that uses JAX as the backend
class HaarWaveletTransformJax(HaarWaveletTransform):
    def __init__(self,
                 first_nside: int,
                 last_nside: int = 1,
                 post_normalise: bool = False,
                 matrix_type: Literal['hadamard', 'sparse_average'] = 'hadamard',
                 normalise_details: bool = True,
                 n_chunks: int = 1):
        super().__init__(
            first_nside=first_nside,
            last_nside=last_nside,
            post_normalise=post_normalise,
            matrix_type=matrix_type,
            normalise_details=normalise_details,
            n_chunks=n_chunks,
            xp=jnp
        )

class HealpixSOPyramid(torch.nn.Module):
    def __init__(
            self,
            nside_fine: int = 16,
            nside_at_levels: Optional[list[int]] = None,
            method: Literal['SO', 'integer'] = 'SO'
    ) -> None:
        super().__init__()
        self.nside_fine = nside_fine
        self.npix_fine = nside2npix(self.nside_fine)
        print(f'Input NSIDE: {self.nside_fine}')
        self.dtype = torch.float32

        # compute nside per level; by default nside is reduced by a factor of
        # 2 per level unless downgrade_factors is set
        if nside_at_levels is None:
            self.n_downscales = int(np.log2(nside_fine))
            self.level_nsides = list(
                reversed([2**n for n in range(self.n_downscales+1)])
            )
        else:
            # assert nside_at_levels[-1] == 1, 'Final nside must be 1.'
            assert nside_at_levels[0] == nside_fine, (
                'First nside must be equal to input map nside.'
            )
            nside_ok = cast(NDArray[np.bool_], isnsideok(nside_at_levels))
            assert nside_ok.all(), 'Invalid nside in nside_at_levels.'

            self.level_nsides = nside_at_levels

        self.out_nside = self.level_nsides[-1]
        self.out_npix = nside2npix(self.out_nside)
            
        self.downscale_factors: list[int] = [
            nside2npix(in_nside) // nside2npix(out_nside)
            for in_nside, out_nside in zip(
                self.level_nsides[:-1], self.level_nsides[1:]
            )
        ]
        print(f'NSIDE levels: {self.level_nsides}')
        print(f'Downscale factors: {self.downscale_factors}') # len(level_nsides)-1

        # for each level, we choose a matrix in SO(n) such that n corresponds
        # to the downscale factor from coarse to fine
        if method == 'SO':
            self.SO_matrices = torch.nn.ModuleList(
                [LearnableSOn(n) for n in self.downscale_factors]
            )
            self.mu_list = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.zeros(n, dtype=self.dtype))
                    for n in self.downscale_factors
                ]
            )
            self.log_sigma_list = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.zeros(n, dtype=self.dtype))
                    for n in self.downscale_factors
                ]
            ) 
        elif method == 'integer':
            self.unimod = Unimodular()
            self.shears = torch.nn.ModuleList(
                [
                    IntShear4(pairs=[(1,0),(2,1),(3,2),(2,0)], k_max=3)
                    for _ in self.downscale_factors
                ]
            )        # # Gaussian stats for loss function, one (mu, sigma) per level

            # 1) Coarse (final out_npix entries): Negative Binomial params
            #    Store as (r, p) in a stable way: logr, logitp
            #    Shapes: (1, out_npix) so they broadcast over batch
            self.nb_logr   = torch.nn.Parameter(
                torch.zeros(1, self.out_npix, dtype=self.dtype)
            )
            self.nb_logitp = torch.nn.Parameter(
                torch.zeros(1, self.out_npix, dtype=self.dtype)
            )

            self.w0 = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.tensor(0., dtype=self.dtype))
                    for _ in self.downscale_factors
                ]
            )
            self.w1 = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.tensor(0.5, dtype=self.dtype))
                    for _ in self.downscale_factors
                ]
            )
        else:
            raise Exception('Unrecognised method.')

        self.method = method

    @property
    def levels(self):
        return len(self.level_nsides) - 1  # number of downsamplings

    @property
    def parents_at_levels(self):
        P_levels = []
        cur = self.npix_fine
        for dscale_factor in self.downscale_factors:
            P_levels.append(cur // dscale_factor)
            cur //= dscale_factor
        return P_levels

    def forward(self, x: torch.Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]], list, Tensor]:
        """
        :param x: (n_batches, n_pix) in nest ordering
        :returns:
          z: (batches, npix) with layout [a(NSIDE=1) | details (coarse->fine)]
          per_level_coefficients: list length=levels, each (B, P, n_coefficients) 
            ordered fine->coarse.
          logdets_at_each_level: value of log determinant Jacobian at each level
          logdet: (n_batches,) value of log determinant Jacobian, 0 by construction
        """
        batches, npix = x.shape
        assert npix == self.npix_fine
        downstream_coefficients = x
        cur_npix = self.npix_fine

        coefficients_fine2coarse: list[tuple[Tensor, Tensor]] = [] 
        a_list: list[Tensor] = []
        logdets_at_each_level: list[float] = []

        # LOGIC
        # - for each healpix map, group the pixels corresponding to the children
        #   of the lower res map (potentially many orders lower)

        # - this group goes into 3rd dimension
            # (shape batches, pix / n_coefficients, n_coefficients)

        # - hit each pixel four vector with matrix vector product, the matrix
        #   being the learnt SOn matrix; this extracts n coefficients which
        #   live in the 3rd dimension; this is done batchwise with matmul

        # - the n coefficients are appended to coefficients_fine2coarse,
            # the a coefficients (first coefficients in 3rd dim) are passed
            # downstream to lower res

        # - at the end, append the coarsest a coefficients + all the n-1 detail
        #   coefficients from each level; this ensures the output shape of z
        #   has the same pixel count as the input shape

        # Apply SO(n) to every n-child group, fine -> coarse
        logdet = torch.zeros(batches, dtype=x.dtype, device=x.device)

        for level, factor in zip(range(self.levels), self.downscale_factors):
            P = cur_npix // factor
            child_pixels = downstream_coefficients.view(batches, P, factor)
            device = child_pixels.device

            # Poisson rate proxy
            s = child_pixels.sum(dim=-1)

            if self.method == 'SO':
                Q = self.SO_matrices[level]()
                y = child_pixels @ Q.T
            else:
                self.unimod.to(device)
                U = self.unimod.U
                # Q = self.shears[level].matrix(device=device) # type: ignore
                y = int_haar4_forward(child_pixels)
                # y = child_pixels @ U.T
                # y = torch.matmul(child_pixels, U.T)

            a = y[..., 0]
            a_for_cond = y[..., :1]
            d = y[..., 1:]

            y = torch.cat([a_for_cond, d], dim=-1)

            per_level_logdet = 0.
            logdets_at_each_level.append(per_level_logdet)
            logdet += per_level_logdet

            coefficients_fine2coarse.append((y, s))
            a_list.append(a)
            downstream_coefficients = a
            cur_npix = P

        a_coarse = downstream_coefficients # (batches, out_npix)

        # Build z: keep a_coarse, then details from coarse->fine
        # (drop the a's there)
        z_parts = [a_coarse.reshape(batches, -1)]  # list[ (B, out_npix) ]
        for y, _ in reversed(coefficients_fine2coarse): # now coarse->fine
            z_parts.append(y[..., 1:].reshape(batches, -1))
        z = torch.cat(z_parts, dim=1)

        per_level_coefficients = coefficients_fine2coarse  # fine to coarse ordering
        return z, per_level_coefficients, logdets_at_each_level, logdet

    def inverse(self, z: torch.Tensor):
        """
        Exact inverse (uses the same SO(n) matrix); logdet=0.
        """
        batches, npix = z.shape
        assert npix == self.npix_fine

        # parents per fine->coarse level
        P_levels = self.parents_at_levels

        # parse z
        # z = [a_coarse | d_coarse ... d_fine ]
        offset = self.out_npix
        a = z[:, :offset]  # (B, out_npix)

        details_coarse2fine = []
        # iterate from high resolution to low resolution
        for npix, dscale_factor in zip(P_levels[::-1], self.downscale_factors[::-1]):
            dsize = npix * (dscale_factor - 1) # detail coefficients always 1 less 
            d = z[:, offset:offset+dsize].view(batches, npix, dscale_factor - 1)
            offset += dsize
            details_coarse2fine.append(d)

        # reconstruct coarse->fine
        downstream_coefficients = a
        inv_logdet = torch.zeros(batches, dtype=z.dtype, device=z.device)

        # reconstruct coarse to fine
        for lvl, d_out in enumerate(details_coarse2fine):
            # y_out is what forward produced after coupling: [a | d_out]
            a_here = downstream_coefficients.unsqueeze(-1) # (B, npix_level, 1)
            y_pre = torch.cat([a_here, d_out], dim=-1)  # (B, P_l, n_coefficients)

            if self.method == 'SO':
                # undo the SO(n) matrix product
                Q = self.SO_matrices[self.levels - 1 - lvl]() # inverse of (· @ Q.T) is (· @ Q)
                coefficients = torch.matmul(y_pre, Q) # (B, npix_level, n_coefficients)
            else:
                # Uinv = self.unimod.Uinv
                Q = self.shears[self.levels - 1 - lvl].inv_matrix() # type: ignore
                Uinv = self.unimod.Uinv
                coefficients = int_haar4_inverse(y_pre)
            # coefficients = torch.matmul(y_pre, Q.t())
            # coefficients = torch.matmul(coefficients, Uinv.t())

            # upsample to next finer 'a'
            downstream_coefficients = coefficients.reshape(batches, -1)

        x_rec = downstream_coefficients
        return x_rec, inv_logdet

class MapDataset(Dataset):
    def __init__(self, D_all: torch.Tensor):
        """
        Interface for pytorch.

        D_all: (N, Npix) torch tensor in NEST order.
        """
        self.D = D_all
    def __len__(self): return self.D.shape[0]
    def __getitem__(self, i): return self.D[i]

def gaussian_nll(y, mu, log_sigma):
    # diagonal Gaussian NLL on last dim (size 4)
    inv_var = torch.exp(-2*log_sigma)
    nll = 0.5 * ((y - mu)**2 * inv_var + 2*log_sigma + math.log(2*math.pi))
    return nll.sum(dim=-1)  # sum over 4 dims

def studentst_nll(y, mu, log_sigma, nu):
    sigma = torch.exp(log_sigma)
    r = (y - mu) / sigma
    nll = 0.5 * (nu + 1) * torch.log(1 + r**2 / nu) + torch.log(sigma)
    return nll.sum(dim=-1)

def skew_gaussian_nll(y, mu, log_sigma, alpha):
    sigma = torch.exp(log_sigma)
    sigma = softplus_pos(sigma)
    delta = torch.tanh(alpha)
    alpha = delta / torch.clamp(torch.sqrt(1 - delta*delta), min=1e-8)

    r = (y - mu) / sigma
    nll = 0.5 * r ** 2 + log_sigma - torch.special.log_ndtr(alpha * r)
    return nll.sum(dim=-1)

def discretised_loglike(y, nb_logr, nb_logitp, w0, w1) -> Tensor:
    '''
    :return: Batchwise loss, shape (n_batches,).
    '''
    # y, local_sums = y # unpack (per_level_coeffs, local sum) tuple

    # y coefficients fine to coarse
    # y a list of tuples [(y,s), ..., (y,s)] over levels
    a_final = y[-1][0][:, :, 0] # (batch, pix, coeffs)
    
    r = torch.exp(nb_logr)            # (1, out_npix)
    p = torch.sigmoid(nb_logitp)      # (1, out_npix)
    coarse_logp = log_negbin_pmf(a_final, r, p).sum(dim=1)  # (B,)

    detail_logps = []
    for level, (per_level_coeffs, s) in enumerate(y):
        d = per_level_coeffs[level][..., 1:]          # (B, Np, C) typically C=3

        # Vectorised over channels: sum over parents then channels
        # log_discrete_laplace should broadcast a scalar log_b[level]
        lp = logp_details_gaussian(d, s, w0[level], w1[level]) 
        detail_logps.append(lp.sum(dim=(1, 2)))     # (B,)

    totlogp_per_batch = coarse_logp + torch.stack(detail_logps, dim=1).sum(dim=1)  # (B,)
    return totlogp_per_batch

def logp_details_gaussian(
        detail_coefficients: Tensor,
        local_sum: Tensor, # poisson rate proxy,
        w0: Tensor,
        w1: Tensor,
        eps: float = 1e-6
) -> Tensor:
    log_sigma = w0 + w1 * torch.log1p(
        torch.sqrt(torch.clamp(local_sum, min=0.0) + eps)
    )
    sigma = torch.exp(log_sigma)
    sigma = sigma.unsqueeze(-1)
    lp = -0.5 * (
        (detail_coefficients/sigma)**2
      + 2 * log_sigma.unsqueeze(-1)
      + math.log(2 * math.pi)
    )
    return lp

def log_negbin_pmf(x: Tensor, r: Tensor, p: Tensor, eps: float = 1e-5) -> Tensor:
    # x: (B, n), integers >= 0; r>0, 0<p<1 (can be broadcast)
    x = torch.clamp(x, min=0).to(torch.float32)
    r = torch.clamp(r, min=eps)
    p = torch.clamp(p, min=eps, max=1-eps)
    return (
        torch.lgamma(x + r) - torch.lgamma(r)
      - torch.lgamma(x + 1) + r*torch.log(p)
      + x*torch.log(1 - p)
    )

def log_discrete_laplace(x, log_b, eps=1e-12):
    b = torch.exp(log_b).clamp_min(eps)
    alpha = torch.exp(-1.0 / b).clamp(max=1-1e-5)   # 0<alpha<1
    logZ = torch.log1p(-alpha) - torch.log1p(alpha) # log((1-alpha)/(1+alpha))
    return -torch.abs(x).to(torch.float32) * (1.0/b) + logZ

def compute_nll(
        per_level_quads: list[tuple[Tensor, Tensor]],
        distribution: Literal['gaussian', 'studentst', 'skew_gaussian'] = 'gaussian',
        **distribution_kwargs
) -> Tensor:
    '''
    Compute the mean NLL per level (averaging across these again) given a choice
    of underlying distribution.

    :param per_level_quads: Coefficients computed at each level, of shape
        (n_batches, npix_level, n_coefficients).
    :param distribution: Underlying distribution for loss function.
    :param **distribution_kwargs: Kwargs to pass to distribution.
    '''
    distribution_to_nll_func = {
        'gaussian': gaussian_nll,
        'studentst': studentst_nll,
        'skew_gaussian': skew_gaussian_nll
    }
    valid_distribution_kwargs = {
        'gaussian': ['mu', 'log_sigma'],
        'studentst': ['mu', 'log_sigma', 'nu'],
        'skew_gaussian': ['mu', 'log_sigma', 'alpha']
    }
    kwargs = valid_distribution_kwargs[distribution]
    for kwarg in kwargs:
        assert kwarg in distribution_kwargs, f'Invalid kwargs for {distribution}.'

    loss_function = distribution_to_nll_func[distribution]
    level_losses = []
    for level, (y, s) in enumerate(per_level_quads):
        nll = loss_function(
            y,
            **{kwarg: val[level] for kwarg, val in distribution_kwargs.items()}
        )
        level_losses.append(nll.mean())
    return torch.stack(level_losses).mean()

def learn_transformation(
        model: HealpixSOPyramid,
        data: Tensor,
        device: str = 'cpu',
        batch_size: int = 128,
        epochs: int = 3,
        lr=0.001,
        validation_frac=0.1,
        min_delta_rel=0.001, 
        patience=100
) -> Tuple[HealpixSOPyramid, dict, dict]:
    data = data.to(device)
    method = model.method

    dataset = MapDataset(data)
    n_simulations = len(dataset)
    n_validation = int(validation_frac * n_simulations)
    n_train = n_simulations - n_validation
    training_set, validation_set = random_split(dataset, [n_train, n_validation]) 

    training_loader = DataLoader(
        training_set,
        batch_size=batch_size, 
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=True
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    steps_since_best = 0
    history = {"train": [], "val": []}

    def compute_loss_SO(per_level_quads: list[tuple[Tensor, Tensor]]) -> Tensor:
        loss = compute_nll(
            per_level_quads, 
            distribution='gaussian',
            mu=model.mu_list, 
            log_sigma=model.log_sigma_list,
        )
        return loss

    def compute_loss_discrete(per_level_quads) -> Tensor:
        logp = discretised_loglike(
            per_level_quads,
            nb_logr=model.nb_logr,
            nb_logitp=model.nb_logitp,
            w0=model.w0,
            w1=model.w1
        )
        loss = (-logp).mean()
        return loss

    if method == 'SO':
        loss_func = compute_loss_SO
    else:
        loss_func = compute_loss_discrete

    def evaluate(loader):
        model.eval()
        tot = 0.0
        cnt = 0
        with torch.no_grad():
            for D in loader:
                _, per_level_quads, *_ = model(D)   # list of (B, P_l, 4)
                loss = loss_func(per_level_quads)
                bs = D.size(0)
                tot += float(loss) * bs
                cnt += bs
        return tot / max(1, cnt)

    # Initial validation (optional)
    val0 = evaluate(validation_loader)
    best_val = val0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        print(f'Start of epoch {epoch}.')
        model.train()
        for D in training_loader:
            _, per_level_quads, *_ = model(D)
            loss = loss_func(per_level_quads)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            history["train"].append(float(loss.detach()))

            # Periodic val check (every ~1 epoch worth of samples / 5, say)
            if len(history["train"]) % max(1, (len(training_loader)//5 or 1)) == 0:
                val = evaluate(validation_loader)
                history["val"].append(val)

                # Early stopping logic (relative improvement)
                rel_impr = (best_val - val) / max(best_val, 1e-12)
                if rel_impr > min_delta_rel:
                    best_val = val
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    steps_since_best = 0
                else:
                    steps_since_best += 1
                    if steps_since_best >= patience:
                        # Restore and exit
                        model.load_state_dict(best_state, strict=True)
                        return (
                            model, 
                            history,
                            {"best_val": best_val, "stopped_early": True}
                        )

    # End of epochs: restore best
    model.load_state_dict(best_state, strict=True)
    return model, history, {"best_val": best_val, "stopped_early": False}
