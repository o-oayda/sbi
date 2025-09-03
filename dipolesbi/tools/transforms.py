from typing import Optional, Tuple, cast, Literal
from blackjax.types import Array
from healpy import isnsideok, nside2npix
from numpy.typing import NDArray
import torch
from torch import Tensor, device
from nflows.transforms.base import Transform
from torch.utils.data import Dataset, DataLoader, random_split
import math
import numpy as np
from dipolesbi.tools.healpix_helpers import build_funnel_steps
from dipolesbi.tools.utils import softplus_pos
import torch.nn.functional as F
from abc import ABC, abstractmethod
import healpy as hp


class InvertibleDataTransform(ABC):
    def __init__(self) -> None:
        pass
    
    def __call__(self, data: NDArray) -> tuple[NDArray, NDArray]:
        return self.forward(data)

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def forward(self, data: NDArray) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def inverse(self, transformed_data: NDArray) -> NDArray:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

class BlankTransform(InvertibleDataTransform):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return 'BlankTransform(No transform to the data.)'

    def forward(self, data: NDArray) -> tuple[NDArray, NDArray]:
        n_batches = data.shape[0]
        return data, np.zeros_like(n_batches)

    def inverse(self, transformed_data: NDArray) -> NDArray:
        return transformed_data

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

    def forward(self, data: NDArray) -> tuple[NDArray, NDArray]:
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

    def inverse(self, transformed_data: NDArray) -> NDArray:
        data = transformed_data * self.sigma + self.mu
        return data

    def clear(self) -> None:
        self.mu = None
        self.sigma = None

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
    def __init__(self, first_nside: int, last_nside: int = 1) -> None:
        self.Q = 0.5 * np.asarray(
            [[1., 1. , 1. ,  1.],
             [1., 1. , -1., -1.],
             [1., -1., 1. , -1.],
             [1., -1., -1.,  1.]]
        )
        self.Q_inv = np.linalg.inv(self.Q)

        self.mu_at_level: list[list[NDArray]] = []
        self.std_at_level: list[list[NDArray]] = []
        self.empty_norm_stats_flag: bool = True

        self.first_nside = first_nside
        self.first_npix = hp.nside2npix(first_nside)
        self.last_nside = last_nside
        self.last_npix = hp.nside2npix(last_nside)

        self.n_levels = int(np.log2(first_nside) - np.log2(last_nside))
        self.downscale_factors = self.n_levels * [4]

    def __repr__(self) -> str:
        return (
            f"HaarWaveletTransform("
            f"first_nside={self.first_nside}, "
            f"last_nside={self.last_nside}, "
            f"first_npix={self.first_npix}, "
            f"last_npix={self.last_npix}, "
            f"n_levels={self.n_levels}, "
            f"downscale_factors={self.downscale_factors}, "
            f"mu_at_level={self.mu_at_level}, "
            f"std_at_level={self.std_at_level}"
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

    def clear(self) -> None:
        self.mu_at_level = []
        self.std_at_level = []
        self.empty_norm_stats_flag = True

    def forward(self, data: NDArray) -> tuple[NDArray, NDArray]:
        return self._cycle_healpix_tree(data)

    def inverse(self, transformed_data: NDArray) -> NDArray:
        return self._reverse_cycle_healpix_tree(transformed_data)

    def _build_surjective_blocks(self, n_chunks: int = 1) -> list[tuple[NDArray, int, int]]:
        detail_sizes = []
        ns = self.first_nside
        while ns > self.last_nside:
            ns //= 2
            npi = 12 * ns**2
            detail_sizes.append(3*npi)
        block_lengths = [self.last_npix] + detail_sizes # a coeffs + 3 details per level
        assert 12 * self.first_nside**2 == sum(block_lengths)

        steps = build_funnel_steps(
            n_coarse=self.last_npix, 
            detail_lengths=block_lengths[1:],
            n_chunks=n_chunks
        )
        return steps

    def _cycle_healpix_tree(
            self, 
            y: NDArray
    ) -> tuple[NDArray, NDArray]:
        batches, npix_fine = y.shape
        cur_npix = npix_fine
        coefficients_fine2coarse: list[NDArray] = [] 
        a_list: list[NDArray] = []
        logdets_at_each_level: list[NDArray] = []
        
        logdet = np.zeros(batches)
        downstream_coefficients = y

        for lvl, factor in enumerate(self.downscale_factors):
            per_level_logdet = np.zeros(())

            P = cur_npix // factor
            child_pixels = downstream_coefficients.reshape(batches, P, factor)

            z = self._forward_matrix_product(child_pixels) # (n_batches, n_pix // 4, 4)
            # outs = []
            # batchsize = 128
            # for i in range(0, batches, batchsize):
            #     outs.append(self.forward_in_batch(child_pixels[i:i+batchsize]))
            # z = jnp.concatenate(outs, axis=0)

            coarse_coeffs = z[..., 0]
            d1 = z[..., 1]
            d2 = z[..., 2]
            d3 = z[..., 3]

            cur_mus = []
            cur_stds = []
            for i, coef in enumerate([coarse_coeffs, d1, d2, d3]):
                if self.empty_norm_stats_flag:
                    mu = coef.mean()
                    sigma = coef.std()
                    cur_mus.append(mu)
                    cur_stds.append(sigma)
                else:
                    mu = self.mu_at_level[lvl][i]
                    sigma = self.std_at_level[lvl][i]

                z[..., i] = (z[..., i] - mu) / sigma

                per_level_logdet += - P * np.log(sigma)

            if self.empty_norm_stats_flag:
                self.mu_at_level.append(cur_mus)
                self.std_at_level.append(cur_stds)

            logdets_at_each_level.append(per_level_logdet)
            logdet += per_level_logdet

            coefficients_fine2coarse.append(z)
            a_list.append(coarse_coeffs)
            downstream_coefficients = z[..., 0]
            cur_npix = P

        a_coarse = downstream_coefficients # (batches, out_npix)
        self.empty_norm_stats_flag = False

        # Build z: keep a_coarse, then details from coarse->fine
        # (drop the a's there)
        z_parts = [a_coarse.reshape(batches, -1)]  # list[ (B, out_npix) ]
        for y in reversed(coefficients_fine2coarse): # now coarse->fine
            z_parts.append(y[..., 1:].reshape(batches, -1))
        z = np.concatenate(z_parts, axis=1)

        return z, logdet

    def _reverse_cycle_healpix_tree(self,
        transformed_data: np.ndarray
    ) -> np.ndarray:
        batches, npix = transformed_data.shape
        P_levels = self.parents_at_levels

        # z = [a_coarse | d_coarse ... d_fine ]
        offset = self.last_npix
        a = transformed_data[:, :offset] # (B, out_npix)

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
            y_pre = np.concatenate([a_here, d_out], axis=-1) # (B, P_l, n_coefficients)

            for i in range(4):
                mu = self.mu_at_level[-1 - lvl][i]
                sigma = self.std_at_level[-1 - lvl][i]
                y_pre[..., i] = y_pre[..., i] * sigma + mu

            data = self._inverse_matrix_product(y_pre)

            # upsample to next finer 'a'
            upstream_data = data.reshape(batches, -1)

        x_rec = upstream_data
        return x_rec

    def _forward_matrix_product(self, v: NDArray) -> NDArray:
        return v @ self.Q

    def _inverse_matrix_product(self, z: NDArray) -> NDArray:
        return z @ self.Q_inv

    def int_haar4_forward(self, x):  # x: (..., 4) integer tensor
        x0, x1, x2, x3 = x[...,0], x[...,1], x[...,2], x[...,3]

        d1 = x1 - x0
        s1 = x0 + (d1 // 2)            # floor division

        d2 = x3 - x2
        s2 = x2 + (d2 // 2)

        D  = s2 - s1
        A  = s1 + (D // 2)

        # output order: [coarse≈average, three details]
        return torch.stack([A, d1, d2, D], dim=-1)

    def int_haar4_inverse(self, y):  # y: (..., 4) with [A, d1, d2, D]
        A, d1, d2, D = y[...,0], y[...,1], y[...,2], y[...,3]

        s1 = A - (D // 2)
        s2 = D + s1

        x0 = s1 - (d1 // 2)
        x1 = d1 + x0
        x2 = s2 - (d2 // 2)
        x3 = d2 + x2

        return torch.stack([x0, x1, x2, x3], dim=-1)

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
