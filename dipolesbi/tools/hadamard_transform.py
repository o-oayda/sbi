from blackjax.types import Array
from dipolesbi.tools.healpix_helpers import split_off_details
from dipolesbi.tools.transforms import InvertibleDataTransform
from typing import Literal, Callable, Optional, cast
import numpy as np
from numpy.typing import NDArray
from jax import numpy as jnp
import healpy as hp
import math


class MaskedSubspaceTransforms:
    def __init__(self) -> None:
        # make lookup table for all mask permutations
        _T_list, _k_list = zip(*[self._transform_for_mask(m) for m in range(16)])
        self.T_TABLE = jnp.stack(_T_list, axis=0) # (16, 4, 4), (i, 4, 4) a matrix mask perm
        self.K_TABLE = jnp.array(_k_list)         # (16,), corresponding n_observed in 4-block

    def _Wk(self, k: int) -> jnp.ndarray:
        '''
        For each masked subspace, construct a matrix transform corresponding
        to a particular orthonormal basis such that log det is zero and the
        detail vectors correspond to zero sums, among other conditions.
        Note that k = 2 and k = 4 are just the Hadamard matrices,
        whereas k = 3 is a 'Helmert' basis basis.
        '''
        if k == 1:
            return jnp.array([[1.0]]) # i.e., get the same pixel out
        if k == 2:
            return (1/jnp.sqrt(2)) * jnp.array(
                [[1,  1],
                 [1, -1]]
            ) # H_2
        if k == 3:
            # Helmert basis: coarse coefficient, then two zero-sum orthonormal contrasts
            return jnp.array(
                [[1/jnp.sqrt(3),  1/jnp.sqrt(3),  1/jnp.sqrt(3)],
                 [1/jnp.sqrt(2), -1/jnp.sqrt(2),  0.0          ],
                 [1/jnp.sqrt(6),  1/jnp.sqrt(6), -2/jnp.sqrt(6)]]
            )
        if k == 4:
            return 0.5 * jnp.array(
                [[ 1,  1,  1,  1],
                 [ 1,  1, -1, -1],
                 [ 1, -1,  1, -1],
                 [ 1, -1, -1,  1]]
            ) # H_4
        raise ValueError("k must be 1, 2, 3 or 4.")

    def _selector(self, idx: list) -> jnp.ndarray:
        # k×4 column selector that picks observed child positions in order
        k = len(idx)
        S = jnp.zeros((k, 4))
        return S.at[jnp.arange(k), jnp.array(idx)].set(1.0)

    def _transform_for_mask(
            self, 
            mask_bits: int
    ) -> tuple[jnp.ndarray, int]:
        # mask bits is an encoded representation of the 4-bit mask (16 possible values),
        # so a number between 0 and 15.
        # bit i (0..3) says child i is observed
        idx = [i for i in range(4) if (mask_bits >> i) & 1]
        k = len(idx)

        if k == 0:
            # Empty block: return all-zero matrix
            T = jnp.zeros((4, 4))
            return T, 0

        Wk = self._Wk(k)            # k×k orthonormal
        S = self._selector(idx)     # k×4 selector of observed columns
        Tk = Wk @ S                 # k×4, rows = [c0; details], columns = children
        # pad to 4×4 so we can keep shapes fixed
        T = jnp.zeros((4, 4)).at[:k, :].set(Tk)
        return T, k

class HadamardTransform(InvertibleDataTransform):
    def __init__(
            self, 
            first_nside: int, 
            last_nside: int = 1,
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

        self.subspace = MaskedSubspaceTransforms()
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
            f"HadamardTransform("
            f"first_nside={self.first_nside}, "
            f"last_nside={self.last_nside}, "
            f"first_npix={self.first_npix}, "
            f"last_npix={self.last_npix}, "
            f"n_levels={self.n_levels}, "
            f"downscale_factors={self.downscale_factors}, "
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
        self.mu_at_level_post = self._make_post_dict()
        self.std_at_level_post = self._make_post_dict()
        self.empty_norm_stats_flag = True

    def compute_mean_and_std(self, data: NDArray) -> None:
        # do a forward cycle to compute stats but don't use outputs
        self._cycle_healpix_tree(data)
        return

    def forward_and_log_det(
            self, 
            data: NDArray, 
            mask: Optional[NDArray] = None
    ) -> tuple[tuple[NDArray, NDArray], NDArray]:
        # Ensure backend array; remember original dtype for optional cast-back
        orig_dtype = getattr(data, 'dtype', None)
        data = self.xp.asarray(data, dtype=self.dtype)
        (z, mask), logdet = self._cycle_healpix_tree(data)
        try:
            if orig_dtype == getattr(self.xp, 'float32'):
                z = z.astype(self.xp.float32)
                logdet = logdet.astype(self.xp.float32)
        except Exception:
            pass
        return (z, mask), logdet

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

    def _cycle_healpix_tree(
            self, 
            y: NDArray,
            mask: Optional[NDArray] = None
    ) -> tuple[tuple[NDArray, NDArray], NDArray]:
        if mask is None:
            mask = self.xp.ones_like(y)
        assert mask is not None # bloody type hinter
        assert mask.shape == y.shape

        batches, npix_fine = y.shape
        cur_npix = npix_fine
        coefficients_fine2coarse: list[NDArray] = [] 
        mask_fine2coarse: list[NDArray] = []
        a_list: list[NDArray] = []
        logdets_at_each_level: list[NDArray] = []
        abslogdet: NDArray = self.xp.zeros(batches)
        
        if self.empty_norm_stats_flag:
            assert batches > 1, 'More than 1 batch needed to define std.'
        
        logdet = self.xp.zeros(batches)
        downstream_coefficients = y
        downstream_mask = mask

        for lvl, factor in enumerate(self.downscale_factors):
            per_level_logdet = self.xp.zeros(())

            P = cur_npix // factor
            child_pixels = downstream_coefficients.reshape(batches, P, factor)
            subspace_mask = downstream_mask.reshape(batches, P, factor)

            # by construction, coarse coeffs lives in [..., 0] and the detail [..., 1:]
            # in the 3 masked 1 valid case, we always get 1 pixel i.e. a downstream coeff
            z, valid = self._forward_matrix_product_masked(child_pixels, subspace_mask) # (B, P // 4, 4)

            coarse_coeffs = z[..., 0]
            coarse_mask = valid[..., 0]

            logdets_at_each_level.append(per_level_logdet)
            logdet += per_level_logdet

            coefficients_fine2coarse.append(z)
            mask_fine2coarse.append(valid)
            a_list.append(coarse_coeffs)
            downstream_coefficients = coarse_coeffs
            downstream_mask = coarse_mask
            cur_npix = P

        a_coarse = downstream_coefficients # (batches, out_npix)
        a_mask = downstream_mask

        if self.empty_norm_stats_flag:
            mean_coarse = a_coarse.mean(axis=0) # across the batch axis
            std_coarse = a_coarse.std(axis=0)
        else:
            mean_coarse = self.mu_at_level_post['coarse']
            std_coarse = self.std_at_level_post['coarse']

        a_coarse = ( a_coarse - mean_coarse ) / std_coarse

        abslogdet += -self.xp.log(std_coarse).sum(axis=-1) # sum over pixels

        if self.empty_norm_stats_flag:
            assert self._post_dict_is_empty()

            self.mu_at_level_post['coarse'] = mean_coarse
            self.std_at_level_post['coarse'] = std_coarse

        # Build z: keep a_coarse, then details from coarse->fine
        # (drop the a's there)
        z_parts = [a_coarse.reshape(batches, -1)]  # list[ (B, out_npix) ]
        z_mask_parts = [a_mask.reshape(batches, -1)]
        lvl_idx = self.n_levels - 1

        if self.empty_norm_stats_flag:
            assert self._post_dict_is_empty(check_coarse=False)

        # I think tje z-scoring is not the best approach now with different
        # basis vectors depending on the masked subspace
        for lvl, (y, msk) in enumerate(
            zip(reversed(coefficients_fine2coarse), reversed(mask_fine2coarse))
        ): # now coarse->fine
            
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

                abslogdet += - self.xp.log(std_d1).sum(axis=-1)
                abslogdet += - self.xp.log(std_d2).sum(axis=-1)
                abslogdet += - self.xp.log(std_d3).sum(axis=-1)

            if self.empty_norm_stats_flag:
                self.mu_at_level_post['detail'][0][lvl_idx - lvl] = mean_d1
                self.mu_at_level_post['detail'][1][lvl_idx - lvl] = mean_d2
                self.mu_at_level_post['detail'][2][lvl_idx - lvl] = mean_d3

                self.std_at_level_post['detail'][0][lvl_idx - lvl] = std_d1
                self.std_at_level_post['detail'][1][lvl_idx - lvl] = std_d2
                self.std_at_level_post['detail'][2][lvl_idx - lvl] = std_d3

            z_parts.append(y[..., 1:].reshape(batches, -1))
            z_mask_parts.append(msk[..., 1:].reshape(batches, -1))
        z = self.xp.concatenate(z_parts, axis=1)
        z_mask = self.xp.concatenate(z_mask_parts, axis=1)

        self.empty_norm_stats_flag = False
        
        # since for each batch the transform jacobian is identical,
        # we only need to extract one scalar (first element) from this 1D array
        self.logdet = abslogdet[0]

        return (z, z_mask), abslogdet

    def _reverse_cycle_healpix_tree(self,
        transformed_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batches, npix = transformed_data.shape
        P_levels = self.parents_at_levels

        # z = [a_coarse | d_coarse ... d_fine ]
        offset = self.last_npix
        a = transformed_data[:, :offset] # (B, out_npix)

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

        # hack slightly and assume forward has been called
        inverse_logdet = -self.logdet * self.xp.ones(batches)

        return x_rec, inverse_logdet

    def _forward_matrix_product(self, v: NDArray) -> NDArray:
        return v @ self.Q # (B, npix, 4) @ (4,4)

    def _forward_matrix_product_masked(
            self, 
            v: NDArray, 
            mask: NDArray
    ) -> tuple[NDArray, NDArray]:
        # for each 4-block, encode the mask as a 4-bit integer
        bitw = self.xp.array([1, 2, 4, 8], dtype=self.xp.int32)
        encoded_mask = (mask.astype(self.xp.int32) * bitw).sum(axis=-1)

        T = self.subspace.T_TABLE[encoded_mask] # lookup a 4x4 transform mat
        T_T = self.xp.swapaxes(T, -1, -2) # transpose matrix, shape (B, P//4, 4, 4)

        coeffs = self.xp.einsum('bnq, bnqQ -> bnQ', v, T_T) # (B, P//4, 4)

        k_rows = self.subspace.K_TABLE[encoded_mask]
        valid = self.xp.arange(4)[None, None, :] < k_rows[..., None]
        return coeffs, valid

    def _inverse_matrix_product(self, z: NDArray) -> NDArray:
        return z @ self.Q_inv

# Convenience subclass that uses JAX as the backend
class HadamardTransformJax(HadamardTransform):
    def __init__(self,
                 first_nside: int,
                 last_nside: int = 1,
                 matrix_type: Literal['hadamard', 'sparse_average'] = 'hadamard',
                 normalise_details: bool = True,
                 n_chunks: int = 1):
        super().__init__(
            first_nside=first_nside,
            last_nside=last_nside,
            matrix_type=matrix_type,
            normalise_details=normalise_details,
            n_chunks=n_chunks,
            xp=jnp
        )
