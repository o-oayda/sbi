from blackjax.types import Array
from dipolesbi.tools.healpix_helpers import split_off_details
from dipolesbi.tools.transforms import InvertibleDataTransform
from typing import Literal, Callable, cast
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

        # realistically this should never be part of the flow and always a pre-transform
        # (1) too slow
        # (2) only accurate (e.g. recovered Poisson ints) with float64
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
        self.encoded_masks_fine2coarse = []

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

    def _masked_moments(self, x, mask, axis=0, eps=1e-6, ddof=1):
        """
        x:  (B, P) values
        m:  (B, P) bool mask (True = valid)
        returns scalar mean, sclalar std, ok where count > ddof
        """
        xp = self.xp

        # masked (nan) pixels will be caught by the finite flag
        # we better hope all nans are in the mask else we've got problems
        finite = xp.isfinite(x)
        valid = mask.astype(bool) & finite

        w = valid.astype(x.dtype)
        n_seen = w.sum(axis=axis, keepdims=True)
        n_seen_safe = xp.maximum(n_seen, 1.)

        # don't do mask * x since nan vals will fuck that up
        # instead zero masked pixels so they are excluded from the sum
        x_masked = xp.where(valid, x, 0.)
        mean = x_masked.sum(axis=axis, keepdims=True) / n_seen_safe

        # same for variance
        xc = xp.where(valid, x - mean, 0.)
        var_num = (xc * xc).sum(axis=axis, keepdims=True)
        denom = xp.maximum(n_seen - ddof, 1.)
        std = xp.sqrt(var_num / denom + eps)

        ok = n_seen > ddof # enough batches to be meaningful std
        return mean, std, ok

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
        raise NotImplementedError
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

    def compute_mean_and_std(self, data: NDArray, mask: NDArray) -> None:
        # do a forward cycle to compute stats but don't use outputs
        self._cycle_healpix_tree(data, mask)
        return

    def forward_and_log_det(
            self, 
            data: NDArray, 
            mask: NDArray
    ) -> tuple[tuple[NDArray, NDArray], NDArray]:
        # Ensure backend array; remember original dtype for optional cast-back
        data = self.xp.asarray(data, dtype=self.dtype)
        (z, mask), logdet = self._cycle_healpix_tree(data, mask)
        return (z, mask), logdet

    def inverse_and_log_det(
            self, 
            transformed_data: NDArray, 
            transformed_mask: NDArray
    ) -> tuple[tuple[NDArray, NDArray], NDArray]:
        transformed_data = self.xp.asarray(transformed_data, dtype=self.dtype)
        (x, mask), logdet = self._reverse_cycle_healpix_tree(
            transformed_data, 
            transformed_mask
        )
        return (x, mask), logdet

    def _build_surjective_blocks(self) -> list[tuple[int, int]]:
        steps = split_off_details(self.first_nside, self.last_nside)
        return steps

    def _cycle_healpix_tree(
            self, 
            y: NDArray,
            mask: NDArray
    ) -> tuple[tuple[NDArray, NDArray], NDArray]:
        assert mask.shape == y.shape

        batches, npix_fine = y.shape
        cur_npix = npix_fine
        coefficients_fine2coarse: list[NDArray] = [] 
        mask_fine2coarse: list[NDArray] = []
        a_list: list[NDArray] = []
        logdets_at_each_level: list[NDArray] = []
        self.encoded_masks_fine2coarse = []
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

            # since the selector * Hadamard transform matrix is zero-padded to
            # 4x4, if we didn't do this and had nans in the masked pixels,
            # despite them being masked thet would still be part of the matrix
            # product and therey blow up the computation --- we need to zero
            child_pixels = self.xp.where(subspace_mask, child_pixels, 0.)

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
            mean_coarse, std_coarse, enough_c = self._masked_moments(
                a_coarse, a_mask, axis=0, ddof=0
            ) # we get shapes (1, P) out, so squeeze down

            # do a blank transform where the stats are undefined
            mean_coarse = self.xp.where(
                self.xp.squeeze(enough_c),
                self.xp.squeeze(mean_coarse),
                0.,
            )
            std_coarse = self.xp.where(
                self.xp.squeeze(enough_c),
                self.xp.squeeze(std_coarse),
                1.,
            )
        else:
            mean_coarse = self.mu_at_level_post['coarse']
            std_coarse  = self.std_at_level_post['coarse']
            enough_c    = self.xp.ones_like(std_coarse, dtype=bool)

        # set masked to 0, these should be removed later, and apply zscore
        a_coarse = self.xp.where(a_mask, (a_coarse - mean_coarse) / std_coarse, 0.)

        # only add the logdet per batch for valid coarse coeffs
        abslogdet += - (
            a_mask.astype(a_coarse.dtype) * self.xp.log(std_coarse)
        ).sum(axis=1)

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
                def _clean_stats(mu_c, std_c, ok_c):
                    '''Blank transform on undefined stats.'''
                    ok_c = self.xp.squeeze(ok_c)
                    mu_c = self.xp.where(ok_c, self.xp.squeeze(mu_c), 0.)
                    std_c = self.xp.where(ok_c, self.xp.squeeze(std_c), 1.)
                    return mu_c, std_c

                # ddof=0 for population std
                mu_d1, s_d1 = _clean_stats(
                    *self._masked_moments(y[..., 1], msk[..., 1], axis=0, ddof=0)
                )
                mu_d2, s_d2 = _clean_stats(
                    *self._masked_moments(y[..., 2], msk[..., 2], axis=0, ddof=0)
                )
                mu_d3, s_d3 = _clean_stats(
                    *self._masked_moments(y[..., 3], msk[..., 3], axis=0, ddof=0)
                )
            else:
                mu_d1 = self.mu_at_level_post['detail'][0][lvl_idx - lvl]
                mu_d2 = self.mu_at_level_post['detail'][1][lvl_idx - lvl]
                mu_d3 = self.mu_at_level_post['detail'][2][lvl_idx - lvl]

                s_d1 = self.std_at_level_post['detail'][0][lvl_idx - lvl]
                s_d2 = self.std_at_level_post['detail'][1][lvl_idx - lvl]
                s_d3 = self.std_at_level_post['detail'][2][lvl_idx - lvl]
            
            if self.normalise_details:
                means = self.xp.stack([mu_d1, mu_d2, mu_d3], axis=-1)  # (P,3)
                stds  = self.xp.stack([s_d1,  s_d2,  s_d3],  axis=-1)  # (P,3)
                y_details = (y[..., 1:] - means[None, :, :]) / stds[None, :, :]

                # zero out invalid rows so they don’t pollute anything
                y_details = self.xp.where(msk[..., 1:], y_details, 0.)

                y_details = self.xp.reshape(y_details, (batches, y.shape[1], -1))
                y_head = self.xp.reshape(y[..., :1], (batches, y.shape[1], 1))
                y = self.xp.concatenate([y_head, y_details], axis=-1)

                # do log det only where it is valid
                logstd = self.xp.log(stds)[None, :, :]              # (1,P,3)
                weight = msk[..., 1:].astype(y.dtype)               # (B,P,3)

                # since we do the same affine transform per batch, this is fine
                if hasattr(self.xp, 'broadcast_to'):
                    logstd_b = self.xp.broadcast_to(logstd, weight.shape)
                else:
                    logstd_b = self.xp.repeat(logstd, axis=0, repeats=batches)

                abslogdet += - (weight * logstd_b).sum(axis=(1, 2))

            if self.empty_norm_stats_flag:
                self.mu_at_level_post['detail'][0][lvl_idx - lvl] = mu_d1
                self.mu_at_level_post['detail'][1][lvl_idx - lvl] = mu_d2
                self.mu_at_level_post['detail'][2][lvl_idx - lvl] = mu_d3

                self.std_at_level_post['detail'][0][lvl_idx - lvl] = s_d1
                self.std_at_level_post['detail'][1][lvl_idx - lvl] = s_d2
                self.std_at_level_post['detail'][2][lvl_idx - lvl] = s_d3

            z_parts.append(y[..., 1:].reshape(batches, -1))
            z_mask_parts.append(msk[..., 1:].reshape(batches, -1))
        z = self.xp.concatenate(z_parts, axis=1)
        z_mask = self.xp.concatenate(z_mask_parts, axis=1)

        self.empty_norm_stats_flag = False
        
        # since for each batch the transform jacobian is identical,
        # we only need to extract one scalar (first element) from this 1D array
        self.logdet = abslogdet

        # we have to hack slightly and assume all masks are the same for
        # all batches, since jax requires me to statically-compute the indexes
        # I slice the data with; this is ok since the mask shouldn't change
        # over runs
        self.keep_idxs = jnp.where(z_mask[0, :])[0]

        return (z, z_mask), abslogdet

    def _reverse_cycle_healpix_tree(self,
        transformed_data: np.ndarray,
        transformed_mask: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        batches, npix = transformed_data.shape
        P_levels = self.parents_at_levels

        # z = [a_coarse | d_coarse ... d_fine ]
        offset = self.last_npix
        a = transformed_data[:, :offset] # (B, out_npix)
        a_mask = transformed_mask[:, :offset].astype(bool)

        # only denormalise on valid pixels
        mu_coarse = self.mu_at_level_post['coarse']
        std_coarse = self.std_at_level_post['coarse']
        a = self.xp.where(a_mask, a * std_coarse[None, :] + mu_coarse[None, :], 0.)

        # iterate from high resolution to low resolution
        details_coarse2fine = []
        mask_coarse2fine = []
        for npix, dscale_factor in zip(P_levels[::-1], self.downscale_factors[::-1]):
            dsize = npix * (dscale_factor - 1) # detail coefficients always 1 less 

            d = transformed_data[:, offset:offset+dsize].reshape(
                batches, npix, dscale_factor - 1
            )
            d_mask = transformed_mask[:, offset:offset+dsize].reshape(
                batches, npix, dscale_factor - 1
            ).astype(bool)

            offset += dsize
            details_coarse2fine.append(d)
            mask_coarse2fine.append(d_mask)

        # encoded masks at each level saved from forward pass
        encoded_fine2coarse = self.encoded_masks_fine2coarse

        # reconstruct coarse->fine
        upstream_data = a

        for lvl, (d_out, d_out_mask, codes_lvl) in enumerate(
            zip(details_coarse2fine, mask_coarse2fine, reversed(encoded_fine2coarse))
        ):
            # y_out is what forward produced after coupling: [a | d_out]
            a_here = upstream_data[..., None]
            a_here = cast(Array, a_here)

            if self.normalise_details:
                lvl_idx = self.n_levels - 1 - lvl
                means = self.xp.stack([
                    self.mu_at_level_post['detail'][0][lvl_idx],
                    self.mu_at_level_post['detail'][1][lvl_idx],
                    self.mu_at_level_post['detail'][2][lvl_idx],
                ], axis=-1)  # (P,3)
                stds = self.xp.stack([
                    self.std_at_level_post['detail'][0][lvl_idx],
                    self.std_at_level_post['detail'][1][lvl_idx],
                    self.std_at_level_post['detail'][2][lvl_idx],
                ], axis=-1)  # (P,3)

                d_out = self.xp.where(
                    d_out_mask, d_out * stds[None, :, :] + means[None, :, :],
                    0.
                )
            
            y_pre = self.xp.concatenate([a_here, d_out], axis=-1)

            data = self._inverse_matrix_product_masked(y_pre, codes_lvl)

            # upsample to next finer 'a'
            upstream_data = data.reshape(batches, -1)

        x_rec = upstream_data

        # hack slightly and assume forward has been called
        inverse_logdet = -self.logdet * self.xp.ones(batches)

        # Recover original mask
        codes_fine = encoded_fine2coarse[0]
        mask_rec = self._decode_codes(codes_fine).reshape(batches, -1)

        return (x_rec, mask_rec), inverse_logdet

    def _decode_codes(self, codes):
        # codes: (B, P_level) uint8/int in [0..15]
        bitw = self.xp.array([1, 2, 4, 8], dtype=self.xp.int32)
        return ((codes.astype(self.xp.int32)[..., None] & bitw) != 0)  # (B, P_level, 4) bool

    def _forward_matrix_product(self, v: NDArray) -> NDArray:
        return v @ self.Q # (B, npix, 4) @ (4,4)

    def _encode_mask(self, mask: NDArray) -> NDArray:
        bitw = self.xp.array([1, 2, 4, 8], dtype=self.xp.int32)
        encoded_mask = (mask.astype(self.xp.int32) * bitw).sum(axis=-1)
        return encoded_mask

    def _forward_matrix_product_masked(
            self, 
            v: NDArray, 
            mask: NDArray
    ) -> tuple[NDArray, NDArray]:
        # for each 4-block, encode the mask as a 4-bit integer
        encoded_mask = self._encode_mask(mask)
        self.encoded_masks_fine2coarse.append(encoded_mask.astype(self.xp.uint8))

        T = self.subspace.T_TABLE[encoded_mask] # lookup a 4x4 transform mat
        T_T = self.xp.swapaxes(T, -1, -2) # transpose matrix, shape (B, P//4, 4, 4)

        coeffs = self.xp.einsum('bnq, bnqQ -> bnQ', v, T_T) # (B, P//4, 4)

        k_rows = self.subspace.K_TABLE[encoded_mask]
        valid = self.xp.arange(4)[None, None, :] < k_rows[..., None]
        return coeffs, valid

    def _inverse_matrix_product(self, z: NDArray) -> NDArray:
        return z @ self.Q_inv

    def _inverse_matrix_product_masked(
        self,
        z: NDArray,
        encoded_mask: NDArray
    ) -> NDArray:
        encoded_mask = encoded_mask.astype(self.xp.int32)
        T = self.subspace.T_TABLE[encoded_mask]
        k_rows = self.subspace.K_TABLE[encoded_mask] 

        row_mask = (
            self.xp.arange(4)[None, None, :] < k_rows[..., None]
        ).astype(z.dtype)  # (B,P,4)
        z_ok = z * row_mask

        x = self.xp.einsum('bnRq, bnR -> bnq', T, z_ok) # (B, nblocks, 4)
        return x


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
