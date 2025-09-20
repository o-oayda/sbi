import unittest
import numpy as np
from dipolesbi.tools.healpix_helpers import split_off_details
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.transforms import HaarWaveletTransform, HaarWaveletTransformJax
from jax import numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)
from dipolesbi.tools.utils import is_integerish_f32


class TestHaarTransform(unittest.TestCase):

    def test_subsequent_transforms(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
        )
        zmap, logdet = transform(dmap)
        self.assertFalse(np.isnan(zmap).any())

        _, subkey = key.split() 
        dmap2 = subkey.poisson(lam, shape=(1, npix))
        zmap2, logdet2 = transform(dmap2)
        self.assertFalse(np.isnan(zmap2).any())

    def test_reconstruction_sparse_no_detail_norm(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
            matrix_type='sparse_average',
            normalise_details=False
        )

        zmap, logdet = transform(dmap)
        dmap_rec, _ = transform.inverse_and_log_det(zmap)

        np.testing.assert_almost_equal(dmap, dmap_rec, err_msg='Recovered != original.')

    def test_logdet_ok(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
        )

        zmap, logdet = transform(dmap)
        
        self.assertEqual(
            logdet.shape[0], 
            n_batches, 
            'Logdet batch dim != input batch dim.'
        )
        self.assertFalse(np.isnan(logdet).any())

    def test_inverse(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered, _ = transform.inverse_and_log_det(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_low_nside(self):
        nside = 8
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered, _ = transform.inverse_and_log_det(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_high_lambda(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered, _ = transform.inverse_and_log_det(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_high_lambda_jax(self):
        # this one fails if jax uses float32 because of floating point errors
        # so we assert now in HaarWaveletTransform that float64 is turned on for jax
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)).astype('float32') # ensure batchwise dim exists
        transform = HaarWaveletTransformJax(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered, _ = transform.inverse_and_log_det(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_high_lambda_sparse(self):
        nside = 32
        batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1,
            matrix_type='sparse_average',
            normalise_details=True
        )

        dmap_transformed, _ = transform(dmap)
        dmap_recovered, _ = transform.inverse_and_log_det(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_detail_unnormalisation_norm_sparse(self):
        nside = 32
        batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1,
            matrix_type='sparse_average',
            normalise_details=True
        )
        dmap_transformed, _ = transform(dmap)

        # grab finest details
        lvl_idx = 0
        n_detail1_lvl0 = transform.mu_at_level_post['detail'][1][lvl_idx].shape[0]
        unnorm_func = transform.make_unnormalise_details_func(level=lvl_idx)

        # coarse details 1s at front
        details_at_lvl0 = dmap_transformed[..., -3*n_detail1_lvl0:]
        details_lvl0_unnormed = unnorm_func(jnp.asarray(details_at_lvl0))

        x = np.asarray(details_lvl0_unnormed)
        diff = np.abs(x - np.rint(x))

        assert np.all(np.isfinite(x)) and is_integerish_f32(x), (
            f'Non-integer out: max diff is {np.max(diff)}'
        )

    def test_detail_unnormalisation_sparse_heirarchy(self):
        nside = 32
        batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1,
            matrix_type='sparse_average',
            normalise_details=True
        )

        dmap_transformed, _ = transform(dmap)
        blocks = split_off_details(initial_nside=nside, output_nside=1)
        zmap = dmap_transformed.copy()

        for lvl, (n_keep, n_drop) in enumerate(blocks):
            y_plus, y_minus = zmap[..., :n_keep], zmap[..., n_keep:]
            y_minus = jnp.asarray(y_minus)
            reconstructed_ints = transform.make_unnormalise_details_func(lvl)(y_minus)
            assert is_integerish_f32(np.asarray(reconstructed_ints))
            zmap = zmap[..., :n_keep]

if __name__ == "__main__":
    unittest.main()
