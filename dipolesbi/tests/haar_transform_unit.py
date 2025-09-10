import unittest
import numpy as np
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.transforms import HaarWaveletTransform
from jax import numpy as jnp


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
            post_normalise=False
        )
        zmap, logdet = transform(dmap)
        self.assertFalse(np.isnan(zmap).any())

        _, subkey = key.split() 
        dmap2 = subkey.poisson(lam, shape=(1, npix))
        zmap2, logdet2 = transform(dmap2)
        self.assertFalse(np.isnan(zmap2).any())

    def test_subsequent_transforms_post_norm(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
            post_normalise=True
        )
        zmap, logdet = transform(dmap)
        self.assertFalse(np.isnan(zmap).any())

        _, subkey = key.split() 
        dmap2 = subkey.poisson(lam, shape=(1, npix))
        zmap2, logdet2 = transform(dmap2)
        self.assertFalse(np.isnan(zmap2).any())

    def test_post_normalisation(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
            post_normalise=True
        )

        zmap, logdet = transform(dmap)

    def test_post_norm_reconstruction_sparse_no_detail_norm(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
            post_normalise=True,
            matrix_type='sparse_average',
            normalise_details=False
        )

        zmap, logdet = transform(dmap)
        dmap_rec = transform.inverse(zmap)

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
            post_normalise=False
        )

        zmap, logdet = transform(dmap)
        
        self.assertEqual(
            logdet.shape[0], 
            n_batches, 
            'Logdet batch dim != input batch dim.'
        )
        self.assertFalse(np.isnan(logdet).any())

    def test_logdet_ok_post_norm(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1, 
            post_normalise=True
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

        dmap = key.poisson(lam, shape=(1, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered = transform.inverse(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_post_norm(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(1000, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1, post_normalise=True)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered = transform.inverse(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_low_nside(self):
        nside = 8
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(1, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered = transform.inverse(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_high_lambda(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(1, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered = transform.inverse(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_inverse_high_lambda_post_norm_sparse(self):
        nside = 32
        batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1,
            post_normalise=True,
            matrix_type='sparse_average',
            normalise_details=True
        )

        dmap_transformed, _ = transform(dmap)
        dmap_recovered = transform.inverse(dmap_transformed)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

    def test_detail_unnormalisation_post_norm_sparse(self):
        nside = 32
        batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(
            first_nside=nside, 
            last_nside=1,
            post_normalise=True,
            matrix_type='sparse_average',
            normalise_details=True
        )

        dmap_transformed, _ = transform(dmap)
        # grab finest details
        lvl_idx = 0
        n_detail1_lvl0 = transform.mu_at_level_post['detail'][1][lvl_idx].shape[0]
        print(n_detail1_lvl0)
        unnorm_func = transform.make_unnormalise_details_func(level=lvl_idx)

        # coarse details 1s at front
        details_at_lvl0 = dmap_transformed[..., -3*n_detail1_lvl0:]
        details_lvl0_unnormed = unnorm_func(jnp.asarray(details_at_lvl0))

        x = np.asarray(details_lvl0_unnormed)
        diff = np.abs(x - np.rint(x))

        def is_integerish_f32(x, ulps=1):
            x = np.asarray(x, dtype=np.float32)
            # handle zeros cleanly
            mag = np.maximum(np.abs(x), np.float32(1.0))
            m = np.floor(np.log2(mag)).astype(np.int32)
            spacing = np.exp2(m - 23).astype(np.float32)   # 1 ulp at each magnitude
            return np.all(np.abs(x - np.rint(x)) <= ulps * spacing)

        assert np.all(np.isfinite(x)) and is_integerish_f32(x), (
            f'Non-integer out: max diff is {np.max(diff)}'
        )

if __name__ == "__main__":
    unittest.main()
