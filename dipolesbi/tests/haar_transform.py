import unittest
import numpy as np
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.transforms import HaarWaveletTransform


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

if __name__ == "__main__":
    unittest.main()
