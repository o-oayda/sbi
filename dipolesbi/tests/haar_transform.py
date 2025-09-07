import unittest
import numpy as np
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.transforms import HaarWaveletTransform


class TestHaarTransform(unittest.TestCase):

    def test_inverse(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(1, npix)) # ensure batchwise dim exists
        transform = HaarWaveletTransform(first_nside=nside, last_nside=1)

        dmap_transformed, _ = transform(dmap)
        dmap_recovered = transform.inverse(dmap_transformed)

        print(dmap[:10])
        print(dmap_transformed[:10])
        print(dmap_recovered[:10])
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

        print(dmap[:10])
        print(dmap_recovered[:10])
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

        print(dmap[:10])
        print(dmap_recovered[:10])
        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')

if __name__ == "__main__":
    unittest.main()
