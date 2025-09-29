import unittest
import warnings
import numpy as np
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.transforms import ZScore


class TestZScoreTransform(unittest.TestCase):
    def test_batchwise_poisson_round_trip_masked(self):
        batches = 64
        npix = 768
        lam = 25
        key = prng_key(2024)

        data = key.poisson(lam, shape=(batches, npix)).astype(np.float64)
        mask_base = np.ones(npix, dtype=bool)
        mask_base[::11] = False
        mask = np.broadcast_to(mask_base, data.shape)
        data[:, ~mask_base] = np.nan

        transform = ZScore(method='batchwise')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            transform.compute_mean_and_std(data, mask)

        masked_values = np.where(mask, data, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            expected_mu = np.nanmean(masked_values, axis=0)
            expected_sigma = np.nanstd(masked_values, axis=0)

        assert type(transform.mu) == np.ndarray; assert type(transform.sigma) == np.ndarray;
        np.testing.assert_allclose(transform.mu[mask_base], expected_mu[mask_base])
        np.testing.assert_allclose(transform.sigma[mask_base], expected_sigma[mask_base])

        # make sure we get default stats out for means of pixels that are totally masked
        np.testing.assert_array_equal(transform.mu[~mask_base], 0.0)
        np.testing.assert_array_equal(transform.sigma[~mask_base], 1.0)

        (zmap, zmask), logdet = transform.forward_and_log_det(data, mask)
        self.assertEqual(logdet.shape, (batches,))
        self.assertTrue(np.isfinite(logdet).all())
        np.testing.assert_equal(zmask, mask)
        self.assertTrue(np.isfinite(zmap[zmask]).all())

        (reconstructed, mask_rec), inv_logdet = transform.inverse_and_log_det(zmap, zmask)
        np.testing.assert_equal(mask_rec, zmask)
        self.assertTrue(np.isfinite(reconstructed[mask]).all())
        np.testing.assert_allclose(reconstructed[mask], data[mask])
        np.testing.assert_allclose(logdet + inv_logdet, np.zeros_like(logdet))

    def test_global_masked_statistics_round_trip(self):
        batches = 32
        npix = 512
        lam = 40
        key = prng_key(1337)

        data = key.poisson(lam, shape=(batches, npix)).astype(np.float64)
        mask_base = np.ones(npix, dtype=bool)
        mask_base[: npix // 5] = False
        mask = np.broadcast_to(mask_base, data.shape)
        data[:, ~mask_base] = np.nan

        transform = ZScore(method='global')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            transform.compute_mean_and_std(data, mask)

        masked_values = np.where(mask, data, np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            expected_mu = np.nanmean(masked_values)
            expected_row_std = np.nanstd(masked_values, axis=1)
        min_std = 1e-14
        expected_row_std = np.where(np.isnan(expected_row_std), min_std, expected_row_std)
        expected_row_std = np.where(expected_row_std < min_std, min_std, expected_row_std)
        expected_sigma = expected_row_std.mean()

        self.assertAlmostEqual(transform.mu, expected_mu) # type: ignore
        self.assertAlmostEqual(transform.sigma, expected_sigma)

        (zmap, zmask), logdet = transform.forward_and_log_det(data, mask)
        self.assertEqual(logdet.shape, (batches,))
        self.assertTrue(np.isfinite(logdet).all())
        np.testing.assert_equal(zmask, mask)
        self.assertTrue(np.isfinite(zmap[zmask]).all())

        (reconstructed, mask_rec), inv_logdet = transform.inverse_and_log_det(zmap, zmask)
        np.testing.assert_equal(mask_rec, zmask)
        self.assertTrue(np.isfinite(reconstructed[mask]).all())
        np.testing.assert_allclose(reconstructed[mask], data[mask])
        np.testing.assert_allclose(logdet + inv_logdet, np.zeros_like(logdet))


if __name__ == '__main__':
    unittest.main()
