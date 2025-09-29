import unittest
from jax.random import PRNGKey
import numpy as np
from dipolesbi.lib.allsky_hadamard import ArchiveHadamardTransform
from dipolesbi.tools.healpix_helpers import split_off_details
from dipolesbi.tools.np_rngkey import prng_key
from dipolesbi.tools.hadamard_transform import HadamardTransform, HadamardTransformJax
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
        transform = HadamardTransform(
            first_nside=nside, 
            last_nside=1, 
        )
        input_mask = np.ones_like(dmap)
        (zmap, zmask), logdet = transform(dmap, input_mask)
        self.assertFalse(np.isnan(zmap).any())
        (dmap_rec, mask_rec), _ = transform.inverse_and_log_det(zmap, zmask)
        np.testing.assert_equal(mask_rec, input_mask)
        np.testing.assert_almost_equal(dmap, dmap_rec)

        _, subkey = key.split() 
        dmap2 = subkey.poisson(lam, shape=(1, npix))
        input_mask2 = np.ones_like(dmap2)
        (zmap2, zmask2), logdet2 = transform(dmap2, input_mask2)
        self.assertFalse(np.isnan(zmap2).any())
        (_, mask_rec2), _ = transform.inverse_and_log_det(zmap2, zmask2)
        np.testing.assert_equal(mask_rec2, input_mask2)

    def test_reconstruction_sparse_no_detail_norm(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(
            first_nside=nside, 
            last_nside=1, 
            matrix_type='sparse_average',
            normalise_details=False
        )
        mask = np.ones_like(dmap)

        (zmap, zmask), logdet = transform(dmap, mask)
        (dmap_rec, mask_rec), _ = transform.inverse_and_log_det(zmap, zmask)

        np.testing.assert_almost_equal(dmap, dmap_rec, err_msg='Recovered != original.')
        np.testing.assert_equal(mask, mask_rec)

    def test_logdet_ok(self):
        nside = 32
        n_batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(
            first_nside=nside, 
            last_nside=1, 
        )
        mask = np.ones_like(dmap)

        (zmap, zmask), logdet = transform(dmap, mask)

        self.assertEqual(
            logdet.shape[0], 
            n_batches, 
            'Logdet batch dim != input batch dim.'
        )
        self.assertFalse(np.isnan(logdet).any())
        (_, mask_rec), _ = transform.inverse_and_log_det(zmap, zmask)
        np.testing.assert_equal(mask, mask_rec)

    def test_inverse(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(first_nside=nside, last_nside=1)
        mask = np.ones_like(dmap)

        (dmap_transformed, zmask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(
            dmap_transformed, zmask
        )

        np.testing.assert_almost_equal(
            dmap * mask,
            dmap_recovered * mask, 
            err_msg='Recovered != original.'
        )
        np.testing.assert_equal(mask.astype(bool), mask_rec)

    def test_single_matrix_round_trip(self):
        B = 100; P = 3072; nside = 64
        transform = HadamardTransform(first_nside=nside, last_nside=1)

        v = np.random.randint(0, 10, size=(B, P, 4))
        m = np.random.randint(0, 2, size=(B, P, 4)).astype(bool)
        z, valid = transform._forward_matrix_product_masked(v, m)
        x = transform._inverse_matrix_product_masked(
            z, (m.astype(np.int32) * np.array([1,2,4,8])).sum(-1)
        )
        assert np.allclose(x * m, v * m)

    def test_encode_decode(self):
        B = 100; P = 3072; nside = 64
        transform = HadamardTransform(first_nside=nside, last_nside=1)

        m = np.random.randint(0, 2, size=(B, P, 4)).astype(bool)
        codes = transform._encode_mask(m)
        decoded = transform._decode_codes(codes)
        np.random.randint(0, 2, size=(B, P, 4)).astype(bool)

        assert np.array_equal(decoded, m.astype(bool))


    def test_inverse_random_mask(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(first_nside=nside, last_nside=1)
        mask = np.random.randint(low=0, high=2, size=dmap.shape)

        (dmap_transformed, zmask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(
            dmap_transformed, zmask
        )

        np.testing.assert_equal(mask, mask_rec)
        np.testing.assert_almost_equal(
            dmap * mask, 
            dmap_recovered * mask, 
            err_msg='Recovered != original.'
        )

    def test_inverse_random_mask_no_nans(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)).astype('float32') # ensure batchwise dim exists
        transform = HadamardTransform(first_nside=nside, last_nside=1)

        # hardcoded requirement for all masks to be the same
        mask = np.random.randint(low=0, high=2, size=dmap.shape[1], dtype=np.bool_) 
        # mask = np.random.randint(low=0, high=2, size=dmap.shape, dtype=np.bool_)
        mask = mask[None, :].repeat(axis=0, repeats=n_batches)
        dmap[~mask] = np.nan

        (dmap_transformed, zmask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(
            dmap_transformed, zmask
        )

        # check no nans have propagated through despite being masked
        assert (~np.isnan(dmap[mask])).all()
        assert (~np.isnan(dmap_transformed[zmask])).all()

        np.testing.assert_equal(mask, mask_rec)
        np.testing.assert_almost_equal(
            dmap[mask], 
            dmap_recovered[mask], 
            err_msg='Recovered != original.'
        )

    def test_inverse_random_mask_jax(self):
        nside = 32
        key = PRNGKey(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = jax.random.poisson(key, lam, shape=(n_batches, npix))
        transform = HadamardTransformJax(first_nside=nside, last_nside=1)
        mask = jax.random.randint(key, minval=0, maxval=2, shape=dmap.shape)

        (dmap_transformed, zmask), _ = transform(dmap, mask) # type: ignore
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(
            dmap_transformed, zmask
        )

        np.testing.assert_equal(np.asarray(mask), np.asarray(mask_rec))
        np.testing.assert_almost_equal(
            np.asarray(dmap * mask), 
            np.asarray(dmap_recovered * mask), 
            err_msg='Recovered != original.'
        )

    def test_inverse_low_nside(self):
        nside = 8
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(first_nside=nside, last_nside=1)

        mask = np.ones_like(dmap)

        (dmap_transformed, transformed_mask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(dmap_transformed, transformed_mask)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')
        np.testing.assert_equal(mask, mask_rec)

    def test_inverse_high_lambda(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(first_nside=nside, last_nside=1)

        mask = np.ones_like(dmap)

        (dmap_transformed, transformed_mask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(dmap_transformed, transformed_mask)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')
        np.testing.assert_equal(mask, mask_rec)

    def test_inverse_high_lambda_jax(self):
        # this one fails if jax uses float32 because of floating point errors
        # so we assert now in HaarWaveletTransform that float64 is turned on for jax
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)).astype('float32') # ensure batchwise dim exists
        transform = HadamardTransformJax(first_nside=nside, last_nside=1)

        mask = np.ones_like(dmap)

        (dmap_transformed, transformed_mask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(dmap_transformed, transformed_mask)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')
        np.testing.assert_equal(mask, mask_rec)

    def test_inverse_high_lambda_sparse(self):
        nside = 32
        batches = 1000
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 5000

        dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
        transform = HadamardTransform(
            first_nside=nside, 
            last_nside=1,
            matrix_type='sparse_average',
            normalise_details=True
        )

        mask = np.ones_like(dmap)

        (dmap_transformed, transformed_mask), _ = transform(dmap, mask)
        (dmap_recovered, mask_rec), _ = transform.inverse_and_log_det(dmap_transformed, transformed_mask)

        np.testing.assert_almost_equal(dmap, dmap_recovered, err_msg='Recovered != original.')
        np.testing.assert_equal(mask, mask_rec)

    def test_equals_unmasked_legacy_implementation(self):
        nside = 32
        key = prng_key(123)
        npix = 12 * nside**2
        lam = 50
        n_batches = 1000

        dmap = key.poisson(lam, shape=(n_batches, npix)) # ensure batchwise dim exists

        transform_new = HadamardTransform(first_nside=nside, last_nside=1)
        mask = np.ones_like(dmap)
        
        transform_old = ArchiveHadamardTransform(first_nside=nside, last_nside=1)

        (zmap_new, zmask), _ = transform_new(dmap, mask)
        zmap_old, _ = transform_old(dmap)

        np.testing.assert_almost_equal(
            zmap_new,
            zmap_old, 
            err_msg='New masked implementation (all sky) != legacy unmasked code.'
        )

# not implemented for now

    # def test_detail_unnormalisation_norm_sparse(self):
    #     nside = 32
    #     batches = 1000
    #     key = prng_key(123)
    #     npix = 12 * nside**2
    #     lam = 5000
    #
    #     dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
    #     transform = HadamardTransform(
    #         first_nside=nside, 
    #         last_nside=1,
    #         matrix_type='sparse_average',
    #         normalise_details=True
    #     )
    #     mask = np.ones_like(dmap)
    #
    #     (dmap_transformed, transformed_mask), _ = transform(dmap, mask)
    #     (_, mask_rec), _ = transform.inverse_and_log_det(dmap_transformed, transformed_mask)
    #     np.testing.assert_equal(mask, mask_rec)
    #
    #     # grab finest details
    #     lvl_idx = 0
    #     n_detail1_lvl0 = transform.mu_at_level_post['detail'][1][lvl_idx].shape[0]
    #     unnorm_func = transform.make_unnormalise_details_func(level=lvl_idx)
    #
    #     # coarse details 1s at front
    #     details_at_lvl0 = dmap_transformed[..., -3*n_detail1_lvl0:]
    #     details_lvl0_unnormed = unnorm_func(jnp.asarray(details_at_lvl0))
    #
    #     x = np.asarray(details_lvl0_unnormed)
    #     diff = np.abs(x - np.rint(x))
    #
    #     assert np.all(np.isfinite(x)) and is_integerish_f32(x), (
    #         f'Non-integer out: max diff is {np.max(diff)}'
    #     )
    #
    # def test_detail_unnormalisation_sparse_heirarchy(self):
    #     nside = 32
    #     batches = 1000
    #     key = prng_key(123)
    #     npix = 12 * nside**2
    #     lam = 5000
    #
    #     dmap = key.poisson(lam, shape=(batches, npix)) # ensure batchwise dim exists
    #     transform = HadamardTransform(
    #         first_nside=nside, 
    #         last_nside=1,
    #         matrix_type='sparse_average',
    #         normalise_details=True
    #     )
    #
    #     mask = np.ones_like(dmap)
    #
    #     (dmap_transformed, transformed_mask), _ = transform(dmap, mask)
    #     (_, mask_rec), _ = transform.inverse_and_log_det(dmap_transformed, transformed_mask)
    #     np.testing.assert_equal(mask, mask_rec)
    #     blocks = split_off_details(initial_nside=nside, output_nside=1)
    #     zmap = dmap_transformed.copy()
    #
    #     for lvl, (n_keep, n_drop) in enumerate(blocks):
    #         y_plus, y_minus = zmap[..., :n_keep], zmap[..., n_keep:]
    #         y_minus = jnp.asarray(y_minus)
    #         reconstructed_ints = transform.make_unnormalise_details_func(lvl)(y_minus)
    #         assert is_integerish_f32(np.asarray(reconstructed_ints))
    #         zmap = zmap[..., :n_keep]

if __name__ == "__main__":
    unittest.main()
