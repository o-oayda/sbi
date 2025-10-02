import numpy as np
import pytest
import healpy as hp

from dipolesbi.tools.healpix_helpers import downgrade_ignore_nan


def _make_block_data(nside: int) -> tuple[np.ndarray, np.ndarray]:
    npix = hp.nside2npix(nside)
    values = np.arange(npix, dtype=np.float64)
    mask = np.ones(npix, dtype=bool)
    return values, mask


def test_downgrade_ignore_nan_basic_sum():
    vals, mask = _make_block_data(nside=2)

    degraded, degraded_mask = downgrade_ignore_nan(vals, mask, nside_out=1)

    blocks = vals.reshape(-1, 4)
    expected = blocks.sum(axis=1)

    np.testing.assert_allclose(degraded, expected)
    assert degraded_mask.dtype == bool
    assert np.all(degraded_mask)


def test_downgrade_ignore_nan_ignores_partial_masked_pixels():
    vals, mask = _make_block_data(nside=2)
    blocks = vals.reshape(-1, 4)
    mask_blocks = mask.reshape(-1, 4)

    blocks[2, 1] = 999.0
    mask_blocks[2, 1] = False
    blocks[7, 3] = 123.0
    mask_blocks[7, 3] = False

    vals = blocks.reshape(-1)
    mask = mask_blocks.reshape(-1)

    degraded, degraded_mask = downgrade_ignore_nan(vals, mask, nside_out=1)

    expected_vals = []
    expected_mask = []
    for block_vals, block_mask in zip(blocks, mask_blocks):
        if block_mask.any():
            expected_vals.append(block_vals[block_mask].sum())
            expected_mask.append(True)
        else:
            expected_vals.append(np.nan)
            expected_mask.append(False)

    np.testing.assert_allclose(degraded, np.asarray(expected_vals))
    np.testing.assert_array_equal(degraded_mask, np.asarray(expected_mask))


def test_downgrade_ignore_nan_full_mask_block_produces_nan():
    vals, mask = _make_block_data(nside=2)
    mask[:4] = False

    degraded, degraded_mask = downgrade_ignore_nan(vals, mask, nside_out=1)

    assert np.isnan(degraded[0])
    assert not degraded_mask[0]

    expected_rest = []
    blocks = vals.reshape(-1, 4)
    mask_blocks = mask.reshape(-1, 4)
    for block_vals, block_mask in zip(blocks[1:], mask_blocks[1:]):
        expected_rest.append(block_vals[block_mask].sum())

    np.testing.assert_allclose(degraded[1:], np.asarray(expected_rest))
    np.testing.assert_array_equal(degraded_mask[1:], np.ones_like(degraded_mask[1:], dtype=bool))


def test_downgrade_ignore_nan_preserves_float32_dtype():
    vals, mask = _make_block_data(nside=2)
    vals32 = vals.astype(np.float32)

    degraded, degraded_mask = downgrade_ignore_nan(vals32, mask, nside_out=1)

    assert degraded.dtype == np.float32
    assert degraded_mask.dtype == bool


def test_downgrade_ignore_nan_invalid_ratio_raises():
    vals, mask = _make_block_data(nside=2)
    with pytest.raises(AssertionError):
        downgrade_ignore_nan(vals, mask, nside_out=3)


def test_downgrade_ignore_nan_batched_inputs():
    vals, mask = _make_block_data(nside=2)

    batch_vals = np.stack([vals, vals + 1000.0], axis=0)
    batch_mask = np.stack([mask, mask.copy()], axis=0)

    batch_mask[1, :4] = False

    degraded, degraded_mask = downgrade_ignore_nan(batch_vals, batch_mask, nside_out=1)

    assert degraded.shape == (2, vals.size // 4)
    assert degraded_mask.shape == (2, vals.size // 4)

    single_out, single_mask = downgrade_ignore_nan(vals, mask, nside_out=1)
    np.testing.assert_allclose(degraded[0], single_out)
    np.testing.assert_array_equal(degraded_mask[0], single_mask)

    expected_second = []
    mask_blocks = batch_mask[1].reshape(-1, 4)
    val_blocks = batch_vals[1].reshape(-1, 4)
    for block_vals, block_mask in zip(val_blocks, mask_blocks):
        if block_mask.any():
            expected_second.append(block_vals[block_mask].sum())
        else:
            expected_second.append(np.nan)

    np.testing.assert_allclose(degraded[1], np.asarray(expected_second))
    assert not degraded_mask[1, 0]
    np.testing.assert_array_equal(
        degraded_mask[1, 1:],
        np.ones_like(degraded_mask[1, 1:], dtype=bool)
    )


def test_downgrade_ignore_nan_broadcast_mask():
    vals, mask = _make_block_data(nside=2)
    batch_vals = np.stack([vals, vals], axis=0)

    out_direct, out_mask_direct = downgrade_ignore_nan(batch_vals, mask, nside_out=1)

    expanded_mask = np.broadcast_to(mask, batch_vals.shape).copy()
    tgt, tgt_mask = downgrade_ignore_nan(batch_vals, expanded_mask, nside_out=1)

    np.testing.assert_allclose(out_direct, tgt)
    np.testing.assert_array_equal(out_mask_direct, tgt_mask)
