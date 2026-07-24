import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import healpy as hp
import numpy as np
import pytest

from dipolesbi.tools.maps import average_smooth_map
from dipolesbi.tools.plotting import smooth_map


def test_average_smooth_map_batch_matches_independent_rows():
    npix = hp.nside2npix(2)
    first = np.arange(npix, dtype=np.float32)
    second = first[::-1].copy()
    first[3] = np.nan
    second[8] = np.nan
    batch = np.stack([first, second])

    smoothed_batch = average_smooth_map(batch, angle_scale=0.5)
    smoothed_rows = np.stack(
        [average_smooth_map(row, angle_scale=0.5) for row in batch]
    )

    assert smoothed_batch.shape == batch.shape
    np.testing.assert_allclose(smoothed_batch, smoothed_rows, equal_nan=True)


def test_smooth_map_preserves_single_map_and_batch_shapes():
    npix = hp.nside2npix(1)
    single = np.arange(npix, dtype=np.float32)
    batch = np.stack([single, single + 1.0])

    smoothed_single = smooth_map(single, only_return_data=True, angle_scale=0.5)
    smoothed_batch = smooth_map(batch, only_return_data=True, angle_scale=0.5)

    assert smoothed_single is not None
    assert smoothed_batch is not None
    assert smoothed_single.shape == single.shape
    assert smoothed_batch.shape == batch.shape


def test_smooth_map_rejects_plotting_a_batch():
    batch = np.ones((2, hp.nside2npix(1)), dtype=np.float32)

    with pytest.raises(ValueError, match="only_return_data=True"):
        smooth_map(batch)
