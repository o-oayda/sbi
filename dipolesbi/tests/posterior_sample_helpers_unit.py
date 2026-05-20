from __future__ import annotations

import numpy as np
import pytest

from dipolesbi.tools import format_posterior_samples, sample_posterior_npz


def test_sample_posterior_npz_returns_consistent_dict_draws(tmp_path):
    path = tmp_path / "samples_rnd-14.npz"
    np.savez(
        path,
        alpha=np.arange(10, dtype=np.float32),
        beta=np.arange(10, dtype=np.float32) + 100,
    )

    draws = sample_posterior_npz(path, n_draws=4, random_state=0)

    assert set(draws) == {"alpha", "beta"}
    assert draws["alpha"].shape == (4,)
    assert np.all(draws["beta"] - draws["alpha"] == 100)


def test_format_posterior_samples_supports_npz_dict_outputs(tmp_path):
    path = tmp_path / "samples_rnd-14.npz"
    np.savez(
        path,
        alpha=np.arange(10, dtype=np.float32),
        beta=np.arange(10, dtype=np.float32) + 100,
        logL=np.arange(10, dtype=np.float32),
        nlive=np.arange(10, dtype=np.float32),
    )

    draws = sample_posterior_npz(path, n_draws=5, random_state=1)
    array = format_posterior_samples(draws, output="array")
    sample_dict = format_posterior_samples(draws, output="dict")

    assert array.shape == (5, 2)
    assert set(sample_dict) == {"alpha", "beta"}
    assert np.array_equal(array[:, 0], sample_dict["alpha"])
    assert np.array_equal(array[:, 1], sample_dict["beta"])


def test_sample_posterior_npz_rejects_mismatched_lengths(tmp_path):
    path = tmp_path / "samples_rnd-14.npz"
    np.savez(path, alpha=np.arange(10), beta=np.arange(9))

    with pytest.raises(ValueError, match="same leading length"):
        sample_posterior_npz(path, n_draws=2, random_state=0)


def test_sample_posterior_npz_rejects_non_npz(tmp_path):
    path = tmp_path / "samples_rnd-14.csv"
    path.write_text("alpha\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NPZ"):
        sample_posterior_npz(path, n_draws=2)
