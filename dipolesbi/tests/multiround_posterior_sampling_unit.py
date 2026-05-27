from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np

from dipolesbi.tools.multiround_inferer import MultiRoundInferer
from dipolesbi.tools.np_rngkey import NPKey


class _DummyInferer:
    _sample_posterior_for_simulations = (
        MultiRoundInferer._sample_posterior_for_simulations
    )
    _reformat_samples = MultiRoundInferer._reformat_samples


class _FakeNestedSamples:
    def __init__(self) -> None:
        self.random_state_types: list[type] = []
        self.replace_values: list[bool] = []

    def sample(self, *, n, random_state, replace):
        self.random_state_types.append(type(random_state))
        self.replace_values.append(replace)
        idx = random_state.choice(5, size=n, replace=replace, p=np.full(5, 0.2))
        return _FakeSampleFrame(idx.astype(np.float32))


class _FakeSampleFrame:
    def __init__(self, theta: np.ndarray) -> None:
        self.theta = theta

    def to_dict(self, orient):
        assert orient == "list"
        return {
            "theta": self.theta.tolist(),
            "logL": np.zeros_like(self.theta).tolist(),
            "logL_birth": np.zeros_like(self.theta).tolist(),
            "nlive": np.ones_like(self.theta).tolist(),
        }


def test_nle_posterior_sampling_uses_npkey_generator_and_replacement():
    nested = _FakeNestedSamples()
    inferer = _DummyInferer()
    inferer.mode = "NLE"
    inferer.current_nested_samples = nested

    key = NPKey.from_seed(123)
    samples_a = inferer._sample_posterior_for_simulations(key, 8)
    samples_b = inferer._sample_posterior_for_simulations(NPKey.from_seed(123), 8)

    assert np.array_equal(samples_a["theta"], samples_b["theta"])
    assert nested.replace_values == [True, True]
    assert nested.random_state_types == [np.random.Generator, np.random.Generator]


def test_npe_posterior_sampling_uses_npkey_choice_not_global_rng():
    inferer = _DummyInferer()
    inferer.mode = "NPE"
    inferer.current_posterior_samples = {
        "theta": np.arange(20, dtype=np.float32),
        "phi": np.arange(20, dtype=np.float32) + 100.0,
    }

    key = NPKey.from_seed(321)
    np.random.seed(0)
    samples_a = inferer._sample_posterior_for_simulations(key, 12)
    np.random.seed(999)
    samples_b = inferer._sample_posterior_for_simulations(NPKey.from_seed(321), 12)

    assert np.array_equal(samples_a["theta"], samples_b["theta"])
    assert np.all(samples_a["phi"] - samples_a["theta"] == 100.0)
