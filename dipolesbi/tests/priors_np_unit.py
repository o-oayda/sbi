import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import pytest

from dipolesbi.tools.priors_np import DipolePriorNP
from dipolesbi.tools.np_rngkey import NPKey


def test_add_prior_inserts_and_samples():
    prior = DipolePriorNP()
    prior.add_prior(
        short_name='amp',
        simulator_kwarg='amplitude',
        low=0.1,
        high=0.9,
        dist_type='uniform',
        index=1
    )

    expected_names = ['N', 'amp', 'D', 'phi', 'theta']
    assert prior.prior_names == expected_names

    assert prior.simulator_kwargs[1] == 'amplitude'

    key = NPKey.from_seed(42)
    samples = prior.sample(key, n_samples=5)
    assert samples['amplitude'].shape == (5,)
    assert np.all(samples['amplitude'] >= 0.1)
    assert np.all(samples['amplitude'] <= 0.9)

    lows = prior.low_ranges
    highs = prior.high_ranges
    assert np.isclose(lows[1], 0.1)
    assert np.isclose(highs[1], 0.9)

    logp = prior.log_prob(samples)
    assert np.isfinite(logp).all()


def test_change_kwarg_reflected_in_jax_prior():
    prior = DipolePriorNP()
    prior.change_kwarg('phi', 'lon')
    prior.change_kwarg('theta', 'lat')

    assert prior.simulator_kwargs[2:4] == ['lon', 'lat']

    jax_prior = prior.to_jax()
    assert jax_prior.simulator_kwargs[2:4] == ['lon', 'lat']


def test_to_jax_preserves_added_prior():
    prior = DipolePriorNP()
    prior.add_prior(
        short_name='amp',
        simulator_kwarg='amplitude',
        low=1.0,
        high=2.0,
        dist_type='uniform',
        index=0
    )

    jax_prior = prior.to_jax()

    assert jax_prior.prior_names[0] == 'amp'
    assert np.isclose(np.array(jax_prior.low_ranges[0]), 1.0)
    assert np.isclose(np.array(jax_prior.high_ranges[0]), 2.0)

    key = jax.random.PRNGKey(0)
    sample = jax_prior.sample(key)
    assert 'amplitude' in sample
    assert sample['amplitude'].shape == ()


def test_add_prior_duplicate_raises():
    prior = DipolePriorNP()
    prior.add_prior('extra', 'extra_param', 0.0, 1.0)
    with pytest.raises(ValueError):
        prior.add_prior('extra', 'another', 0.0, 1.0)
