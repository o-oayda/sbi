import numpy as np
import pytest

from dipolesbi.pipelines.catwise_components import (
    build_catwise_config,
    build_prior,
    make_model_sim_wrapper,
    process_native_sample,
)
from dipolesbi.tools.np_rngkey import NPKey


def run(backend='jax'):
    entries = [
        ('N', 'log10_n_initial_samples', 7.4, 7.6, 'uniform'),
        ('etaWX', 'w1_extra_error', 0, 8, 'uniform'),
        ('D', 'observer_speed', 0, 8, 'uniform'),
        ('phi', 'dipole_longitude', 0, 360, 'uniform'),
        ('theta', 'dipole_latitude', -90, 90, 'polar'),
        ('lambda_clus', 'lambda_clus', 0, 8, 'uniform'),
    ]
    return {
        'model': {
            'simulate_clustering': 'poisson',
            'cat_w1_max': 17.0,
            'cat_w12_min': 0.5,
            'magnitude_error_dist': 'gaussian',
            'use_common_extra_error': True,
            'max_cluster_children_per_parent': 6,
        },
        'observation': {
            'native_nside': 2,
            'downscale_nside': 1,
            'native_count_summary': 'log_dispersion',
            'base_mask_version': 'S21',
            'w1_min': 9.0,
            'w1_max': 16.4,
            'w12_min': 0.8,
        },
        'paths': {
            's21_catalogue': '/tmp/catalogue',
            'source_catalogue': None,
            'hashgrid_cache': None,
        },
        'execution': {'backend': backend, 'chunk_size': 25, 'jax_batch_size': 2},
        'fixed_parameters': {},
        'priors': [
            dict(
                zip(('short_name', 'simulator_kwarg', 'low', 'high', 'dist_type'), row)
            )
            for row in entries
        ],
    }


def test_config_and_prior():
    cfg = build_catwise_config(run())
    assert cfg.downscale_nside is None
    assert cfg.use_common_extra_error and cfg.max_cluster_children_per_parent == 6
    prior = build_prior(run())
    assert prior.prior_names == ['N', 'etaWX', 'D', 'phi', 'theta', 'lambda_clus']
    assert prior.prior_dict['theta']['dist_type'] == 'polar'


def test_shared_processing():
    obs = run()['observation']
    counts = np.arange(48, dtype=float) + 1
    mask = np.ones(48, dtype=bool)
    mask[:4] = False
    output, output_mask = process_native_sample(counts, mask, obs)
    assert output.shape == (13,)
    assert not output_mask[0] and np.isnan(output[0])
    assert output[1] == sum(counts[4:8])
    assert output[-1] == pytest.approx(
        np.log(np.var(counts[4:], ddof=1) / np.mean(counts[4:]))
    )
    with pytest.raises(ValueError, match='shape'):
        process_native_sample(counts[:-1], mask, obs)


@pytest.mark.parametrize('backend', ['jax', 'numpy'])
def test_adapters_apply_cuts_and_identical_processing(backend):
    class Fake:
        downscale_nside = None

        def __init__(self):
            self.calls = []

        def generate_dipole(self, **kw):
            self.calls.append(kw)
            return np.arange(48, dtype=float) + 1, np.ones(48, dtype=bool)

        def batch_generate_dipole(self, theta, key, batch_size, show_progress):
            self.calls.append(theta)
            return np.tile(np.arange(48) + 1.0, (2, 1)), np.ones((2, 48), dtype=bool)

    fake = Fake()
    wrapper = make_model_sim_wrapper(fake, run(backend), np.ones(48, dtype=bool))
    x, mask = wrapper(NPKey.from_seed(4), {'w1_extra_error': np.array([2.0, 3.0])})
    assert x.shape == mask.shape == (2, 13)
    assert np.array_equal(x[0], x[1])
    assert np.all(fake.calls[0]['w1_max'] == 16.4)
    assert np.all(fake.calls[0]['w12_min'] == 0.8)
    assert 'w2_extra_error' not in fake.calls[0]
    with pytest.raises(ValueError, match='conflict'):
        wrapper(NPKey.from_seed(0), {'w1_min': np.array([9.0])})
    with pytest.raises(ValueError, match='mask'):
        make_model_sim_wrapper(fake, run(backend), np.zeros(48, dtype=bool))(
            NPKey.from_seed(0), {'w1_extra_error': np.array([2.0, 3.0])}
        )


def test_scenario_uses_resolved_contract(tmp_path):
    import yaml
    from pathlib import Path
    from dipolesbi.pipelines.catwise_components import build_scenario

    cfg = run()
    config_path = (
        Path(__file__).resolve().parents[2] / 'workflow/configs/catwise/inference.yaml'
    )
    cfg['inference'] = yaml.safe_load(config_path.read_text())
    cfg['inference']['multiround'].update(
        map_ndim=12,
        summary_ndim=1,
        native_count_summary='log_dispersion',
        simulations_per_round=3,
        effective_simulations=9,
    )
    cfg['execution'].update(seed=7, requested_simulations=10, n_rounds=3)
    cfg['reference_theta'] = {'observer_speed': 2.0}
    scenario = build_scenario(cfg, build_prior(cfg), str(tmp_path))
    assert scenario.multiround.simulations_per_round == 3
    assert scenario.multiround.summary_ndim == 1
    assert scenario.multiround.reference_theta == cfg['reference_theta']
    assert scenario.flow.architecture.count('surjective_MAF') == 1
    assert scenario.training.learning_rate == 1e-4
