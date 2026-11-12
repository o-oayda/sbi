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
            'native_summaries': ['log_dispersion'],
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
        'execution': {
            'backend': backend,
            'chunk_size': 25,
            'jax_batch_size': 2,
        },
        'fixed_parameters': {},
        'priors': [
            dict(
                zip(
                    (
                        'short_name',
                        'simulator_kwarg',
                        'low',
                        'high',
                        'dist_type',
                    ),
                    row,
                )
            )
            for row in entries
        ],
    }


def test_config_and_prior():
    cfg = build_catwise_config(run())
    assert cfg.downscale_nside is None
    assert (
        cfg.use_common_extra_error and cfg.max_cluster_children_per_parent == 6
    )
    prior = build_prior(run())
    assert prior.prior_names == [
        'N',
        'etaWX',
        'D',
        'phi',
        'theta',
        'lambda_clus',
    ]
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
            return np.tile(np.arange(48) + 1.0, (2, 1)), np.ones(
                (2, 48), dtype=bool
            )

    fake = Fake()
    wrapper = make_model_sim_wrapper(
        fake, run(backend), np.ones(48, dtype=bool)
    )
    x, mask = wrapper(
        NPKey.from_seed(4), {'w1_extra_error': np.array([2.0, 3.0])}
    )
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
        Path(__file__).resolve().parents[2]
        / 'workflow/configs/catwise/inference.yaml'
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
    assert scenario.multiround.native_count_summary == 'log_dispersion'
    assert scenario.multiround.simulations_per_round == 3
    assert scenario.multiround.summary_ndim == 1
    assert scenario.multiround.reference_theta == cfg['reference_theta']
    assert scenario.flow.architecture.count('surjective_MAF') == 1
    assert scenario.training.learning_rate == 1e-4


def test_numpy_adapter_passes_allocated_worker_count(monkeypatch):
    import dipolesbi.pipelines.catwise_components as components

    config = run('numpy')
    config['execution']['threads'] = 4
    calls = {}

    def dispatch(theta, model_callable, *, n_workers, rng_key, ui):
        from joblib.parallel import get_active_backend

        backend, _ = get_active_backend()
        calls.update(
            workers=n_workers,
            inner_threads=backend.inner_max_num_threads,
            theta=theta,
            ui=ui,
        )
        result = model_callable(
            **{name: values[0] for name, values in theta.items()},
            rng_key=rng_key.fold_in(0),
        )
        return np.tile(result[0], (4, 1)), np.tile(result[1], (4, 1))

    class Model:
        downscale_nside = None

        def generate_dipole(self, **kwargs):
            return np.arange(48) + 1.0, np.ones(48, dtype=bool)

    monkeypatch.setattr(components, 'batch_simulate', dispatch)
    wrapper = make_model_sim_wrapper(Model(), config)
    values, mask = wrapper(NPKey.from_seed(7), {'observer_speed': np.ones(4)})
    assert values.shape == mask.shape == (4, 13)
    assert calls['workers'] == 4
    assert calls['inner_threads'] == 1
    assert np.all(calls['theta']['w1_max'] == 16.4)


def test_numpy_pool_preserves_seeds_order_and_single_draw_shape(
    tmp_path, monkeypatch
):
    import json
    import os
    import time
    from pathlib import Path

    monkeypatch.setenv('OMP_NUM_THREADS', '32')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '32')

    class Model:
        downscale_nside = None

        def __init__(self, directory):
            self.directory = str(directory)

        def generate_dipole(self, rng_key, observer_speed, **kwargs):
            info = {
                'pid': os.getpid(),
                'omp': os.environ.get('OMP_NUM_THREADS'),
                'blas': os.environ.get('OPENBLAS_NUM_THREADS'),
            }
            Path(self.directory, f'{os.getpid()}.json').write_text(
                json.dumps(info)
            )
            time.sleep(0.05)
            counts = (
                rng_key.poisson(lam=4.0, shape=(48,)) + observer_speed + 1.0
            )
            return counts, np.ones(48, dtype=bool)

    config = run('numpy')
    directory = tmp_path / 'workers'
    directory.mkdir()
    model = Model(directory)
    params = {'observer_speed': np.arange(8, dtype=float)}
    key = NPKey.from_seed(42)
    parallel = make_model_sim_wrapper(model, config, n_workers=2)
    parallel_x, parallel_mask = parallel(key, params)
    workers = [
        json.loads(path.read_text()) for path in directory.glob('*.json')
    ]
    assert len(workers) == 2
    assert all(worker['pid'] != os.getpid() for worker in workers)
    assert all(worker['omp'] == worker['blas'] == '1' for worker in workers)
    serial = make_model_sim_wrapper(model, config, n_workers=1)
    serial_x, serial_mask = serial(key, params)
    np.testing.assert_array_equal(parallel_x, serial_x)
    np.testing.assert_array_equal(parallel_mask, serial_mask)
    single_x, single_mask = parallel(key, {'observer_speed': np.array([0.0])})
    assert single_x.shape == single_mask.shape == (1, 13)
    np.testing.assert_array_equal(single_x[0], serial_x[0])


@pytest.mark.parametrize('workers', [0, -1, True, 1.5])
def test_numpy_worker_count_must_be_positive_integer(workers):
    with pytest.raises(ValueError, match='positive integer'):
        make_model_sim_wrapper(object(), run('numpy'), n_workers=workers)


@pytest.mark.parametrize('n', [1, 2])
def test_numpy_dispatch_preserves_parameter_dtype(n):
    class Model:
        downscale_nside = None

        def generate_dipole(self, observer_speed, **kwargs):
            assert isinstance(observer_speed, np.float32)
            return np.arange(48) + 1.0, np.ones(48, dtype=bool)

    wrapper = make_model_sim_wrapper(Model(), run('numpy'), n_workers=1)
    x, mask = wrapper(
        NPKey.from_seed(8), {'observer_speed': np.ones(n, dtype=np.float32)}
    )
    assert x.shape == mask.shape == (n, 13)
