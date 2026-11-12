"""GLASS workflow integration without running expensive CAMB preparation."""

from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from dipolesbi.pipelines.experiment_config import resolve_config
from catsim import CatwiseConfig
from catsim.configs import default_glass_bias, default_glass_dndz

from dipolesbi.pipelines.catwise_components import (
    build_catwise_config,
    build_prior,
)
from dipolesbi.pipelines.catwise_config import (
    CatwiseConfigError,
    DEFAULT_EXPERIMENT,
    load_catwise_run,
)
from dipolesbi.tests.based_catwise_pipeline_unit import prepared_run
from dipolesbi.tools.model_config_io import (
    load_model_config,
    save_model_config,
)


@pytest.fixture
def glass_config(tmp_path):
    directory = tmp_path / 'configs'
    shutil.copytree(DEFAULT_EXPERIMENT.parent, directory)
    experiment = directory / 's21_nle_gaussian_glass.yaml'
    edit(
        experiment,
        lambda config: config['execution'].update(jax_platforms='cpu'),
    )
    return experiment, directory / 'site.example.yaml'


def edit(path, update):
    data, _ = resolve_config(path)
    if isinstance(data.get("parameters"), dict):
        data["parameters"] = [
            {"short_name": key, **entry}
            for key, entry in data["parameters"].items()
        ]
    update(data)
    path.write_text(yaml.safe_dump(data))


def test_glass_config_and_model_roundtrip(glass_config, tmp_path):
    run = load_catwise_run(*glass_config)
    assert run['execution']['backend'] == 'numpy'
    assert run['execution']['jax_platforms'] == 'cpu'
    assert [p['short_name'] for p in run['priors']] == [
        'N',
        'etaWX',
        'D',
        'phi',
        'theta',
    ]
    assert 'lambda_clus' not in run['reference_theta']
    assert run['fixed_parameters'] == {'lambda_clus': 0.0}
    prior = build_prior(run)
    assert prior.ndim == 5
    config = build_catwise_config(run)
    defaults = CatwiseConfig(17.0, 0.5, 'gaussian')
    assert config.position_model == 'glass'
    assert config.glass_lmax == defaults.glass_lmax
    assert config.glass_ncorr == defaults.glass_ncorr
    assert config.glass_cosmology == defaults.glass_cosmology
    np.testing.assert_array_equal(
        config.glass_shell_edges, defaults.glass_shell_edges
    )
    assert config.glass_dndz is default_glass_dndz
    assert config.glass_bias is default_glass_bias
    assert config.glass_dndz(1.2) == defaults.glass_dndz(1.2)
    path = tmp_path / 'model_config.json'
    save_model_config(config, path)
    restored = load_model_config(path)
    assert restored == config
    assert restored.glass_bias(2.0) == 2.27


def test_existing_isotropic_config_serializes_new_callable_defaults(
    glass_config, tmp_path
):
    experiment, site = glass_config
    run = load_catwise_run(experiment.parent / DEFAULT_EXPERIMENT.name, site)
    config = build_catwise_config(run)
    assert config.position_model == 'isotropic'
    assert build_prior(run).ndim == 6
    path = tmp_path / 'model_config.json'
    save_model_config(config, path)
    restored = load_model_config(path)
    assert restored.glass_dndz is default_glass_dndz
    assert restored.glass_bias is default_glass_bias


@pytest.mark.parametrize(
    'field,value,message',
    [
        ('glass_shell_edges', [0.0, 0.2, 0.1], 'strictly increasing'),
        ('glass_shell_edges', [0.0, float('nan'), 1.0], 'finite'),
        ('glass_shell_edges', [0.0], 'too short'),
        ('glass_lmax', 192, 'must not exceed'),
        ('glass_lmax', True, 'integer'),
        ('glass_ncorr', -1, 'minimum'),
        ('glass_cosmology', {'wrong': 1.0}, 'Additional properties'),
        ('glass_cosmology', {'H0': 0.0}, 'positive'),
        ('glass_dndz', 'lambda z: z', 'does not match'),
    ],
)
def test_invalid_glass_settings_fail_before_simulator(
    glass_config, field, value, message
):
    experiment, site = glass_config
    edit(
        experiment.parent / 'model_glass.yaml',
        lambda m: m.update({field: value}),
    )
    with pytest.raises(CatwiseConfigError, match=message):
        load_catwise_run(experiment, site)


def test_glass_rejects_jax_and_additional_clustering(glass_config):
    experiment, site = glass_config
    edit(experiment, lambda c: c['execution'].update(backend='jax'))
    with pytest.raises(CatwiseConfigError, match='backend: numpy'):
        load_catwise_run(experiment, site)
    edit(
        experiment,
        lambda c: c['execution'].update(backend='numpy', jax_platforms='cuda'),
    )
    assert (
        load_catwise_run(experiment, site)['execution']['jax_platforms']
        == 'cuda'
    )
    edit(
        experiment.parent / 'model_glass.yaml',
        lambda m: m.update(simulate_clustering='poisson'),
    )
    with pytest.raises(CatwiseConfigError):
        load_catwise_run(experiment, site)


def test_glass_rejects_lambda_prior_and_requires_explicit_zero(glass_config):
    experiment, site = glass_config
    edit(experiment, lambda c: c.update(priors='priors.yaml'))
    with pytest.raises(CatwiseConfigError, match='fixed_parameters'):
        load_catwise_run(experiment, site)
    edit(experiment, lambda c: c.update(priors='priors_glass.yaml'))
    edit(
        experiment.parent / 'priors_glass.yaml',
        lambda c: c.update(fixed_parameters={}),
    )
    with pytest.raises(CatwiseConfigError, match='fixed_parameters'):
        load_catwise_run(experiment, site)


def test_optional_cache_and_custom_named_functions(glass_config):
    experiment, site = glass_config
    edit(site, lambda c: c['paths'].update(glass_cache='../glass-cache'))
    edit(
        experiment.parent / 'model_glass.yaml',
        lambda m: m.update(
            glass_bias='math:exp',
            glass_shell_edges=[0.1, 0.3, 0.8],
            glass_lmax=16,
            glass_ncorr=None,
            glass_cosmology={'H0': 70.0},
        ),
    )
    run = load_catwise_run(experiment, site)
    config = build_catwise_config(run)
    assert config.glass_cache_dir == str(site.parent.parent / 'glass-cache')
    assert config.glass_bias(0.0) == 1.0
    assert config.glass_shell_edges == (0.1, 0.3, 0.8)
    assert config.glass_lmax == 16 and config.glass_ncorr is None
    assert config.glass_cosmology['H0'] == 70.0
    assert config.glass_cosmology['ombh2'] == 0.0224
    edit(
        experiment.parent / 'model_glass.yaml',
        lambda m: m.update(glass_bias='math:pi'),
    )
    with pytest.raises(ValueError, match='callable'):
        build_catwise_config(load_catwise_run(experiment, site))


def test_nonimportable_callable_has_clear_serialization_error(tmp_path):
    config = CatwiseConfig(17.0, 0.5, 'gaussian', glass_bias=lambda z: 1.0)
    with pytest.raises(TypeError, match='importable named functions'):
        save_model_config(config, tmp_path / 'model.json')


def test_glass_runner_handoff_uses_five_priors_and_zero_lambda(
    prepared_run, tmp_path, monkeypatch
):
    from dipolesbi.pipelines import based_catwise as pipeline
    from dipolesbi.tools.np_rngkey import NPKey

    _, paths, experiment, site = prepared_run
    run = load_catwise_run(
        experiment.parent / 's21_nle_gaussian_glass.yaml', site
    )
    native_mask = np.load(paths['native_mask'])
    calls = {}

    class Model:
        def __init__(self, config):
            calls['config'] = config
            self.downscale_nside = None
            self.native_mask = native_mask

        def initialise_data(self):
            from types import SimpleNamespace

            self._glass_model = SimpleNamespace(
                bin_edges=np.array([0.0, 1.0, 2.0]),
                fractions=np.array([0.4, 0.6]),
                effective_bias_per_shell=np.ones(2),
                lmax=3,
                nside=64,
                ncorr=1,
                Cls=[np.ones(4), np.ones(4) * 2, np.ones(4) * 0.1],
                Cls_discrete=[np.ones(4) * 0.8, np.ones(4) * 1.5, np.empty(0)],
                cache_keys={
                    'cls': 'test-raw',
                    'cls_discrete': 'test-discrete',
                },
            )
            calls['prepared'] = True

        def generate_dipole(self, **kwargs):
            calls['theta'] = kwargs
            return np.arange(native_mask.size) % 3, native_mask

    class Inferer:
        def __init__(self, mode, prior, simulator, observation, **kwargs):
            assert prior.ndim == 5
            result, mask = simulator(
                NPKey.from_seed(42), {'w1_extra_error': np.array([2.0])}
            )
            assert result.shape == mask.shape == (1, 193)
            save_model_config(kwargs['model_config'], tmp_path / 'model.json')

        def run(self):
            calls['ran'] = True

    monkeypatch.setattr(pipeline, 'Catwise', Model)
    monkeypatch.setattr(pipeline, 'MultiRoundInferer', Inferer)
    pipeline.run_inference(run, paths['processed'], tmp_path / 'run')
    assert calls['prepared'] and calls['ran']
    assert calls['config'].position_model == 'glass'
    assert calls['theta']['lambda_clus'] == 0.0
    assert load_model_config(tmp_path / 'model.json').position_model == 'glass'
    assert (tmp_path / 'run/glass_power_spectrum.png').is_file()
    assert (tmp_path / 'run/glass_power_spectra.npz').is_file()
