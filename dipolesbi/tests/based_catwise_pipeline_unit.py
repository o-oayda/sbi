from pathlib import Path
import shutil

import healpy as hp
import numpy as np
import pytest
import yaml
from astropy.table import Table

from dipolesbi.pipelines import based_catwise as pipeline
from dipolesbi.pipelines.catwise_config import load_catwise_run
from dipolesbi.pipelines.prepare_catwise_observation import prepare_catwise_observation
from dipolesbi.pipelines.validate_racs_data import DataValidationError, sha256
from dipolesbi.tools.np_rngkey import NPKey


@pytest.fixture
def prepared_run(tmp_path):
    configs = Path(__file__).resolve().parents[2] / 'workflow/configs/catwise'
    config_dir = tmp_path / 'configs'
    shutil.copytree(configs, config_dir)
    catalogue = tmp_path / 'catalogue.fits'
    Table(
        {'w1': [10.0, 11.0, 11.0], 'l': [0.0, 0.0, 90.0], 'b': [0.0, 0.0, 0.0]}
    ).write(catalogue)
    mask_path = tmp_path / 'mask.npy'
    np.save(mask_path, np.ones(hp.nside2npix(64), dtype=np.int8))
    observation_path = config_dir / 'observation.yaml'
    obs = yaml.safe_load(observation_path.read_text())
    obs.update(catalogue_sha256=sha256(catalogue), mask_sha256=sha256(mask_path))
    observation_path.write_text(yaml.safe_dump(obs))
    site_path = config_dir / 'site.yaml'
    site_path.write_text(
        yaml.safe_dump(
            {
                'paths': {
                    's21_catalogue': str(catalogue),
                    's21_mask': str(mask_path),
                    'output_root': str(tmp_path / 'results'),
                }
            }
        )
    )
    experiment = config_dir / 's21_nle_gaussian_poisson.yaml'
    run = load_catwise_run(experiment, site_path)
    paths = prepare_catwise_observation(run, tmp_path / 'prepared')
    return run, paths, experiment, site_path


@pytest.mark.parametrize('backend', ['numpy', 'jax'])
def test_run_inference_handoff(prepared_run, tmp_path, monkeypatch, backend):
    run, paths, _, _ = prepared_run
    run['execution']['backend'] = backend
    native_mask = np.load(paths['native_mask'])
    calls = {}

    class FakeModel:
        def __init__(self, config):
            calls['config'] = config
            self.downscale_nside = config.downscale_nside
            self.native_mask = native_mask
            self.mask_map = native_mask

        def initialise_data(self):
            calls['initialised'] = True

        def generate_dipole(self, **kwargs):
            calls['params'] = kwargs
            return (
                np.where(native_mask, np.arange(native_mask.size) % 3, np.nan),
                native_mask,
            )

        def batch_generate_dipole(self, theta, key, batch_size, show_progress):
            calls['params'] = theta
            return (
                np.where(native_mask, np.arange(native_mask.size) % 3, np.nan)[None, :],
                native_mask[None, :],
            )

    class FakeInferer:
        def __init__(self, mode, prior, simulator, observed, **kwargs):
            calls.update(
                mode=mode, prior=prior, observed=observed, inferer_kwargs=kwargs
            )
            calls['simulated'] = simulator(
                NPKey.from_seed(1), {'w1_extra_error': np.array([4.0])}
            )

        def run(self):
            calls['ran'] = True

    monkeypatch.setattr(pipeline, 'Catwise', FakeModel)
    monkeypatch.setattr(pipeline, 'CatwiseJax', FakeModel)
    monkeypatch.setattr(pipeline, 'MultiRoundInferer', FakeInferer)
    output = tmp_path / 'run output'
    pipeline.run_inference(
        run,
        paths['processed'],
        output,
        command=['catwise-run', '--out-dir', str(output)],
    )
    assert calls['initialised'] and calls['ran'] and calls['mode'] == 'NLE'
    assert calls['config'].downscale_nside is None
    assert calls['prior'].prior_names == [
        'N',
        'etaWX',
        'D',
        'phi',
        'theta',
        'lambda_clus',
    ]
    assert calls['simulated'][0].shape == (1, 193)
    assert np.all(calls['params']['w1_max'] == 16.4)
    assert calls['inferer_kwargs']['multi_round_config'].simulations_per_round == 3333
    assert not calls['inferer_kwargs']['use_ui']
    resolved = yaml.safe_load((output / 'experiment.resolved.yaml').read_text())
    assert resolved['execution']['effective_simulations'] == 49995
    assert resolved['inference']['multiround']['plot_save_dir'] == str(output)
    assert "'" in (output / 'run_command.txt').read_text()
    for name in (
        'prepared_observation.npz',
        'reference_observation_native.npz',
        'native_mask.npy',
        'validation-report.yaml',
    ):
        assert (output / name).exists()


def test_bad_simulator_mask_fails_before_inferer(prepared_run, tmp_path, monkeypatch):
    run, paths, _, _ = prepared_run

    class FakeModel:
        def __init__(self, config):
            self.mask_map = np.zeros(hp.nside2npix(64), dtype=bool)

        def initialise_data(self):
            pass

    def forbidden(*args, **kwargs):
        pytest.fail('Inferer was constructed before checking the simulator mask')

    monkeypatch.setattr(pipeline, 'CatwiseJax', FakeModel)
    monkeypatch.setattr(pipeline, 'MultiRoundInferer', forbidden)
    with pytest.raises(DataValidationError, match='Simulator native mask'):
        pipeline.run_inference(run, paths['processed'], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_invalid_observation_fails_before_model(prepared_run, tmp_path, monkeypatch):
    run, paths, _, _ = prepared_run
    mask = np.load(paths['native_mask'])
    mask[0] = ~mask[0]
    np.save(paths['native_mask'], mask)

    def forbidden(*args, **kwargs):
        pytest.fail('Model was constructed before observation validation')

    monkeypatch.setattr(pipeline, 'CatwiseJax', forbidden)
    with pytest.raises(DataValidationError, match='native mask'):
        pipeline.run_inference(run, paths['processed'], tmp_path / 'out')
