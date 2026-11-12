"""Check the native pseudo spectrum against an independent masked transform."""

import shutil

import healpy as hp
import numpy as np
import pytest
import yaml

from dipolesbi.pipelines.catwise_components import process_native_sample
from dipolesbi.pipelines.catwise_config import (
    DEFAULT_EXPERIMENT,
    load_catwise_run,
)
from dipolesbi.pipelines.catwise_pseudo_cl import native_pseudo_cl


@pytest.mark.parametrize('nside, lmax', [(8, 12), (64, 128)])
def test_pseudo_cl_matches_masked_anafast(nside, lmax):
    pytest.importorskip('pymaster')
    rng = np.random.default_rng(14)
    counts = rng.poisson(10, hp.nside2npix(nside)).astype(float)
    mask = rng.random(counts.size) > 0.25
    counts[~mask] = np.nan
    observation = {'pseudo_cl_range': [2, lmax]}
    actual = native_pseudo_cl(counts, mask, observation)
    delta = np.zeros_like(counts)
    delta[mask] = counts[mask] / counts[mask].mean() - 1
    expected = hp.anafast(hp.reorder(delta, n2r=True), lmax=lmax, iter=3)
    np.testing.assert_allclose(actual, expected[2:], rtol=1e-5, atol=1e-10)
    # Masked NaNs do not leak into the estimator; the input remains intact.
    assert np.all(np.isnan(counts[~mask]))


def test_summary_appends_spectrum_and_keeps_mask():
    pytest.importorskip('pymaster')
    observation = {
        'native_nside': 8,
        'downscale_nside': 4,
        'native_count_summary': 'log_dispersion',
        'native_pseudo_cl': True,
        'pseudo_cl_range': [2, 12],
    }
    counts = np.random.default_rng(4).poisson(10, hp.nside2npix(8))
    mask = np.ones(counts.size, dtype=bool)
    actual, retain = process_native_sample(counts, mask, observation)
    baseline, baseline_mask = process_native_sample(
        counts, mask, {**observation, 'native_pseudo_cl': False}
    )
    np.testing.assert_array_equal(actual[:193], baseline)
    np.testing.assert_array_equal(retain[:193], baseline_mask)
    np.testing.assert_allclose(
        actual[193:], native_pseudo_cl(counts, mask, observation)
    )
    assert actual.shape == retain.shape == (204,)
    assert retain[193:].all()


def test_glass_summary_range_follows_model():
    run = load_catwise_run(
        DEFAULT_EXPERIMENT.parent / 's21_nle_gaussian_glass.yaml',
        DEFAULT_EXPERIMENT.parent / 'site.example.yaml',
    )
    assert run['observation']['pseudo_cl_range'] == [
        run['model']['glass_lmin'],
        run['model']['glass_lmax'],
    ]
    assert run['inference']['multiround']['summary_ndim'] == 128
    summary = run['summary']
    assert summary['total_ndim'] == 320
    assert summary['native_summary_ndim'] == 128
    assert [part['name'] for part in summary['components']] == [
        'downscaled_count_map',
        'log_dispersion',
        'pseudo_cl',
    ]
    spectrum = summary['components'][-1]
    assert spectrum['feature_slice'] == [193, 320]
    assert spectrum['ell_min'] == 2
    assert spectrum['ell_max'] == 128
    assert spectrum['mask_coupled']
    assert not spectrum['shot_noise_subtracted']


def test_custom_multipoles_change_summary_dimension(tmp_path):
    configs = tmp_path / 'configs'
    shutil.copytree(DEFAULT_EXPERIMENT.parent, configs)
    model_path = configs / 'model_glass.yaml'
    model = yaml.safe_load(model_path.read_text())
    model.update(glass_lmin=5, glass_lmax=20)
    model_path.write_text(yaml.safe_dump(model))
    run = load_catwise_run(
        configs / 's21_nle_gaussian_glass.yaml', configs / 'site.example.yaml'
    )
    assert run['observation']['pseudo_cl_range'] == [5, 20]
    assert run['inference']['multiround']['summary_ndim'] == 17
    assert run['summary']['total_ndim'] == 209
    assert run['summary']['components'][-1]['feature_slice'] == [193, 209]


def test_poisson_summary_lists_only_enabled_components():
    run = load_catwise_run(
        DEFAULT_EXPERIMENT, DEFAULT_EXPERIMENT.parent / 'site.example.yaml'
    )
    assert [part['name'] for part in run['summary']['components']] == [
        'downscaled_count_map',
        'log_dispersion',
    ]
    assert run['summary']['total_ndim'] == 193
    assert run['summary']['native_summary_ndim'] == 1
