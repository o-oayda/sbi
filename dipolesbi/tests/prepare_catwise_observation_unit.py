from pathlib import Path

import healpy as hp
import numpy as np
import pytest
from astropy.table import Table
from catsim.utils.healsphere import Mask, downgrade_ignore_nan

from dipolesbi.pipelines.prepare_catwise_observation import (
    build_native_retain_mask,
    load_prepared_observation,
    prepare_catwise_observation,
    validate_catwise_inputs,
)
from dipolesbi.pipelines.validate_racs_data import DataValidationError, sha256


@pytest.fixture
def run(tmp_path):
    catalogue = tmp_path / 'catalogue.fits'
    # Endpoints must be excluded; catalogue is already colour-selected.
    Table(
        {
            'w1': [9.0, 9.1, 10.0, 10.0, 16.4],
            'l': [0.0, 0.0, 0.0, 90.0, 180.0],
            'b': [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).write(catalogue)
    mask = np.ones(hp.nside2npix(64), dtype=np.int8)
    mask[17] = 0
    mask_path = tmp_path / 'mask.npy'
    np.save(mask_path, mask)
    return {
        'paths': {'s21_catalogue': str(catalogue), 's21_mask': str(mask_path)},
        'observation': {
            'observation_id': 'test',
            'catalogue_version': 'S21',
            'base_mask_version': 'S21',
            'catalogue_sha256': sha256(catalogue),
            'mask_sha256': sha256(mask_path),
            'native_nside': 64,
            'downscale_nside': 4,
            'native_summaries': ['log_dispersion'],
            'w1_min': 9.0,
            'w1_max': 16.4,
            'w12_min': 0.8,
            'mask_north_ecliptic': False,
        },
    }


def test_mask_order_and_ecliptic_toggle(run):
    retain = build_native_retain_mask(run)
    ring = np.load(run['paths']['s21_mask'])
    np.testing.assert_array_equal(
        retain, hp.reorder(ring, r2n=True).astype(bool)
    )
    run['observation']['mask_north_ecliptic'] = True
    expected = retain.copy()
    expected[Mask(nside=64).north_ecliptic_mask()] = False
    np.testing.assert_array_equal(build_native_retain_mask(run), expected)


def test_prepare_and_reload(run, tmp_path):
    paths = prepare_catwise_observation(run, tmp_path / 'out')
    assert paths['native_plot'].read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    with np.load(paths['native']) as native:
        x = native['x0']
        pix = hp.ang2pix(64, [0.0, 90.0], [0.0, 0.0], lonlat=True, nest=True)
        assert x[pix[0]] == 2
        assert x[pix[1]] == 1
        assert np.nansum(x) == 3
    x, mask, metadata = load_prepared_observation(run, paths['processed'])
    assert x.shape == mask.shape == (193,)
    assert metadata['feature_layout']['coarse_nest_pixels'] == [0, 192]
    assert mask[-1] and np.isfinite(x[-1])
    with np.load(paths['native']) as native:
        expected_coarse, expected_mask = downgrade_ignore_nan(
            native['x0'], native['mask'], 4
        )
    np.testing.assert_array_equal(mask[:-1], expected_mask)
    np.testing.assert_allclose(x[:-1], expected_coarse, equal_nan=True)
    valid = np.load(paths['native'])['x0']
    valid = valid[np.isfinite(valid)]
    np.testing.assert_allclose(
        x[-1], np.log(np.var(valid, ddof=1) / np.mean(valid)), rtol=1e-5
    )
    run['observation']['w1_max'] = 16.3
    with pytest.raises(DataValidationError, match='metadata'):
        load_prepared_observation(run, paths['processed'])


def test_digest_and_mask_validation(run):
    validate_catwise_inputs(run)
    path = run['paths']['s21_mask']
    mask = np.load(path)
    mask[0] = 2
    np.save(path, mask)
    with pytest.raises(DataValidationError, match='checksum'):
        validate_catwise_inputs(run)
    run['observation']['mask_sha256'] = sha256(Path(path))
    with pytest.raises(DataValidationError, match='binary'):
        validate_catwise_inputs(run)


def test_saved_mask_tampering_rejected(run, tmp_path):
    paths = prepare_catwise_observation(run, tmp_path / 'out')
    mask = np.load(paths['native_mask'])
    mask[0] = ~mask[0]
    np.save(paths['native_mask'], mask)
    with pytest.raises(DataValidationError, match='native mask'):
        load_prepared_observation(run, paths['processed'])


def test_missing_required_column(run):
    path = run['paths']['s21_catalogue']
    Table({'w1': [10.0], 'l': [0.0]}).write(path, overwrite=True)
    run['observation']['catalogue_sha256'] = sha256(Path(path))
    with pytest.raises(DataValidationError, match='missing columns: b'):
        validate_catwise_inputs(run)


def test_preexisting_mask_is_verified_and_not_rewritten(run, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    path = output / 'native_mask.npy'
    np.save(path, build_native_retain_mask(run))
    # A fixed old timestamp makes the check independent of filesystem resolution.
    import os

    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    before = path.stat().st_mtime_ns
    prepare_catwise_observation(run, output, mask_path=path)
    assert path.stat().st_mtime_ns == before
    mask = np.load(path)
    mask[0] = ~mask[0]
    np.save(path, mask)
    with pytest.raises(DataValidationError, match='Supplied native mask'):
        prepare_catwise_observation(run, output, mask_path=path)
    with pytest.raises(DataValidationError, match='output_dir/native_mask'):
        prepare_catwise_observation(
            run, output, mask_path=tmp_path / 'other.npy'
        )
