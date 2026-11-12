"""Diagnostics use actual shell spectra and never alter inference data."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dipolesbi.pipelines.catwise_diagnostics import (
    cached_glass_inputs,
    save_density_diagnostics,
    save_glass_diagnostics,
    spectrum_arrays,
)
from dipolesbi.pipelines.catwise_config import (
    DEFAULT_EXPERIMENT,
    load_catwise_run,
)


def prepared():
    return SimpleNamespace(
        bin_edges=np.array([0.0, 1.0, 2.0, 3.0]),
        fractions=np.array([0.2, 0.3, 0.5]),
        effective_bias_per_shell=np.array([2.0, 1.0, 1.0]),
        lmax=3,
        nside=64,
        ncorr=1,
        Cls=[np.ones(4) * i for i in range(1, 7)],
        Cls_discrete=[np.ones(4) * i for i in range(1, 6)] + [np.empty(0)],
        cache_keys={'cls': 'abc', 'cls_discrete': 'def'},
    )


def test_glass_pair_order_and_projected_cross_terms():
    data = spectrum_arrays(prepared())
    np.testing.assert_array_equal(
        data['pairs'], [[0, 0], [1, 1], [1, 0], [2, 2], [2, 1], [2, 0]]
    )
    np.testing.assert_allclose(data['raw_projected'], 5.96)
    np.testing.assert_allclose(data['discrete_projected'], 3.56)
    assert data['discrete_lengths'][-1] == 0
    np.testing.assert_array_equal(data['discrete'][-1], 0.0)


def test_glass_projection_matches_explicit_shell_sum():
    model = prepared()
    rng = np.random.default_rng(42)
    model.Cls = [rng.random(4) for _ in model.Cls]
    model.Cls_discrete = [rng.random(4) for _ in model.Cls_discrete]
    model.Cls_discrete[-1] = np.empty(0)
    data = spectrum_arrays(model)
    weights = model.fractions * model.effective_bias_per_shell
    for name in ('raw', 'discrete'):
        expected = np.zeros(model.lmax + 1)
        for (i, j), cl in zip(data['pairs'], data[name]):
            expected += weights[i] * weights[j] * cl * (1 if i == j else 2)
        np.testing.assert_allclose(
            data[f'{name}_projected'], expected, rtol=1e-13, atol=0
        )


def test_spectrum_snapshot_and_plot_are_from_prepared_inputs(tmp_path):
    model = prepared()
    original = model.Cls[0].copy()
    save_glass_diagnostics(model, tmp_path)
    with np.load(
        tmp_path / 'glass_power_spectra.npz', allow_pickle=False
    ) as data:
        np.testing.assert_array_equal(data['raw'][0], original)
        assert (
            json.loads(data['metadata_json'].item())['cache_keys']['cls']
            == 'abc'
        )
        np.testing.assert_allclose(data['raw_projected'], 5.96)
    assert (
        (tmp_path / 'glass_power_spectrum.png')
        .read_bytes()
        .startswith(b'\x89PNG')
    )
    np.testing.assert_array_equal(model.Cls[0], original)
    assert not plt.get_fignums()


def test_density_smoothing_uses_existing_utility_and_preserves_mask(tmp_path):
    from dipolesbi.tools.plotting import smooth_map
    from dipolesbi.pipelines import catwise_diagnostics as module

    native = np.arange(48, dtype=float)
    mask = np.ones(48, dtype=bool)
    mask[2] = False
    path = tmp_path / 'native.npz'
    np.savez(path, x0=native, mask=mask)
    before = path.read_bytes()
    captured = {}

    def record(output, values):
        captured[output.name] = np.array(values)

    from unittest.mock import patch

    with patch.object(module, 'save_healpix_plot', record):
        save_density_diagnostics(path, tmp_path)
    expected = native.copy()
    expected[~mask] = np.nan
    np.testing.assert_array_equal(
        captured['reference_observation_native.png'], expected
    )
    np.testing.assert_array_equal(
        captured['reference_observation_native_smoothed.png'],
        smooth_map(expected, angle_scale=1.0, only_return_data=True),
    )
    assert path.read_bytes() == before


def test_cache_reader_matches_raw_and_discrete_without_camb(tmp_path):
    run = load_catwise_run(
        DEFAULT_EXPERIMENT.parent / 's21_nle_gaussian_glass_with_cls.yaml',
        DEFAULT_EXPERIMENT.parent / 'site.example.yaml',
    )
    run['model'].update(
        glass_shell_edges=[0.0, 1.0, 2.0, 3.0], glass_lmax=3, glass_ncorr=1
    )
    model = prepared()
    metadata = {
        'product': 'cls',
        'lmax': 3,
        'shell_edges': [0.0, 1.0, 2.0, 3.0],
        'cosmology': run['model']['glass_cosmology'],
    }
    discrete = {
        'product': 'cls_discrete',
        'lmax': 3,
        'nside': 64,
        'ncorr': 1,
        'matter_key': 'abc',
    }
    for name, meta, values in [
        ('cls_abc.npz', metadata, model.Cls),
        ('cls_discrete_def.npz', discrete, model.Cls_discrete),
    ]:
        np.savez(
            tmp_path / name,
            metadata=json.dumps(meta),
            **{f'cl_{i:06d}': cl for i, cl in enumerate(values)},
        )
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    loaded = cached_glass_inputs(run, tmp_path)
    np.testing.assert_array_equal(loaded.Cls[5], model.Cls[5])
    assert loaded.cache_keys == model.cache_keys
    assert loaded.fractions.sum() == pytest.approx(1.0)
    assert set(before) == {p.name for p in tmp_path.iterdir()}
    assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    # A different CAMB/cache version with identical declared settings is ambiguous.
    np.savez(
        tmp_path / 'cls_discrete_other.npz', metadata=json.dumps(discrete)
    )
    with pytest.raises(ValueError, match='found 2'):
        cached_glass_inputs(run, tmp_path)


def test_cache_reader_does_not_rebuild_missing_spectra(tmp_path):
    run = load_catwise_run(
        DEFAULT_EXPERIMENT.parent / 's21_nle_gaussian_glass_with_cls.yaml',
        DEFAULT_EXPERIMENT.parent / 'site.example.yaml',
    )
    with pytest.raises(ValueError, match='No spectra were recomputed'):
        cached_glass_inputs(run, tmp_path)
    assert not list(tmp_path.iterdir())


def test_spectrum_plot_uses_plain_cl_and_marks_lmax(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from dipolesbi.pipelines.catwise_diagnostics import plot_glass_spectra

    figures = []
    monkeypatch.setattr(
        Figure, 'savefig', lambda fig, *args, **kwargs: figures.append(fig)
    )
    data = spectrum_arrays(prepared())
    plot_glass_spectra(data, tmp_path / 'spectrum.png')
    for axis in figures[0].axes[:2]:
        assert axis.get_ylabel() == r'$C_\ell$'
        assert axis.get_xscale() == axis.get_yscale() == 'log'
        marker = next(
            line for line in axis.lines if line.get_linestyle() == ':'
        )
        np.testing.assert_array_equal(marker.get_xdata(), [3, 3])
        label = next(text for text in axis.texts if r'\max' in text.get_text())
        assert label.get_rotation() == 90
        assert label.get_position()[0] == 3
    # No ell-dependent or 2*pi rescaling of either projected curve.
    np.testing.assert_array_equal(
        figures[0].axes[0].lines[0].get_ydata(), data['raw_projected'][1:]
    )
    np.testing.assert_array_equal(
        figures[0].axes[0].lines[1].get_ydata(), data['discrete_projected'][1:]
    )
    # Every shell keeps both raw and discretized auto-spectra.
    assert len(figures[0].axes[1].lines) == 2 * 3 + 1
