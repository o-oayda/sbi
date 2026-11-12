"""Export CatWISE observation plots and the actual GLASS input spectra."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from scipy.integrate import quad
import yaml

from dipolesbi.pipelines.prepare_catwise_observation import save_healpix_plot

SMOOTHING_SOLID_ANGLE = (
    1.0  # sr, matching smooth_map's default moving average.
)


def save_density_diagnostics(native_path, output_dir):
    """Plot native observed counts, retaining the mask during smoothing."""
    from dipolesbi.tools.plotting import smooth_map

    output = Path(output_dir)
    with np.load(native_path, allow_pickle=False) as data:
        native = np.array(data['x0'], dtype=float, copy=True)
        mask = np.asarray(data['mask'], dtype=bool)
    if native.ndim != 1 or native.shape != mask.shape:
        raise ValueError(
            'Native density map and mask must be matching vectors.'
        )
    native[~mask] = np.nan
    smoothed = smooth_map(
        native, angle_scale=SMOOTHING_SOLID_ANGLE, only_return_data=True
    )
    save_healpix_plot(output / 'reference_observation_native.png', native)
    save_healpix_plot(
        output / 'reference_observation_native_smoothed.png', smoothed
    )


def spectrum_arrays(prepared):
    """Retain shell pairs in GLASS order and form the biased population input."""
    import glass

    edges = np.asarray(prepared.bin_edges, dtype=float)
    weights = glass.position_weights(
        np.asarray(prepared.fractions),
        bias=np.asarray(prepared.effective_bias_per_shell),
    )
    nfields = len(edges) - 1
    if weights.shape != (nfields,) or not np.all(np.isfinite(weights)):
        raise ValueError('GLASS shell weights must match the shell edges.')
    count = nfields * (nfields + 1) // 2
    pairs = np.array(
        [(i, j) for i in range(nfields) for j in range(i, -1, -1)]
    )
    arrays = {
        'ell': np.arange(prepared.lmax + 1),
        'shell_edges': edges,
        'fractions': np.asarray(prepared.fractions),
        'bias': np.asarray(prepared.effective_bias_per_shell),
        'pairs': pairs,
    }
    for name, spectra in (
        ('raw', prepared.Cls),
        ('discrete', prepared.Cls_discrete),
    ):
        if len(spectra) != count:
            raise ValueError(
                'GLASS spectrum count does not match the shell count.'
            )
        matrix = np.zeros((count, prepared.lmax + 1))
        lengths = []
        for index, cl in enumerate(spectra):
            cl = np.asarray(cl)
            if (
                cl.ndim != 1
                or cl.size not in (0, prepared.lmax + 1)
                or not np.all(np.isfinite(cl))
            ):
                raise ValueError(
                    'GLASS spectrum has invalid dimensions or values.'
                )
            matrix[index, : cl.size] = cl
            lengths.append(cl.size)
        arrays[name] = matrix
        arrays[f'{name}_lengths'] = np.array(lengths)
        arrays[f'{name}_projected'] = glass.effective_cls(
            spectra, weights, lmax=prepared.lmax
        )
    arrays['metadata_json'] = json.dumps(
        {
            'format_version': 1,
            'cache_keys': prepared.cache_keys,
            'source': (
                'existing_cache'
                if hasattr(prepared, 'cache_files')
                else 'prepared_catsim_model'
            ),
            'cache_files': getattr(prepared, 'cache_files', {}),
            'nside': prepared.nside,
            'lmax': prepared.lmax,
            'ncorr': prepared.ncorr,
            'ordering': 'GLASS: (0,0), (1,1), (1,0), (2,2), (2,1), (2,0), ...',
            'projection': 'sum_ij f_i b_i f_j b_j C_ell_ij; before mask, shot noise and selection',
        },
        sort_keys=True,
    )
    return arrays


def plot_glass_spectra(arrays, path):
    """Compare raw and discretized input C_ell; never use the auxiliary gls."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    ell = np.asarray(arrays['ell'])
    metadata = json.loads(str(np.asarray(arrays['metadata_json']).item()))
    lmax = metadata['lmax']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')

    def draw(axis, cl, **kwargs):
        valid = (ell > 0) & np.isfinite(cl) & (cl > 0)
        axis.loglog(ell[valid], cl[valid], **kwargs)

    try:
        draw(
            axes[0], arrays['raw_projected'], color='black', label='Raw input'
        )
        draw(
            axes[0],
            arrays['discrete_projected'],
            color='tab:orange',
            linestyle='--',
            label='Discretized input',
        )
        axes[0].set_title('Population-weighted input spectrum')
        axes[0].legend()
        edges = arrays['shell_edges']
        norm = Normalize(edges[0], edges[-1])
        cmap = plt.get_cmap('viridis')
        for index, (i, j) in enumerate(arrays['pairs']):
            if i != j:
                continue
            color = cmap(norm((edges[i] + edges[i + 1]) / 2))
            draw(axes[1], arrays['raw'][index], color=color, alpha=0.7)
            draw(
                axes[1],
                arrays['discrete'][index],
                color=color,
                linestyle='--',
                alpha=0.7,
            )
        axes[1].set_title('Matter shell auto-spectra')
        axes[1].legend(
            handles=[
                Line2D([], [], color='black', label='Raw'),
                Line2D(
                    [], [], color='black', linestyle='--', label='Discretized'
                ),
            ]
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes[1],
            label='Shell redshift',
        )
        for axis in axes:
            axis.set_xlabel(r'Multipole $\ell$')
            axis.set_ylabel(r'$C_\ell$')
            axis.axvline(lmax, color='0.35', linestyle=':', linewidth=1.2)
            axis.text(
                lmax,
                0.5,
                rf'$\ell_{{\max}}={lmax}$',
                transform=axis.get_xaxis_transform(),
                rotation=90,
                ha='center',
                va='center',
                color='0.25',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'alpha': 0.85,
                    'pad': 2,
                },
            )
            axis.set_xlim(1, lmax * 1.3)
            axis.grid(False, which='both')
        fig.suptitle('GLASS inputs: before sky mask and Poisson shot noise')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def save_glass_diagnostics(prepared, output_dir):
    """Snapshot the spectra actually loaded/prepared by catsim before simulation."""
    if prepared is None:
        raise ValueError('GLASS model has not been prepared.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = spectrum_arrays(prepared)
    np.savez_compressed(output / 'glass_power_spectra.npz', **arrays)
    plot_glass_spectra(arrays, output / 'glass_power_spectrum.png')


def cached_glass_inputs(run, cache_dir=None):
    """Read one unambiguous matching raw/discrete cache pair without running CAMB."""
    from dipolesbi.pipelines.catwise_components import build_catwise_config

    config = build_catwise_config(run)
    cache = Path(
        cache_dir
        or config.glass_cache_dir
        or Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        / 'catsim'
        / 'glass'
    ).expanduser()
    entries = []
    for path in cache.glob('*.npz'):
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata'].item()))
                if isinstance(metadata, dict):
                    entries.append((path, metadata))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    matches = []
    for raw_path, raw in entries:
        if (
            raw.get('product') != 'cls'
            or raw.get('lmax') != config.glass_lmax
            or raw.get('cosmology') != config.glass_cosmology
            or raw.get('shell_edges') != list(config.glass_shell_edges)
        ):
            continue
        raw_key = raw_path.stem.removeprefix('cls_')
        for discrete_path, discrete in entries:
            if (
                discrete.get('product') == 'cls_discrete'
                and discrete.get('matter_key') == raw_key
                and discrete.get('lmax') == config.glass_lmax
                and discrete.get('nside') == run['observation']['native_nside']
                and discrete.get('ncorr') == config.glass_ncorr
            ):
                matches.append((raw_path, discrete_path, raw_key))
    if len(matches) != 1:
        raise ValueError(
            f'Expected one matching GLASS raw/discrete cache pair in {cache}; found {len(matches)}. No spectra were recomputed.'
        )
    raw_path, discrete_path, raw_key = matches[0]
    nfields = len(config.glass_shell_edges) - 1
    count = nfields * (nfields + 1) // 2

    def read(path):
        with np.load(path, allow_pickle=False) as data:
            return [
                np.array(data[f'cl_{i:06d}'], copy=True) for i in range(count)
            ]

    mass = np.array(
        [
            quad(config.glass_dndz, lo, hi)[0]
            for lo, hi in zip(
                config.glass_shell_edges[:-1], config.glass_shell_edges[1:]
            )
        ]
    )
    if not np.all(np.isfinite(mass)) or np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError(
            'Configured GLASS redshift distribution has invalid shell weights.'
        )
    bias = np.array(
        [
            (
                quad(
                    lambda z: config.glass_dndz(z) * config.glass_bias(z),
                    lo,
                    hi,
                )[0]
                / weight
                if weight > 0
                else 0.0
            )
            for lo, hi, weight in zip(
                config.glass_shell_edges[:-1],
                config.glass_shell_edges[1:],
                mass,
            )
        ]
    )
    return SimpleNamespace(
        Cls=read(raw_path),
        Cls_discrete=read(discrete_path),
        bin_edges=config.glass_shell_edges,
        fractions=mass / mass.sum(),
        effective_bias_per_shell=bias,
        nside=run['observation']['native_nside'],
        lmax=config.glass_lmax,
        ncorr=config.glass_ncorr,
        cache_keys={
            'cls': raw_key,
            'cls_discrete': discrete_path.stem.removeprefix('cls_discrete_'),
        },
        cache_files={'cls': str(raw_path), 'cls_discrete': str(discrete_path)},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--glass-cache-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        output = args.result_dir
        run = yaml.safe_load((output / 'experiment.resolved.yaml').read_text())
        save_density_diagnostics(
            output / 'reference_observation_native.npz', output
        )
        if run['model'].get('position_model') == 'glass':
            snapshot = output / 'glass_power_spectra.npz'
            if snapshot.exists():
                with np.load(snapshot, allow_pickle=False) as data:
                    plot_glass_spectra(
                        {key: np.array(data[key]) for key in data.files},
                        output / 'glass_power_spectrum.png',
                    )
            else:
                save_glass_diagnostics(
                    cached_glass_inputs(run, args.glass_cache_dir), output
                )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
