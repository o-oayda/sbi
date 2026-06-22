from pathlib import Path
from catsim.racs_jax import RacsJax
from dipolesbi.tools.model_config_io import load_model_config
from catsim import Racs, smooth_map
from dipolesbi.tools.posterior_samples import format_posterior_samples, sample_posterior_csv
from dipolesbi.pipelines.based_racs import build_mask, build_real_sample
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.samples import CatalogueToMap
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
import jax
import numpy as np
matplotlib.use('TkAgg')


def build_temperature_corrected_real_sample(model, flux_min, temp_beta):
    product = model.product
    cat = DataLoader(*product.data_loader_args).load().copy()

    sbid = np.asarray(cat[product.columns.tile_id], dtype=np.int64)
    tile_indices = np.array(
        [model._tile_index_from_sbid.get(int(source_sbid), -1) for source_sbid in sbid],
        dtype=np.int32,
    )

    temperature = np.full(len(cat), np.nan, dtype=np.float64)
    valid_tile = tile_indices >= 0
    temperature[valid_tile] = np.asarray(
        model.tile_temperature_by_index,
        dtype=np.float64,
    )[tile_indices[valid_tile]]

    enhancement = np.ones(len(cat), dtype=np.float64)
    finite_temperature = np.isfinite(temperature)
    hot_temperature = np.maximum(
        temperature[finite_temperature] - model.cfg.paf_reference_temp_c,
        0.0,
    )
    enhancement[finite_temperature] = 1.0 - temp_beta * hot_temperature
    enhancement = np.maximum(enhancement, 1e-6)

    flux = np.asarray(cat[product.columns.total_flux], dtype=np.float64)
    corrected_flux_column = f'{product.columns.total_flux}_temperature_corrected'
    cat[corrected_flux_column] = flux / enhancement
    cat['temperature_enhancement'] = enhancement
    cat['Temperature_C'] = temperature

    c2map = CatalogueToMap(cat)
    c2map.make_cut(corrected_flux_column, minimum=flux_min, maximum=None)
    if product.columns.source_name in cat.colnames:
        c2map.crossmatch_local_sources(
            'equatorial',
            radius=5,
            source_name_A_column=product.columns.source_name,
        )
    corrected_map = c2map.make_density_map(
        coordinate_system='equatorial',
        nside=model.nside,
        nest=True,
    ).astype('float32')
    corrected_map[~model.mask_map.astype(bool)] = np.nan
    return corrected_map


key = jax.random.PRNGKey(0)
# root = Path('racs-mid-17-06-26/20260617_144129_SEED0_NLE')
root = Path('/home/oliver/Documents/sbi/racs-elevation/20260621_132306_SEED0_NLE')

cfg = load_model_config(root / 'model_config.json')
cfg.paf_temperature_data_dir = '~/Documents/dipole-utils/data/paf_temps'
cfg.downscale_nside = None
cfg.mask_map = build_mask(64)

model = RacsJax(cfg)
model.initialise_data()
x0, mask = build_real_sample(model, flux_min=15, save_map_plot=False)

samples = format_posterior_samples(
    sample_posterior_csv(root / 'samples_rnd-19.csv', n_draws=500), output='dict'
)
single_sample = {key: val[0] for key, val in samples.items()}
median_temp_beta = float(np.median(samples['temp_beta']))

dmap, mask = model.generate_dipole(**single_sample)
all_dmap, all_mask = model.batch_generate_dipole(samples, key=key, batch_size=8)
av_dmap = np.mean(all_dmap, axis=0)
temperature_corrected_x0 = build_temperature_corrected_real_sample(
    model,
    flux_min=15,
    temp_beta=median_temp_beta,
)
# model.batch_generate_dipole()

smooth_map(dmap, title='Posterior draw RACS-mid1', min=10, max=11)
smooth_map(x0, title='RACS-mid1 real data (x0)', min=10, max=11)
smooth_map(
    temperature_corrected_x0,
    title=f'RACS-mid1 temperature-corrected ($\\beta$={median_temp_beta:.2g})',
    min=10,
    max=11,
)
smooth_map(av_dmap, title='Average posterior predictive')
plt.show()
