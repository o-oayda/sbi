# type: ignore
from pathlib import Path
from catsim.racs_jax import RacsJax
from catsim.utils.healsphere import downgrade_ignore_nan
from dipoleutils.utils.crossmatch import CrossMatch
from dipolesbi.tools.model_config_io import load_model_config
from catsim import Racs
from dipolesbi.tools.posterior_samples import format_posterior_samples, sample_posterior_csv
from dipolesbi.pipelines.based_racs import build_mask, build_real_sample
from dipoleutils.utils.data_loader import DataLoader
from dipoleutils.utils.samples import CatalogueToMap
from dipoleutils.utils.mask import Masker
from dipoleutils.utils.plotting import plot_binned_mean, plot_binned_quantile
from dipolesbi.pipelines.summary_stats import _flux_temperature_edges
from dipolesbi.tools.plotting import smooth_map
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
import jax
import numpy as np
from catsim.racs_temperature import evaluate_temperature_response
matplotlib.use('TkAgg')



def evaluate_temperature_enhancement(
        temperatures,
        beta,
        model,
        xp=np,
        reference_temperature = 25,
    ):
    return evaluate_temperature_response(
        temperatures, beta, reference_temperature, model=model, xp=xp
    )


def scale_line_to_reference_point(
        line,
        line_at_reference,
        point_temperatures,
        point_values,
        reference_temperature,
    ):
    point_temperatures = np.asarray(point_temperatures, dtype=float)
    point_values = np.asarray(point_values, dtype=float)
    finite = np.isfinite(point_temperatures) & np.isfinite(point_values)
    if not np.any(finite):
        raise ValueError('Cannot scale line without a finite reference point.')

    point_temperatures = point_temperatures[finite]
    point_values = point_values[finite]
    closest = np.argmin(np.abs(point_temperatures - reference_temperature))
    return np.asarray(line) * point_values[closest] / line_at_reference


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

    temp_model = model.cfg.temperature_model
    enhancement = evaluate_temperature_response(
        temperature,
        temp_beta,
        model.cfg.paf_reference_temp_c,
        model=temp_model,
        xp=np
    )

    flux = np.asarray(cat[product.columns.total_flux], dtype=np.float64)
    corrected_flux_column = f'{product.columns.total_flux}_temperature_corrected'
    cat[corrected_flux_column] = flux / enhancement
    cat['temperature_enhancement'] = enhancement
    cat['Temperature_C'] = temperature

    comparison_flux_min = (
        flux_min
        if model.cfg.flux_temperature_min_mjy is None
        else model.cfg.flux_temperature_min_mjy
    )
    comparison_view = CatalogueToMap(cat.copy())
    comparison_view.make_cut(
        product.columns.total_flux,
        minimum=comparison_flux_min,
        maximum=None,
    )
    if product.columns.source_name in cat.colnames:
        comparison_view.crossmatch_local_sources(
            'equatorial',
            radius=5,
            source_name_A_column=product.columns.source_name,
        )
    comparison_cat = comparison_view.get_catalogue()
    source_pixels = hp.ang2pix(
        model.nside,
        np.asarray(comparison_cat[product.columns.ra], dtype=np.float64),
        np.asarray(comparison_cat[product.columns.dec], dtype=np.float64),
        lonlat=True,
        nest=True,
    )
    comparison_keep = (
        np.asarray(model.mask_map, dtype=bool)[source_pixels]
        & np.isfinite(comparison_cat[product.columns.total_flux])
        & np.isfinite(comparison_cat['Temperature_C'])
    )
    comparison_cat = comparison_cat[comparison_keep]

    c2map = CatalogueToMap(cat)
    pmap = c2map.make_parameter_map('PSF_Maj', 'equatorial')
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
    return corrected_map, comparison_cat, pmap


def crossmatch_nvss(comparison_cat, flux_min):
    racs_view = CatalogueToMap(comparison_cat.copy())
    racs_view.make_cut('Total_flux', minimum=flux_min, maximum=None)
    nvss = DataLoader('nvss').load()
    nvss_view = CatalogueToMap(nvss)
    nvss_view.make_cut('integrated_flux', minimum=flux_min, maximum=1000)
    xmatch = CrossMatch(
        racs_view.get_catalogue(),
        nvss_view.get_catalogue(),
        coordinate_system='equatorial',
    )
    xmatch.cross_match(
        radius=5,
        source_name_A_column='Source_ID',
        source_name_B_column='source_name',
    )
    return xmatch.get_common_sources()


SURVEY = 'low3'
MODEL = 'hot_linear'
RESULTS_DICT = {
    'mid1': {
        'hot_exponential': 'racs-mid1-exp/20260716_171822_SEED0_NLE',
        'hot_quadratic': 'racs-mid1-quad/20260716_171905_SEED0_NLE',
        'hot_linear': 'racs-mid1-15mjy-sanity/20260716_143109_SEED0_NLE'
    },
    'low3': {
        'hot_exponential': '',
        'hot_quadratic': '',
        'hot_linear': 'racs-low3-15mjy-sanity/20260716_092116_SEED0_NLE'
    }
}
SURVEY_FREQUENCY_MHZ = {
    'low3': 943.5,
    'mid1': 1367.5,
}
root = Path.home() / 'Documents' / 'sbi' / 'archive' / RESULTS_DICT[SURVEY][MODEL]

DEFAULT_FLUX_TEMPERATURE_N_BINS = 10
DEFAULT_FLUX_TEMPERATURE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SELECTED_QUANTILE = 0.5
n_sim_bins = 10

selected_quantile_matches = np.flatnonzero(np.isclose(
    DEFAULT_FLUX_TEMPERATURE_QUANTILES,
    SELECTED_QUANTILE,
))
if selected_quantile_matches.size != 1:
    raise ValueError(
        f'SELECTED_QUANTILE={SELECTED_QUANTILE} is not uniquely present in '
        f'{DEFAULT_FLUX_TEMPERATURE_QUANTILES}.'
    )
selected_quantile_index = int(selected_quantile_matches[0])
selected_quantile_label = f'$q={SELECTED_QUANTILE:g}$ quantile'

key = jax.random.PRNGKey(0)
# root = Path.home() / 'Documents' / 'sbi' / 'racs-low3-moretemp-quad' / '20260714_184606_SEED0_NLE'
# root = Path.home() / 'Documents' / 'sbi' / 'racs-low3-15mjy-sanity' / '20260715_223437_SEED0_NLE'
# root = Path.home() / 'Documents' / 'sbi' / 'racs-low3-15mjy-sanity' / '20260716_092116_SEED0_NLE' #5mjy temp cut
# root = Path.home() / 'Documents' / 'sbi' / 'racs-low3-moretemp' / '20260714_151824_SEED0_NLE'

cfg = load_model_config(root / 'model_config.json')
cfg.paf_temperature_data_dir = '~/Documents/dipole-utils/data/paf_temps'
cfg.downscale_nside = None
cfg.mask_map = build_mask(64)

model = RacsJax(cfg)
model.initialise_data()

samples = format_posterior_samples(
    sample_posterior_csv(root / 'samples_rnd-19.csv', n_draws=500), output='dict'
)
median_temp_beta = float(np.median(samples['temp_beta']))

temperature_edges = _flux_temperature_edges(
    model,
    n_bins=n_sim_bins,
)
sim_dmap, sim_mask, temperature_summaries = (
    model.batch_generate_dipole_with_flux_temperature_summary(
        samples,
        key,
        batch_size=8,
        temperature_edges=temperature_edges,
        quantiles=DEFAULT_FLUX_TEMPERATURE_QUANTILES,
        show_progress=True,
    )
)
temperature_bin_centres = 0.5 * (temperature_edges[:-1] + temperature_edges[1:])
temperature_quantile_summaries = temperature_summaries.reshape(
    len(samples['temp_beta']),
    temperature_bin_centres.size,
    len(DEFAULT_FLUX_TEMPERATURE_QUANTILES),
)
posterior_predictive_quantiles = temperature_quantile_summaries[
    :, :, selected_quantile_index
]
posterior_predictive_mean = np.mean(posterior_predictive_quantiles, axis=0)
posterior_predictive_std = np.std(posterior_predictive_quantiles, axis=0, ddof=1)

# Return the masked/crossmatched catalogue used by the configured flux summary.
temperature_corrected_x0, cat, pmap = build_temperature_corrected_real_sample(
    model,
    flux_min=15,
    temp_beta=median_temp_beta,
)

CUT_MJY = (
    15 if cfg.flux_temperature_min_mjy is None
    else cfg.flux_temperature_min_mjy
)
# CUT_MJY = 15
cut_cat = cat[cat[model.product.columns.total_flux] > CUT_MJY]
# cut_cat = cat[cat[f'{model.product.columns.total_flux}_temperature_corrected'] > CUT_MJY]

# Crossmatched RACS/NVSS flux ratios. Scale NVSS from 1.4 GHz to the
# selected RACS survey frequency assuming S_nu proportional to nu**-0.8.
xmatch_table = crossmatch_nvss(cat, CUT_MJY)
scaled_nvss_flux = np.asarray(xmatch_table['B_integrated_flux'], dtype=float) * (
    SURVEY_FREQUENCY_MHZ[SURVEY] / 1400.0
) ** -0.8
racs_flux = np.asarray(xmatch_table['A_Total_flux'], dtype=float)
xmatch_temperature = np.asarray(xmatch_table['A_Temperature_C'], dtype=float)
xmatch_flux_ratio = racs_flux / scaled_nvss_flux
xmatch_valid = (
    np.isfinite(xmatch_temperature)
    & np.isfinite(xmatch_flux_ratio)
    & (scaled_nvss_flux > 0)
)

x = np.unique(np.append(
    np.linspace(np.nanmin(cat['Temperature_C']), np.nanmax(cat['Temperature_C'])),
    cfg.paf_reference_temp_c,
))
y = evaluate_temperature_enhancement(
    x, median_temp_beta, model=cfg.temperature_model,
    xp=np, reference_temperature=cfg.paf_reference_temp_c
)
y_at_reference = evaluate_temperature_enhancement(
    cfg.paf_reference_temp_c,
    median_temp_beta,
    model=cfg.temperature_model,
    xp=np,
    reference_temperature=cfg.paf_reference_temp_c,
)

#### WITH CUTS
fig, (quantile_ax, ratio_ax) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
# plot_binned_quantile(cut_cat['Temperature_C'], cut_cat[f'{model.product.columns.total_flux}_temperature_corrected'], label='Temperature corrected')
bin_temperatures, selected_flux_quantile, _, _ = plot_binned_quantile(
    cut_cat['Temperature_C'],
    cut_cat[f'{model.product.columns.total_flux}'],
    quantile=SELECTED_QUANTILE,
    color='tab:orange',
    label=f'Real {SURVEY} {selected_quantile_label}',
    bins=10,
    ax=quantile_ax,
)
quantile_ax.plot(
    temperature_bin_centres,
    posterior_predictive_mean,
    marker='o',
    color='tab:green',
    label=f'Simulated {selected_quantile_label}',
)
quantile_ax.fill_between(
    temperature_bin_centres,
    posterior_predictive_mean - posterior_predictive_std,
    posterior_predictive_mean + posterior_predictive_std,
    color='tab:green',
    alpha=0.2,
    label=r'Simulated $1\sigma$',
)
quantile_ax.set_xlabel('PAF temp')
quantile_ax.set_ylabel(f'Flux {selected_quantile_label} per temp bin')
quantile_ax.set_title(f'{SURVEY} sources > {CUT_MJY} mJy')
quantile_ax.legend()

ratio_bin_temperatures, mean_flux_ratio, _, _ = plot_binned_quantile(
    xmatch_temperature[xmatch_valid],
    xmatch_flux_ratio[xmatch_valid],
    bins=temperature_edges,
    n_bootstrap=500,
    ax=ratio_ax,
    color='tab:blue',
    label='Median RACS-NVSS ratio',
    linestyle=None
)
scaled_ratio_response = scale_line_to_reference_point(
    y,
    y_at_reference,
    ratio_bin_temperatures,
    mean_flux_ratio,
    cfg.paf_reference_temp_c,
)
ratio_ax.plot(
    x,
    scaled_ratio_response,
    linestyle='--',
    color='tab:red',
    label='Learned instrument response',
)
ratio_ax.set_xlabel('PAF temp')
ratio_ax.set_ylabel('RACS-NVSS flux ratio')
ratio_ax.set_title('Crossmatched sources to NVSS')
ratio_ax.legend()
fig.tight_layout()
plt.show()

catalogue = DataLoader(*model.product.data_loader_args).load()
x0, mask = build_real_sample(
    model,
    catalogue,
    flux_min=15,
    local_source_crossmatch_radius_arcsec=5.0,
    save_map_plot=False,
)

# compute per pixel mean and std, assume gaussian, and z-score x0 pixels
# but downscale first to resemble an average
NSIDE_LOW = 32
SMOOTH_SCALE_STR = 0.5
x0_low, _ = downgrade_ignore_nan(x0, mask, nside_out=NSIDE_LOW)
sim_low, _ = downgrade_ignore_nan(sim_dmap, sim_mask, nside_out=NSIDE_LOW)
x0_low = smooth_map(x0_low, only_return_data=True, angle_scale=SMOOTH_SCALE_STR)
sim_low = smooth_map(sim_low, only_return_data=True, angle_scale=SMOOTH_SCALE_STR)

sim_av_count = np.mean(sim_low, axis=0)
sim_std = np.std(sim_low, axis=0)
assert len(sim_av_count) == hp.nside2npix(NSIDE_LOW)
z_x0 = (x0_low - sim_av_count) / sim_std

hp.projview(z_x0, cmap='coolwarm', nest=True, unit='$z$-scored mid1 residuals: data $-$ simulations',
            min=-4, max=4, coord=['C'], rlabel='Equatorial', sub=121)
hp.projview(z_x0, cmap='coolwarm', nest=True, unit='$z$-scored mid1 residuals: data $-$ simulations',
            min=-4, max=4, coord=['C', 'G'], rlabel='Galactic', sub=122)
plt.show()

# all_dmap, all_mask = model.batch_generate_dipole(samples, key=key, batch_size=8)
# av_dmap = np.mean(all_dmap, axis=0)
# model.batch_generate_dipole()

# smooth_map(dmap, title='Posterior draw RACS-mid1', min=10, max=11)
# smooth_map(x0, title='RACS-mid1 real data (x0)', min=10, max=11)
# smooth_map(
#     temperature_corrected_x0,
#     title=f'RACS-mid1 temperature-corrected ($\\beta$={median_temp_beta:.2g})',
#     min=10,
#     max=11,
# )
# smooth_map(av_dmap, title='Average posterior predictive')
