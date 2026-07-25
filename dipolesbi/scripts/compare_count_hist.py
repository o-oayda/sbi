from catsim import RacsLow3, RacsLow3Config
from dipolesbi.tools import sample_posterior_csv, format_posterior_samples
from dipolesbi.tools.plotting import smooth_map
from dipolesbi.tools.posterior_samples import sample_posterior_npz
from dipolesbi.tools.utils import batch_simulate
from dipolesbi.pipelines.based_racs import build_mask, build_real_sample
from dipoleutils.utils.data_loader import DataLoader
import healpy as hp
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


N_DRAWS: int = 100
N_CPUS: int = 12
# RUN: str = 'racs/20260519_105023_SEED0_NLE' # no clustering, ln Z ~ -1018
# RUN: str = 'racs/20260519_164312' + '_SEED0_NLE' # clustering, ln Z ~ -1008
# RUN: str = 'racs/20260520_103733' + '_SEED0_NPE'
RUN: str = 'racs/20260519_213520' + '_SEED0_NLE'
ROUND: int = 14

if RUN.split('_')[-1] == 'NLE':
    path_to_samples = f'{RUN}/samples_rnd-14.csv'
    samples = sample_posterior_csv(path_to_samples, n_draws=N_DRAWS)
else:
    path_to_samples = f'{RUN}/samples_rnd-14.npz'
    samples = sample_posterior_npz(path_to_samples, n_draws=N_DRAWS)

samples_fmt = format_posterior_samples(samples, output='dict')
# samples_fmt['lambda_clus'] = 0.8 * np.ones((N_DRAWS,))
# temp_intercept = [-samples_fmt['temp_slope'][i]+1 for i in range(len(samples_fmt['temp_slope']))]
# samples_fmt['temp_intercept'] = temp_intercept
single_sample = {key: val[0] for key, val in samples_fmt.items()}


# temp intercept!!!!
# for some reason poisson clustering model is WAY too overdispersed despite the
# evidence being higher than a non-clustering model

# need a better way to make the model config portable otherwise I'm guessing
# at the params

config = RacsLow3Config(
    flux_min=15,
    nside=64,
    chunk_size=2_500_000,
    use_float32=False,
    cluster_count_model='poisson',
    downscale_nside=None, # don't downscale for comparison
    store_final_samples=True,
    alpha_mean=0.8,
    alpha_sigma=0.2,
    fractional_error_flux_min_mjy=10,
    paf_temperature_data_dir='/home/oliver/Documents/dipole-utils/data/paf_temps',
    mask_map=build_mask(64)
)
model = RacsLow3(config)
model.initialise_data()
catalogue = DataLoader(*model.product.data_loader_args).load()
x0, mask0 = build_real_sample(
    model,
    catalogue,
    15,
    local_source_crossmatch_radius_arcsec=5.0,
)
x, mask = batch_simulate(samples_fmt, model.generate_dipole, n_workers=N_CPUS)
x_av = np.nanmean(x, axis=0)
x_single, mask = model.generate_dipole(**single_sample)

bins = np.arange(np.nanmin(x), np.nanmax(x))
x_flat = x.flatten()

plt.hist(x_flat, bins=bins, alpha=0.3, density=True, label='Simulation', color='tab:blue')
plt.hist(x_flat, bins=bins, histtype='step', density=True, color='tab:blue')
# plt.hist(x_single, bins, alpha=0.3, density=True, label='Single sim.')
plt.hist(x0, bins=bins, density=True, alpha=0.3, label='Real', color='tab:orange')
plt.hist(x0, bins=bins, density=True, histtype='step', color='tab:orange')

x = bins
y = poisson(np.nanmean(x0)).pmf(bins)
plt.scatter(x+0.5, y, c='tab:red', label=r'$\mathrm{Pois}(\bar{\lambda}_{\mathrm{real}})$')
plt.legend()
plt.show()
