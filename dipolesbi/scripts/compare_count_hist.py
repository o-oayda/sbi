from catsim import RacsLow3, RacsLow3Config
from dipolesbi.tools import sample_posterior_csv, format_posterior_samples
from dipolesbi.tools.plotting import smooth_map
from dipolesbi.tools.utils import batch_simulate
from dipolesbi.scripts.based_racs_low3 import _build_mask
import healpy as hp
import matplotlib; matplotlib.use('macosx')
import matplotlib.pyplot as plt


N_DRAWS: int = 100
N_CPUS: int = 12
# RUN: str = 'racs/20260517_190530_SEED0_NLE'
RUN: str = 'racs/20260518_122920_SEED0_NLE'
ROUND: int = 14
path_to_samples = f'{RUN}/samples_rnd-14.csv'

samples = sample_posterior_csv(path_to_samples, n_draws=N_DRAWS)
samples_fmt = format_posterior_samples(samples, output='dict')

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
    mask_map=_build_mask(64)
)
model = RacsLow3(config)
model.initialise_data()
x, mask = model.generate_dipole(**samples_fmt)
# x, mask = batch_simulate(samples_fmt, model.generate_dipole, n_workers=N_CPUS)
