from typing import Literal
from catsim import Catwise, CatwiseConfig
from catsim.simulator import downgrade_ignore_nan
from dipolesbi.tools.jax_ns import run_ns_from_chkpt
import jax
import numpy as np


DOWNSCALE_NSIDE = 4
CATWISE_VERSION: Literal['S21', 'S22'] = 'S22'
PATH_TO_CHKPT = 'S22_NLE/20260119_154826_SEED0_NLE/nflow_checkpoint.npz'
PRNG_SEED = 42
JOINT_SAMPLE: Literal['NVSS', 'RACS'] = 'NVSS'

config = CatwiseConfig(
    cat_w1_max=17.0, 
    cat_w12_min=0.5,
    magnitude_error_dist='gaussian',
    downscale_nside=DOWNSCALE_NSIDE,
    base_mask_version=CATWISE_VERSION,
    s21_catalogue_path=(
        '/home/oliver/Documents/catsim/src/catsim/data/'
        'catwise_agns_masked_final_w1lt16p5_alpha.fits'
    )
)

model = Catwise(config)
model.initialise_data()

if CATWISE_VERSION == 'S21':
    x0, mask = model.make_real_sample()
elif CATWISE_VERSION == 'S22':
    x0 = np.asarray(
        np.load('dipolesbi/catwise/catwise_S22.npy'), dtype=np.float32
    )
    mask = model.binary_mask
    x0[~mask] = np.nan

    x0, mask = downgrade_ignore_nan(x0, mask, DOWNSCALE_NSIDE)

out = run_ns_from_chkpt(
    path_to_chkpt=PATH_TO_CHKPT,
    data=x0,
    mask=mask,
    jax_key=jax.random.PRNGKey(PRNG_SEED)
)
