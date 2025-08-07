from dipolesbi.catwise.maps import Catwise 
from typing import Literal
from dipolesbi.tools.inference import LikelihoodFreeInferer
from dipolesbi.tools.plotting import smooth_map, sky_probability
import matplotlib.pyplot as plt
from corner import corner
from dipolesbi.tools.constants import CMB_PHI_GAL, CMB_THETA_GAL
import numpy as np
from dipolesbi.tools.simulator import Simulator
import torch


SPEED_MULTIPLIER = 2
ERROR_SCALE_W1 = 2.05
ERROR_SCALE_W2 = 2.2
N_SAMPLES = 36_000_000
ERROR_DIST = 'gaussian'
SHAPE_PARAM = 1
SAMPLE: Literal['real', 'simulated'] = 'simulated'
DIPOLE_LONGITUDE = 215
DIPOLE_LATITUDE = 40
POSTERIOR_FILE = 'based_posterior_catwise_0p5_17p0_gaussian_etaw1w2.pkl'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
model = Catwise(cat_w1_max=17.0, cat_w12_min=0.5, magnitude_error_dist=ERROR_DIST)
model.initialise_data()

if SAMPLE == 'simulated':
    truths = [
        N_SAMPLES, ERROR_SCALE_W1, ERROR_SCALE_W2, SHAPE_PARAM, SPEED_MULTIPLIER,
        DIPOLE_LONGITUDE, DIPOLE_LATITUDE
    ]
    if ERROR_DIST != 'students-t':
        truths.pop(3)

    model.generate_dipole(
        n_initial_samples=N_SAMPLES,
        observer_speed=SPEED_MULTIPLIER,
        w1_extra_error=ERROR_SCALE_W1,
        w2_extra_error=ERROR_SCALE_W2,
        dipole_longitude=DIPOLE_LONGITUDE,
        dipole_latitude=DIPOLE_LATITUDE,
        log10_magnitude_error_shape_param=SHAPE_PARAM
    )
    smooth_map(model.density_map)
    plt.show()
    dmap = model.density_map
    mask = np.isnan(dmap)
    dmap[mask] = 0
else:
    truths = None

    catwise = model
    model.make_real_sample()
    dmap = catwise.real_density_map
    smooth_map(dmap)
    plt.show()
    mask = np.isnan(dmap)
    dmap[mask] = 0

inferer = LikelihoodFreeInferer()
inferer.load_posterior(POSTERIOR_FILE)
samples = inferer.sample_amortized_posterior(x_obs=dmap, n_samps=100_000)
sky_probability(samples, truth_star=[CMB_PHI_GAL, CMB_THETA_GAL], lonlat=True)
plt.show()

labels = [
    r'$N_{\mathrm{init.}}$', r'$\eta_{W1}$', r'$\eta_{W2}$', r'$\nu$',
    r'$v_{\mathrm{obs.}} / v_{\mathrm{CMB}}$', r'$l$ ($^\circ$)', r'$b$ ($^\circ$)'
]
if ERROR_DIST != 'students-t':
    labels.pop(3)
corner(
    samples.numpy(),
    truths=truths,
    labels=labels,
    label_kwargs={
        'size': 20
    },
    truth_color='cornflowerblue'
)
plt.show()

inferer.posterior_predictive_check(
    n_samples=5,
    model_callable=model.generate_dipole,
    x_real=dmap,
    mask=mask
)
#
# inferer.load_simulation('catwise_0p5_17p0_error_scale')
# inferer._check_for_mask_nans()
# inferer.run_simulation_based_calibration()
