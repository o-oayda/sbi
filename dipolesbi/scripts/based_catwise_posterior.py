from dipolesbi.catwise.maps import Catwise, CatwiseReal
from typing import Literal
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.inference import LikelihoodFreeInferer
from dipolesbi.tools.simulator import Simulator
from sbi.utils import BoxUniform
from dipolesbi.tools.plotting import smooth_map, sky_probability
import matplotlib.pyplot as plt
import torch
from corner import corner
from dipolesbi.tools.constants import CMB_L, CMB_B, CMB_PHI_GAL, CMB_THETA_GAL, CMB_BETA
import healpy as hp
import numpy as np

SPEED_MULTIPLIER = 2
ERROR_SCALE = 2.05
N_SAMPLES = 36_000_000
ERROR_DIST = 'students-t'
SHAPE_PARAM = 1
SAMPLE: Literal['real', 'simulated'] = 'simulated'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
sim = Catwise(cat_w1_max=17.0, cat_w12_min=0.5, magnitude_error_dist=ERROR_DIST)
sim.initialise_data()

sim.generate_dipole(
    n_initial_samples=N_SAMPLES,
    observer_speed=SPEED_MULTIPLIER*CMB_BETA,
    w1_extra_error=ERROR_SCALE,
    w2_extra_error=ERROR_SCALE,
    dipole_longitude=215,
    dipole_latitude=40,
    log10_magnitude_error_shape_param=SHAPE_PARAM
)
smooth_map(sim.density_map)
plt.show()
dmap = sim.density_map
mask = np.isnan(dmap)
dmap[mask] = 0

# catwise = CatwiseReal()
# dmap = catwise.density_map
# smooth_map(dmap)
# plt.show()
# mask = torch.isnan(dmap)
# dmap[mask] = 0

prior = DipolePrior(
    mean_count_range=[30_000_000, 40_000_000],
    speed_range=[0, 8]
)
prior.add_prior(
    prior=BoxUniform(
        low= 0 * torch.ones(1),
        high=8 * torch.ones(1)
    ),
    short_name='etaW1',
    simulator_kwarg='w1_extra_error',
    index=1
)
prior.add_prior(
    prior=BoxUniform(
        low= 0 * torch.ones(1),
        high=8 * torch.ones(1)
    ),
    short_name='etaW2',
    simulator_kwarg='w2_extra_error',
    index=2
)
prior.add_prior(
    prior=BoxUniform(
        low=-1 * torch.ones(1),
        high=3 * torch.ones(1)
    ),
    short_name='nu',
    simulator_kwarg='log10_magnitude_error_shape_param',
    index=3
)

inferer = LikelihoodFreeInferer()
inferer.load_posterior('based_posterior_catwise_0p5_17p0_error_scale.pkl')
inferer.posterior.prior.custom_prior = prior
inferer.posterior.to('cpu')
samples = inferer.sample_amortized_posterior(x_obs=dmap, n_samps=20_000)

# hack rn
# samples[:, -1] = samples[:, -1] % torch.pi

sky_probability(samples, truth_star=[CMB_PHI_GAL, CMB_THETA_GAL])
plt.show()

# transform samples
# samples[:, -2] = np.rad2deg(samples[:, -2])
# samples[:, -1] = np.rad2deg(np.pi / 2 - samples[:, -1])
# samples[:, -3] = samples[:, -3] / 0.00123

corner(
    samples,
    # truths=[None, None, 1, CMB_L, CMB_B],
    truths=[N_SAMPLES, ERROR_SCALE, SPEED_MULTIPLIER, CMB_L, CMB_B],
    labels=[
        r'$N_{\mathrm{init.}}$',
        r'$\eta$',
        r'$v_{\mathrm{obs.}} / v_{\mathrm{CMB}}$',
        r'$l$ ($^\circ$)',
        r'$b$ ($^\circ$)'
    ],
    label_kwargs={
        'size': 20
    },
    truth_color='cornflowerblue'
)
plt.show()

# inferer.posterior_predictive_check(
#     n_samples=10,
#     x=dmap.to('cuda'),
#     simulator=sim.simulator
# )
#
# inferer.load_simulation('catwise_0p5_17p0_error_scale')
# inferer._check_for_mask_nans()
# inferer.run_simulation_based_calibration()
