from dipolesbi.catwise.maps import CatwiseSim, CatwiseReal
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.inference import Inference
from dipolesbi.tools.plotting import smooth_map
import matplotlib.pyplot as plt
import torch
from corner import corner
from dipolesbi.tools.constants import CMB_L, CMB_B
import healpy as hp
import numpy as np

# sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
# sim.initialise_data()
# sim.generate_dipole(n_initial_samples=30_000_000)
# smooth_map(sim.density_map)
# dmap = sim.density_map
# mask = torch.isnan(dmap)
# dmap[mask] = 0

catwise = CatwiseReal()
dmap = catwise.density_map
smooth_map(dmap)
plt.show()

mask = torch.isnan(dmap)
dmap[mask] = 0

inferer = Inference()
inferer.load_posterior('based_posterior_catwise_0p5_17p0.pkl')
samples = inferer.sample_amortized_posterior(x_obs=dmap)

# transform samples
samples[:, -2] = np.rad2deg(samples[:, -2])
samples[:, -1] = np.rad2deg(np.pi / 2 - samples[:, -1])
samples[:, -3] = samples[:, -3] / 0.00123

corner(
    samples,
    truths=[None, 1, CMB_L, CMB_B],
    labels=[
        r'$N_{\mathrm{init.}}$',
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