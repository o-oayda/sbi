# %%
from tools.maps import SkyMap
from tools.inference import Inference
import healpy as hp
import matplotlib.pyplot as plt
from tools.plotting import smooth_map, sky_probability
from corner import corner
# %%
dipole_hyperparameters={
    'flux_percentage_noise': 'ecliptic',
    'minimum_flux_cut': 3
}
sim = SkyMap()
N = 10_000_000
V = 0.005
PHI = 2.93
THETA = 1.6
sim.generate_dipole_from_base(
    observer_direction=[PHI, THETA],
    n_initial_points=N,
    observer_speed=V,
    **dipole_hyperparameters
)
sim.mask_pixels(fill_value=0, equator_mask=30)
smooth_map(sim.density_map, coord=['C', 'G'])
plt.show()
# %%
inference = Inference()
inference.load_simulation('sim_ecliptic_mask')
inference.load_posterior('based_posterior_sim_ecliptic_mask.pkl')
samples = inference.sample_amortized_posterior(
    sim.density_map, n_samps=50_000
)
corner(
    samples,
    labels=['$N$', '$v$', '$\phi$', '$\\theta$'],
    label_kwargs={'size': 15},
    truths=[N, V, PHI, THETA]
)
sky_probability(
    samples,
    truth_star=[PHI, THETA],
    coord=['C', 'G']
)
# %%
