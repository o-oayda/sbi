# %%
from tools.maps import SkyMap
import torch
import healpy as hp
from tools.models import DipolePoisson
import corner
import matplotlib.pyplot as plt
from tools.utils import sky_probability
# %%
N, V, PHI, THETA = [10_000_000, 0.009, 5, 1]
N_SIM = 2000

sim = SkyMap()
sim.generate_dipole_from_base(
    observer_direction=(PHI, THETA),
    n_initial_points=N,
    observer_speed=V
)
sim.mask_pixels(fill_value=0)

# hp.projview(sim.density_map.numpy(), nest=True)
# print(f'Expected amplitude: {sim.expected_amplitude:.3g}')
# # %%
# labels = [r'$\bar{N}$', r'$\mathcal{D}$', r'$\phi$', r'$\theta$']
# model = DipolePoisson(sim.density_map)
# model.run_dynesty()
# %%
# samples = model.dresults.samples_equal()
# corner.corner(
#     samples,
#     truths=[None, sim.expected_amplitude, 5, 1],
#     labels=labels,
#     label_kwargs={'size': 15}
# )
# plt.show()
# sky_probability(samples, truth_star=[5, 1])
# %%
model = DipolePoisson(
    sim.density_map,
    mean_count_range=[8_000_000, 12_000_000],
    amplitude_range=[0, 0.01]
)
model.run_sbi(dipole_method='base', n_simulations=N_SIM, n_workers=32)
model.save_posterior(f'based_posterior_N{N_SIM}.pkl')
# %%
# labels = [r'$\bar{N}$', r'$\mathcal{D}$', r'$\phi$', r'$\theta$']
# samples = model.sample_amortized_posterior(x_obs=sim.density_map)
# corner.corner(
#     samples,
#     truths=[N, V, PHI, THETA],
#     labels=labels,
#     label_kwargs={'size': 15}
# )
# plt.show()
# sky_probability(samples, smooth=0.1, truth_star=[5, 1])
# %%
