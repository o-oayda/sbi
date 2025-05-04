# %%
import healpy as hp
import matplotlib.pyplot as plt
from tools.maps import SkyMap
from tools.models import DipolePoisson
from corner import corner
import torch
from tools.plotting import sky_probability
# %%
D = 0.025
PHI =  5
THETA = 1
NBAR = 200
truths = [NBAR, D, PHI, THETA]
DEVICE = 'cpu'

simulation_class = SkyMap()
simulation_class.generate_dipole(torch.as_tensor(truths))
simulation_class.mask_pixels(fill_value=0)
dmap = simulation_class.density_map
hp.projview(dmap.numpy(), nest=True)
plt.show()
# %%
model = DipolePoisson(
    dmap, amplitude_range=[0, 0.05], mean_count_range=[0.8*NBAR, 1.2*NBAR]
)
# %%
model.run_sbi(
    n_simulations=50_000, device=DEVICE, mask_fill_value=0, equator_mask=30
)
# %%
labels = [r'$\bar{N}$', r'$\mathcal{D}$', r'$\phi$', r'$\theta$']
samples = model.sample_amortized_posterior(x_obs=dmap.to(DEVICE))
corner(samples, truths=truths, labels=labels, label_kwargs={'size': 15})
plt.show()
sky_probability(samples, smooth=0.1, truth_star=[PHI, THETA])
 # %%
model.density_map = dmap
model.run_dynesty()
corner(model.dresults.samples_equal(), truths=truths, labels=labels, label_kwargs={'size': 15})
plt.show()
# %%
from contextlib import contextmanager
import torch
from tools.plotting import sky_probability
@contextmanager
def open_samples(file_path):
    try:
        samples = torch.load(file_path)
        yield samples
    finally:
        del samples

with (
    open_samples('sbi_samples.pt') as sbi_samples,
    open_samples('ns_samples.pt') as ns_samples
):
    # Use sbi_samples and ns_samples here
    sky_probability(sbi_samples, smooth=0.1, truth_star=[5,1])
    sky_probability(ns_samples, smooth=0.12, truth_star=[5,1])
# %%
