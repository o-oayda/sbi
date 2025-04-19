# %%
import healpy as hp
import matplotlib.pyplot as plt
from maps import SkyMap
from models import DipolePoisson
from corner import corner
import torch
# %%
D = 0.007
PHI =  5
THETA = 1
NBAR = 50
truths = [NBAR, D, PHI, THETA]

simulation_class = SkyMap()
simulation_class.generate_dipole(torch.as_tensor(truths))
simulation_class.mask_pixels(equator_mask=0, fill_value=0)
dmap = simulation_class.density_map
hp.projview(dmap.numpy())
plt.show()
# %%
model = DipolePoisson(
    dmap, amplitude_range=[0, 0.1], mean_count_range=[0.8*NBAR, 1.2*NBAR]
)
# %%
model.run_sbi(
    n_simulations=20_000, device='cuda', mask_fill_value=0, equator_mask=30
)
# %%
samples = model.sample_amortized_posterior(x_obs=dmap.to('cuda'))
corner(samples, truths=truths)
plt.show()
 # %%
model.density_map = dmap
model.run_dynesty()
corner(model.dresults.samples_equal(), truths=truths)
plt.show()