# %%
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from funcs import simulation, DipolePoisson
from corner import corner

D = 0.05
PHI =  5
THETA = 1
NBAR = 50
truths = [NBAR, D, PHI, THETA]

dmap = simulation(truths)
hp.projview(dmap.numpy())
plt.show()
# %%
model = DipolePoisson(dmap.cuda(), device='cpu')
# %%
# COMMENTS
# - running straight NPE SBI with high dim data (e.g. nside=64)
    # does not give accurate predictions; NPE does not work well with high dim
# - reduce nside improves this, but there is a limit to how much nside
    # should be reuced
# - using an embedding network to reduce the dimensionality of the data (I think)
    # works better; I tried a CNE initially but switched to multi-layer FC perceptron
    # which works better
# - The FC embedding network needs n_sims > 20_000 it seems for decent posteriors
# - This was all tested with Nbar = 50, D = 0.05, phi = 5 and theta = 1
# - Higher n_sims is pushing the memory limit of my RTX 3070 so will need to cluster
model.run_sbi(n_simulations=35_000, posterior_device='cpu')
corner(model.samples, truths=truths)
plt.show()
# %%
model.density_map = model.density_map.detach().cpu().numpy()
model.run_dynesty()
corner(model.dresults.samples_equal(), truths=truths)
plt.show()
# %%
