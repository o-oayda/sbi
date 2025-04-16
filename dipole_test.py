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
model = DipolePoisson(dmap, device='gpu')
# %%
model.run_dynesty()
corner(model.dresults.samples_equal(), truths=truths)
plt.show()
# %%
model.run_sbi(n_simulations=20000, posterior_device='cpu')
corner(model.samples, truths=truths)
plt.show()
# %%
