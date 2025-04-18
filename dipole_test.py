# %%
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from funcs import simulation, DipolePoisson
from corner import corner

D = 0.007
PHI =  5
THETA = 1
NBAR = 50
truths = [NBAR, D, PHI, THETA]

dmap = simulation(truths)
# hp.projview(dmap.numpy())
# plt.show()
# %%
# Having constrictive priors completely decimates the accuracy of the posterior
    # for whatever reason
    # e.g. D ~ [0, 0.1] & Nbar ~ [0.8 mean 1.2 mean] is ok; but,
    # D ~ [0, 0.1] & Nbar ~ [0.9 mean 1.1 mean] is not?!?
model = DipolePoisson(
    dmap, device='cuda', amplitude_range=[0, 0.1],
    mean_count_range=[0.8*NBAR, 1.2*NBAR]
)
# %%
model.run_sbi(n_simulations=20_000, posterior_device='cpu')
# %%
samples = model.sample_amortized_posterior(x_obs=dmap.to('cuda'))
corner(samples, truths=truths)
plt.show()
 # %%
model.density_map = dmap.detach().cpu().numpy()
model.run_dynesty()
corner(model.dresults.samples_equal(), truths=truths)
plt.show()
# %%
