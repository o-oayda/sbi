# %%
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.utils import Sample1DHistogram
import matplotlib.pyplot as plt
import torch
import healpy as hp
from dipolesbi.tools.plotting import smooth_map
# %%
sim = Catwise(cat_w1_max=16.8, cat_w12_min=0.7)
sim.load_catalogue()
# %%
sim.create_spectral_index_distribution()
# %%
sampler = Sample1DHistogram()
sampler.load_data('catwise/data/spectral_index/')
# %%
samples = -sampler.sample(1_000_000)
plt.hist(samples, density=True, bins=200)
plt.yscale('log')
plt.show()
# %%
sim.create_error_map()
# %%
w1_error_map = torch.load('catwise/data/error_map/w1_error_map.pt')
w2_error_map = torch.load('catwise/data/error_map/w2_error_map.pt')
# %%
hp.projview(w2_error_map.numpy(), nest=True, cbar=True)
# %%
sim = Catwise(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
# %%
sim.generate_dipole(n_initial_samples=29_000_000)
 # %%
smooth_map(sim.density_map)
# %%
