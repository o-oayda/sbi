# %%
from catwise.maps import CatwiseSim
from tools.utils import Sample1DHistogram
import matplotlib.pyplot as plt
import torch
import healpy as hp
from tools.plotting import smooth_map
# %%
sim = CatwiseSim()
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
sim.initialise_data()
# %%
sim = CatwiseSim()
sim.initialise_data()
# %%
sim.generate_dipole()
 # %%
smooth_map(sim.density_map)
# %%
