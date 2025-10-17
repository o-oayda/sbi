import os
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig
from dipolesbi.tools.utils import ParameterMap
from dipolesbi.tools.plotting import smooth_map


config = CatwiseConfig(
    cat_w1_max=17.0, 
    cat_w12_min=0.5,
    magnitude_error_dist='gaussian',
    store_final_samples=True
)
catwise = Catwise(config)
catwise.initialise_data()
dmap, mask = catwise.generate_dipole(log10_n_initial_samples=7.5)
print(f'Number of sources: {int(np.nansum(dmap))}')

spectral_idxs = catwise.final_alpha_samples
print(f'Mean alpha: {np.mean(spectral_idxs)}')
plt.hist(spectral_idxs, bins=100)
plt.xlabel('Spectral index')
plt.yscale('log')
plt.show()

plt.hist(catwise.final_w12_samples, bins=100)
plt.yscale('log')
plt.ylabel('W12 colour')
plt.show()
