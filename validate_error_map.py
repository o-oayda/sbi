# %%
from dipolesbi.catwise.maps import Catwise
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dipolesbi.tools.utils import ParameterMap
# %%

sim = Catwise(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
sim.mask_pixels()

error_map = sim.w1_error_map.numpy()
mask = sim.mask_map.numpy()
error_map[mask == 1] = np.nan
hp.projview(error_map, nest=True)
plt.show()
# %%
if not sim.catalogue_is_loaded:
    sim.load_catalogue()
sim.mask_pixels()
sim.make_masked_catalogue()

# make cuts
cut = (sim.masked_catalogue['w1'] < 16.4) & (sim.masked_catalogue['w12'] > 0.8)
cut_cat = sim.masked_catalogue[cut]

l, b = cut_cat['l'], cut_cat['b']
w1_mag = cut_cat['w1']
w1_error = cut_cat['w1e']
w12_mag = cut_cat['w12']
w12_error = cut_cat['w12e']

pixel_indices = hp.ang2pix(64, l, b, lonlat=True, nest=True)
error = w12_error / w12_mag
median_error_per_pixel = ParameterMap(pixel_indices, error, nside=64).get_map()

# does the high error tail contribute to the discrepancy between the simulator
# and actual catwise?
# plt.hist(pixel_error_dict[10_001], bins=30)
# plt.show()

hp.projview(median_error_per_pixel * 100, nest=True)
plt.show()

# NOTE!
plt.hist2d(
    w12_mag,
    w12_error * 100 / w12_mag,
    norm='log',
    bins=200
)
plt.show()
# %%
from dipolesbi.catwise.maps import Catwise
import healpy as hp
import matplotlib.pyplot as plt
from dipolesbi.tools.plotting import smooth_map

sim = Catwise(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
sim.generate_dipole(n_initial_samples=30_000_000)
# %%
colour_errs = sim.final_w12_frac_errors
pix_inds = sim.final_pixel_indices

colours = ParameterMap(pix_inds, colour_errs, nside=64)
colour_map = colours.get_map()

hp.projview(colour_map * 100, nest=True)
plt.show()
# %%
plt.figure()
plt.hist(colour_map * 100, bins=50, color='tab:orange', alpha=0.4, label='Simulated colour error')
plt.hist(median_error_per_pixel * 100, bins=50, alpha=0.4, label='Real colour errors')
plt.legend()
plt.show()
# %%
