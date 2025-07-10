# %%
from dipolesbi.catwise.maps import CatwiseSim
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
sim.mask_pixels()

error_map = sim.w1_error_map.numpy()
mask = sim.mask_map.numpy()
error_map[mask == 1] = np.nan
hp.projview(error_map, nest=True)
plt.show()
# %%
sim.load_catalogue()

# make cuts
cut = (sim.catalogue['w1'] < 17.0) & (sim.catalogue['w12'] > 0.5)
cut_cat = sim.catalogue[cut]

l, b = cut_cat['l'], cut_cat['b']
w1_mag = cut_cat['w1']
w1_error = cut_cat['w1e']
w12_mag = cut_cat['w12']
w12_error = cut_cat['w12e']

pixel_indices = hp.ang2pix(64, l, b, lonlat=True, nest=True)
# Create a dictionary to store w1_error values for each pixel
pixel_error_dict = defaultdict(list)
for idx, pix in enumerate(pixel_indices):
    pixel_error_dict[pix].append( (w12_error[idx] / w12_mag[idx] ) * 100)

num_pixels = hp.nside2npix(64)
median_error_per_pixel = np.full(num_pixels, np.nan)
for pix in range(num_pixels):
    errors = pixel_error_dict[pix]
    if errors:
        median_error_per_pixel[pix] = np.median(errors)

median_error_per_pixel[mask == 1] = np.nan

for pix in range(num_pixels):
    if mask[pix] == 1:
        pixel_error_dict[pix] = np.nan

# does the high error tail contribute to the discrepancy between the simulator
# and actual catwise?
# plt.hist(pixel_error_dict[10_001], bins=30)
# plt.show()

hp.projview(median_error_per_pixel, nest=True)
plt.show()

# NOTE!
plt.hist2d(
    cut_cat['w12'],
    cut_cat['w12e'] * 100 / cut_cat['w12'],
    norm='log',
    bins=200
)
plt.show()
# %%
from dipolesbi.tools.utils import ParameterMap
from dipolesbi.catwise.maps import CatwiseSim
import healpy as hp
import matplotlib.pyplot as plt
from dipolesbi.tools.plotting import smooth_map

sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()
sim.generate_dipole(n_initial_samples=30_000_000)
colour_errs = sim.final_w12_frac_errors
pix_inds = sim.final_pixel_indices

colours = ParameterMap(pix_inds, colour_errs, nside=64)
colour_map = colours.get_map()

hp.projview(colour_map * 100, nest=True)
plt.show()
# %%
plt.figure()
plt.hist(colour_map * 100, bins=50, color='tab:orange', alpha=0.4, label='Simulated colour error')
plt.hist(median_error_per_pixel, bins=50, alpha=0.4, label='Real colour errors')
plt.legend()
plt.show()