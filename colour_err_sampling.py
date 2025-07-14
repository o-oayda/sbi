# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic_2d
from astropy.table import Table
import healpy as hp
from dipolesbi.tools.utils import ParameterMap
# %%
catalogue = Table.read('dipolesbi/catwise/catwise2020_corr_w12-0p5_w1-17p0.fits')
# %%
mask = hp.reorder(
    np.load('dipolesbi/catwise/CatWISE_Mask_nside64.npy'),
    r2n=True
)
error_map = np.load('w1_errormap.npy') # nest ordering
masked_pixels = np.where(mask == 0)[0]
error_map[masked_pixels] = np.nan
masked_pixels_set = set(masked_pixels)
pixel_indicies = hp.ang2pix(
    64,
    catalogue['l'],
    catalogue['b'],
    lonlat=True,
    nest=True,
)
boolean_exclusion = [idx not in masked_pixels_set for idx in pixel_indicies]
masked_catalogue = catalogue[boolean_exclusion]
mag = masked_catalogue['w12']
sigma = masked_catalogue['w12e']
n_exp = np.log10( masked_catalogue['w1cov'] )
covmap = ParameterMap(pixel_indicies[boolean_exclusion], masked_catalogue['w1cov'], 64)
# %%

# 1. Mock galaxy data (replace with your real catalog)
# np.random.seed(42)
# N = 100000
# mag = np.random.uniform(13, 18, N)                    # magnitudes
# n_exp = np.random.randint(5, 20, N)                   # coverage (WISE: 5–20 exposures)
# sigma = 0.02 * (10 ** (0.2 * (mag - 15.5))) * np.sqrt(8 / n_exp)  # artificial "true" sigma
# sigma += np.random.normal(0, 0.002, N)                # noise scatter

# 2. Bin edges
# mag_bins = np.linspace(min(mag), max(mag), 50)
mag_bins = np.linspace(0.5, 5, 100)
# exp_bins = np.linspace(min(n_exp), max(n_exp), 50)
exp_bins = np.linspace(1.5, 4., 100)
sigma_bins = np.linspace(min(sigma), 0.1, 25)

# 3. Compute 2D median grid
stat, x_edges, y_edges, _ = binned_statistic_2d(
    mag, n_exp, sigma, statistic='median', bins=[mag_bins, exp_bins]
)
count, *_ = binned_statistic_2d(
    mag, n_exp, sigma, statistic='count', bins=[mag_bins, exp_bins]
)
stat[count < 3] = np.nan

# %%
from scipy.stats import norm
plt.figure(figsize=(10, 6))

mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
sigma_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])
exp_centers = 0.5 * (exp_bins[:-1] + exp_bins[1:])

im = plt.pcolormesh(
    mag_bins,            # x edges
    exp_bins,            # y edges
    stat.T,       # transpose so sigma is vertical axis
    shading='auto',
    cmap='viridis'
)
plt.xticks(
    ticks=mag_centers[::5],  # skip some for readability
    labels=[f"{x:.1f}" for x in mag_centers[::5]]
)
plt.yticks(
    ticks=exp_centers[::4],
    labels=[f"{y:.1f}" for y in exp_centers[::4]]
)
plt.xlabel('W12 colour mag')
plt.ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.tight_layout()
cbar = plt.colorbar()
cbar.ax.set_title('Median raw error')

plt.axvline(x=0.8, linestyle='--', c='black')

plt.show()

# Compute probability for each bin
prob_matrix = np.full_like(stat, np.nan, dtype=float)

for i in range(stat.shape[0]):
    for j in range(stat.shape[1]):
        median_sigma = stat[i, j]
        if np.isnan(median_sigma):
            continue
        mag_lo = mag_bins[i]
        mag_hi = mag_bins[i+1]
        mag_center = 0.5 * (mag_lo + mag_hi)
        # For bins above 0.8, P(mag < 0.8)
        if mag_center > 0.8:
            # Probability mag < 0.8
            z = (0.8 - mag_center) / median_sigma
            prob = -z
            # prob = norm.cdf(z)
        # For bins below 0.8, P(mag > 0.8)
        elif mag_center < 0.8:
            z = (0.8 - mag_center) / median_sigma
            prob = z
            # prob = 1 - norm.cdf(z)
        else:
            prob = 0.5  # exactly at 0.8
        prob_matrix[i, j] = prob

# Optional: plot the probability matrix
plt.figure(figsize=(10, 6))
im = plt.pcolormesh(
    mag_bins, exp_bins, prob_matrix.T,
    shading='auto', cmap='plasma_r', vmax=10
)
plt.xlabel('W12 colour mag')
plt.ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.title('Probability of scattering past colour cut')
cbar = plt.colorbar(im)
cbar.ax.set_title('$\sigma$')
plt.axvline(x=0.8, linestyle='--', c='black')
plt.tight_layout()
plt.show()
# %%
