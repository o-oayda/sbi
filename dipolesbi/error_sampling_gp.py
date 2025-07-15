# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binned_statistic_2d
from astropy.table import Table
import healpy as hp
from dipolesbi.tools.utils import ParameterMap
from astropy.coordinates import SkyCoord
import astropy.units as u
# %%
catalogue = Table.read('dipolesbi/catwise/catwise2020_corr_w12-0p5_w1-17p0.fits')
mask = hp.reorder(
    np.load('dipolesbi/catwise/CatWISE_Mask_nside64.npy'),
    r2n=True
)
# error_map = np.load('w1_errormap.npy') # nest ordering
masked_pixels = np.where(mask == 0)[0]
# error_map[masked_pixels] = np.nan
masked_pixels_set = set(masked_pixels)

## there's reasonable justification for masking out the north ecl pole,
# on account of the discontinuity in the mag-coverage-error distribution
ecl_north_pole = SkyCoord(lon=0*u.deg, lat=90*u.deg, frame='geocentrictrueecliptic') # type: ignore
gal_north_pole = ecl_north_pole.transform_to('galactic')

# Convert to healpy angles (theta, phi)
theta = np.deg2rad(90 - gal_north_pole.b.deg)  # colatitude # type: ignore
phi = np.deg2rad(gal_north_pole.l.deg)         # longitude # type: ignore

vec_north_ecl_gal = hp.ang2vec(theta, phi)

nside = 64
radius_deg = 5 # minimum integer radius at which the coverage discontinuity is masked
radius_rad = np.deg2rad(radius_deg)

disc_pixels = hp.query_disc(
    nside=nside,
    vec=vec_north_ecl_gal,
    radius=radius_rad,
    nest=True
)

masked_pixels_set.update(disc_pixels)

pixel_indicy = hp.ang2pix(
    64,
    catalogue['l'],
    catalogue['b'],
    lonlat=True,
    nest=True,
)
boolean_exclusion = [idx not in masked_pixels_set for idx in pixel_indicy]
masked_catalogue = catalogue[boolean_exclusion]
mag = masked_catalogue['w2']
sigma = masked_catalogue['w2e']
n_exp = np.log10( masked_catalogue['w2cov'] )
covmap = ParameterMap(pixel_indicy[boolean_exclusion], masked_catalogue['w2cov'], 64)

hp.projview( covmap.get_map(), norm='log', nest=True )
# %%
# Bin edges
mag_bins = np.linspace(9, 17, 100)
exp_bins = np.linspace(1.5, 4., 100)
sigma_bins = np.linspace(min(sigma), 0.1, 25)

# Compute 2D median grid
stat, x_edges, y_edges, _ = binned_statistic_2d(
    mag, n_exp, sigma, statistic='median', bins=[mag_bins, exp_bins] # type: ignore
)
count, *_ = binned_statistic_2d(
    mag, n_exp, sigma, statistic='count', bins=[mag_bins, exp_bins] # type: ignore
)
stat[count < 10] = np.nan

# %%
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
plt.xlabel('W1 mag')
plt.ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.tight_layout()
cbar = plt.colorbar()
cbar.ax.set_title('Median raw error')

plt.axvline(x=16.4, linestyle='--', c='black')

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
        # For bins above 16.4, P(mag < 16.4)
        if mag_center > 16.4:
            # Probability mag < 16.4
            z = (16.4 - mag_center) / median_sigma
            prob = -z
        # For bins below 16.4, P(mag > 16.4)
        elif mag_center < 16.4:
            z = (16.4 - mag_center) / median_sigma
            prob = z
        else:
            prob = 0.5  # exactly at 16.4
        prob_matrix[i, j] = prob

# Plot the probability matrix
plt.figure(figsize=(10, 6))
im = plt.pcolormesh(
    mag_bins, exp_bins, prob_matrix.T,
    shading='auto', cmap='plasma_r', vmax=10
)
plt.xlabel('W1 mag')
plt.ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.title(r'Probability (in units of $\sigma$) wrt mag=16.4')
cbar = plt.colorbar(im)
cbar.ax.set_title(r'$\sigma$')
plt.axvline(x=16.4, linestyle='--', c='black')
plt.tight_layout()
plt.show()

# %%
# KNN Interpolation Example and Visual Inspection
print("Setting up KNN interpolation for missing values...")

from sklearn.neighbors import NearestNeighbors

# Extract valid data points from the binned statistic
median_raw_error = stat  # This is your 2D grid from binned_statistic_2d
W1_mag_grid, log_COV_grid = np.meshgrid(mag_centers, exp_centers, indexing='ij')

# Find valid (non-NaN) points
mask = ~np.isnan(median_raw_error)
valid_indices = np.where(mask)
X_train = np.column_stack([W1_mag_grid[valid_indices], log_COV_grid[valid_indices]])
y_train = median_raw_error[valid_indices]

print(f"Training KNN with {len(X_train)} valid points...")

# Setup KNN interpolator
nbrs = NearestNeighbors(
    n_neighbors=4,  # Use 4 nearest neighbors
    algorithm='kd_tree',  # Fastest for 2D data
    leaf_size=30
)
nbrs.fit(X_train)

def knn_interpolate(X_pred):
    """KNN interpolation with inverse distance weighting"""
    distances, indices = nbrs.kneighbors(X_pred)
    
    # Inverse distance weighting
    weights = 1 / (distances + 1e-8)  # Small epsilon to avoid division by zero
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    
    # Weighted prediction
    return np.sum(weights * y_train[indices], axis=1)

# Fill NaN values in the original grid
print("Filling NaN values...")
filled_grid = median_raw_error.copy()
nan_mask = np.isnan(median_raw_error)

# Initialize variables for later use
X_predict = None
filled_values = None

if np.any(nan_mask):
    nan_indices = np.where(nan_mask)
    X_predict = np.column_stack([W1_mag_grid[nan_indices], log_COV_grid[nan_indices]])
    
    # Predict values for NaN locations
    filled_values = knn_interpolate(X_predict)
    filled_grid[nan_mask] = filled_values
    
    print(f"Filled {len(filled_values)} NaN values")

# %%
# Create visualization plots
fig, axes = plt.subplots(2, 1, figsize=(6, 10))

# Plot 1: Original data with NaN regions
im1 = axes[0].pcolormesh(
    mag_bins, exp_bins, stat.T,
    shading='auto', cmap='viridis'
)
axes[0].set_title('Original Data (with NaN regions)')
axes[0].set_xlabel('W1 mag')
axes[0].set_ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.colorbar(im1, ax=axes[0], label='Median raw error')

# Plot 2: KNN-filled data
im2 = axes[1].pcolormesh(
    mag_bins, exp_bins, filled_grid.T,
    shading='auto', cmap='viridis'
)
axes[1].set_title('KNN-Filled Data')
axes[1].set_xlabel('W1 mag')
axes[1].set_ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.colorbar(im2, ax=axes[1], label='Median raw error')

plt.tight_layout()
plt.show()

# %%
# Large-scale KNN prediction for 30 million samples
print("Setting up large-scale KNN prediction system...")

def large_scale_knn_predict(X_samples, chunk_size=100000, verbose=True):
    """
    Efficiently predict error values for millions of samples using KNN interpolation
    
    Parameters:
    -----------
    X_samples : array-like, shape (n_samples, 2)
        Input samples with columns [W1_mag, log_COV]
    chunk_size : int, default=100000
        Number of samples to process at once (adjust based on RAM)
    verbose : bool, default=True
        Whether to show progress updates
        
    Returns:
    --------
    predictions : array, shape (n_samples,)
        Predicted error values for all samples
    """
    n_samples = len(X_samples)
    
    # Pre-allocate the result array
    predictions = np.empty(n_samples)
    
    if verbose:
        print(f"Processing {n_samples:,} samples in chunks of {chunk_size:,}...")
    
    # Process in chunks to manage memory
    for i in range(0, n_samples, chunk_size):
        chunk_end = min(i + chunk_size, n_samples)
        chunk = X_samples[i:chunk_end]
        
        # Progress update
        if verbose and i % (chunk_size * 10) == 0:
            progress = (i / n_samples) * 100
            print(f"  Progress: {progress:.1f}% ({i:,}/{n_samples:,} samples)")
        
        # KNN prediction for chunk
        distances, indices = nbrs.kneighbors(chunk)
        
        # Inverse distance weighting (vectorized)
        weights = 1 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1)[:, np.newaxis]
        
        # Weighted prediction - store directly in pre-allocated array
        predictions[i:chunk_end] = np.sum(weights * y_train[indices], axis=1)
    
    if verbose:
        print("  Progress: 100.0% - Complete!")
    
    return predictions

# %%
# Example: Generate and predict for 30 million samples
print("\nExample: Large-scale prediction demonstration...")

# For demonstration, let's use a smaller sample size first
demo_size = 1_000_000  # 1 million for demo (change to 30_000_000 for full scale)

print(f"Generating {demo_size:,} random samples...")
np.random.seed(42)  # For reproducibility

# Generate samples within the data range
mag_min, mag_max = mag_bins[0], mag_bins[-1]
cov_min, cov_max = exp_bins[0], exp_bins[-1]

# Generate samples uniformly within the data bounds
demo_mags = np.random.uniform(mag_min, mag_max, demo_size)
demo_covs = np.random.uniform(cov_min, cov_max, demo_size)
demo_samples = np.column_stack([demo_mags, demo_covs])

print(f"Sample range - Mag: [{demo_mags.min():.2f}, {demo_mags.max():.2f}], "
      f"log COV: [{demo_covs.min():.2f}, {demo_covs.max():.2f}]")

# Time the prediction
import time
start_time = time.time()

# Predict error values
demo_predictions = large_scale_knn_predict(demo_samples, chunk_size=50000)

end_time = time.time()
prediction_time = end_time - start_time

print(f"\nPrediction Results:")
print(f"  Samples processed: {len(demo_predictions):,}")
print(f"  Processing time: {prediction_time:.2f} seconds")
print(f"  Rate: {len(demo_predictions)/prediction_time:,.0f} samples/second")
print(f"  Predicted errors - Min: {demo_predictions.min():.6f}, "
      f"Max: {demo_predictions.max():.6f}, Mean: {demo_predictions.mean():.6f}")

# Estimate performance for 30 million samples
samples_per_second = len(demo_predictions) / prediction_time
time_for_30m = 30_000_000 / samples_per_second

print(f"\nEstimated performance for 30 million samples:")
print(f"  Estimated time: {time_for_30m:.1f} seconds ({time_for_30m/60:.1f} minutes)")
print(f"  Memory usage per chunk: ~{50000 * 8 * 2 / 1024**2:.1f} MB")

# %%
# Production function for 30 million samples
def predict_30_million_samples(
        mag_range=(9, 17),
        cov_range=(1.5, 4.0)
    ):
    """
    Generate and predict error values for 30 million samples
    
    Parameters:
    -----------
    mag_range : tuple, default=(9, 17)
        Range for W1 magnitude sampling
    cov_range : tuple, default=(1.5, 4.0)
        Range for log coverage sampling
    save_results : bool, default=True
        Whether to save results to disk
    filename : str, default='knn_predictions_30M.npz'
        Filename for saving results
    """
    print("=" * 60)
    print("LARGE-SCALE KNN PREDICTION: 30 MILLION SAMPLES")
    print("=" * 60)
    
    # Generate 30 million samples
    print("Generating 30 million samples...")
    np.random.seed(42)
    
    mags_30m = np.random.uniform(mag_range[0], mag_range[1], 30_000_000)
    covs_30m = np.random.uniform(cov_range[0], cov_range[1], 30_000_000)
    samples_30m = np.column_stack([mags_30m, covs_30m])
    
    print(f"Generated samples - Mag: [{mags_30m.min():.2f}, {mags_30m.max():.2f}], "
          f"log COV: [{covs_30m.min():.2f}, {covs_30m.max():.2f}]")
    
    # Make predictions
    print("\nStarting KNN predictions...")
    start_time = time.time()
    
    predictions_30m = large_scale_knn_predict(samples_30m, chunk_size=100_000)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nFinal Results:")
    print(f"  Total samples: {len(predictions_30m):,}")
    print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Average rate: {len(predictions_30m)/total_time:,.0f} samples/second")
    print(f"  Prediction stats - Min: {predictions_30m.min():.6f}, "
          f"Max: {predictions_30m.max():.6f}, Mean: {predictions_30m.mean():.6f}")
    
    return mags_30m, covs_30m, predictions_30m

# Uncomment the line below to run the full 30 million sample prediction
mags_30m, covs_30m, predictions_30m = predict_30_million_samples()

print("\n" + "="*60)
print("LARGE-SCALE KNN SYSTEM READY!")
print("="*60)
print("To process 30 million samples, run:")
print("  mags, covs, preds = predict_30_million_samples()")
print("="*60)

# %%
# Grid-based comparison: Original vs KNN predictions
print("Comparing original grid with KNN predictions...")

# Get KNN predictions for all grid points
X_grid_all = np.column_stack([W1_mag_grid.ravel(), log_COV_grid.ravel()])

knn_predictions_grid, x_edges, y_edges, _ = binned_statistic_2d(
    mags_30m, covs_30m, predictions_30m, statistic='median',
    bins=[mag_bins, exp_bins] # type: ignore
)
# knn_predictions_grid = knn_interpolate(X_grid_all).reshape(W1_mag_grid.shape)

# Calculate residuals (KNN - Original) only where original data exists
residuals_grid = np.full_like(median_raw_error, np.nan)
valid_mask = ~np.isnan(median_raw_error)
residuals_grid[valid_mask] = knn_predictions_grid[valid_mask] - median_raw_error[valid_mask]

# Calculate statistics for valid regions
valid_residuals = residuals_grid[valid_mask]
rmse_grid = np.sqrt(np.mean(valid_residuals**2))
mae_grid = np.mean(np.abs(valid_residuals))
bias_grid = np.mean(valid_residuals)

print(f"Grid comparison statistics:")
print(f"  RMSE: {rmse_grid:.6f}")
print(f"  MAE: {mae_grid:.6f}")
print(f"  Bias: {bias_grid:.6f}")
print(f"  Max absolute residual: {np.max(np.abs(valid_residuals)):.6f}")

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original data
im1 = axes[0, 0].pcolormesh(
    mag_bins, exp_bins, median_raw_error.T,
    shading='auto', cmap='viridis'
)
axes[0, 0].set_title('Original Binned Data')
axes[0, 0].set_xlabel('W1 mag')
axes[0, 0].set_ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.colorbar(im1, ax=axes[0, 0], label='Median raw error')

# Plot 2: KNN predictions on grid
im2 = axes[0, 1].pcolormesh(
    mag_bins, exp_bins, knn_predictions_grid.T,
    shading='auto', cmap='viridis'
)
axes[0, 1].set_title('KNN Predictions on Grid')
axes[0, 1].set_xlabel('W1 mag')
axes[0, 1].set_ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.colorbar(im2, ax=axes[0, 1], label='Median raw error')

# Plot 3: Residuals (KNN - Original)
max_abs_residual = np.nanmax(np.abs(residuals_grid))
im3 = axes[1, 0].pcolormesh(
    mag_bins, exp_bins, residuals_grid.T,
    shading='auto', cmap='RdBu_r', 
    vmin=-max_abs_residual, vmax=max_abs_residual
)
axes[1, 0].set_title(f'Residuals (KNN - Original)\nRMSE: {rmse_grid:.6f}')
axes[1, 0].set_xlabel('W1 mag')
axes[1, 0].set_ylabel(r'$\log_{10} \mathrm{COV}_{W_1}$')
plt.colorbar(im3, ax=axes[1, 0], label='Residuals')

# Plot 4: Scatter plot of predictions vs original (grid points only)
valid_original = median_raw_error[valid_mask]
valid_predictions = knn_predictions_grid[valid_mask]

axes[1, 1].scatter(valid_original, valid_predictions, alpha=0.6, s=20)
axes[1, 1].plot([valid_original.min(), valid_original.max()], 
                [valid_original.min(), valid_original.max()], 
                'r--', lw=2, label='Perfect prediction')
axes[1, 1].set_xlabel('Original Grid Values')
axes[1, 1].set_ylabel('KNN Predictions')
axes[1, 1].set_title(f'Grid Point Comparison\nR² = {1 - np.sum(valid_residuals**2) / np.sum((valid_original - np.mean(valid_original))**2):.4f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Detailed residual analysis
print("\nDetailed residual analysis:")

# Show residual statistics by magnitude bins
mag_bin_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
cov_bin_centers = 0.5 * (exp_bins[:-1] + exp_bins[1:])

print(f"Residual statistics by magnitude range:")
print(f"{'Mag Range':<12} {'Count':<8} {'Mean Res':<12} {'Std Res':<12} {'Max |Res|':<12}")
print("-" * 60)

for i in range(0, len(mag_bin_centers), 10):  # Every 10th bin for readability
    if i < len(mag_bin_centers):
        # Get residuals for this magnitude range
        mag_slice = residuals_grid[i, :]
        valid_slice = ~np.isnan(mag_slice)
        
        if np.any(valid_slice):
            slice_residuals = mag_slice[valid_slice]
            mag_range = f"{mag_bin_centers[i]:.1f}"
            count = len(slice_residuals)
            mean_res = np.mean(slice_residuals)
            std_res = np.std(slice_residuals)
            max_abs_res = np.max(np.abs(slice_residuals))
            
            print(f"{mag_range:<12} {count:<8} {mean_res:<12.6f} {std_res:<12.6f} {max_abs_res:<12.6f}")

print(f"\nOverall grid comparison summary:")
print(f"  Grid points with original data: {np.sum(valid_mask):,}")
print(f"  Grid points filled by KNN: {np.sum(nan_mask):,}")
print(f"  Total grid points: {median_raw_error.size:,}")
print(f"  KNN interpolation accuracy (RMSE): {rmse_grid:.6f}")
print(f"  KNN interpolation bias: {bias_grid:.6f}")

# %%
# Single point comparison: Original vs KNN prediction
print("\n" + "="*60)
print("SINGLE POINT COMPARISON")
print("="*60)

# Pick a specific grid point for comparison
# Let's choose a point in the middle of the grid that has original data
i_mid = len(mag_centers) // 2  # Middle magnitude index
j_mid = len(exp_centers) // 2  # Middle coverage index

# Find a nearby point that has original data if the middle point is NaN
search_radius = 5
found_valid_point = False
i_selected = i_mid  # Initialize with default values
j_selected = j_mid

for radius in range(search_radius + 1):
    for i_offset in range(-radius, radius + 1):
        for j_offset in range(-radius, radius + 1):
            i_test = i_mid + i_offset
            j_test = j_mid + j_offset
            
            # Check bounds
            if (0 <= i_test < len(mag_centers) and 
                0 <= j_test < len(exp_centers)):
                
                # Check if this point has original data
                if not np.isnan(median_raw_error[i_test, j_test]):
                    i_selected = i_test
                    j_selected = j_test
                    found_valid_point = True
                    break
        if found_valid_point:
            break
    if found_valid_point:
        break

if found_valid_point:
    # Get the selected grid point coordinates
    mag_selected = mag_centers[i_selected]
    cov_selected = exp_centers[j_selected]
    
    # Get original value
    original_value = median_raw_error[i_selected, j_selected]
    
    # Get KNN prediction for this exact point
    test_point = np.array([[mag_selected, cov_selected]])
    knn_predicted_value = knn_interpolate(test_point)[0]
    
    # Calculate difference
    difference = knn_predicted_value - original_value
    relative_error = (difference / original_value) * 100
    
    print(f"Selected grid point:")
    print(f"  Grid indices: ({i_selected}, {j_selected})")
    print(f"  W1 magnitude: {mag_selected:.3f}")
    print(f"  log10(COV): {cov_selected:.3f}")
    print(f"")
    print(f"Values:")
    print(f"  Original (binned) value: {original_value:.8f}")
    print(f"  KNN predicted value:    {knn_predicted_value:.8f}")
    print(f"  Difference (KNN - Orig): {difference:.8f}")
    print(f"  Relative error:         {relative_error:.4f}%")
    print(f"")
    print(f"Assessment:")
    if abs(relative_error) < 1.0:
        print(f"  Excellent agreement (< 1% error)")
    elif abs(relative_error) < 5.0:
        print(f"  Good agreement (< 5% error)")
    elif abs(relative_error) < 10.0:
        print(f"  Moderate agreement (< 10% error)")
    else:
        print(f"  Poor agreement (> 10% error)")
    
    # Also test a few neighboring points for context
    print(f"\nNeighboring points comparison:")
    print(f"{'Direction':<12} {'Original':<12} {'KNN Pred':<12} {'Diff':<12} {'Rel Err %':<12}")
    print("-" * 72)
    
    neighbors = [
        ("Center", i_selected, j_selected),
        ("Left", i_selected-1, j_selected),
        ("Right", i_selected+1, j_selected),
        ("Up", i_selected, j_selected-1),
        ("Down", i_selected, j_selected+1)
    ]
    
    for direction, i_test, j_test in neighbors:
        if (0 <= i_test < len(mag_centers) and 
            0 <= j_test < len(exp_centers) and
            not np.isnan(median_raw_error[i_test, j_test])):
            
            mag_test = mag_centers[i_test]
            cov_test = exp_centers[j_test]
            original_test = median_raw_error[i_test, j_test]
            
            test_point = np.array([[mag_test, cov_test]])
            knn_test = knn_interpolate(test_point)[0]
            
            diff_test = knn_test - original_test
            rel_err_test = (diff_test / original_test) * 100
            
            print(f"{direction:<12} {original_test:<12.6f} {knn_test:<12.6f} {diff_test:<12.6f} {rel_err_test:<12.2f}")
    
else:
    print("Could not find a valid grid point with original data for comparison.")

print("="*60)

# %%
# Performance comparison: KNN vs RegularGridInterpolator
print("\n" + "="*60)
print("PERFORMANCE COMPARISON: KNN vs RegularGridInterpolator")
print("="*60)

from scipy.interpolate import RegularGridInterpolator

# Setup RegularGridInterpolator
print("Setting up RegularGridInterpolator...")
print(f"Grid shape: {filled_grid.shape}")
print(f"Grid covers: Mag [{mag_bins[0]:.2f}, {mag_bins[-1]:.2f}], COV [{exp_bins[0]:.2f}, {exp_bins[-1]:.2f}]")

# Create coordinate arrays for the grid
mag_coords = 0.5 * (mag_bins[:-1] + mag_bins[1:])  # bin centers
cov_coords = 0.5 * (exp_bins[:-1] + exp_bins[1:])  # bin centers

# Create the interpolator with the filled grid
rgi = RegularGridInterpolator(
    (mag_coords, cov_coords), 
    filled_grid,
    method='linear',
    bounds_error=False,
    fill_value=np.nan
)

def large_scale_rgi_predict(X_samples, chunk_size=500000, verbose=True):
    """
    Efficiently predict using RegularGridInterpolator
    
    Parameters:
    -----------
    X_samples : array-like, shape (n_samples, 2)
        Input samples with columns [W1_mag, log_COV]
    chunk_size : int, default=500000
        Number of samples to process at once (can be larger than KNN)
    verbose : bool, default=True
        Whether to show progress updates
        
    Returns:
    --------
    predictions : array, shape (n_samples,)
        Predicted error values for all samples
    """
    n_samples = len(X_samples)
    
    # Pre-allocate the result array
    predictions = np.empty(n_samples, dtype=np.float32)
    
    if verbose:
        print(f"Processing {n_samples:,} samples in chunks of {chunk_size:,}...")
    
    # Process in chunks
    for i in range(0, n_samples, chunk_size):
        chunk_end = min(i + chunk_size, n_samples)
        chunk = X_samples[i:chunk_end]
        
        # Progress update
        if verbose and i % (chunk_size * 10) == 0:
            progress = (i / n_samples) * 100
            print(f"  Progress: {progress:.1f}% ({i:,}/{n_samples:,} samples)")
        
        # RGI prediction for chunk
        predictions[i:chunk_end] = rgi(chunk).astype(np.float32)
    
    if verbose:
        print("  Progress: 100.0% - Complete!")
    
    return predictions

# %%
# Performance comparison on the same demo samples
print(f"\nTesting both methods on {demo_size:,} samples...")

# Test KNN method
print("\n1. KNN Method:")
knn_start = time.time()
knn_predictions = large_scale_knn_predict(demo_samples, chunk_size=50000, verbose=True)
knn_end = time.time()
knn_time = knn_end - knn_start

# Test RegularGridInterpolator method
print("\n2. RegularGridInterpolator Method:")
rgi_start = time.time()
rgi_predictions = large_scale_rgi_predict(demo_samples, chunk_size=200000, verbose=True)
rgi_end = time.time()
rgi_time = rgi_end - rgi_start

# Compare results
print(f"\n" + "="*60)
print("PERFORMANCE RESULTS")
print("="*60)

print(f"Sample size: {demo_size:,}")
print(f"")
print(f"KNN Method:")
print(f"  Time: {knn_time:.2f} seconds")
print(f"  Rate: {demo_size/knn_time:,.0f} samples/second")
print(f"  Memory per chunk: ~{50000 * 8 * 2 / 1024**2:.1f} MB")

print(f"")
print(f"RegularGridInterpolator Method:")
print(f"  Time: {rgi_time:.2f} seconds")
print(f"  Rate: {demo_size/rgi_time:,.0f} samples/second")
print(f"  Memory per chunk: ~{200000 * 8 * 2 / 1024**2:.1f} MB")

print(f"")
print(f"Speed comparison:")
if rgi_time < knn_time:
    speedup = knn_time / rgi_time
    print(f"  RegularGridInterpolator is {speedup:.1f}x FASTER")
else:
    speedup = rgi_time / knn_time
    print(f"  KNN is {speedup:.1f}x FASTER")

# %%
# Compare prediction quality
print(f"\nPREDICTION QUALITY COMPARISON")
print("="*60)

# Calculate differences
prediction_diff = np.abs(knn_predictions - rgi_predictions)
valid_mask = ~(np.isnan(knn_predictions) | np.isnan(rgi_predictions))

print(f"Valid predictions: {np.sum(valid_mask):,} / {len(valid_mask):,}")
print(f"")
print(f"Prediction Statistics:")
print(f"  KNN - Mean: {np.nanmean(knn_predictions):.6f}, Std: {np.nanstd(knn_predictions):.6f}")
print(f"  RGI - Mean: {np.nanmean(rgi_predictions):.6f}, Std: {np.nanstd(rgi_predictions):.6f}")
print(f"")
print(f"Difference Statistics:")
print(f"  Mean absolute difference: {np.mean(prediction_diff[valid_mask]):.6f}")
print(f"  Max absolute difference: {np.max(prediction_diff[valid_mask]):.6f}")
print(f"  RMS difference: {np.sqrt(np.mean(prediction_diff[valid_mask]**2)):.6f}")

# Calculate correlation
correlation = np.corrcoef(knn_predictions[valid_mask], rgi_predictions[valid_mask])[0, 1]
print(f"  Correlation coefficient: {correlation:.6f}")

# %%
# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: KNN vs RGI scatter
valid_knn = knn_predictions[valid_mask]
valid_rgi = rgi_predictions[valid_mask]

# Sample for plotting (too many points otherwise)
plot_indices = np.random.choice(len(valid_knn), min(10000, len(valid_knn)), replace=False)
axes[0].scatter(valid_knn[plot_indices], valid_rgi[plot_indices], alpha=0.5, s=1)
axes[0].plot([valid_knn.min(), valid_knn.max()], [valid_knn.min(), valid_knn.max()], 'r--', lw=2)
axes[0].set_xlabel('KNN Predictions')
axes[0].set_ylabel('RGI Predictions')
axes[0].set_title(f'KNN vs RGI (r={correlation:.4f})')
axes[0].grid(True, alpha=0.3)

# Plot 2: Difference histogram
axes[1].hist(prediction_diff[valid_mask], bins=50, alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Absolute Difference')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Prediction Differences')
axes[1].grid(True, alpha=0.3)

# Plot 3: Speed comparison bar chart
methods = ['KNN', 'RegularGrid']
times = [knn_time, rgi_time]
rates = [demo_size/knn_time, demo_size/rgi_time]

bars = axes[2].bar(methods, rates, color=['skyblue', 'lightcoral'])
axes[2].set_ylabel('Samples/Second')
axes[2].set_title('Processing Speed Comparison')
axes[2].grid(True, alpha=0.3)

# Add value labels on bars
for bar, rate in zip(bars, rates):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
# Estimate performance for 30 million samples
print(f"\nESTIMATED PERFORMANCE FOR 30 MILLION SAMPLES")
print("="*60)

knn_30m_time = 30_000_000 / (demo_size/knn_time)
rgi_30m_time = 30_000_000 / (demo_size/rgi_time)

print(f"KNN Method:")
print(f"  Estimated time: {knn_30m_time:.1f} seconds ({knn_30m_time/60:.1f} minutes)")
print(f"  Memory usage: Moderate (neighbor search)")

print(f"")
print(f"RegularGridInterpolator Method:")
print(f"  Estimated time: {rgi_30m_time:.1f} seconds ({rgi_30m_time/60:.1f} minutes)")
print(f"  Memory usage: Low (direct grid lookup)")

print(f"")
print(f"Recommendation:")
if rgi_30m_time < knn_30m_time:
    time_saved = knn_30m_time - rgi_30m_time
    print(f"  Use RegularGridInterpolator - saves {time_saved:.1f} seconds ({time_saved/60:.1f} minutes)")
else:
    time_saved = rgi_30m_time - knn_30m_time
    print(f"  Use KNN - saves {time_saved:.1f} seconds ({time_saved/60:.1f} minutes)")

print(f"  Quality difference is minimal (correlation = {correlation:.4f})")

# %%
