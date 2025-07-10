# %%
from dipolesbi.catwise.maps import CatwiseSim, CatwiseReal
import matplotlib.pyplot as plt
import numpy as np
from fastkde import fastKDE

sim = CatwiseSim(cat_w1_max=17.0, cat_w12_min=0.5)
sim.initialise_data()

# draw samples from joint 2D dist
x_samples, y_samples = sim.colour_mag_sampler.sample(n_samples=30_000_000)
plt.hist2d(
    x_samples,
    y_samples,
    bins=200,
    norm='log',
    range=[[15, 16.4], [14.0, 16.0]]
)
x = np.linspace(0, 17, 1000)
plt.plot(x, x - 0.8, c='black', linestyle='--')
plt.show()

# compare with empirical dist
sim.load_catalogue()
x_real, y_real = sim.catalogue['w1'], sim.catalogue['w2']
plt.hist2d(
    x_real,
    y_real,
    bins=200,
    norm='log',
    range=[[15, 16.4], [14.0, 16.0]]
)
plt.plot(x, x - 0.8, c='black', linestyle='--')
plt.show()

# %%
X = np.vstack([x_real, y_real]).T
rng = np.random.default_rng()

# 1) KDE on a *fine* grid -----------------------------------------------
PDF  = fastKDE.pdf(
          X[:,0], X[:,1],
        #   num_points = (513, 513)   # 4× the previous resolution
       )

dens       = PDF.data
y_centres  = PDF.coords[PDF.dims[0]].values
x_centres  = PDF.coords[PDF.dims[1]].values
ny, nx     = dens.shape
dy, dx     = np.diff(y_centres)[0], np.diff(x_centres)[0]

from scipy.interpolate import interpn
grid_axes = (y_centres, x_centres)          # (ny, nx) order
interp = lambda xy: interpn(
          grid_axes, dens,
          np.column_stack([xy[:, 1], xy[:, 0]]),   # <-- swap to (y, x)
          method='linear', bounds_error=False, fill_value=0.0
        )

p_max = dens.max()
lo    = np.array([x_centres[0] - dx/2,  y_centres[0] - dy/2])
hi    = np.array([x_centres[-1] + dx/2, y_centres[-1] + dy/2])

def sample_bilinear(n):
    out = np.empty((n, 2))
    k   = 0
    while k < n:
        cand = rng.uniform(lo, hi, size=(n-k, 2))
        accept_prob = interp(cand) / p_max
        keep = rng.random(len(cand)) < accept_prob
        m    = keep.sum()
        out[k:k+m] = cand[keep]
        k += m
    return out

# %%
new = sample_bilinear(1_000_000)

# 4) Quick visual check --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=110)

vmin, vmax = 0, None
h1 = ax1.hist2d(X[:,0], X[:,1], bins=400, cmap='Blues', norm='log')
h2 = ax2.hist2d(new[:,0], new[:,1], bins=400, cmap='Blues', norm='log')
# vmax = max(h1[0].max(), h2[0].max())
# ax1.cla()
# ax2.cla()
# ax1.hist2d(X[:,0], X[:,1], bins=400, cmap='Blues', vmin=vmin, vmax=vmax)
ax1.set_title("Original data")
# ax2.hist2d(new[:,0], new[:,1], bins=400, cmap='Blues', vmin=vmin, vmax=vmax)
ax2.set_title("Resampled data")

plt.tight_layout(); plt.show()
# %%
