import argparse
import os
import time
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dipolesbi.catwise.maps import Catwise
from dipolesbi.tools.configs import CatwiseConfig
from dipolesbi.tools.np_rngkey import NPKey
from dipolesbi.tools.plotting import smooth_map
from dipolesbi.tools.utils import HidePrints


mpl.rcParams['text.usetex'] = True
COLORBAR_FONTSIZE = {
    'cbar_label': 20,
    'cbar_tick_label': 18
}
rng_key = NPKey.from_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--only_measured_alpha',
    action='store_true',
    help=(
        'Plot only the measured spectral indices'
        '(do not additionally add the true alphas in CatSIM)'
    )
)
parser.add_argument(
    '--save',
    action='store_true',
    help='Save the plot to the default figure directory instead of showing it'
)
args = parser.parse_args()

def solid_edge_histogram(data, ax=None, **kwargs) -> None:
    if ax is None:
        ax = plt.gca()

    label = kwargs.pop('label')
    alpha = kwargs.pop('alpha', 0.4)

    fill_kwargs = dict(kwargs)
    fill_kwargs['alpha'] = alpha
    ax.hist(data, label=None, **fill_kwargs)

    edge_kwargs = dict(kwargs)
    edge_kwargs.pop('alpha', None)
    edge_kwargs['histtype'] = 'step'
    ax.hist(data, label=label, **edge_kwargs)


def uniform_bins_with_alignment(
    start_edge: float,
    end_edge: float,
    target_edge: float,
    approx_edges: int,
    edge_window: int = 10,
) -> np.ndarray:
    """
    Return uniformly spaced bin edges between ``start_edge`` and ``end_edge`` where one edge
    lies as close as practical to ``target_edge`` while keeping roughly ``approx_edges`` edges.

    The routine searches edge counts within ``edge_window`` of ``approx_edges``, always produces a
    uniform grid, and picks the option whose edge is nearest to the requested value (e.g. the
    minimum measured spectral index). Use this when you need the binning to remain uniform and
    near the original resolution but want an edge to align with a specific measurement.
    """
    approx_edges = max(approx_edges, 3)
    edge_window = max(edge_window, 0)
    start_edge = float(start_edge)
    end_edge = float(end_edge)
    target_edge = float(target_edge)

    best_bins = None
    best_error = np.inf
    best_edge_count = None

    search_min = max(3, approx_edges - edge_window)
    search_max = approx_edges + edge_window + 1
    for edge_count in range(search_min, search_max):
        candidate_bins = np.linspace(start_edge, end_edge, edge_count)
        idx = np.abs(candidate_bins - target_edge).argmin()
        error = abs(candidate_bins[idx] - target_edge)

        if error < best_error:
            best_error = error
            best_bins = candidate_bins
            best_edge_count = edge_count
        elif np.isclose(error, best_error):
            if (
                best_edge_count is None
                or abs(edge_count - approx_edges) < abs(best_edge_count - approx_edges)
            ):
                best_error = error
                best_bins = candidate_bins
                best_edge_count = edge_count

    if best_bins is None:
        return np.linspace(start_edge, end_edge, approx_edges)

    return best_bins

# simulated data
with HidePrints():
    config = CatwiseConfig(
        cat_w1_max=17.0, 
        cat_w12_min=0.5,
        magnitude_error_dist='gaussian',
        store_final_samples=True,
        use_common_extra_error=True
    )
    catwise = Catwise(config)
    catwise.initialise_data()
t0 = time.time()
dmap, mask = catwise.generate_dipole( # median params from free gauss extra err
    log10_n_initial_samples=7.55,
    w1_extra_error=3.57,
    observer_speed=2.07,
    dipole_longitude=221.,
    dipole_latitude=44.,
    rng_key=rng_key
)
t1 = time.time()
print(t1 - t0)

# empirical CatWISE2020 sample
real_dmap, real_mask = catwise.make_real_sample()
longitudes = catwise.real_catalogue['l']
latitudes = catwise.real_catalogue['b']
source_indices = hp.ang2pix(
    64, longitudes, latitudes, lonlat=True, nest=True
).astype(np.int32)
source_ismasked = catwise.mask_map[source_indices] == 0
masked_catalogue = catwise.real_catalogue[source_ismasked]

print(
    f'Number of sources (real): {int(np.nansum(dmap))}\n'
    f'Number of sources (sim): {int(np.nansum(real_dmap))}'
)

assert catwise.final_w12_samples is not None
sim_colour = catwise.final_w12_samples # after error
real_colour = masked_catalogue['w12']
bins = np.linspace(0.8, 3, 100)
# solid_edge_histogram(sim_colour, bins=bins, color='tab:blue', label='Simulated')
# solid_edge_histogram(real_colour, bins=bins, color='tab:orange', label='Real')
# plt.yscale('log')
# plt.ylabel('W12 colour')
# plt.legend()
# plt.show()

sim_spec_idxs = catwise.final_alpha_samples
real_spec_idxs = -masked_catalogue['alpha_W1']
sim_meas_spec_idxs = catwise.final_measured_alpha_samples
min_real_spec_idx = float(np.nanmin(real_spec_idxs))

mean_sim_true = float(np.mean(sim_spec_idxs))
mean_sim_meas = float(np.mean(sim_meas_spec_idxs))
mean_real_meas = float(np.mean(real_spec_idxs))
COL_SIM_TRUE = 'silver'
COL_SIM_MEAS = 'tomato'
COL_REAL_MEAS = 'cornflowerblue'
PLOT_TRUE = not args.only_measured_alpha

print(
    f'Mean sim. alpha (true): {mean_sim_true} (n={len(sim_spec_idxs)})\n'
    f'Mean sim. alpha (meas): {mean_sim_meas} (n={len(sim_meas_spec_idxs)})\n'
    f'Mean real alpha: {mean_real_meas} (n={len(real_spec_idxs)})'
)

if PLOT_TRUE:
    bins = uniform_bins_with_alignment(
        start_edge=float(np.nanmin(sim_spec_idxs)),
        end_edge=7.0,
        target_edge=min_real_spec_idx,
        approx_edges=150,
        edge_window=10,
    )
else:
    bins = np.linspace(min(real_spec_idxs), 7, 150)

fig, ax = plt.subplots(figsize=(4.5, 5.0))
if PLOT_TRUE:
    solid_edge_histogram(
        sim_spec_idxs,
        ax=ax,
        bins=bins,
        color=COL_SIM_TRUE,
        label=r'CatSIM (true $\alpha$)',
        alpha=0.2,
        lw=1.5,
        linestyle='--'
    )
solid_edge_histogram(
    sim_meas_spec_idxs,
    ax=ax,
    bins=bins,
    color=COL_SIM_MEAS,
    label=r'CatSIM (measured $\alpha$)',
    alpha=0.2,
    lw=1.5
)
solid_edge_histogram(
    real_spec_idxs,
    ax=ax,
    bins=bins,
    color=COL_REAL_MEAS,
    label=r'CatWISE (measured $\alpha$)',
    alpha=0.2,
    lw=1.5
)
ax.set_xlabel(r'Spectral index $\alpha$')
ax.set_ylabel('Count')
ax.set_yscale('log')
ax.legend()
x_min, x_max = bins[0], bins[-1]
span = x_max - x_min
main_x_pad_fraction = 0.05  # fraction of data span padded on either side of main axis
main_x_pad_abs = 0.1
pad = max(main_x_pad_fraction * span, main_x_pad_abs) if span > 0 else main_x_pad_abs
ax.set_xlim(x_min - pad, x_max + pad)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

counts_real, _ = np.histogram(real_spec_idxs, bins=bins)
counts_sim_meas, _ = np.histogram(sim_meas_spec_idxs, bins=bins)
counts_sim_true = None
if PLOT_TRUE:
    counts_sim_true, _ = np.histogram(sim_spec_idxs, bins=bins)

zoom_left = 0.2
zoom_right = 2.2
zoom_y_margin = 1.1  # multiplicative slack for zoom inset upper limit

max_count = 1
def _max_in_range(counts: np.ndarray) -> float:
    if not counts.size:
        return 1.0
    mask = (bins[:-1] >= zoom_left) & (bins[:-1] <= zoom_right)
    if not np.any(mask):
        return float(np.max(counts)) if counts.size else 1.0
    return float(np.max(counts[mask])) if counts[mask].size else 1.0

max_count = max(
    max_count,
    _max_in_range(counts_real),
    _max_in_range(counts_sim_meas),
    _max_in_range(counts_sim_true) if counts_sim_true is not None else 1,
)

divider = make_axes_locatable(ax)
inset_ax = divider.append_axes("top", size="50%", pad=0.0)
if PLOT_TRUE:
    solid_edge_histogram(
        sim_spec_idxs,
        ax=inset_ax,
        bins=bins,
        color=COL_SIM_TRUE,
        label=r'CatSIM (true $\alpha$)',
        alpha=0.2,
        lw=1.5,
        linestyle='--'
    )
solid_edge_histogram(
    sim_meas_spec_idxs,
    ax=inset_ax,
    bins=bins,
    color=COL_SIM_MEAS,
    label=r'CatSIM (measured $\alpha$)',
    alpha=0.2,
    lw=1.5
)
solid_edge_histogram(
    real_spec_idxs,
    ax=inset_ax,
    bins=bins,
    color=COL_REAL_MEAS,
    label=r'CatWISE (measured $\alpha$)',
    alpha=0.2,
    lw=1.5
)
inset_ax.set_xlim(zoom_left, zoom_right)
inset_ax.set_yscale('log')
lower_ylim = max(1, max_count / 5)
inset_ax.set_ylim(lower_ylim, max_count * zoom_y_margin)
inset_ax.tick_params(
    axis='x',
    which='both',
    labelsize=8,
    labelbottom=False,
    labeltop=True,
    top=True,
    bottom=False
)
inset_ax.tick_params(axis='y', which='both', length=0, labelleft=False)
inset_ax.xaxis.set_label_position('top')
inset_ax.xaxis.set_ticks_position('top')
inset_ax.set_xlabel(r'Spectral index $\alpha$ (zoom)')
inset_indicator = ax.indicate_inset_zoom(inset_ax, edgecolor='gray')
for connector in getattr(inset_indicator, 'connectors', []):
    if connector is not None:
        connector.set_visible(False)

if args.save:
    figure_dir = os.path.join(
        os.path.expanduser('~'),
        'Documents',
        'papers',
        'catwise_sbi',
        'figures'
    )
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, 'catwise_alpha_histogram.pdf')
    print(f'Saving alpha plot to {figure_path}...')
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)

w1_values = masked_catalogue['w1']
valid_real_alpha = np.isfinite(real_spec_idxs)
mag_selection = (
    (w1_values >= 16.3)
    & (w1_values < 16.4)
    & valid_real_alpha
)
selected_real_alphas = real_spec_idxs[mag_selection]
print(
    f'Number of CatWISE sources with 16.3 ≤ W1 < 16.4: '
    f'{selected_real_alphas.size}'
)

if selected_real_alphas.size > 0:
    fig_mag, ax_mag = plt.subplots(figsize=(4.5, 3.8))
    solid_edge_histogram(
        selected_real_alphas,
        ax=ax_mag,
        bins=40,
        color=COL_REAL_MEAS,
        label=r'CatWISE (measured $\alpha$)',
        alpha=0.3,
        lw=1.2
    )
    ax_mag.set_xlabel(r'Spectral index $\alpha$')
    ax_mag.set_ylabel('Count')
    ax_mag.set_yscale('log')
    ax_mag.set_title(r'CatWISE $\alpha$ for $16.3 \leq W1 < 16.4$')
    ax_mag.legend()
    print(np.mean(selected_real_alphas))
    plt.close(fig_mag)

smooth_map(
    dmap,
    cmap='magma',
    unit='Averaged source count (CatSIM)',
    format='%.3g',
    fontsize=COLORBAR_FONTSIZE
)
if args.save:
    figure_dir = os.path.join(
        os.path.expanduser('~'),
        'Documents',
        'papers',
        'catwise_sbi',
        'figures'
    )
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, 'catsim_dmap.pdf')
    print(f'Saving catsim dmap plot to {figure_path}...')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
plt.show()
