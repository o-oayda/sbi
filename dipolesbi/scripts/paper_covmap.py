import argparse
import os
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from catsim import CatwiseConfig, Catwise
from dipolesbi.tools.utils import ParameterMap
from dipolesbi.tools.plotting import smooth_map


mpl.rcParams['text.usetex'] = True

COLORBAR_FONTSIZE = {
    'cbar_label': 20,
    'cbar_tick_label': 18
}


def main():
    parser = argparse.ArgumentParser(description="Generate CatWISE coverage and error maps.")
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='If set, save plots to papers/catwise_sbi/figures; otherwise display interactively.'
    )
    parser.add_argument(
        '--fit_inverse_sqrt',
        action='store_true',
        help='Overlay least-squares $k \\times \\mathrm{coverage}^{-1/2}$ fit on scatter plots.'
    )
    parser.add_argument(
        '--use_raw_error',
        action='store_true',
        help='Use raw W1 error rather than percentage in scatter histograms.'
    )
    args = parser.parse_args()

    config = CatwiseConfig(
        cat_w1_max=17.0, 
        cat_w12_min=0.5,
        magnitude_error_dist='gaussian'
    )
    catwise = Catwise(config)
    catwise.determine_masked_pixels()
    catwise.make_real_sample()

    pixel_indices = hp.ang2pix(
        64,
        catwise.real_catalogue['l'], 
        catwise.real_catalogue['b'],
        nest=True,
        lonlat=True
    )
    coverage_parameter = catwise.real_catalogue['w1cov']
    coverage_map = ParameterMap(pixel_indices, coverage_parameter, nside=64).get_map()
    coverage_map[~catwise.binary_mask] = np.nan

    raw_error = catwise.real_catalogue['w1e']
    if args.use_raw_error:
        error_values = raw_error
        error_label = 'W1 Error'
    else:
        error_values = np.divide(
            raw_error,
            catwise.real_catalogue['w1'],
            out=np.zeros_like(raw_error),
            where=catwise.real_catalogue['w1'] != 0
        ) * 100.0
        error_label = r'W1 Error (\%)'
    error_map = ParameterMap(pixel_indices, error_values, nside=64).get_map()
    error_map[~catwise.binary_mask] = np.nan

    figure_dir = os.path.join(
        os.path.expanduser('~'),
        'Documents',
        'papers',
        'catwise_sbi',
        'figures'
    )
    if args.save_plots:
        os.makedirs(figure_dir, exist_ok=True)

    def maybe_save(fig, filename, description):
        if fig is None:
            return
        if isinstance(fig, tuple):  # healpy may return (fig, ax)
            fig = fig[0]
        if args.save_plots:
            output_path = os.path.join(figure_dir, filename)
            print(f"Saving {description} to {output_path}...")
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)

    hp.projview(
        coverage_map,
        nest=True,
        norm='log',
        unit='W1 Coverage',
        format='%.3g',
        cmap='magma',
        fontsize=COLORBAR_FONTSIZE
    )
    coverage_fig = plt.gcf()
    maybe_save(coverage_fig, 'catwise_coverage_map.pdf', 'coverage plot')

    hp.projview(
        error_map,
        nest=True,
        unit=error_label,
        format='%.3g',
        cmap='magma',
        fontsize=COLORBAR_FONTSIZE
    )
    error_fig = plt.gcf()
    maybe_save(error_fig, 'catwise_error_map.pdf', 'error plot')

    smooth_map(
        catwise.real_density_map,
        cmap='magma',
        unit='Averaged source count (CatWISE)',
        min=54.7,
        max=58,
        format='%.3g',
        fontsize=COLORBAR_FONTSIZE
    )
    density_fig = plt.gcf()
    maybe_save(density_fig, 'catwise_density_map.pdf', 'density plot')

    def plot_scatter(ax, x_values, y_values, title_suffix: str = "", mask: np.ndarray | None = None, y_label: str | None = None):
        base_mask = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & (x_values > 0)
            & (y_values > 0)
        )
        if mask is not None:
            base_mask &= mask
        if not np.any(base_mask):
            ax.set_xlabel('W1 Coverage')
            ax.set_ylabel(y_label or r'W1 Error (\%)')
            ax.set_title(title_suffix.strip())
            return

        x_valid = x_values[base_mask]
        y_valid = y_values[base_mask]
        x_min = np.percentile(x_valid, 0.5)
        y_min = np.percentile(y_valid, 0.5)
        x_max = np.percentile(x_valid, 99.5)
        y_max = np.percentile(y_valid, 99.5)
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_min -= x_span * 0.05
        x_max += x_span * 0.05
        y_min -= y_span * 0.05
        y_max += y_span * 0.05
        in_range = (
            (x_valid >= x_min)
            & (x_valid <= x_max)
            & (y_valid >= y_min)
            & (y_valid <= y_max)
        )
        x_clipped = x_valid[in_range]
        y_clipped = y_valid[in_range]
        hist = ax.hist2d(
            x_clipped,
            y_clipped,
            bins=100,
            norm=mpl.colors.LogNorm(),
            cmap='magma'
        )
        plt.colorbar(hist[3], ax=ax, label='Count')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if x_span > 0 and y_span > 0:
            ax.set_aspect((x_max - x_min) / (y_max - y_min))
        ax.set_xlabel('W1 Coverage')
        ax.set_ylabel(y_label or r'W1 Error (\%)')
        if title_suffix:
            ax.set_title(title_suffix)

        if args.fit_inverse_sqrt:
            predictor = x_clipped ** (-0.5)
            denom = np.dot(predictor, predictor)
            if denom > 0:
                scale = np.dot(y_clipped, predictor) / denom
                coverage_grid = np.linspace(x_min, x_max, 200)
                fit_line = scale * coverage_grid ** (-0.5)
                ax.plot(
                    coverage_grid,
                    fit_line,
                    color='tab:red',
                    linewidth=2,
                    label=r'Least squares fit: $k \, \mathrm{coverage}^{-1/2}$'
                )
                ax.legend()

    # Map-based scatter
    map_scatter_fig, map_scatter_ax = plt.subplots()
    map_mask = catwise.binary_mask.ravel()
    plot_scatter(
        map_scatter_ax,
        coverage_map.ravel(),
        error_map.ravel(),
        title_suffix='Map pixels',
        mask=map_mask,
        y_label=error_label
    )

    # Catalogue sample scatter
    sample_scatter_fig, sample_scatter_ax = plt.subplots()
    catalogue_mask = map_mask[pixel_indices]
    plot_scatter(
        sample_scatter_ax,
        coverage_parameter,
        error_values,
        title_suffix='Catalogue samples (masked)',
        mask=catalogue_mask,
        y_label=error_label
    )

    unmasked_scatter_fig, unmasked_scatter_ax = plt.subplots()
    plot_scatter(
        unmasked_scatter_ax,
        coverage_parameter,
        error_values,
        title_suffix='Catalogue samples (unmasked)',
        y_label=error_label
    )

    if not args.save_plots:
        plt.show()


if __name__ == "__main__":
    main()
