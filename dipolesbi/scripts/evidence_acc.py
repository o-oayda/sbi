import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import argparse
from matplotlib.ticker import ScalarFormatter, FixedLocator
import matplotlib.gridspec as gridspec


def main():
    rcParams['text.usetex'] = True
    parser = argparse.ArgumentParser(description="Accumulate evidence from subdirectories.")
    parser.add_argument(
        '--save_dir', 
        help='Directory containing subdirectories with epoch_lnZ.npy'
    )
    parser.add_argument(
        '--smoothing_window',
        type=int,
        default=2,
        help='Window size for moving average smoothing of mean $\\ln \\mathcal{Z}$ per epoch.'
    )
    parser.add_argument(
        '--connect_runs',
        action='store_true',
        help='If provided, draw connecting lines between $\\ln \\mathcal{Z}$ points within each run.'
    )
    parser.add_argument(
        '--skip_delta',
        action='store_true',
        help='Disable the $\\Delta \\ln \\mathcal{Z}$ summary panel.'
    )
    parser.add_argument(
        '--save_plot',
        action='store_true',
        help='If set, save the figure under papers/catwise_sbi/figures instead of showing.'
    )
    args = parser.parse_args()

    base_dir = os.path.join('exp_out', args.save_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Could not find directory: {base_dir}")

    runs = []
    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        epoch_path = os.path.join(subdir_path, 'epoch_lnZ.npy')
        true_path = os.path.join(subdir_path, 'true_lnZ.npy')
        if not os.path.isdir(subdir_path):
            continue
        if not (os.path.isfile(epoch_path) and os.path.isfile(true_path)):
            continue

        epoch_data = np.load(epoch_path, allow_pickle=True)
        lnZ_trace = np.asarray(epoch_data[0], dtype=float)
        lnZ_err_trace = np.asarray(epoch_data[1], dtype=float)

        true_data = np.asarray(np.load(true_path, allow_pickle=True), dtype=float)
        if true_data.size < 2:
            raise ValueError(f"true_lnZ.npy in {subdir_path} must contain value and error.")

        runs.append(
            {
                'name': subdir,
                'lnZ': lnZ_trace,
                'lnZ_err': lnZ_err_trace,
                'true_lnZ': float(true_data[0]),
                'true_lnZ_err': float(true_data[1]),
            }
        )

    if not runs:
        raise ValueError(f"No runs with epoch_lnZ.npy and true_lnZ.npy found in {base_dir}")

    true_lnZ_values = np.array([run['true_lnZ'] for run in runs], dtype=float)
    true_lnZ_mean = np.mean(true_lnZ_values)
    true_lnZ_std = np.std(true_lnZ_values)
    print(
        f"True $\\ln \\mathcal{{Z}}$ across runs: mean={true_lnZ_mean:.4f}, "
        f"std={true_lnZ_std:.4f} (n={len(true_lnZ_values)})"
    )
    for idx, run in enumerate(runs[:3]):
        print(
            f"Run {idx}: {run['name']} true $\\ln \\mathcal{{Z}}$="
            f"{run['true_lnZ']:.4f} ± {run['true_lnZ_err']:.4f}"
        )

    if args.skip_delta:
        fig = plt.figure(figsize=(5, 3.5))
        ax0 = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax0 = fig.add_subplot(gs[0])
    
    # Left panel: error bar plots
    max_length = 0
    num_rounds = 0
    point_label_added = False
    for run in runs:
        y = run['lnZ']
        yerr = run['lnZ_err']
        if len(y) == 0:
            continue
        max_length = max(max_length, len(y))
        x = np.arange(1, len(y) + 1)
        if args.connect_runs:
            ax0.plot(x, y, color='cornflowerblue', alpha=0.2)
        label = r'NLE $\ln \mathcal{Z}$ per round' if not point_label_added else None
        ax0.scatter(x, y, color='cornflowerblue', alpha=0.5, s=10, label=label)
        point_label_added = True
    if max_length > 0:
        sum_vals = np.zeros(max_length, dtype=float)
        sum_sq = np.zeros(max_length, dtype=float)
        counts = np.zeros(max_length, dtype=float)

        for run in runs:
            y = run['lnZ']
            for idx, value in enumerate(y):
                sum_vals[idx] += value
                sum_sq[idx] += value**2
                counts[idx] += 1

        mean_trace = np.full(max_length, np.nan, dtype=float)
        std_trace = np.full(max_length, np.nan, dtype=float)

        mean_mask = counts > 0
        std_mask = counts > 1

        mean_trace[mean_mask] = sum_vals[mean_mask] / counts[mean_mask]

        variance = np.zeros(max_length, dtype=float)
        variance[:] = np.nan
        safe_counts = counts[std_mask]
        variance_vals = (sum_sq[std_mask] / safe_counts) - mean_trace[std_mask] ** 2
        variance_vals = np.maximum(
            variance_vals * safe_counts / (safe_counts - 1),
            0.0
        )
        variance[std_mask] = variance_vals
        std_trace = np.sqrt(variance)

        def smooth_series(values, valid_mask):
            window = max(1, min(args.smoothing_window, len(values)))
            if window <= 1:
                return values
            kernel = np.ones(window, dtype=float)
            filled_vals = np.where(valid_mask, values, 0.0)
            weights = valid_mask.astype(float)
            smooth_num = np.convolve(filled_vals, kernel, mode='same')
            smooth_den = np.convolve(weights, kernel, mode='same')
            smoothed = np.full_like(values, np.nan, dtype=float)
            positive = smooth_den > 0
            smoothed[positive] = smooth_num[positive] / smooth_den[positive]
            smoothed[~valid_mask] = np.nan
            return smoothed

        mean_trace = smooth_series(mean_trace, mean_mask)
        std_trace = smooth_series(std_trace, std_mask)

        epochs = np.arange(1, len(mean_trace) + 1)
        num_rounds = len(mean_trace)
        ax0.plot(
            epochs,
            mean_trace,
            color='tomato',
            linewidth=2,
            label=r'NLE $\ln \mathcal{Z}$ moving average'
        )

        upper_band = mean_trace + std_trace
        lower_band = mean_trace - std_trace
        valid_band = (~np.isnan(upper_band)) & (~np.isnan(lower_band)) & (~np.isnan(mean_trace))
        ax0.fill_between(
            epochs,
            lower_band,
            upper_band,
            where=valid_band,
            color='tomato',
            alpha=0.15,
            label=r'$\pm 1\sigma$ interval'
        )
    ax0.set_xlabel('Round')
    ax0.set_ylabel(r'$\ln \mathcal{Z}$')
    if num_rounds or max_length:
        upper = num_rounds if num_rounds else max_length
        ticks = np.arange(1, upper + 1)
        ax0.set_xlim(0.5, upper + 0.5)
        ax0.xaxis.set_major_locator(FixedLocator(ticks))
    else:
        ax0.xaxis.set_major_locator(FixedLocator([1]))
    # ax0.set_title('Evidence evolution over training epochs')

    if not args.skip_delta:
        ax1 = fig.add_subplot(gs[1])
        deltas = []
        for run in runs:
            if len(run['lnZ']) == 0:
                deltas.append(np.nan)
                continue
            final_lnZ = run['lnZ'][-1]
            delta = final_lnZ - run['true_lnZ']
            deltas.append(delta)

        x_positions = np.zeros(len(runs))
        ax1.scatter(
            x_positions,
            deltas,
            color='cornflowerblue',
            alpha=0.4,
            label=r'$\Delta \ln \mathcal{Z}$'
        )

        mean_delta = np.nanmean(deltas)
        std_delta = np.nanstd(deltas)

        ax1.errorbar(
            0,
            mean_delta,
            yerr=std_delta,
            fmt='o',
            alpha=1.0,
            color='tomato',
            label=r'Mean $\pm 1\sigma$',
            capsize=5,
            zorder=10
        )
        ax1.axhline(0.0, color='grey', linestyle='--', linewidth=1)
        ax1.set_xticks([0])
        label_round = num_rounds if num_rounds else max_length
        ax1.set_xticklabels([f'Round {label_round}'])
        ax1.xaxis.set_major_locator(FixedLocator([0]))
        ax1.set_title(r'$\Delta \ln \mathcal{Z}$')
        ax1.set_ylabel(r'$\Delta \ln \mathcal{Z}$')
        ax1.legend()
        ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    # Highlight mean true $\ln \mathcal{Z}$ ± 1 sigma on left panel
    lower = true_lnZ_mean - true_lnZ_std
    upper = true_lnZ_mean + true_lnZ_std
    ax0.axhspan(lower, upper, color='grey', alpha=0.4, label=r'True $\ln \mathcal{Z} \pm 1\sigma$ (known likelihood)')
    ax0.legend(loc='lower right')
    
    plt.tight_layout()
    figure_dir = os.path.join(
        os.path.expanduser('~'),
        'Documents',
        'papers',
        'catwise_sbi',
        'figures'
    )
    if args.save_plot:
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, 'evidence_evolution.pdf')
        print(f'Saving coverage plot to {figure_path}...')
        plt.savefig(figure_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    main()
