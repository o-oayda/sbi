import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec


save_dir = 'exp_out/nside_16_dequantise32'

def main():
    parser = argparse.ArgumentParser(description="Accumulate evidence from subdirectories.")
    parser.add_argument('--save_dir', default=save_dir, help='Directory containing subdirectories with epoch_lnZ.npy')
    args = parser.parse_args()

    lnZ = []
    lnZ_err = []

    for subdir in os.listdir(args.save_dir):
        subdir_path = os.path.join(args.save_dir, subdir)
        file_path = os.path.join(subdir_path, 'epoch_lnZ.npy')
        if os.path.isdir(subdir_path) and os.path.isfile(file_path):
            data = np.load(file_path, allow_pickle=True)
            lnZ.append(data[0])
            lnZ_err.append(data[1])

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    
    # Left panel: error bar plots
    ax0 = fig.add_subplot(gs[0])
    for _, (y, yerr) in enumerate(zip(lnZ, lnZ_err)):
        x = range(len(y))
        ax0.errorbar(x, y, yerr=yerr, color='tab:blue', alpha=0.3, capsize=3)
    ax0.set_xlabel('Index')
    ax0.set_ylabel('lnZ value')
    ax0.set_title('Evidence Accumulation')

    # Right panel: final lnZ values
    ax1 = fig.add_subplot(gs[1])
    final_lnZ = [y[-1] for y in lnZ]
    
    # Plot individual final lnZ values
    ax1.scatter([0]*len(final_lnZ), final_lnZ, color='grey', alpha=0.3)
    
    # Plot mean and std as a single point with error bar
    mean_lnZ = np.mean(final_lnZ)
    std_lnZ = np.std(final_lnZ)
    ax1.errorbar(
        0, 
        mean_lnZ, 
        yerr=std_lnZ, 
        fmt='o', 
        alpha=0.7, 
        color='tab:red', 
        label='Mean ± 1 std', 
        capsize=5, 
        zorder=10
    )
    
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Final'])
    ax1.set_title('Final lnZ')
    ax1.set_ylabel('lnZ value')
    ax1.legend()
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    # Add horizontal dashed line at y=-14262 to both panels
    for ax in [ax0, ax1]:
        ax.axhline(-14262, color='black', linestyle='--', linewidth=1, label='y = -14262')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
