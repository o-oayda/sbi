import os
import argparse
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from anesthetic import NestedSamples, read_csv as nested_read_csv
from getdist import MCSamples, plots


def get_param_names_and_labels(path: str) -> Tuple[List[str], List[str]]:
    key_names: List[str] = []
    key_labels: List[str] = []

    with open(path, 'r') as f:
        lines = f.readlines()

    inside_block = False
    for line in lines:
        if line.strip().startswith('DipolePriorNP'):
            inside_block = True
            continue
        if inside_block:
            if line.strip().startswith(')'):
                break
            match = re.search(r"\s*([A-Za-z0-9_]+):\s+kwarg='([^']+)'", line)
            if match:
                key_labels.append(match.group(1))
                key_names.append(match.group(2))

    return key_names, key_labels

def get_containing_folder(parent_dir: str, sub_dir: str) -> str:
    # resolve timestamp+seed folder name
    containing_folders = [
        f for f in os.listdir(os.path.join(parent_dir, sub_dir)) 
        if os.path.isdir(os.path.join(parent_dir, sub_dir, f))
    ]
    if len(containing_folders) != 1:
        raise ValueError("Expected exactly one containing folder inside FULL_DIR")
    return containing_folders[0]

def load_mc_samples(parent_dir: str, sub_dir: str) -> Tuple[MCSamples, List[str], List[str], Optional[Tuple[float, Optional[float]]]]:
    containing_folder = get_containing_folder(parent_dir, sub_dir)
    full_dir = os.path.join(parent_dir, sub_dir, containing_folder)
    csv_path = os.path.join(full_dir, 'samples.csv')
    npz_path = os.path.join(full_dir, 'samples.npz')

    config_path = os.path.join(full_dir, 'configs.txt')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Expected configs.txt alongside samples in {full_dir}")

    key_names, key_labels = get_param_names_and_labels(config_path)

    evidence: Optional[Tuple[float, Optional[float]]] = None

    if os.path.exists(csv_path):
        nested_samples = nested_read_csv(csv_path)
        weights = np.asarray(nested_samples.get_weights())
        raw_samples = nested_samples[key_names].to_numpy()
        logZ_mean = float(nested_samples.logZ())
        logZ_err: Optional[float]
        try:
            logZ_err = float(np.asarray(nested_samples.logZ(100)).std())
        except Exception:
            logZ_err = None
        evidence = (logZ_mean, logZ_err)
    elif os.path.exists(npz_path):
        data = np.load(npz_path)
        missing = [name for name in key_names if name not in data.files]
        if missing:
            raise KeyError(
                f"Missing parameters {missing} in {npz_path}. Available keys: {list(data.files)}"
            )
        raw_samples = np.column_stack([data[key] for key in key_names])
        weights = np.ones(raw_samples.shape[0]) / raw_samples.shape[0]
    else:
        raise FileNotFoundError(
            f"Neither samples.csv nor samples.npz found in {full_dir}"
        )

    return MCSamples(
        samples=raw_samples,
        weights=weights,
        sampler='nested',
        names=key_names,
        labels=key_labels
    ), key_names, key_labels, evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        nargs='+',
        required=True
    )
    args = parser.parse_args()

    PARENT_DIR = 'fiducial_results'

    reference_names: Optional[List[str]] = None
    reference_labels: Optional[List[str]] = None
    mc_samples: List[MCSamples] = []
    legend_labels: List[str] = []

    for sub_dir in args.save_dir:
        sample, names, labels, evidence = load_mc_samples(PARENT_DIR, sub_dir)

        if reference_names is None:
            reference_names = names
            reference_labels = labels
        else:
            if names != reference_names:
                raise ValueError(
                    f"Parameter names for {sub_dir} differ from previous directories"
                )
            if labels != reference_labels:
                raise ValueError(
                    f"Parameter labels for {sub_dir} differ from previous directories"
                )

        mc_samples.append(sample)
        legend_labels.append(sub_dir)

        if evidence is not None:
            mean, err = evidence
            if err is not None:
                print(f"{sub_dir}: logZ = {mean:.4g} ± {err:.4g}")
            else:
                print(f"{sub_dir}: logZ = {mean:.4g} (uncertainty unavailable)")
        else:
            print(f"{sub_dir}: logZ unavailable for provided samples")

    if reference_names is None:
        raise ValueError('No save directories processed')

    safe_name_parts = [sub_dir.replace(os.sep, '_') for sub_dir in legend_labels]
    plot_filename = '__'.join(safe_name_parts)
    save_path = os.path.join(PARENT_DIR, f'{plot_filename}.png')

    g = plots.get_subplot_plotter()
    g.triangle_plot(
        mc_samples,
        params=reference_names,
        filled=True,
        marker_args={'lw': 1}, # type: ignore
        legend_labels=legend_labels
    )
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
