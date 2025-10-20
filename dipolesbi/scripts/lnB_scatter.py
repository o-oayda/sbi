import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter


rc("text", usetex=True)
plt.rcParams["font.family"] = "sans-serif"


LABEL_OVERRIDES = {
    "free_dipole": r"Free dipole",
    "cmb_dipole": r"CMB dipole",
    "cmb_velocity": r"CMB velocity",
    "cmb_direction": r"CMB direction",
}

LEGEND_LABELS = {
    "true": r"Analytic likelihood ($N_{\mathrm{side}}=64$)",
    "nle": r"NLE likelihood ($N_{\mathrm{side}}=4$)",
}


def format_model_label(name: str) -> str:
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    return " ".join(word.capitalize() for word in name.split("_"))


DEFAULT_RESULTS_DIR = Path("lnB_exp")
DEFAULT_OUTPUT = Path("lnB_exp/bayes_factor_scatter.pdf")


def gather_bayes_factors(results_root: Path):
    """Load lnZ results from each run and accumulate ln Bayes factors."""
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory '{results_root}' does not exist.")

    bf_data = {"true": defaultdict(list), "nle": defaultdict(list)}

    for run_dir in sorted(results_root.iterdir()):
        lnz_path = run_dir / "lnZ_results.json"
        if not lnz_path.is_file():
            continue

        with lnz_path.open("r") as fp:
            lnz = json.load(fp)

        free = lnz.get("free_dipole")
        if free is None:
            # Skip runs without the reference model.
            continue

        for key in ("true", "nle"):
            free_entry = free.get(key)
            if not free_entry or "lnZ" not in free_entry:
                continue

            ref_lnZ = free_entry["lnZ"]

            for model_name, model_data in lnz.items():
                entry = model_data.get(key)
                if not entry or "lnZ" not in entry:
                    continue
                bf_data[key][model_name].append(entry["lnZ"] - ref_lnZ)

    return bf_data


def make_scatter_plot(bf_data, output_path: Path):
    """Create the column scatter plot comparing true and nle ln Bayes factors."""
    models = sorted({model for key in bf_data for model in bf_data[key]})
    if "free_dipole" in models:
        models.remove("free_dipole")
        models.insert(0, "free_dipole")

    x_positions = np.arange(len(models), dtype=float)
    offsets = {"true": -0.12, "nle": 0.12}
    colors = {"true": "cornflowerblue", "nle": "tomato"}
    labels_done = set()

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    for idx, model in enumerate(models):
        for key in ("true", "nle"):
            values = bf_data[key].get(model, [])
            if not values:
                continue
            x = np.full(len(values), x_positions[idx] + offsets[key])
            ax.scatter(
                x,
                values,
                alpha=0.15,
                color=colors[key],
                label=None,
                edgecolor="none",
            )

            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            error_label = LEGEND_LABELS[key] if key not in labels_done else None
            ax.errorbar(
                x_positions[idx] + offsets[key],
                mean_val,
                yerr=std_val,
                fmt="o",
                color=colors[key],
                ecolor=colors[key],
                elinewidth=1.5,
                capsize=4,
                alpha=0.9,
                label=error_label,
            )
            labels_done.add(key)

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_model_label(m) for m in models], rotation=20)
    ax.set_ylabel(r"$\ln B$ versus free dipole")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )
    ax.set_facecolor("white")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot of ln Bayes factors relative to the free_dipole model."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing run subdirectories with lnZ_results.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the generated plot.",
    )
    args = parser.parse_args()

    bf_data = gather_bayes_factors(args.results_dir)
    if not bf_data["true"] and not bf_data["nle"]:
        raise RuntimeError(
            f"No Bayes factor data found in '{args.results_dir}'. "
            "Ensure lnZ_results.json files are present."
        )

    make_scatter_plot(bf_data, args.output)


if __name__ == "__main__":
    main()
