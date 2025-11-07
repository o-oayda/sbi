#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment_root> [posterior_cli_args...]" >&2
    exit 1
fi

EXPERIMENT_ROOT=$1
shift || true

if [[ ! -d "${EXPERIMENT_ROOT}" ]]; then
    echo "Error: '${EXPERIMENT_ROOT}' is not a directory." >&2
    exit 1
fi

EXPERIMENT_ROOT=$(cd "${EXPERIMENT_ROOT}" && pwd)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}"

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

RESULTS_DIR="${EXPERIMENT_ROOT}/results"
mkdir -p "${RESULTS_DIR}"

logz_rows=()

shopt -s nullglob
run_dirs=("${EXPERIMENT_ROOT}"/*/)
shopt -u nullglob

if [[ ${#run_dirs[@]} -eq 0 ]]; then
    echo "No subdirectories found in ${EXPERIMENT_ROOT}; nothing to do." >&2
    exit 0
fi

for run_path in "${run_dirs[@]}"; do
    run_path=${run_path%/}
    run_name=$(basename "${run_path}")
    if [[ "${run_name}" == "results" ]]; then
        continue
    fi
    config_path="${run_path}/configs.txt"

    if [[ ! -f "${config_path}" ]]; then
        echo "Warning: Skipping ${run_name} (missing configs.txt)." >&2
        continue
    fi

    model_identifier=$("${PYTHON_EXECUTABLE}" -c "
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
text = config_path.read_text()
match = re.search(r\"model_identifier=['\\\"]([^'\\\"]+)['\\\"]\", text)
if match:
    print(match.group(1), end='')
" "${config_path}")

    if [[ -z "${model_identifier:-}" ]]; then
        echo "Warning: Skipping ${run_name} (model_identifier not found)." >&2
        continue
    fi

    sanitized_model=$(echo "${model_identifier}" | tr -c '[:alnum:]_-' '_')
    latexified_model=$("${PYTHON_EXECUTABLE}" -m dipolesbi.tools.model_labels "${model_identifier}")
    corner_path="${RESULTS_DIR}/${sanitized_model}${run_name}_corner.pdf"
    sky_path="${RESULTS_DIR}/${sanitized_model}${run_name}_sky.pdf"

    echo "Processing ${run_name} (model: ${model_identifier})"

    command=(
        "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.posterior_cli
        "${run_path}"
        --corner "${corner_path}"
        --sky-prob "${sky_path}"
        --legend "${latexified_model}"
        --sky-truth 264 48 238 29 237 42
        --sky-truth-labels CMB S21 D23
        --sky-top-quad
        --corner-no-legend
        --sky-smooth 0.1
        --logz-average-start 3 --logz-average-simple
    )
    if (( $# > 0 )); then
        command+=("$@")
    fi

    run_log=$(mktemp)
    if ! "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.cli_capture \
        --log "${run_log}" \
        --cwd "${PROJECT_ROOT}" \
        --env "JAX_PLATFORMS=cpu" \
        -- "${command[@]}"; then
        status=$?
        cat "${run_log}"
        echo "Error: posterior_cli failed for ${run_name} (exit ${status}). Skipping." >&2
        rm -f "${run_log}"
        continue
    fi

    cat "${run_log}"

    if ! parse_output=$(
        "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.logz_summary extract "${run_log}"
    ); then
        echo "Warning: Failed to parse logZ summary for ${run_name}; check log output." >&2
        rm -f "${run_log}"
        continue
    fi

    IFS=$'\t' read -r parsed_mean parsed_std parsed_dkl_mean parsed_dkl_std parsed_dg_mean parsed_dg_std <<<"${parse_output}"
    printf -v logz_row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
        "$run_name" "$latexified_model" "$parsed_mean" "$parsed_std" "$model_identifier" \
        "$parsed_dkl_mean" "$parsed_dkl_std" "$parsed_dg_mean" "$parsed_dg_std"
    logz_rows+=("$logz_row")
    rm -f "${run_log}"
done

if (( ${#logz_rows[@]} )); then
    printf '%s\n' "${logz_rows[@]}" | "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.logz_summary write --output-dir "${RESULTS_DIR}"
    cp -f "${RESULTS_DIR}/logb_summary.tex" "${HOME}/Documents/papers/catwise_sbi/logb_summary.tex"
fi

best_corner=$(ls "${RESULTS_DIR}"/free_gauss_extra_err_*_corner.pdf 2>/dev/null | head -n 1 || true)
best_sky=$(ls "${RESULTS_DIR}"/free_gauss_extra_err_*_sky.pdf 2>/dev/null | head -n 1 || true)
if [[ -z "${best_corner}" ]]; then
    best_run_dir=$(for dir_path in "${run_dirs[@]}"; do
        dir_base=${dir_path%/}
        dir_base=${dir_base##*/}
        if [[ "${dir_base}" == *"free_gauss_extra_err"* ]]; then
            printf '%s\n' "${dir_base}"
        fi
    done | head -n 1)
    if [[ -n "${best_run_dir}" ]]; then
        candidate="${RESULTS_DIR}/free_gauss_extra_err_${best_run_dir}_corner.pdf"
        if [[ -f "${candidate}" ]]; then
            best_corner="${candidate}"
        fi
    fi
fi

if [[ -f "${best_corner}" ]]; then
    papers_figures_dir="${HOME}/Documents/papers/catwise_sbi/figures"
    mkdir -p "${papers_figures_dir}"
    dest_corner="${papers_figures_dir}/free_gauss_extra_err_corner.pdf"
    dest_sky="${papers_figures_dir}/free_gauss_extra_err_sky.pdf"

    cp -f "${best_corner}" "${dest_corner}"
    echo "Copied free_gauss_extra_err corner plot to ${dest_corner}"

    cp -f "${best_sky}" "${dest_sky}"
    echo "Copied free_gauss_extra_err sky plot to ${dest_sky}"
else
    echo "Warning: free_gauss_extra_err corner plot not found; skipped copy." >&2
fi
