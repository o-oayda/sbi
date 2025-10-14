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
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

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
    corner_path="${RESULTS_DIR}/${sanitized_model}${run_name}_corner.png"
    sky_path="${RESULTS_DIR}/${sanitized_model}${run_name}_sky.png"

    echo "Processing ${run_name} (model: ${model_identifier})"

    run_log=$(mktemp)
    if ! (
        cd "${PROJECT_ROOT}"
        JAX_PLATFORMS=cpu "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.posterior_cli \
            "${run_path}" \
            --corner "${corner_path}" \
            --sky-prob "${sky_path}" \
            --legend "${latexified_model}" \
            --sky-truth 264 45 238 29 237 42 \
            --sky-truth-labels CMB Secrest+21 Dam+23 \
            --sky-smooth 0.1 \
            --logz-average-start 3 --logz-average-simple \
            "$@"
    ) >"${run_log}" 2>&1; then
        status=$?
        cat "${run_log}"
        echo "Error: posterior_cli failed for ${run_name} (exit ${status})." >&2
        rm -f "${run_log}"
        exit "${status}"
    fi

    cat "${run_log}"

    if ! parse_output=$(
        "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.logz_summary extract "${run_log}"
    ); then
        echo "Warning: Failed to parse logZ summary for ${run_name}; check log output." >&2
        rm -f "${run_log}"
        continue
    fi

    IFS=$'\t' read -r parsed_mean parsed_std <<<"${parse_output}"
    printf -v logz_row '%s\t%s\t%s\t%s' "$run_name" "$latexified_model" "$parsed_mean" "$parsed_std"
    logz_rows+=("$logz_row")
    rm -f "${run_log}"
done

if (( ${#logz_rows[@]} )); then
    printf '%s\n' "${logz_rows[@]}" | "${PYTHON_EXECUTABLE}" -m dipolesbi.tools.logz_summary write --output-dir "${RESULTS_DIR}"
fi
