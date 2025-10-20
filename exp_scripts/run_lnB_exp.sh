#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --nside <int> --downscale-nside <int> --out-dir <path>

Runs lnB_exp.py for seeds 0 through 24 with the supplied arguments.
EOF
}

NSIDE=""
COARSE_NSIDE=""
OUT_DIR=""

while (($#)); do
    case "$1" in
        --nside)
            NSIDE="$2"
            shift 2
            ;;
        --downscale-nside)
            COARSE_NSIDE="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$NSIDE" || -z "$COARSE_NSIDE" || -z "$OUT_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

for SEED in {0..24}; do
    echo "Running lnB_exp.py with ssnle_seed=${SEED}"
    JAX_PLATFORMS=cpu python dipolesbi/scripts/lnB_exp.py \
        --nside "${NSIDE}" \
        --downscale_nside "${COARSE_NSIDE}" \
        --ssnle_seed "${SEED}" \
        --out_dir "${OUT_DIR}"
done

echo "Completed runs for seeds 0 through 24."
