#!/bin/bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --nside <INT> --downscale_nside <INT> --mode <MODE[,MODE...]> --descriptor <NAME> [--resume]

Examples:
  $0 --nside 16 --downscale_nside 8 --mode 'NLE,NPE' --descriptor cold_start
  $0 --nside 16 --downscale_nside 8 --mode NLE --descriptor test_run --resume
EOF
}

NSIDE=""
DOWNSCALE_NSIDE=""
MODE_INPUT=""
DESCRIPTOR=""
RESUME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nside)
            NSIDE=${2:-}
            shift 2
            ;;
        --downscale_nside)
            DOWNSCALE_NSIDE=${2:-}
            shift 2
            ;;
        --mode)
            MODE_INPUT=${2:-}
            shift 2
            ;;
        --descriptor)
            DESCRIPTOR=${2:-}
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            echo "Unexpected positional argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$NSIDE" || -z "$DOWNSCALE_NSIDE" || -z "$MODE_INPUT" || -z "$DESCRIPTOR" ]]; then
    echo "Error: --nside, --downscale_nside, --mode, and --descriptor are required." >&2
    usage
    exit 1
fi

# First, check NSIDE is a positive integer
if ! [[ "$NSIDE" =~ ^[0-9]+$ ]] || [ "$NSIDE" -le 0 ]; then
    echo "Error: NSIDE must be a positive integer." >&2
    exit 1
fi

# Next, check NSIDE is a power of 2 using bitwise AND:
# For powers of 2, NSIDE in binary has exactly one bit set (e.g., 8 is 1000).
# NSIDE - 1 flips all bits after the single set bit (e.g., 8-1=7 is 0111).
# The bitwise AND of NSIDE and NSIDE-1 will be zero only for powers of 2:
#   8 & 7 = 1000 & 0111 = 0000
if [ $((NSIDE & (NSIDE - 1))) -ne 0 ]; then
    echo "Error: NSIDE must be a power of 2."
    exit 1
fi

if ! [[ "$DOWNSCALE_NSIDE" =~ ^[0-9]+$ ]] || [ "$DOWNSCALE_NSIDE" -le 0 ]; then
    echo "Error: DOWNSCALE_NSIDE must be a positive integer." >&2
    exit 1
fi

if [ $((DOWNSCALE_NSIDE & (DOWNSCALE_NSIDE - 1))) -ne 0 ]; then
    echo "Error: DOWNSCALE_NSIDE must be a power of 2." >&2
    exit 1
fi

if [ "$DOWNSCALE_NSIDE" -gt "$NSIDE" ]; then
    echo "Error: DOWNSCALE_NSIDE must be less than or equal to NSIDE." >&2
    exit 1
fi

# Validate MODE is non-empty and convert comma separated list to canonical form
if [[ -z "${MODE_INPUT// }" ]]; then
    echo "Error: MODE must be a non-empty string."
    exit 1
fi

IFS=',' read -r -a MODE_ARRAY <<< "$MODE_INPUT"
VALIDATED_MODES=()
for raw_mode in "${MODE_ARRAY[@]}"; do
    trimmed_mode=${raw_mode//[[:space:]]/}
    if [[ -z "$trimmed_mode" ]]; then
        echo "Error: MODE entries must be non-empty."
        exit 1
    fi
    VALIDATED_MODES+=("$trimmed_mode")
done
MODE=$(IFS=','; echo "${VALIDATED_MODES[*]}")

# Validate DESCRIPTOR is non-empty and not just whitespace
if [[ -z "${DESCRIPTOR// }" ]]; then
    echo "Error: DESCRIPTOR must be a non-empty string."
    exit 1
fi

OUTDIR="exp_out/nside_${NSIDE}_${DESCRIPTOR}"

LOGFILE="${OUTDIR}/run.log"
mkdir -p "$OUTDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$DESCRIPTOR] $1" | tee -a "$LOGFILE"
}

eval $(poetry env activate)

log "Starting multiround runs for NSIDE=$NSIDE"

START_SEED=0
END_SEED=24
RUN_SEEDS=()

if [ "$RESUME" -eq 1 ]; then
    log "Resume mode enabled. Checking for completed seeds..."
    COMPLETED_SEEDS=()
    RUN_SEEDS=()
    LAST_COMPLETED_SEED=-1
    for d in "$OUTDIR"/*_SEED*; do
        if [ -d "$d" ]; then
            SEED_NUM=$(basename "$d" | sed -n 's/.*_SEED\([0-9]\+\)$/\1/p')
            if [ -n "$SEED_NUM" ] && [ -f "$d/lnZ_evolution.png" ]; then
                COMPLETED_SEEDS+=("$SEED_NUM")
                if [ "$SEED_NUM" -gt "$LAST_COMPLETED_SEED" ]; then
                    LAST_COMPLETED_SEED=$SEED_NUM
                fi
            fi
        fi
    done
    if [ "$LAST_COMPLETED_SEED" -eq "$END_SEED" ]; then
        if [ -f "$OUTDIR"/*_SEED${END_SEED}/lnZ_evolution.png ]; then
            log "All seeds completed (last: $END_SEED). Nothing to resume."
            exit 0
        fi
    fi
    START_SEED=$((LAST_COMPLETED_SEED + 1))
    SKIPPED_SEEDS=$(printf "%s " "${COMPLETED_SEEDS[@]}")
    log "Seeds already completed and skipped: $SKIPPED_SEEDS"
    log "Resuming from SEED=$START_SEED."
fi

for SEED in $(seq $START_SEED $END_SEED)
do
    log "Running SEED=$SEED..."
    python dipolesbi/scripts/run_multiround.py --nside "$NSIDE" --downscale_nside "$DOWNSCALE_NSIDE" --mode "$MODE" --ssnle_seed "$SEED" --out_dir "$OUTDIR" >> "$LOGFILE" 2>&1
    if [ $? -eq 0 ]; then
        log "SEED=$SEED completed successfully."
        RUN_SEEDS+=("$SEED")
    else
        log "Error: SEED=$SEED failed. Stopping script."
        exit 1
    fi
done
SUMMARY_MSG="All runs completed for NSIDE=$NSIDE. Seeds run in this session: $(printf '%s ' "${RUN_SEEDS[@]}")"
log "$SUMMARY_MSG"
