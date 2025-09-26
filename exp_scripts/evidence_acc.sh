#!/bin/bash

# Parse arguments
RESUME=0
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        RESUME=1
    fi
done

# Remove --resume from positional parameters
set -- $(printf '%s\n' "$@" | grep -v -- '--resume')

# e.g. exp_scripts/evidence_acc.sh 16 'cold_start_only_post_100k' --resume
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <NSIDE> <DESCRIPTOR> [--resume]"
    exit 1
fi

NSIDE=$1
MODE=$2
DESCRIPTOR=$3

# First, check NSIDE is a positive integer
if ! [[ "$NSIDE" =~ ^[0-9]+$ ]] || [ "$NSIDE" -le 0 ]; then
    echo "Error: NSIDE must be a positive integer."
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
    python dipolesbi/scripts/run_multiround.py --nside "$NSIDE" --mode "$MODE" --ssnle_seed "$SEED" --out_dir "$OUTDIR" >> "$LOGFILE" 2>&1
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

