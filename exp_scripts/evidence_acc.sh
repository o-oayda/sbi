#!/bin/bash


# Usage: ./evidence_acc.sh <NSIDE>
if [ "$#" -ne 2 ]; then

    echo "Usage: $0 <NSIDE> <DESCRIPTOR>"
    exit 1
fi

NSIDE=$1
DESCRIPTOR=$2

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
for SEED in {0..24}
do
    log "Running SEED=$SEED..."
    python dipolesbi/scripts/run_multiround.py --nside "$NSIDE" --ssnle_seed "$SEED" --out_dir "$OUTDIR" >> "$LOGFILE" 2>&1
    if [ $? -eq 0 ]; then
        log "SEED=$SEED completed successfully."
    else
        log "Error: SEED=$SEED failed. Stopping script."
        exit 1
    fi
done
log "All runs completed for NSIDE=$NSIDE."
