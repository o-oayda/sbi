#!/bin/bash


# Usage: ./evidence_acc.sh <NSIDE>
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <NSIDE>"
    exit 1
fi

NSIDE=$1
OUTDIR="nside_${NSIDE}"

LOGFILE="${OUTDIR}/run.log"
mkdir -p "$OUTDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
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
