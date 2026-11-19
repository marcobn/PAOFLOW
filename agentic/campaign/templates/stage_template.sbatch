#!/bin/bash
# One stage of a campaign on Slurm (template). The guards come from LESSONS_LEARNED.md:
#   - cores: a multiple of the billing granularity, pools balanced over the k-points (balanced_pools);
#   - time limit: about 1.5x the expected time, and pw.x max_seconds below it so that a long run can restart;
#   - MPI drivers through python -m mpi4py (or run_main): any exception aborts every rank instead of hanging;
#   - success means "JOB DONE" (pw.x) or the expected result file, not a zero exit code alone;
#   - results copied to permanent storage at the end of every stage (scratch is purged).
# Name every job of a run with the run's prefix (CampaignRun.job_prefix): sbatch -J <prefix><stage> ...
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH -n <cores>
#SBATCH -t <hh:mm:ss>
#SBATCH -o <run dir>/jobs/logs/%x.%j.out
set -uo pipefail
source ~/<environment script> > /dev/null 2>&1
export OMP_NUM_THREADS=1
log() { echo "$(date '+%F %T') $*"; }

RUN=<run dir>
NAME=<prefix>                 # input and output stem
NPOOL=<pools>                 # one pool per k-point when the billed cores allow it
RESULTS=<permanent dir>

cd "$RUN" || { log "ERROR: no run directory $RUN"; exit 1; }
if grep -q 'JOB DONE' "$NAME.out" 2>/dev/null; then log "$NAME done before"; exit 0; fi

# A displaced cell may start from the converged density of its undisplaced reference (same basis):
# mkdir -p tmp_$NAME/$NAME.save && cp tmp_<ref>/<ref>.save/charge-density.* tmp_$NAME/$NAME.save/

log "$NAME on $SLURM_NTASKS cores, -nk $NPOOL"
srun pw.x -nk "$NPOOL" -in "$NAME.in" > "$NAME.out" 2>&1
# a Python stage instead:  srun python -m mpi4py driver.py > driver.log 2>&1 || { log "ERROR: driver.py"; exit 1; }

if ! grep -q 'JOB DONE' "$NAME.out"; then
    grep -E 'Error in routine|convergence NOT achieved|Maximum CPU time exceeded' -A2 "$NAME.out" | head -20
    log "ERROR: $NAME did not finish"
    exit 1
fi
grep -E 'convergence has been achieved|PWSCF.*WALL' "$NAME.out"

mkdir -p "$RESULTS"
rsync -a --exclude 'tmp_*' --exclude '*.save' "$RUN"/ "$RESULTS"/   # never --delete
log "$NAME done, results in $RESULTS"
