#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="tbmodel-it"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=2000

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPICH_GPU_SUPPORT_ENABLED=0
ulimit -s unlimited

MINICONDA_PATH="${MINICONDA_PATH:-/home/jayn/miniconda3}"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate paoflow_new

set -euo pipefail

export PARALLEL_EXEC="srun -n ${SLURM_NTASKS:-1}"

cd "$SLURM_SUBMIT_DIR"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$script_dir/job.sh" --paoflow-test
