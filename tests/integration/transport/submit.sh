#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="transport-it"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=2500

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MINICONDA_PATH=${MINICONDA_PATH:-$HOME/miniconda3}

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate paoflow_new

cd "$SLURM_SUBMIT_DIR"

export QE_BIN=/home/jayn/qe-7.4.1/bin

./job.sh --all --build-assets
