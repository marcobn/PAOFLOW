#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="transport-it"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=2500

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPICH_GPU_SUPPORT_ENABLED=0
ulimit -s unlimited

MINICONDA_PATH=${MINICONDA_PATH:-/home/jayn/miniconda3}

source "$MINICONDA_PATH/etc/profile.d/conda.sh"
conda activate paoflow_new

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export QE_BIN=${QE_BIN:-/home/jayn/qe-7.4.1/bin}
export PARALLEL_EXEC="srun -n ${SLURM_NTASKS:-4}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

chmod u+x "$script_dir/create_assets.sh"
chmod u+x "$script_dir/build_tar.sh"

create_status=0
build_status=0

"$script_dir/create_assets.sh" --all || create_status=$?
"$script_dir/build_tar.sh" --all || build_status=$?

if [[ $create_status -ne 0 || $build_status -ne 0 ]]; then
	exit 1
fi
