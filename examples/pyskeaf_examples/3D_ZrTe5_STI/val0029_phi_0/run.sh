#!/bin/bash
#SBATCH -J skeaf_val0029_phi_0
#SBATCH -o %j.out
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --cpus-per-task=2
#SBATCH -t 03:00:00

# export OMP_NUM_THREADS=1  # avoid accidental overthreading

source ~/codes/anaconda3/etc/profile.d/conda.sh
conda activate paoflow_develop


module load QuantumESPRESSO/7.3.1-foss-2023a


mpirun -np 32 python main.py > paoflow.out