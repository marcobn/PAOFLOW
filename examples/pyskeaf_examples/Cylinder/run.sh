#!/bin/bash
#SBATCH -J skeaf_cylinder
#SBATCH -o %j.out
#SBATCH -p regular
#SBATCH -n 16
#SBATCH --cpus-per-task=1
#SBATCH -t 00:02:00

# export OMP_NUM_THREADS=1  # avoid accidental overthreading

source ~/codes/anaconda3/etc/profile.d/conda.sh
conda activate paoflow_develop


module load QuantumESPRESSO/7.3.1-foss-2023a


mpirun -np 16 python main.py > paoflow.out