#!/bin/bash
#SBATCH -J proj_MgB2
#SBATCH -o %j.out
#SBATCH -p regular
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:02:00


# source ~/codes/anaconda3/etc/profile.d/conda.sh
# conda activate paoflow_develop

module load QuantumESPRESSO/7.3.1-foss-2023a

# srun pw.x < scf.in > scf.out
# srun pw.x < nscf.in > nscf.out
srun projwfc.x < proj.in > proj.out
# srun python main.py > paoflow.out
# srun python main_pyskeaf.py > paoflow_pyskeaf.out
