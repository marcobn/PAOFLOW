#!/bin/bash
#SBATCH -J scf_FeP
#SBATCH -o %j.out
#SBATCH -p regular
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH -t 00:03:00


module load QuantumESPRESSO/7.3.1-foss-2023a

# srun pw.x < scf.in > scf.out
#srun pw.x < nscf.in > nscf.out
#srun pw.x < bans.in > bans.out;
#srun bands.x < bands.in > bands.out;
srun projwfc.x < proj.in > proj.out;
