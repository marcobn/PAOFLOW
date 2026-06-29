#!/bin/bash
#SBATCH -J proj_FeP
#SBATCH -o %j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c.chen.ye@rug.nl
#SBATCH -p fat
#SBATCH -n 100
#SBATCH -t 00:15:00


module purge
module load 2022
module load QuantumESPRESSO/7.1-foss-2022a

#mpirun -np 64 /home/cchenye/codes/qe/qe-6.6/bin/pw.x < scf.in > scf.out
#mpirun -np 64 /home/cchenye/codes/qe/qe-6.6/bin/pw.x < nscf.in > nscf.out
#mpirun -np 64 /home/cchenye/codes/qe/qe-6.6/bin/pw.x < bans.in > bans.out;
#mpirun -np 64 /home/cchenye/codes/qe/qe-6.6/bin/bands.x < bands.in > bands.out;
mpirun -np 64 /home/cchenye/codes/qe/qe-6.6/bin/projwfc.x < proj.in > proj.out;
