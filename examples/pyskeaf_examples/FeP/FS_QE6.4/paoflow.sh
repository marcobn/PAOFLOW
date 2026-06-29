#!/bin/bash
#SBATCH -J fermi_surface_FeP
#SBATCH -o %j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c.chen.ye@rug.nl
#SBATCH -p fat
#SBATCH -n 100
#SBATCH -t 00:15:00

python main.py > paoflow.out
