#!/bin/bash
export OMP_NUM_THREADS=8
PW=/home/anooja/Work/software/qe-7.4.1/bin/pw.x
PP=/home/anooja/Work/software/qe-7.4.1/bin/projwfc.x
cd "$(dirname "$0")"
$PW <scf.in  >output/qe/scf.out  2>&1 || { echo "SCF FAILED"; exit 1; }
$PW <nscf.in >output/qe/nscf.out 2>&1 || { echo "NSCF FAILED"; exit 1; }
$PP <proj.in >output/qe/proj.out 2>&1 || { echo "PROJ FAILED"; exit 1; }
echo "QE CHAIN DONE"
