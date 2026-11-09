#!/bin/bash
# QE steps of the electron-phonon workflows (run from this directory): NP=8 bash run_qe.sh
# The PAOFLOW_* variables (set by the paoflow-studio Runs tab) take precedence: the MPI
# prefix, the Quantum ESPRESSO bin folder and the pw.x / ph.x options (PAOFLOW.utils.qe_launch).
set -e
NP=${NP:-8}
MPI=${PAOFLOW_MPI_QE-mpirun -np $NP}
BIN=${PAOFLOW_QE_BIN:+$PAOFLOW_QE_BIN/}
PW=${PW:-${BIN}pw.x}; PH=${PH:-${BIN}ph.x}
OPT=${PAOFLOW_QE_OPTIONS:-}
log() { echo "$(date +%T) $*"; }

cd phonon
log scf;  $MPI $PW $OPT -in scf.in > scf.out
log ph.x; $MPI $PH $OPT -in ph.in > ph.out        # irreducible q, fildvscf -> _ph0/
cd ../nscf
mkdir -p Si2.save
cp ../phonon/Si2.save/charge-density.dat ../phonon/Si2.save/data-file-schema.xml Si2.save/
log nscf; $MPI $PW $OPT -in nscf.in > nscf.out
cd ../optics
mkdir -p Si2.save
cp ../phonon/Si2.save/charge-density.dat ../phonon/Si2.save/data-file-schema.xml Si2.save/
log optics nscf; $MPI $PW -nk $NP -in nscf.in > nscf.out   # one k point per pool
cd ..
log done
