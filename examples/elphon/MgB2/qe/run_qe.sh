#!/bin/bash
# QE steps of the electron-phonon workflows (run from this directory): NP=8 bash run_qe.sh
set -e
NP=${NP:-8}
PW=${PW:-pw.x}; PH=${PH:-ph.x}
log() { echo "$(date +%T) $*"; }

cd phonon
log scf;  mpirun -np $NP $PW -in scf.in > scf.out
log ph.x; mpirun -np $NP $PH -in ph.in > ph.out        # irreducible q, fildvscf -> _ph0/
cd ../nscf
mkdir -p mgb2.save
cp ../phonon/mgb2.save/charge-density.dat ../phonon/mgb2.save/data-file-schema.xml mgb2.save/
log nscf; mpirun -np $NP $PW -in nscf.in > nscf.out
cd ..
log done
