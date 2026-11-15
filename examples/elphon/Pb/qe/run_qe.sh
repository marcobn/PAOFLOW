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
mkdir -p pb.save
cp ../phonon/pb.save/charge-density.dat ../phonon/pb.save/data-file-schema.xml pb.save/
log nscf; mpirun -np $NP $PW -in nscf.in > nscf.out
cd ../optics
mkdir -p pb.save
cp ../phonon/pb.save/charge-density.dat ../phonon/pb.save/data-file-schema.xml pb.save/
log optics nscf; mpirun -np $NP $PW -nk $NP -in nscf.in > nscf.out
cd ..
log done
