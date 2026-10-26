#!/bin/bash
# 3C-SiC reference for the long-range (Frohlich) coupling: scf 6^3 -> ph.x 3^3 (epsil, dvscf) -> pp.py-style save/
# -> nscf 3^3 (4 bands for EPW, 8 bands for PAOFLOW) -> epw.x epbwrite+epwwrite (lpolar, sp3 Wannier) ->
# epw.x epwread + prtgkk on a k/q list (dense 12^3 points).  QE 7.5 / EPW 6.0, PAOFLOW PBE pseudos.
set -euo pipefail
QE=/Users/marco/Local/Programs/qe-7.5/bin
PY=/Users/marco/anaconda3/envs/work/bin/python
NP=8
HERE=$(cd "$(dirname "$0")" && pwd)
A0=8.237; ECUT=40; NK=3; NQ=3
log() { echo "$(date '+%F %T')  $*"; }
pwin() {  # $1 calc, $2 nbnd ('' = none), $3 kpoints block
cat <<EOT
&CONTROL
  calculation = '$1', prefix = 'sic', pseudo_dir = '$HERE/', outdir = './', verbosity = 'high'
/
&SYSTEM
  ibrav = 2, celldm(1) = $A0, nat = 2, ntyp = 2, ecutwfc = $ECUT, ${2:+nbnd = $2,}
/
&ELECTRONS
  conv_thr = 1.0d-12, mixing_beta = 0.7, diago_full_acc = .true.
/
ATOMIC_SPECIES
  Si 28.0855 Si.upf
  C  12.0107 C.upf
ATOMIC_POSITIONS {alat}
  Si 0.00 0.00 0.00
  C  0.25 0.25 0.25
$3
EOT
}
mkdir -p phonon epw paoflow
cd phonon
log "scf"
pwin scf "" "K_POINTS {automatic}
  6 6 6 0 0 0" > scf.in
mpirun -np $NP $QE/pw.x -nk $NP -in scf.in > scf.out 2>&1
log "ph.x $NQ^3 (epsil)"
cat > ph.in <<EOT
SiC phonons
&INPUTPH
  prefix = 'sic', outdir = './', fildyn = 'sic.dyn', fildvscf = 'dvscf', ldisp = .true., epsil = .true.
  nq1 = $NQ, nq2 = $NQ, nq3 = $NQ, tr2_ph = 1.0d-16
/
EOT
mpirun -np $NP $QE/ph.x -nk $NP -in ph.in > ph.out 2>&1
mkdir -p save/sic.phsave
NQIRR=$(ls sic.dyn[0-9]* | /usr/bin/grep -v dyn0 | wc -l | tr -d ' ')
for i in $(seq 1 $NQIRR); do
  if [ $i -eq 1 ]; then cp _ph0/sic.dvscf1 save/sic.dvscf_q1; else cp _ph0/sic.q_$i/sic.dvscf1 save/sic.dvscf_q$i; fi
  cp sic.dyn$i save/sic.dyn_q$i
done
cp _ph0/sic.phsave/patterns.*.xml save/sic.phsave/
log "ph done: $NQIRR irreducible q"
KBLOCK=$($PY -c "from PAOFLOW.gen.epw_inputs import kpoints_card; print(kpoints_card(($NK,$NK,$NK)), end='')")
# nscf for PAOFLOW (8 bands)
cd ../paoflow && mkdir -p sic.save && cp ../phonon/sic.save/charge-density.dat ../phonon/sic.save/data-file-schema.xml sic.save/ && cp $HERE/Si.upf $HERE/C.upf sic.save/
log "nscf 8 bands (PAOFLOW)"
pwin nscf 8 "$KBLOCK" > nscf.in
mpirun -np $NP $QE/pw.x -nk $NP -in nscf.in > nscf.out 2>&1
# nscf for EPW (4 valence bands: isolated group, no disentanglement)
cd ../epw && mkdir -p sic.save && cp ../phonon/sic.save/charge-density.dat ../phonon/sic.save/data-file-schema.xml sic.save/ && cp $HERE/Si.upf $HERE/C.upf sic.save/
log "nscf 4 bands (EPW)"
pwin nscf 4 "$KBLOCK" > nscf.in
mpirun -np $NP $QE/pw.x -nk $NP -in nscf.in > nscf.out 2>&1
# fine-grid lists (crystal coordinates on the 12^3 grid): 2 k, 7 q along (1,1,0)
cat > kpts.txt <<EOT
2 crystal
0.0 0.0 0.0 1.0
0.083333333333 0.166666666667 0.25 1.0
EOT
cat > qpts.txt <<EOT
7 crystal
0.083333333333 0.083333333333 0.0 1.0
0.166666666667 0.166666666667 0.0 1.0
0.25 0.25 0.0 1.0
0.333333333333 0.333333333333 0.0 1.0
0.5 0.5 0.0 1.0
0.083333333333 0.0 0.0 1.0
0.25 0.166666666667 0.083333333333 1.0
EOT
epwin() {  # $1: stage (1 write, 2 read+prtgkk)
cat <<EOT
&inputepw
  prefix = 'sic', outdir = './', dvscf_dir = '../phonon/save', amass(1) = 28.0855, amass(2) = 12.0107
  elph = .true., lpolar = .true., lphase = .true., vme = 'dipole', etf_mem = 1
  nbndsub = 4, nk1 = $NK, nk2 = $NK, nk3 = $NK, nq1 = $NQ, nq2 = $NQ, nq3 = $NQ
  iverbosity = 0, fsthick = 100.0, temps = 300, degaussw = 0.1, elecselfen = .true.
EOT
if [ "$1" = 1 ]; then cat <<EOT
  epbwrite = .true., epbread = .false., epwwrite = .true., epwread = .false.
  wannierize = .true., num_iter = 500, iprint = 2, proj(1) = 'Si:sp3'
  wdata(1) = 'use_ws_distance = T'
  nkf1 = 6, nkf2 = 6, nkf3 = 6, nqf1 = 6, nqf2 = 6, nqf3 = 6
/
EOT
else cat <<EOT
  epbwrite = .false., epbread = .false., epwwrite = .false., epwread = .true.
  wannierize = .false., prtgkk = .true.
  filkf = './kpts.txt', filqf = './qpts.txt'
/
EOT
fi
}
epwin 1 > epw1.in; epwin 2 > epw2.in
log "epw.x epbwrite+epwwrite (Wannier sp3, lpolar)"
mpirun -np $NP $QE/epw.x -nk $NP -in epw1.in > epw1.out 2>&1 || { log "epw1 failed"; tail -20 epw1.out; exit 1; }
log "epw.x epwread + prtgkk"
mpirun -np 1 $QE/epw.x -nk 1 -in epw2.in > epw2.out 2>&1 || { log "epw2 failed"; tail -20 epw2.out; exit 1; }
/usr/bin/grep -c "" epw2.out
log "done"
