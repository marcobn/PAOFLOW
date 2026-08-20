#!/bin/bash
#
# Bi2Se3(0001) surface spectrum: QE -> PAOFLOW -> NEGF surface Green's function.
#
# The nscf step is the expensive one (15 atoms, spin-orbit, 300 bands, 108
# k-points with symmetry switched off). Everything after it is cheap.

set -euo pipefail

PW_EXEC="${PW_EXEC:-/home/anooja/Work/software/qe-7.4.1/bin/pw.x}"
PP_EXEC="${PP_EXEC:-/home/anooja/Work/software/qe-7.4.1/bin/projwfc.x}"
NPROC="${NPROC:-1}"

qe_output_dir="output/qe"
pao_output_dir="output/paoflow"
mkdir -p "$qe_output_dir" "$pao_output_dir"

# Regenerate scf.in / nscf.in from the structural parameters. Skip if you have
# edited the inputs by hand.
python build_inputs.py

mpirun -n "$NPROC" "$PW_EXEC" <scf.in  >"$qe_output_dir/scf.out"
mpirun -n "$NPROC" "$PW_EXEC" <nscf.in >"$qe_output_dir/nscf.out"
mpirun -n "$NPROC" "$PP_EXEC" <proj.in >"$qe_output_dir/proj.out"

# Sanity gate: bulk PAO bands vs the QE eigenvalues. Cheap, and it catches a bad
# projection before the NEGF sweep runs.
python check_pao_bands.py >"$pao_output_dir/pao_bands.out"

# The NEGF sweep parallelises over the energy grid.
mpirun -n "$NPROC" python main.py >"$pao_output_dir/surface_bands.out"

python plot_surface_bands.py
