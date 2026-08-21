#!/bin/bash

PW_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/pw.x"
PP_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/projwfc.x"

# This QE build is serial ("Serial version" in the output header). Do NOT wrap
# pw.x in mpirun: it launches N independent copies, only one of which receives
# stdin, and the rest die with "could not find namelist &control". Scale with
# OpenMP threads instead.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
qe_output_dir="output/qe/"
pao_output_dir="output/paoflow/"
mkdir -p "$qe_output_dir"
mkdir -p "$pao_output_dir"

# Regenerate scf.in / nscf.in from build_slab.py (NLAYERS lives there).
python build_slab.py

"$PW_EXEC" <scf.in >"$qe_output_dir/scf.out"
"$PW_EXEC" <nscf.in >"$qe_output_dir/nscf.out"
"$PP_EXEC" <proj.in >"$qe_output_dir/proj.out"

# Serial: site_projected_bands reads v_k, which bands_calc leaves scattered
# across MPI ranks.
mpirun -n 1 python main.py > "$pao_output_dir/slab_bands.out"
