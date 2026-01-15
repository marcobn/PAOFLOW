#!/bin/bash

PW_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/pw.x"
PP_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/projwfc.x"
qe_output_dir="output/qe/"
pao_output_dir="output/paoflow/"
mkdir -p "$qe_output_dir"
mkdir -p "$pao_output_dir"

"$PW_EXEC" <scf.in >"$qe_output_dir/scf.out"
"$PW_EXEC" <nscf.in >"$qe_output_dir/nscf.out"
"$PP_EXEC" <proj.in >"$qe_output_dir/proj.out"
mpirun -n 1 python main_conductor.py > "$pao_output_dir/conductor.out"
