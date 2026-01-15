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

python main_conductor.py conductor_bulk.yaml > "$pao_output_dir/conductor_bulk.out"
python main_conductor.py conductor_lcr.yaml > "$pao_output_dir/conductor_lcr.out"
python main_conductor.py conductor_lead_Al.yaml > "$pao_output_dir/conductor_lead_Al.out"

python main_current.py current.yaml > "$pao_output_dir/current.out"
