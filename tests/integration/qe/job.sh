#!/usr/bin/env bash
set -euo pipefail

PW_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/pw.x"
PP_EXEC="/home/anooja/Work/software/qe-7.4.1/bin/projwfc.x"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_inputs() {
  local exdir="$1"
  local outdir="$exdir/output/qe"
  local outdir_rel="output/qe"
  local input
  local cmd

  mkdir -p "$outdir"

  mapfile -t inputs < <(find "$exdir" -maxdepth 1 -type f -name "*.in" -printf "%f\n" | sort)
  if [[ ${#inputs[@]} -eq 0 ]]; then
    return
  fi

  # Run scf/nscf/proj first when present for consistent workflows.
  for input in scf.in nscf.in proj.in; do
    if [[ -f "$exdir/$input" ]]; then
      if [[ "$input" == *proj* ]]; then
        cmd="$PP_EXEC"
      else
        cmd="$PW_EXEC"
      fi
      (cd "$exdir" && "$cmd" < "$input" > "$outdir_rel/${input%.in}.out")
    fi
  done

  for input in "${inputs[@]}"; do
    case "$input" in
      scf.in|nscf.in|proj.in)
        continue
        ;;
    esac
    if [[ "$input" == *proj* ]]; then
      cmd="$PP_EXEC"
    else
      cmd="$PW_EXEC"
    fi
    (cd "$exdir" && "$cmd" < "$input" > "$outdir_rel/${input%.in}.out")
  done

  # Keep only XML and UPF files in each .save directory.
  while IFS= read -r -d '' savedir; do
    find "$savedir" -type f ! -name "*.xml" ! -name "*.UPF" -delete
  done < <(find "$exdir" -maxdepth 1 -type d -name "*.save" -print0)
}

for exdir in "$root_dir"/example*/; do
  if [[ -d "$exdir" ]]; then
    run_inputs "$exdir"
  fi
done
