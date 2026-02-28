#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] job.sh [options] example01 example02 ...

Run transport integration jobs for selected examples.

Options:
  --qe              Run QE only.
  --paoflow         Run PAOFLOW transport only.
  --all             Run both QE and PAOFLOW transport.
  --build-assets    Build transport_test_assets tar.gz after runs complete.
  --assets-out PATH Output tar.gz path (default: tests/integration/transport/_assets/transport_test_assets_dev.tar.gz).
  --examples LIST   Comma-separated list of example directories.
  -h, --help        Show this help message.

Default behavior: run all examples with QE + PAOFLOW transport (same as --all).

Environment overrides:
  QE_BIN        Directory containing pw.x and projwfc.x.
  PW_EXEC       Full path to pw.x.
  PP_EXEC       Full path to projwfc.x.
  PARALLEL_EXEC Optional launcher command (e.g. 'mpirun -np 8' or 'srun -n 8').
  PYTHON_EXEC   Python interpreter to run scripts (default: python).
  ASSETS_OUT    Same as --assets-out.
EOF
}

resolve_root_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "$SLURM_SUBMIT_DIR/tests/integration/transport" ]]; then
      echo "$SLURM_SUBMIT_DIR/tests/integration/transport"
      return
    fi
  fi

  echo "$script_dir"
}

resolve_exec() {
  local env_var="$1"
  local fallback_name="$2"
  local qe_bin="${QE_BIN:-}"

  if [[ -n "${!env_var:-}" ]]; then
    if [[ -x "${!env_var}" ]]; then
      echo "${!env_var}"
      return
    fi
    return 1
  fi

  if [[ -n "$qe_bin" && -x "$qe_bin/$fallback_name" ]]; then
    echo "$qe_bin/$fallback_name"
    return
  fi

  if command -v "$fallback_name" >/dev/null 2>&1; then
    command -v "$fallback_name"
    return
  fi

  return 1
}

log_job() {
  local msg="$1"
  echo "[$(timestamp)] $msg" | tee -a "$JOB_LOG"
}

run_qe_exec() {
  local exe="$1"
  local input_file="$2"

  if [[ ${#PARALLEL_EXEC_CMD[@]} -gt 0 ]]; then
    "${PARALLEL_EXEC_CMD[@]}" "$exe" < "$input_file"
  else
    "$exe" < "$input_file"
  fi
}

collect_job_dirs() {
  local example_dir="$1"
  local -a dirs=()

  dirs+=("$example_dir")

  while IFS= read -r -d '' d; do
    dirs+=("$d")
  done < <(find "$example_dir" -type f -name "*.in" -printf '%h\0' | sort -zu)

  while IFS= read -r -d '' d; do
    dirs+=("$d")
  done < <(
    find "$example_dir" -type f \( -name "main.py" -o -name "main_conductor.py" -o -name "main_current.py" \) -printf '%h\0' | sort -zu
  )

  printf '%s\n' "${dirs[@]}" | awk '!seen[$0]++'
}

run_qe_dir() {
  local jobdir="$1"
  local label="$2"

  mapfile -t inputs < <(find "$jobdir" -maxdepth 1 -type f -name "*.in" -printf "%f\n" | sort)
  if [[ ${#inputs[@]} -eq 0 ]]; then
    log_job "QE: $label - skipped (no .in files)"
    return 0
  fi

  for input in scf.in nscf.in proj.in; do
    if [[ -f "$jobdir/$input" ]]; then
      local cmd
      local out
      if [[ "$input" == *proj* ]]; then
        cmd="$PP_EXEC"
      else
        cmd="$PW_EXEC"
      fi
      out="$jobdir/${input%.in}.out"
      (cd "$jobdir" && run_qe_exec "$cmd" "$input" > "$out")
    fi
  done

  while IFS= read -r -d '' savedir; do
    find "$savedir" -type f ! -name "*.xml" ! -name "*.UPF" -delete
  done < <(find "$jobdir" -maxdepth 1 -type d -name "*.save" -print0)

  find "$jobdir" -maxdepth 1 -type f \( -name "*pdos_*" -o -name "*.wfc*" -o -name "*.xml" \) -delete

  log_job "QE: $label - OK"
}

run_transport_dir() {
  local jobdir="$1"
  local label="$2"

  local had_script=false
  local script

  if [[ -f "$jobdir/main.py" ]]; then
    had_script=true
    (cd "$jobdir" && "$PYTHON_EXEC" main.py)
  fi

  if [[ -f "$jobdir/main_conductor.py" ]]; then
    had_script=true
    if [[ -f "$jobdir/conductor.yaml" ]]; then
      (cd "$jobdir" && "$PYTHON_EXEC" main_conductor.py)
    else
      while IFS= read -r -d '' script; do
        (cd "$jobdir" && "$PYTHON_EXEC" main_conductor.py "$(basename "$script")")
      done < <(find "$jobdir" -maxdepth 1 -type f -name "conductor*.yaml" -print0 | sort -z)
    fi
  fi

  if [[ -f "$jobdir/main_current.py" ]]; then
    had_script=true
    if [[ -f "$jobdir/current.yaml" ]]; then
      (cd "$jobdir" && "$PYTHON_EXEC" main_current.py)
    else
      while IFS= read -r -d '' script; do
        (cd "$jobdir" && "$PYTHON_EXEC" main_current.py "$(basename "$script")")
      done < <(find "$jobdir" -maxdepth 1 -type f -name "current*.yaml" -print0 | sort -z)
    fi
  fi

  if [[ "$had_script" == false ]]; then
    log_job "PAOFLOW: $label - skipped (no transport entrypoints)"
    return 0
  fi

  local outdir="$jobdir/output/paoflow"
  if [[ -d "$outdir" ]]; then
    rm -rf "$jobdir/Reference"
    mkdir -p "$jobdir/Reference"
    find "$outdir" -maxdepth 1 -type f \( -name "*.dat" -o -name "projectability.txt" \) -exec cp {} "$jobdir/Reference/" \;
  fi

  log_job "PAOFLOW: $label - OK"
}

run_qe=true
run_paoflow=true
mode_set=false
examples_arg=""
extra_examples=()
build_assets=false
assets_out=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qe)
      if [[ "$mode_set" = false ]]; then
        run_qe=true
        run_paoflow=false
        mode_set=true
      else
        run_qe=true
      fi
      ;;
    --paoflow)
      if [[ "$mode_set" = false ]]; then
        run_qe=false
        run_paoflow=true
        mode_set=true
      else
        run_paoflow=true
      fi
      ;;
    --all)
      run_qe=true
      run_paoflow=true
      mode_set=true
      ;;
    --build-assets)
      build_assets=true
      ;;
    --assets-out)
      assets_out="$2"
      shift
      ;;
    --assets-out=*)
      assets_out="${1#*=}"
      ;;
    --examples)
      examples_arg="$2"
      shift
      ;;
    --examples=*)
      examples_arg="${1#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_examples+=("$1")
      ;;
  esac
  shift
done

root_dir="$(resolve_root_dir)"
log_dir="$root_dir/logs"
mkdir -p "$log_dir"

JOB_LOG="$log_dir/job.$(date +%Y%m%d_%H%M%S).log"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
if [[ "$run_qe" = true ]]; then
  PW_EXEC="${PW_EXEC:-$(resolve_exec PW_EXEC pw.x || true)}"
  PP_EXEC="${PP_EXEC:-$(resolve_exec PP_EXEC projwfc.x || true)}"
  if [[ -z "${PW_EXEC:-}" || -z "${PP_EXEC:-}" ]]; then
    echo "QE requested but pw.x/projwfc.x are not configured." >&2
    exit 2
  fi

  PARALLEL_EXEC="${PARALLEL_EXEC:-}"
  PARALLEL_EXEC_CMD=()
  if [[ -n "$PARALLEL_EXEC" ]]; then
    read -r -a PARALLEL_EXEC_CMD <<< "$PARALLEL_EXEC"
    if [[ ${#PARALLEL_EXEC_CMD[@]} -eq 0 ]]; then
      echo "PARALLEL_EXEC was set but is empty after parsing." >&2
      exit 2
    fi
    if ! command -v "${PARALLEL_EXEC_CMD[0]}" >/dev/null 2>&1; then
      echo "PARALLEL_EXEC command not found: ${PARALLEL_EXEC_CMD[0]}" >&2
      exit 2
    fi
  fi
fi

declare -a examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a examples <<< "$examples_arg"
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  examples=("${extra_examples[@]}")
else
  while IFS= read -r d; do
    examples+=("$(basename "$d")")
  done < <(find "$root_dir" -maxdepth 1 -type d -name "example*" | sort)
fi

for ex in "${examples[@]}"; do
  exdir="$root_dir/$ex"
  if [[ ! -d "$exdir" ]]; then
    log_job "SKIP: $ex (missing directory)"
    continue
  fi

  while IFS= read -r jobdir; do
    rel="${jobdir#$root_dir/}"
    if [[ "$run_qe" = true ]]; then
      run_qe_dir "$jobdir" "$rel"
    fi
    if [[ "$run_paoflow" = true ]]; then
      run_transport_dir "$jobdir" "$rel"
    fi
  done < <(collect_job_dirs "$exdir")
done

if [[ "$build_assets" = true ]]; then
  out="${assets_out:-${ASSETS_OUT:-$root_dir/_assets/transport_test_assets_dev.tar.gz}}"
  mkdir -p "$(dirname "$out")"
  "$PYTHON_EXEC" -m tests.integration.transport.build_assets --transport-root "$root_dir" --out "$out"
  log_job "Built assets: $out"
fi

log_job "Done."
