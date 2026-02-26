#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] job.sh [options] example01 example02 ...
Example: QE_BIN=/home/anooja/Work/software/qe-7.4.1/bin ./job.sh --all example01 example02


Run PAOFLOW and/or QE jobs for the integration examples.

Options:
  --qe              Run QE only.
  --paoflow         Run PAOFLOW only.
  --all             Run both QE and PAOFLOW.
  --build-assets    Build qe_test_assets tar.gz after runs complete.
  --assets-out PATH Output tar.gz path (default: tests/integration/qe/_assets/qe_test_assets_dev.tar.gz).
  --examples LIST   Comma-separated list of example directories.
  -h, --help        Show this help message.

Default behavior: run all examples with PAOFLOW only.

Environment overrides:
  QE_BIN        Directory containing pw.x and projwfc.x.
  PW_EXEC       Full path to pw.x.
  PP_EXEC       Full path to projwfc.x.
  PYTHON_EXEC   Python interpreter to run main.py (default: python).
  ASSETS_OUT    Same as --assets-out.
EOF
}

resolve_root_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "$SLURM_SUBMIT_DIR/tests/integration/qe" ]]; then
      echo "$SLURM_SUBMIT_DIR/tests/integration/qe"
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

log_block_start() {
  local label="$1"
  printf '\n[%s] %s\n' "$(timestamp)" "$label" >> "$JOB_LOG"
}

# Collect "job folders" under an example directory.
# Includes:
#   - the example directory itself
#   - any subdirectory (at any depth) containing *.in and/or main.py
collect_job_dirs() {
  local example_dir="$1"
  local -a dirs=()

  dirs+=("$example_dir")

  # QE triggers: any dir containing *.in
  while IFS= read -r -d '' d; do
    dirs+=("$d")
  done < <(find "$example_dir" -type f -name "*.in" -printf '%h\0' | sort -zu)

  # PAOFLOW triggers: any dir containing main.py
  while IFS= read -r -d '' d; do
    dirs+=("$d")
  done < <(find "$example_dir" -type f -name "main.py" -printf '%h\0' | sort -zu)

  # Unique + stable order
  printf '%s\n' "${dirs[@]}" | awk '!seen[$0]++'
}

run_qe_dir() {
  local jobdir="$1"
  local label="$2"   # e.g. example01/subA

  local tmp_log
  tmp_log="${QE_LOG}.tmp"

  # Collect *.in in THIS jobdir only (not deeper) because jobdir is already discovered by recursion.
  mapfile -t inputs < <(find "$jobdir" -maxdepth 1 -type f -name "*.in" -printf "%f\n" | sort)
  if [[ ${#inputs[@]} -eq 0 ]]; then
    log_job "QE: $label - skipped (no .in files)"
    return 0
  fi

  log_block_start "QE: $label"
  : > "$tmp_log"

  # Run priority inputs first if present in this folder.
  for input in scf.in nscf.in proj.in; do
    if [[ -f "$jobdir/$input" ]]; then
      local cmd
      if [[ "$input" == *proj* ]]; then
        cmd="$PP_EXEC"
      else
        cmd="$PW_EXEC"
      fi
      if ! (cd "$jobdir" && "$cmd" < "$input") >> "$tmp_log" 2>&1; then
        log_job "QE: $label - FAILED on $input"
        printf '[%s] QE failed: %s (%s)\n' "$(timestamp)" "$label" "$input" >> "$QE_LOG"
        tail -n 200 "$tmp_log" >> "$QE_LOG"
        echo "--- QE error (last 40 lines) ---" | tee -a "$JOB_LOG"
        tail -n 40 "$tmp_log" | tee -a "$JOB_LOG"
        return 1
      fi
    fi
  done

  # Run the rest of the *.in files in this folder.
  for input in "${inputs[@]}"; do
    case "$input" in
      scf.in|nscf.in|proj.in)
        continue
        ;;
    esac

    local cmd
    if [[ "$input" == *proj* ]]; then
      cmd="$PP_EXEC"
    else
      cmd="$PW_EXEC"
    fi

    if ! (cd "$jobdir" && "$cmd" < "$input") >> "$tmp_log" 2>&1; then
      log_job "QE: $label - FAILED on $input"
      printf '[%s] QE failed: %s (%s)\n' "$(timestamp)" "$label" "$input" >> "$QE_LOG"
      tail -n 200 "$tmp_log" >> "$QE_LOG"
      echo "--- QE error (last 40 lines) ---" | tee -a "$JOB_LOG"
      tail -n 40 "$tmp_log" | tee -a "$JOB_LOG"
      return 1
    fi
  done

  # Cleanup in this jobdir (and its immediate *.save dirs within it)
  while IFS= read -r -d '' savedir; do
    find "$savedir" -type f ! -name "*.xml" ! -name "*.UPF" -delete
  done < <(find "$jobdir" -maxdepth 1 -type d -name "*.save" -print0)

  find "$jobdir" -maxdepth 1 -type f \( -name "*pdos_*" -o -name "*.wfc*" -o -name "*.xml" \) -delete

  log_job "QE: $label - OK"
  printf '[%s] QE ok: %s\n' "$(timestamp)" "$label" >> "$QE_LOG"
  rm -f "$tmp_log"
}

run_paoflow_dir() {
  local jobdir="$1"
  local label="$2"

  local main_py="$jobdir/main.py"
  local outputdir
  local output_path

  if [[ ! -f "$main_py" ]]; then
    log_job "PAOFLOW: $label - skipped (no main.py)"
    return 0
  fi

  log_block_start "PAOFLOW: $label"
  printf '[%s] PAOFLOW start: %s\n' "$(timestamp)" "$label" >> "$PAOFLOW_LOG"

  outputdir="$($PYTHON_EXEC - <<'PY' "$main_py"
import re
import sys
text = open(sys.argv[1], "r", encoding="utf-8").read()
m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
print(m.group(1) if m else "output")
PY
)"

  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then
    output_path="$outputdir"
  else
    output_path="$jobdir/$outputdir"
  fi

  if ! (cd "$jobdir" && "$PYTHON_EXEC" "main.py") >> "$PAOFLOW_LOG" 2>&1; then
    log_job "PAOFLOW: $label - FAILED"
    printf '[%s] PAOFLOW failed: %s\n' "$(timestamp)" "$label" >> "$PAOFLOW_LOG"
    echo "--- PAOFLOW error (last 40 lines) ---" | tee -a "$JOB_LOG"
    tail -n 40 "$PAOFLOW_LOG" | tee -a "$JOB_LOG"
    return 1
  fi

  # Move outputdir -> Reference inside the same jobdir
  if [[ -d "$output_path" ]]; then
    rm -rf "$jobdir/Reference"
    mv "$output_path" "$jobdir/Reference"
  else
    log_job "PAOFLOW: $label - WARNING (output directory missing at $output_path)"
  fi

  log_job "PAOFLOW: $label - OK"
  printf '[%s] PAOFLOW done: %s\n' "$(timestamp)" "$label" >> "$PAOFLOW_LOG"
}

cleanup_generated_dirs() {
  local root="$1"

  while IFS= read -r -d '' refdir; do
    rm -rf "$refdir"
  done < <(find "$root" -type d -name "Reference" -print0)

  while IFS= read -r -d '' savedir; do
    rm -rf "$savedir"
  done < <(find "$root" -type d -name "*.save" -print0)
}

# -------- argument parsing --------
run_qe=false
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
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --assets-out" >&2
        exit 2
      fi
      assets_out="$2"
      shift
      ;;
    --assets-out=*)
      assets_out="${1#*=}"
      ;;
    --examples)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --examples" >&2
        exit 2
      fi
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

# repo root is needed for `python -m tests.integration.qe.build_assets`
repo_root="$(cd "$root_dir/../../.." && pwd)"

JOB_LOG="$log_dir/job.log"
QE_LOG="$log_dir/qe.log"
PAOFLOW_LOG="$log_dir/paoflow.log"

: > "$JOB_LOG"
: > "$QE_LOG"
: > "$PAOFLOW_LOG"

log_job "Root directory: $root_dir"
log_job "Repo root: $repo_root"
log_job "Run QE: $run_qe"
log_job "Run PAOFLOW: $run_paoflow"
log_job "Build assets: $build_assets"

if [[ "$run_qe" = true ]]; then
  if ! PW_EXEC="$(resolve_exec PW_EXEC pw.x)"; then
    echo "QE executable pw.x not found (set PW_EXEC or QE_BIN)" | tee -a "$JOB_LOG"
    exit 2
  fi
  if ! PP_EXEC="$(resolve_exec PP_EXEC projwfc.x)"; then
    echo "QE executable projwfc.x not found (set PP_EXEC or QE_BIN)" | tee -a "$JOB_LOG"
    exit 2
  fi
fi

PYTHON_EXEC="${PYTHON_EXEC:-python}"

# Default output tarball path for assets.
if [[ -z "${assets_out}" ]]; then
  assets_out="${ASSETS_OUT:-$root_dir/_assets/qe_test_assets_dev.tar.gz}"
fi

declare -a examples
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a examples <<< "$examples_arg"
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  examples=("${extra_examples[@]}")
else
  mapfile -t examples < <(find "$root_dir" -maxdepth 1 -type d -name "example*" -printf "%f\n" | sort)
fi

if [[ ${#examples[@]} -eq 0 ]]; then
  echo "No examples found to run." | tee -a "$JOB_LOG"
  exit 1
fi

failures=0
failed_examples=()

for ex in "${examples[@]}"; do
  ex_failed=false
  exdir="$root_dir/$ex"
  if [[ ! -d "$exdir" ]]; then
    log_job "Example $ex not found under $root_dir"
    failures=$((failures + 1))
    ex_failed=true
    failed_examples+=("$ex")
    continue
  fi

  # Collect all sub-job directories under this example
  mapfile -t jobdirs < <(collect_job_dirs "$exdir")

  if [[ ${#jobdirs[@]} -eq 0 ]]; then
    log_job "Example $ex - skipped (no job folders found)"
    continue
  fi

  for jobdir in "${jobdirs[@]}"; do
    # Label in logs relative to root_dir, e.g. example01/subA
    label="${jobdir#"$root_dir"/}"

    qe_ok=true
    if [[ "$run_qe" = true ]]; then
      if ! run_qe_dir "$jobdir" "$label"; then
        qe_ok=false
        failures=$((failures + 1))
        ex_failed=true
      fi
    fi

    if [[ "$run_paoflow" = true && "$qe_ok" = true ]]; then
      if ! run_paoflow_dir "$jobdir" "$label"; then
        failures=$((failures + 1))
        ex_failed=true
      fi
    fi
  done

  if [[ "$ex_failed" = true ]]; then
    failed_examples+=("$ex")
  fi
done

if [[ "$build_assets" = true ]]; then
  if [[ $failures -gt 0 ]]; then
    if [[ ${#failed_examples[@]} -gt 0 ]]; then
      log_job "WARNING: $failures failure(s); asset bundle will be incomplete for: ${failed_examples[*]}"
    else
      log_job "WARNING: $failures failure(s); asset bundle may be incomplete."
    fi
  fi

  log_job "Building asset bundle: $assets_out"
  mkdir -p "$(dirname "$assets_out")"

  if ! (cd "$repo_root" && "$PYTHON_EXEC" -m tests.integration.qe.build_assets --qe-root "$root_dir" --out "$assets_out") >> "$JOB_LOG" 2>&1; then
    log_job "Asset bundle build FAILED"
    echo "--- asset build error (last 60 lines) ---" | tee -a "$JOB_LOG"
    tail -n 60 "$JOB_LOG" | tee -a "$JOB_LOG"
    exit 1
  fi

  log_job "Asset bundle build OK: $assets_out"

  log_job "Cleaning generated Reference and *.save directories under examples"
  cleanup_generated_dirs "$root_dir"
  log_job "Cleanup complete"
fi

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures failure(s)."
  exit 1
fi

log_job "Completed successfully."
