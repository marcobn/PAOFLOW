#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] job.sh [options] graphene graphene2 ...

Run TBmodel PAOFLOW jobs for selected examples.

Options:
  --build-assets    Build tbmodel_test_assets tar.gz after runs complete.
  --assets-out PATH Output tar.gz path (default: tests/integration/TBmodel/_assets/tbmodel_test_assets_dev.tar.gz).
  --examples LIST   Comma-separated list of example scripts (without .py extension).
  -h, --help        Show this help message.

Default behavior: run all TBmodel examples.

Environment overrides:
  PYTHON_EXEC   Python interpreter to run scripts (default: python).
  PARALLEL_EXEC Optional launcher command (e.g. 'mpirun -np 8' or 'srun -n 8').
  ASSETS_OUT    Same as --assets-out.
EOF
}

resolve_root_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -d "$SLURM_SUBMIT_DIR/tests/integration/TBmodel" ]]; then
      echo "$SLURM_SUBMIT_DIR/tests/integration/TBmodel"
      return
    fi
  fi

  echo "$script_dir"
}

log_job() {
  local msg="$1"
  echo "[$(timestamp)] $msg" | tee -a "$JOB_LOG"
}

infer_outputdir() {
  local script_path="$1"
  local script_dir
  script_dir="$(dirname "$script_path")"

  local outdir
  outdir="$($PYTHON_EXEC - <<'PY' "$script_path"
import os
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8").read()
m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
stem = os.path.splitext(os.path.basename(sys.argv[1]))[0]
print(m.group(1) if m else stem)
PY
)"

  outdir="${outdir%/}"
  if [[ "$outdir" = /* ]]; then
    echo "$outdir"
  else
    echo "$script_dir/$outdir"
  fi
}

run_tbmodel_script() {
  local script_path="$1"
  local name="$2"

  log_job "PAOFLOW: $name - start"

  if [[ ${#PARALLEL_EXEC_CMD[@]} -gt 0 ]]; then
    if ! (cd "$(dirname "$script_path")" && "${PARALLEL_EXEC_CMD[@]}" "$PYTHON_EXEC" "$(basename "$script_path")"); then
      log_job "PAOFLOW: $name - FAILED"
      return 1
    fi
  elif ! (cd "$(dirname "$script_path")" && "$PYTHON_EXEC" "$(basename "$script_path")"); then
    log_job "PAOFLOW: $name - FAILED"
    return 1
  fi

  log_job "PAOFLOW: $name - OK"
}

build_assets=false
assets_out=""
examples_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
export MPLBACKEND="${MPLBACKEND:-Agg}"

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

if [[ -z "$assets_out" ]]; then
  assets_out="${ASSETS_OUT:-$root_dir/_assets/tbmodel_test_assets_dev.tar.gz}"
fi

examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a examples <<< "$examples_arg"
fi
if [[ ${#extra_examples[@]} -gt 0 ]]; then
  examples+=("${extra_examples[@]}")
fi

if [[ ${#examples[@]} -eq 0 ]]; then
  while IFS= read -r -d '' script; do
    name="$(basename "$script" .py)"
    case "$name" in
      assets|build_assets|compare|conftest|jobs|runner)
        continue
        ;;
    esac
    examples+=("$name")
  done < <(
    find "$root_dir" -maxdepth 1 -type f -name "*.py" \
      ! -name "test_*" ! -name "_*" ! -name "__init__.py" -print0 | sort -z
  )
fi

if [[ ${#examples[@]} -eq 0 ]]; then
  echo "No TBmodel examples found under $root_dir" >&2
  exit 1
fi

failures=0
failed_examples=()

for name in "${examples[@]}"; do
  script_path="$root_dir/$name.py"
  if [[ ! -f "$script_path" ]]; then
    log_job "PAOFLOW: $name - skipped (missing $script_path)"
    continue
  fi

  if ! run_tbmodel_script "$script_path" "$name"; then
    failures=$((failures + 1))
    failed_examples+=("$name")
  fi
 done

if [[ "$build_assets" = true ]]; then
  "$PYTHON_EXEC" "$root_dir/build_assets.py" --tbmodel-root "$root_dir" --out "$assets_out"
  log_job "Assets written to $assets_out"
fi

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures failure(s): ${failed_examples[*]}"
  exit 1
fi

log_job "Completed successfully."
