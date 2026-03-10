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
  local script_dir repo_root candidate

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    candidate="$SLURM_SUBMIT_DIR/tests/integration/TBmodel"
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  fi

  repo_root="$(cd "$script_dir/../../.." && pwd)"
  candidate="$repo_root/tests/integration/TBmodel"

  if [[ -d "$candidate" ]]; then
    echo "$candidate"
    return
  fi

  echo "ERROR: could not locate tests/integration/TBmodel." >&2
  echo "Checked:" >&2
  [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && echo "  $SLURM_SUBMIT_DIR/tests/integration/TBmodel" >&2
  echo "  $candidate" >&2
  exit 1
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

cleanup_output_dir() {
  local outdir="$1"

  if [[ -z "$outdir" || ! -d "$outdir" ]]; then
    return 0
  fi

  local resolved root_resolved
  resolved="$(realpath "$outdir")"
  root_resolved="$(realpath "$root_dir")"

  if [[ "$resolved" != "$root_resolved"* ]]; then
    log_job "Skipping output directory outside TBmodel root: $outdir"
    return 0
  fi

  rm -rf "$outdir"
  log_job "Removed output directory $outdir"
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

repo_root="$(cd "$root_dir/../../.." && pwd)"

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
successful_output_dirs=()

for name in "${examples[@]}"; do
  script_path="$root_dir/$name.py"
  if [[ ! -f "$script_path" ]]; then
    log_job "PAOFLOW: $name - skipped (missing $script_path)"
    failures=$((failures + 1))
    failed_examples+=("$name")
    continue
  fi

  if ! run_tbmodel_script "$script_path" "$name"; then
    failures=$((failures + 1))
    failed_examples+=("$name")
  else
    successful_output_dirs+=("$(infer_outputdir "$script_path")")
  fi
done

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures failure(s): ${failed_examples[*]}"
  if [[ "$build_assets" = true ]]; then
    log_job "Assets were not created and output directories were not removed because one or more examples failed."
  fi
  exit 1
fi

if [[ "$build_assets" = true ]]; then
  mkdir -p "$(dirname "$assets_out")"

  (cd "$repo_root" && "$PYTHON_EXEC" \
    "$repo_root/.github/assets_generation/TBmodel/build_assets.py" \
    --tbmodel-root "$root_dir" \
    --out "$assets_out")

  if [[ ! -f "$assets_out" ]]; then
    log_job "All examples ran successfully, but asset tar was not created at $assets_out"
    exit 1
  fi

  declare -A output_dirs_seen=()
  for outdir in "${successful_output_dirs[@]}"; do
    if [[ -n "$outdir" && -z "${output_dirs_seen[$outdir]:-}" ]]; then
      output_dirs_seen["$outdir"]=1
      cleanup_output_dir "$outdir"
    fi
  done

  log_job "All examples ran successfully and asset tar has been created at $assets_out"
else
  log_job "All examples ran successfully."
fi
