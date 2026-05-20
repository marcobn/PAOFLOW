#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] job.sh [options] [example names...]

Options:
  --paoflow-examples     Run PAOFLOW from examples/TBmodel_examples and write Reference folders.
  --paoflow-test         Run PAOFLOW from tests/integration/TBmodel and create paoflow_assets.tar.gz.
  --all                  Run --paoflow-test.
  --paoflow-assets-out PATH
                         Output path for paoflow assets tar.gz.
  --examples LIST        Comma-separated list of example names.
  -h, --help             Show this help message.
EOF
}

log_job() {
  echo "[$(timestamp)] $1" | tee -a "$JOB_LOG"
}

resolve_examples_root() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  repo_root="$(cd "$script_dir/../../.." && pwd)"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/examples/TBmodel_examples" ]]; then
    echo "$SLURM_SUBMIT_DIR/examples/TBmodel_examples"
    return
  fi
  echo "$repo_root/examples/TBmodel_examples"
}

resolve_tests_root() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  repo_root="$(cd "$script_dir/../../.." && pwd)"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/tests/integration/TBmodel" ]]; then
    echo "$SLURM_SUBMIT_DIR/tests/integration/TBmodel"
    return
  fi
  echo "$repo_root/tests/integration/TBmodel"
}

infer_outputdir_from_script() {
  local script_path="$1"
  "$PYTHON_EXEC" - <<'PY' "$script_path"
import os, re, sys
text = open(sys.argv[1], "r", encoding="utf-8").read()
m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
if m:
    print(m.group(1))
else:
    print(os.path.splitext(os.path.basename(sys.argv[1]))[0])
PY
}

run_python_script() {
  local script_path="$1" logfile="$2"
  if [[ ${#PARALLEL_EXEC_CMD[@]} -gt 0 ]]; then
    (cd "$(dirname "$script_path")" && "${PARALLEL_EXEC_CMD[@]}" "$PYTHON_EXEC" "$(basename "$script_path")") >> "$logfile" 2>&1
  else
    (cd "$(dirname "$script_path")" && "$PYTHON_EXEC" "$(basename "$script_path")") >> "$logfile" 2>&1
  fi
}

collect_example_script() {
  local examples_root="$1" ex_name="$2"
  if [[ -f "$examples_root/$ex_name/$ex_name.py" ]]; then
    echo "$examples_root/$ex_name/$ex_name.py"
  elif [[ -f "$examples_root/$ex_name.py" ]]; then
    echo "$examples_root/$ex_name.py"
  else
    find "$examples_root" -type f -name "$ex_name.py" | head -n 1
  fi
}

collect_test_script() {
  local tests_root="$1" ex_name="$2"
  if [[ -f "$tests_root/$ex_name.py" ]]; then
    echo "$tests_root/$ex_name.py"
  fi
}

run_paoflow_example_script() {
  local script_path="$1" label="$2"
  run_python_script "$script_path" "$PAOFLOW_EXAMPLES_LOG"
  local outputdir outputpath
  outputdir="$(infer_outputdir_from_script "$script_path")"
  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then outputpath="$outputdir"; else outputpath="$(dirname "$script_path")/$outputdir"; fi
  if [[ -d "$outputpath" ]]; then
    rm -rf "$(dirname "$script_path")/Reference"
    mv "$outputpath" "$(dirname "$script_path")/Reference"
  fi
  log_job "PAOFLOW(examples): $label - OK"
}

run_paoflow_test_script() {
  local script_path="$1" label="$2" staging_dir="$3"
  run_python_script "$script_path" "$PAOFLOW_TEST_LOG"
  local outputdir outputpath
  outputdir="$(infer_outputdir_from_script "$script_path")"
  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then outputpath="$outputdir"; else outputpath="$(dirname "$script_path")/$outputdir"; fi
  if [[ -d "$outputpath" ]]; then
    mkdir -p "$staging_dir/$label/Reference"
    cp -a "$outputpath/." "$staging_dir/$label/Reference/"
    rm -rf "$outputpath"
  fi
  log_job "PAOFLOW(test): $label - OK"
}

build_paoflow_assets_tar() {
  local staging_dir="$1"
  mkdir -p "$(dirname "$PAOFLOW_ASSETS_OUT")"
  tar -C "$staging_dir" -czf "$PAOFLOW_ASSETS_OUT" .
  log_job "Built PAOFLOW assets: $PAOFLOW_ASSETS_OUT"
}

run_paoflow_examples=false
run_paoflow_test=false
examples_arg=""
paoflow_assets_out_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paoflow-examples) run_paoflow_examples=true ;;
    --paoflow-test) run_paoflow_test=true ;;
    --all) run_paoflow_test=true ;;
    --paoflow-assets-out) paoflow_assets_out_arg="$2"; shift ;;
    --paoflow-assets-out=*) paoflow_assets_out_arg="${1#*=}" ;;
    --examples) examples_arg="$2"; shift ;;
    --examples=*) examples_arg="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) extra_examples+=("$1") ;;
  esac
  shift
done

if [[ "$run_paoflow_examples" = false && "$run_paoflow_test" = false ]]; then
  run_paoflow_examples=true
  run_paoflow_test=true
fi

EXAMPLES_ROOT="$(resolve_examples_root)"
TESTS_ROOT="$(resolve_tests_root)"
LOG_DIR="$TESTS_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_LOG="$LOG_DIR/job.log"
PAOFLOW_EXAMPLES_LOG="$LOG_DIR/paoflow_examples.log"
PAOFLOW_TEST_LOG="$LOG_DIR/paoflow_test.log"
: > "$JOB_LOG"
: > "$PAOFLOW_EXAMPLES_LOG"
: > "$PAOFLOW_TEST_LOG"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PARALLEL_EXEC="${PARALLEL_EXEC:-}"
if [[ -z "$PARALLEL_EXEC" && -n "${SLURM_NTASKS:-}" ]] && command -v srun >/dev/null 2>&1; then
  PARALLEL_EXEC="srun -n ${SLURM_NTASKS}"
fi
PARALLEL_EXEC_CMD=()
[[ -n "$PARALLEL_EXEC" ]] && read -r -a PARALLEL_EXEC_CMD <<< "$PARALLEL_EXEC"

PAOFLOW_ASSETS_OUT="${paoflow_assets_out_arg:-${PAOFLOW_ASSETS_OUT:-$TESTS_ROOT/_assets/paoflow_assets.tar.gz}}"

declare -a examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a examples <<< "$examples_arg"
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  examples=("${extra_examples[@]}")
else
  mapfile -t examples < <(find "$TESTS_ROOT" -maxdepth 1 -type f -name '*.py' ! -name 'test_*' ! -name '__init__.py' ! -name 'assets.py' ! -name 'compare.py' ! -name 'conftest.py' ! -name 'jobs.py' ! -name 'runner.py' -printf '%f\n' | sed 's/\.py$//' | sort)
fi

failures=0

for ex in "${examples[@]}"; do
  if [[ "$run_paoflow_examples" = true ]]; then
    ex_script="$(collect_example_script "$EXAMPLES_ROOT" "$ex")"
    if [[ -z "$ex_script" || ! -f "$ex_script" ]]; then
      log_job "PAOFLOW(examples): $ex - missing script"
      failures=$((failures + 1))
    else
      run_paoflow_example_script "$ex_script" "$ex" || failures=$((failures + 1))
    fi
  fi

  if [[ "$run_paoflow_test" = true ]]; then
    test_script="$(collect_test_script "$TESTS_ROOT" "$ex")"
    if [[ -z "$test_script" || ! -f "$test_script" ]]; then
      log_job "PAOFLOW(test): $ex - missing script"
      failures=$((failures + 1))
      continue
    fi
    if [[ -z "${PAOFLOW_TEST_STAGING_DIR:-}" ]]; then
      PAOFLOW_TEST_STAGING_DIR="$(mktemp -d)"
      trap 'rm -rf "$PAOFLOW_TEST_STAGING_DIR"' EXIT
    fi
    run_paoflow_test_script "$test_script" "$ex" "$PAOFLOW_TEST_STAGING_DIR" || failures=$((failures + 1))
  fi
done

if [[ "$run_paoflow_test" = true && $failures -eq 0 && -n "${PAOFLOW_TEST_STAGING_DIR:-}" ]]; then
  build_paoflow_assets_tar "$PAOFLOW_TEST_STAGING_DIR"
fi

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures failure(s)."
  exit 1
fi

log_job "Completed successfully."
