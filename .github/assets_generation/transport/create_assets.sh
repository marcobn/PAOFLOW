#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] create_assets.sh [options] [example01 example02 ...]

Options:
  --qe                   Run QE on examples and create/update *.save folders.
  --paoflow-examples     Run PAOFLOW from examples and write Reference folders.
  --paoflow-test         Run PAOFLOW from tests and stage Reference folders.
  --all                  Run --qe + --paoflow-examples + --paoflow-test.
                         If no mode option is passed, the script runs all modes.
  --skip-qe-if-save-exists
                         Skip QE in a job directory when '*.save' already exists.
                         Default: disabled.
  --paoflow-test-staging-dir PATH
                         Directory where PAOFLOW test outputs are staged.
                         Default: PAOFLOW_TEST_STAGING_DIR or tests/integration/transport/_assets/staging.
  --examples LIST        Comma-separated list of example selectors.
                         Default: all example* directories under EXAMPLES_ROOT.
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
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/examples/transport_examples" ]]; then
    echo "$SLURM_SUBMIT_DIR/examples/transport_examples"
    return
  fi
  echo "$repo_root/examples/transport_examples"
}

resolve_tests_root() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  repo_root="$(cd "$script_dir/../../.." && pwd)"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/tests/integration/transport" ]]; then
    echo "$SLURM_SUBMIT_DIR/tests/integration/transport"
    return
  fi
  echo "$repo_root/tests/integration/transport"
}

resolve_exec() {
  local env_var="$1"
  local fallback_name="$2"
  local qe_bin="${QE_BIN:-}"
  if [[ -n "${!env_var:-}" && -x "${!env_var}" ]]; then
    echo "${!env_var}"
    return
  fi
  if [[ -n "$qe_bin" && -x "$qe_bin/$fallback_name" ]]; then
    echo "$qe_bin/$fallback_name"
    return
  fi
  command -v "$fallback_name"
}

normalize_example_selector() {
  local selector="$1"
  selector="${selector%/}"
  selector="${selector#./}"

  if [[ "$selector" == "$EXAMPLES_ROOT/"* ]]; then
    selector="${selector#"$EXAMPLES_ROOT"/}"
  fi

  if [[ "$selector" == examples/transport_examples/* ]]; then
    selector="${selector#examples/transport_examples/}"
  fi

  printf '%s\n' "${selector%%/*}"
}

expand_examples_from_selectors() {
  local -a selectors=("$@")
  local selector normalized match

  for selector in "${selectors[@]}"; do
    [[ -n "$selector" ]] || continue

    if [[ "$selector" == *['*''?''[']* ]]; then
      shopt -s nullglob
      local -a matches=("$EXAMPLES_ROOT"/$selector)
      shopt -u nullglob

      if [[ ${#matches[@]} -eq 0 ]]; then
        log_job "WARN: selector '$selector' matched no examples under $EXAMPLES_ROOT"
        continue
      fi

      for match in "${matches[@]}"; do
        normalized="$(normalize_example_selector "$match")"
        if [[ -d "$EXAMPLES_ROOT/$normalized" ]]; then
          printf '%s\n' "$normalized"
        fi
      done
      continue
    fi

    normalized="$(normalize_example_selector "$selector")"
    if [[ -d "$EXAMPLES_ROOT/$normalized" ]]; then
      printf '%s\n' "$normalized"
    else
      log_job "WARN: selector '$selector' does not resolve to an example under $EXAMPLES_ROOT"
    fi
  done | awk '!seen[$0]++'
}

run_qe_exec() {
  local exe="$1"
  local input_file="$2"
  if [[ ${#PARALLEL_EXEC_CMD[@]} -gt 0 ]]; then
    "${PARALLEL_EXEC_CMD[@]}" "$exe" -in "$input_file"
  else
    "$exe" -in "$input_file"
  fi
}

infer_qe_outdir_from_input() {
  local input_file="$1"
  "$PYTHON_EXEC" - <<'PY' "$input_file"
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"\boutdir\s*=\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
if match:
    print(match.group(1).strip())
PY
}

ensure_qe_outdir() {
  local jobdir="$1"
  local input_file="$2"
  local outdir

  outdir="$(infer_qe_outdir_from_input "$jobdir/$input_file")"
  [[ -n "$outdir" ]] || return 0

  if [[ "$outdir" = /* ]]; then
    mkdir -p "$outdir"
  else
    mkdir -p "$jobdir/$outdir"
  fi
}

collect_example_jobs() {
  local base_dir="$1"
  local -a dirs=()
  if find "$base_dir" -maxdepth 1 -type f \( -name '*.in' -o -name 'main.py' -o -name 'main_conductor.py' -o -name 'main_current.py' \) | grep -q .; then
    dirs+=("$base_dir")
  fi
  while IFS= read -r -d '' d; do dirs+=("$d"); done < <(find "$base_dir" -type f -name '*.in' -printf '%h\0' | sort -zu)
  while IFS= read -r -d '' d; do dirs+=("$d"); done < <(find "$base_dir" -type f \( -name 'main.py' -o -name 'main_conductor.py' -o -name 'main_current.py' \) -printf '%h\0' | sort -zu)
  printf '%s\n' "${dirs[@]}" | awk '!seen[$0]++'
}

collect_test_jobs() {
  local base_dir="$1"
  find "$base_dir" -type f \( -name 'main.py' -o -name 'main_conductor.py' -o -name 'main_current.py' \) -printf '%h\n' | sort -u
}

infer_transport_outputdir() {
  local jobdir="$1"
  "$PYTHON_EXEC" - <<'PY' "$jobdir"
import re
import sys
from pathlib import Path

jobdir = Path(sys.argv[1])
for name in ("main.py", "main_conductor.py", "main_current.py"):
    path = jobdir / name
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    match = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
    if match:
        print(match.group(1))
        raise SystemExit(0)
print("output/paoflow")
PY
}

run_transport_scripts() {
  local jobdir="$1"
  local logfile="$2"
  local ran=false

  if [[ -f "$jobdir/main.py" ]]; then
    ran=true
    (cd "$jobdir" && "$PYTHON_EXEC" main.py) >> "$logfile" 2>&1
  fi

  if [[ -f "$jobdir/main_conductor.py" ]]; then
    ran=true
    if [[ -f "$jobdir/conductor.yaml" ]]; then
      (cd "$jobdir" && "$PYTHON_EXEC" main_conductor.py) >> "$logfile" 2>&1
    else
      while IFS= read -r -d '' yaml_file; do
        (cd "$jobdir" && "$PYTHON_EXEC" main_conductor.py "$(basename "$yaml_file")") >> "$logfile" 2>&1
      done < <(find "$jobdir" -maxdepth 1 -type f -name 'conductor*.yaml' -print0 | sort -z)
    fi
  fi

  if [[ -f "$jobdir/main_current.py" ]]; then
    ran=true
    if [[ -f "$jobdir/current.yaml" ]]; then
      (cd "$jobdir" && "$PYTHON_EXEC" main_current.py) >> "$logfile" 2>&1
    else
      while IFS= read -r -d '' yaml_file; do
        (cd "$jobdir" && "$PYTHON_EXEC" main_current.py "$(basename "$yaml_file")") >> "$logfile" 2>&1
      done < <(find "$jobdir" -maxdepth 1 -type f -name 'current*.yaml' -print0 | sort -z)
    fi
  fi

  [[ "$ran" = true ]]
}

run_qe_dir() {
  local jobdir="$1"
  local label="$2"

  if [[ "$skip_qe_if_save_exists" = true ]] && find "$jobdir" -type d -name '*.save' | grep -q .; then
    log_job "QE: $label - skipped (existing .save detected)"
    return 0
  fi

  mapfile -t inputs < <(find "$jobdir" -maxdepth 1 -type f -name '*.in' -printf '%f\n' | sort)
  if [[ ${#inputs[@]} -eq 0 ]]; then
    log_job "QE: $label - skipped"
    return 0
  fi

  for input in scf.in nscf.in proj.in; do
    if [[ -f "$jobdir/$input" ]]; then
      local cmd="$PW_EXEC"
      [[ "$input" == *proj* ]] && cmd="$PP_EXEC"
      ensure_qe_outdir "$jobdir" "$input"
      if ! (cd "$jobdir" && run_qe_exec "$cmd" "$input") >> "$QE_LOG" 2>&1; then
        log_job "QE: $label - FAILED on $input"
        return 1
      fi
    fi
  done

  while IFS= read -r -d '' savedir; do
    find "$savedir" -type f ! -name '*.xml' ! -name '*.UPF' ! -iname '*wfc*' -delete
  done < <(find "$jobdir" -type d -name '*.save' -print0)

  find "$jobdir" -type f -name '.hub*' -delete
  log_job "QE: $label - OK"
}

run_paoflow_example_dir() {
  local jobdir="$1"
  local label="$2"

  if ! run_transport_scripts "$jobdir" "$PAOFLOW_EXAMPLES_LOG"; then
    log_job "PAOFLOW(examples): $label - skipped"
    return 0
  fi

  local outputdir outpath
  outputdir="$(infer_transport_outputdir "$jobdir")"
  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then
    outpath="$outputdir"
  else
    outpath="$jobdir/$outputdir"
  fi

  if [[ ! -d "$outpath" ]]; then
    log_job "PAOFLOW(examples): $label - FAILED (missing output dir $outpath)"
    return 1
  fi

  rm -rf "$jobdir/Reference"
  mkdir -p "$jobdir/Reference"
  cp -a "$outpath/." "$jobdir/Reference/"
  log_job "PAOFLOW(examples): $label - OK"
}

overlay_qe_save_dirs() {
  local source_jobdir="$1"
  local test_jobdir="$2"
  local copied_list_file="$3"

  [[ -d "$source_jobdir" ]] || return 0

  while IFS= read -r -d '' savedir; do
    local dst="$test_jobdir/$(basename "$savedir")"
    [[ -e "$dst" ]] && continue
    cp -a "$savedir" "$dst"
    printf '%s\n' "$dst" >> "$copied_list_file"
  done < <(find "$source_jobdir" -type d -name '*.save' -print0 | sort -z)
}

run_paoflow_test_dir() {
  local jobdir="$1"
  local label="$2"
  local source_jobdir="$3"
  local staging_dir="$4"
  local copied_list outputdir outpath

  copied_list="$(mktemp)"
  overlay_qe_save_dirs "$source_jobdir" "$jobdir" "$copied_list"

  if ! run_transport_scripts "$jobdir" "$PAOFLOW_TEST_LOG"; then
    while IFS= read -r copied_path; do
      [[ -n "$copied_path" ]] && rm -rf "$copied_path"
    done < "$copied_list"
    rm -f "$copied_list"
    log_job "PAOFLOW(test): $label - skipped"
    return 0
  fi

  outputdir="$(infer_transport_outputdir "$jobdir")"
  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then
    outpath="$outputdir"
  else
    outpath="$jobdir/$outputdir"
  fi

  if [[ ! -d "$outpath" ]]; then
    while IFS= read -r copied_path; do
      [[ -n "$copied_path" ]] && rm -rf "$copied_path"
    done < "$copied_list"
    rm -f "$copied_list"
    log_job "PAOFLOW(test): $label - FAILED (missing output dir $outpath)"
    return 1
  fi

  mkdir -p "$staging_dir/$label/Reference"
  rm -rf "$staging_dir/$label/Reference"
  mkdir -p "$staging_dir/$label/Reference"
  cp -a "$outpath/." "$staging_dir/$label/Reference/"
  rm -rf "$outpath"

  while IFS= read -r copied_path; do
    [[ -n "$copied_path" ]] && rm -rf "$copied_path"
  done < "$copied_list"
  rm -f "$copied_list"
  log_job "PAOFLOW(test): $label - OK"
}

run_qe=false
run_paoflow_examples=false
run_paoflow_test=false
skip_qe_if_save_exists=false
examples_arg=""
paoflow_test_staging_dir_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qe) run_qe=true ;;
    --paoflow-examples) run_paoflow_examples=true ;;
    --paoflow-test) run_paoflow_test=true ;;
    --all) run_qe=true; run_paoflow_examples=true; run_paoflow_test=true ;;
    --skip-qe-if-save-exists) skip_qe_if_save_exists=true ;;
    --paoflow-test-staging-dir) paoflow_test_staging_dir_arg="$2"; shift ;;
    --paoflow-test-staging-dir=*) paoflow_test_staging_dir_arg="${1#*=}" ;;
    --examples) examples_arg="$2"; shift ;;
    --examples=*) examples_arg="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) extra_examples+=("$1") ;;
  esac
  shift
done

if [[ "$run_qe" = false && "$run_paoflow_examples" = false && "$run_paoflow_test" = false ]]; then
  run_qe=true
  run_paoflow_examples=true
  run_paoflow_test=true
fi

EXAMPLES_ROOT="$(resolve_examples_root)"
TESTS_ROOT="$(resolve_tests_root)"
REPO_ROOT="$(cd "$EXAMPLES_ROOT/../.." && pwd)"
LOG_DIR="$EXAMPLES_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_LOG="$LOG_DIR/job.log"
QE_LOG="$LOG_DIR/qe.log"
PAOFLOW_EXAMPLES_LOG="$LOG_DIR/paoflow_examples.log"
PAOFLOW_TEST_LOG="$LOG_DIR/paoflow_test.log"

: > "$JOB_LOG"
: > "$QE_LOG"
: > "$PAOFLOW_EXAMPLES_LOG"
: > "$PAOFLOW_TEST_LOG"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
PAOFLOW_TEST_STAGING_DIR="${paoflow_test_staging_dir_arg:-${PAOFLOW_TEST_STAGING_DIR:-$TESTS_ROOT/_assets/staging}}"

if [[ "$run_qe" = true ]]; then
  PW_EXEC="$(resolve_exec PW_EXEC pw.x)"
  PP_EXEC="$(resolve_exec PP_EXEC projwfc.x)"
  PARALLEL_EXEC="${PARALLEL_EXEC:-}"
  if [[ -z "$PARALLEL_EXEC" && -n "${SLURM_NTASKS:-}" ]] && command -v srun >/dev/null 2>&1; then
    PARALLEL_EXEC="srun -n ${SLURM_NTASKS}"
  fi
  PARALLEL_EXEC_CMD=()
  [[ -n "$PARALLEL_EXEC" ]] && read -r -a PARALLEL_EXEC_CMD <<< "$PARALLEL_EXEC"
else
  PARALLEL_EXEC_CMD=()
fi

declare -a examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a selected_examples <<< "$examples_arg"
  mapfile -t examples < <(expand_examples_from_selectors "${selected_examples[@]}")
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  mapfile -t examples < <(expand_examples_from_selectors "${extra_examples[@]}")
else
  mapfile -t examples < <(find "$EXAMPLES_ROOT" -maxdepth 1 -type d -name 'example*' -printf '%f\n' | sort)
fi

failures=0

for ex in "${examples[@]}"; do
  exdir="$EXAMPLES_ROOT/$ex"
  test_exdir="$TESTS_ROOT/$ex"

  if [[ ! -d "$exdir" ]]; then
    log_job "WARN: missing example directory: $exdir"
    failures=$((failures + 1))
    continue
  fi

  if [[ "$run_qe" = true ]]; then
    while IFS= read -r jobdir; do
      label="${jobdir#"$EXAMPLES_ROOT"/}"
      run_qe_dir "$jobdir" "$label" || failures=$((failures + 1))
    done < <(collect_example_jobs "$exdir")
  fi

  if [[ "$run_paoflow_examples" = true ]]; then
    while IFS= read -r jobdir; do
      label="${jobdir#"$EXAMPLES_ROOT"/}"
      run_paoflow_example_dir "$jobdir" "$label" || failures=$((failures + 1))
    done < <(collect_example_jobs "$exdir")
  fi

  if [[ "$run_paoflow_test" = true ]]; then
    if [[ ! -d "$test_exdir" ]]; then
      log_job "WARN: missing transport test directory: $test_exdir"
      failures=$((failures + 1))
      continue
    fi

    while IFS= read -r test_jobdir; do
      label="${test_jobdir#"$TESTS_ROOT"/}"
      source_jobdir="$EXAMPLES_ROOT/$label"
      run_paoflow_test_dir "$test_jobdir" "$label" "$source_jobdir" "$PAOFLOW_TEST_STAGING_DIR" || failures=$((failures + 1))
    done < <(collect_test_jobs "$test_exdir")
  fi
done

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures failure(s)."
  exit 1
fi

log_job "Completed successfully."
