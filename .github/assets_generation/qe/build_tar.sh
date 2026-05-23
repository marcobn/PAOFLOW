#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] build_assets_tar.sh [options] [example01 example02 ...]

Options:
  --qe                   Create qe_assets.tar.gz from existing *.save folders only.
  --paoflow-test         Create paoflow_assets.tar.gz from staged test Reference folders.
  --all                  Create all selected tar files.
                         If no mode option is passed, the script runs --qe and --paoflow-test.
  --qe-assets-out PATH   Output path for QE assets tar.gz.
                         Default: QE_ASSETS_OUT or examples/qe_examples/_assets/qe_assets.tar.gz.
  --paoflow-assets-out PATH
                         Output path for PAOFLOW test assets tar.gz.
                         Default: PAOFLOW_ASSETS_OUT or tests/integration/qe/_assets/paoflow_assets.tar.gz.
  --paoflow-test-staging-dir PATH
                         Directory containing staged PAOFLOW test Reference folders.
                         Default: PAOFLOW_TEST_STAGING_DIR or tests/integration/qe/_assets/staging.
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
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/examples/qe_examples" ]]; then
    echo "$SLURM_SUBMIT_DIR/examples/qe_examples"
    return
  fi
  echo "$repo_root/examples/qe_examples"
}

resolve_tests_root() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  repo_root="$(cd "$script_dir/../../.." && pwd)"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/tests/integration/qe" ]]; then
    echo "$SLURM_SUBMIT_DIR/tests/integration/qe"
    return
  fi
  echo "$repo_root/tests/integration/qe"
}

normalize_example_selector() {
  local selector="$1"
  selector="${selector%/}"
  selector="${selector#./}"

  if [[ "$selector" == "$EXAMPLES_ROOT/"* ]]; then
    selector="${selector#"$EXAMPLES_ROOT"/}"
  fi

  if [[ "$selector" == examples/qe_examples/* ]]; then
    selector="${selector#examples/qe_examples/}"
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

build_qe_assets_tar() {
  mkdir -p "$(dirname "$QE_ASSETS_OUT")"

  if ! PYTHONPATH="$REPO_ROOT" "$PYTHON_EXEC" "$REPO_ROOT/.github/assets_generation/qe/build_assets.py" \
    --qe-root "$EXAMPLES_ROOT" \
    --out "$QE_ASSETS_OUT"; then
    log_job "WARN: QE assets tar was not created. No valid *.save folders may exist."
    return 1
  fi

  log_job "Built QE assets: $QE_ASSETS_OUT"
}

build_reference_tar_from_roots() {
  local out_tar="$1"
  local description="$2"
  shift 2
  local -a roots=("$@")
  local manifest
  manifest="$(mktemp)"

  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue

    for ex in "${examples[@]}"; do
      if [[ -d "$root/$ex" ]]; then
        while IFS= read -r -d '' refdir; do
          printf '%s\0' "$refdir" >> "$manifest"
        done < <(find "$root/$ex" -type d -name Reference -print0)
      else
        log_job "WARN: missing selected $description directory: $root/$ex"
      fi
    done
  done

  if [[ ! -s "$manifest" ]]; then
    rm -f "$manifest"
    log_job "WARN: no $description Reference folders found. Skipping tar creation."
    return 1
  fi

  mkdir -p "$(dirname "$out_tar")"
  tar --null -T "$manifest" \
    --transform "s|^$EXAMPLES_ROOT/||" \
    --transform "s|^$TESTS_ROOT/||" \
    --transform "s|^$PAOFLOW_TEST_STAGING_DIR/||" \
    -czf "$out_tar"

  rm -f "$manifest"
  log_job "Built $description assets: $out_tar"
}

run_qe=false
run_paoflow_test=false
examples_arg=""
qe_assets_out_arg=""
paoflow_assets_out_arg=""
paoflow_test_staging_dir_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qe) run_qe=true ;;
    --paoflow-test) run_paoflow_test=true ;;
    --all) run_qe=true; run_paoflow_test=true ;;
    --qe-assets-out) qe_assets_out_arg="$2"; shift ;;
    --qe-assets-out=*) qe_assets_out_arg="${1#*=}" ;;
    --paoflow-assets-out) paoflow_assets_out_arg="$2"; shift ;;
    --paoflow-assets-out=*) paoflow_assets_out_arg="${1#*=}" ;;
    --paoflow-test-staging-dir) paoflow_test_staging_dir_arg="$2"; shift ;;
    --paoflow-test-staging-dir=*) paoflow_test_staging_dir_arg="${1#*=}" ;;
    --examples) examples_arg="$2"; shift ;;
    --examples=*) examples_arg="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) extra_examples+=("$1") ;;
  esac
  shift
done

if [[ "$run_qe" = false && "$run_paoflow_test" = false ]]; then
  run_qe=true
  run_paoflow_test=true
fi

EXAMPLES_ROOT="$(resolve_examples_root)"
TESTS_ROOT="$(resolve_tests_root)"
REPO_ROOT="$(cd "$EXAMPLES_ROOT/../.." && pwd)"
LOG_DIR="$EXAMPLES_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_LOG="$LOG_DIR/build_assets_tar.log"
: > "$JOB_LOG"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
QE_ASSETS_OUT="${qe_assets_out_arg:-${QE_ASSETS_OUT:-$EXAMPLES_ROOT/_assets/qe_assets.tar.gz}}"
PAOFLOW_ASSETS_OUT="${paoflow_assets_out_arg:-${PAOFLOW_ASSETS_OUT:-$TESTS_ROOT/_assets/paoflow_assets.tar.gz}}"
PAOFLOW_TEST_STAGING_DIR="${paoflow_test_staging_dir_arg:-${PAOFLOW_TEST_STAGING_DIR:-$TESTS_ROOT/_assets/staging}}"

declare -a examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a _raw_examples <<< "$examples_arg"
  mapfile -t examples < <(expand_examples_from_selectors "${_raw_examples[@]}")
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  mapfile -t examples < <(expand_examples_from_selectors "${extra_examples[@]}")
else
  mapfile -t examples < <(find "$EXAMPLES_ROOT" -maxdepth 1 -type d -name 'example*' -printf '%f\n' | sort)
fi

if [[ ${#examples[@]} -eq 0 ]]; then
  log_job "No examples selected."
  exit 1
fi

log_job "Resolved EXAMPLES_ROOT: $EXAMPLES_ROOT"
log_job "Resolved TESTS_ROOT: $TESTS_ROOT"
log_job "Selected examples: ${examples[*]}"
log_job "PAOFLOW test staging dir: $PAOFLOW_TEST_STAGING_DIR"

failures=0

if [[ "$run_qe" = true ]]; then
  build_qe_assets_tar || failures=$((failures + 1))
fi

if [[ "$run_paoflow_test" = true ]]; then
  build_reference_tar_from_roots "$PAOFLOW_ASSETS_OUT" "PAOFLOW test" "$PAOFLOW_TEST_STAGING_DIR" || failures=$((failures + 1))
fi

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures tar creation warning/failure(s)."
  exit 1
fi

log_job "Completed successfully."
