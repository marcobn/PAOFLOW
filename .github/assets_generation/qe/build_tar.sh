#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] build_tar.sh [options] [example01 example02 ...]

Options:
  --qe-test              Create qe_test_assets.tar.gz from example *.save folders,
                         staged PAOFLOW test Reference outputs, and BASIS data.
  --repack               Repack qe_test_assets.tar.gz from an unpacked copy under
                         .github/assets_generation/qe/_assets/. Only example*/,
                         job Reference/, job *.save/, and required BASIS/ entries
                         are included; unrelated files in _assets are ignored.
  --all                  Create qe_test_assets.tar.gz.
                         If no mode option is passed, the script runs --qe-test.
  --clean-paoflow-test-staging
                         Remove staged PAOFLOW test outputs for the selected examples
                         after qe_test_assets.tar.gz is created successfully.
  --assets-out PATH      Output path for the combined QE test assets tar.gz.
                         Default: QE_TEST_ASSETS_OUT or .github/assets_generation/qe/_assets/qe_test_assets.tar.gz.
  --paoflow-test-staging-dir PATH
                         Directory containing staged PAOFLOW test outputs.
                         Default: PAOFLOW_TEST_STAGING_DIR or tests/integration/qe/_assets/staging.
  --examples LIST        Comma-separated list of example selectors.
                         Default: all example* directories under EXAMPLES_ROOT.
                         Not supported with --repack.
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

  if [[ "$selector" == "$SELECTOR_ROOT/"* ]]; then
    selector="${selector#"$SELECTOR_ROOT"/}"
  fi

  if [[ "$selector" == examples/qe_examples/* ]]; then
    selector="${selector#examples/qe_examples/}"
  fi

  if [[ "$selector" == .github/assets_generation/qe/_assets/* ]]; then
    selector="${selector#.github/assets_generation/qe/_assets/}"
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
      local -a matches=("$SELECTOR_ROOT"/$selector)
      shopt -u nullglob

      if [[ ${#matches[@]} -eq 0 ]]; then
        log_job "WARN: selector '$selector' matched no examples under $SELECTOR_ROOT"
        continue
      fi

      for match in "${matches[@]}"; do
        normalized="$(normalize_example_selector "$match")"
        if [[ -d "$SELECTOR_ROOT/$normalized" ]]; then
          printf '%s\n' "$normalized"
        fi
      done
      continue
    fi

    normalized="$(normalize_example_selector "$selector")"
    if [[ -d "$SELECTOR_ROOT/$normalized" ]]; then
      printf '%s\n' "$normalized"
    else
      log_job "WARN: selector '$selector' does not resolve to an example under $SELECTOR_ROOT"
    fi
  done | awk '!seen[$0]++'
}

build_qe_test_assets_tar() {
  local examples_csv
  local qe_root
  local reference_root
  local basis_root
  local mode_label

  mkdir -p "$(dirname "$QE_TEST_ASSETS_OUT")"
  examples_csv="$(IFS=,; printf '%s' "${examples[*]}")"

  if [[ "$repack_qe_test" = true ]]; then
    qe_root="$ASSETS_DIR"
    reference_root="$ASSETS_DIR"
    basis_root="$ASSETS_DIR/BASIS"
    mode_label="repack"
  else
    qe_root="$EXAMPLES_ROOT"
    reference_root="$PAOFLOW_TEST_STAGING_DIR"
    basis_root="$REPO_ROOT/BASIS"
    mode_label="source"
  fi

  if ! PYTHONPATH="$REPO_ROOT" "$PYTHON_EXEC" "$REPO_ROOT/.github/assets_generation/qe/build_assets.py" \
    --qe-root "$qe_root" \
    --reference-root "$reference_root" \
    --basis-root "$basis_root" \
    $( [[ "$repack_qe_test" = true ]] && printf '%s ' --repack-layout )\
    --examples "$examples_csv" \
    --out "$QE_TEST_ASSETS_OUT"; then
    log_job "WARN: QE test assets tar was not created in $mode_label mode. No valid *.save folders or Reference folders were found."
    return 1
  fi

  log_job "Built QE test assets ($mode_label): $QE_TEST_ASSETS_OUT"
}

clean_reference_staging() {
  local ex
  local cleaned=0

  for ex in "${examples[@]}"; do
    if [[ -e "$PAOFLOW_TEST_STAGING_DIR/$ex" ]]; then
      rm -rf "$PAOFLOW_TEST_STAGING_DIR/$ex"
      cleaned=$((cleaned + 1))
    fi
  done

  if [[ $cleaned -gt 0 ]]; then
    log_job "Removed staged PAOFLOW test outputs for $cleaned selected example(s)."
  else
    log_job "No staged PAOFLOW test outputs were removed."
  fi
}

run_qe_test=false
repack_qe_test=false
clean_paoflow_test_staging=false
examples_arg=""
assets_out_arg=""
paoflow_test_staging_dir_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qe-test) run_qe_test=true ;;
    --repack) repack_qe_test=true; run_qe_test=true ;;
    --all) run_qe_test=true ;;
    --clean-paoflow-test-staging) clean_paoflow_test_staging=true ;;
    --assets-out) assets_out_arg="$2"; shift ;;
    --assets-out=*) assets_out_arg="${1#*=}" ;;
    --paoflow-test-staging-dir) paoflow_test_staging_dir_arg="$2"; shift ;;
    --paoflow-test-staging-dir=*) paoflow_test_staging_dir_arg="${1#*=}" ;;
    --examples) examples_arg="$2"; shift ;;
    --examples=*) examples_arg="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) extra_examples+=("$1") ;;
  esac
  shift
done

if [[ "$run_qe_test" = false ]]; then
  run_qe_test=true
fi

if [[ "$repack_qe_test" = true && ( -n "$examples_arg" || ${#extra_examples[@]} -gt 0 ) ]]; then
  printf '%s\n' 'ERROR: --repack always rebuilds the full unpacked asset tree; --examples is not supported in repack mode.' >&2
  exit 1
fi

EXAMPLES_ROOT="$(resolve_examples_root)"
TESTS_ROOT="$(resolve_tests_root)"
REPO_ROOT="$(cd "$EXAMPLES_ROOT/../.." && pwd)"
ASSETS_DIR="$REPO_ROOT/.github/assets_generation/qe/_assets"
SELECTOR_ROOT="$EXAMPLES_ROOT"
if [[ "$repack_qe_test" = true ]]; then
  SELECTOR_ROOT="$ASSETS_DIR"
fi
LOG_DIR="$EXAMPLES_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_LOG="$LOG_DIR/build_tar.log"
: > "$JOB_LOG"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
QE_TEST_ASSETS_OUT="${assets_out_arg:-${QE_TEST_ASSETS_OUT:-$ASSETS_DIR/qe_test_assets.tar.gz}}"
PAOFLOW_TEST_STAGING_DIR="${paoflow_test_staging_dir_arg:-${PAOFLOW_TEST_STAGING_DIR:-$TESTS_ROOT/_assets/staging}}"

declare -a examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a _raw_examples <<< "$examples_arg"
  mapfile -t examples < <(expand_examples_from_selectors "${_raw_examples[@]}")
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  mapfile -t examples < <(expand_examples_from_selectors "${extra_examples[@]}")
else
  mapfile -t examples < <(find "$SELECTOR_ROOT" -maxdepth 1 -type d -name 'example*' -printf '%f\n' | sort)
fi

if [[ ${#examples[@]} -eq 0 ]]; then
  log_job "No examples selected."
  exit 1
fi

log_job "Resolved EXAMPLES_ROOT: $EXAMPLES_ROOT"
log_job "Resolved TESTS_ROOT: $TESTS_ROOT"
log_job "Selected examples: ${examples[*]}"
log_job "Selector root: $SELECTOR_ROOT"
log_job "PAOFLOW test staging dir: $PAOFLOW_TEST_STAGING_DIR"
log_job "QE test assets out: $QE_TEST_ASSETS_OUT"
log_job "Repack mode: $repack_qe_test"
if [[ "$repack_qe_test" = true ]]; then
  log_job "Repack root: $ASSETS_DIR"
fi

failures=0

if [[ "$run_qe_test" = true ]]; then
  if build_qe_test_assets_tar; then
    if [[ "$clean_paoflow_test_staging" = true ]]; then
      if [[ "$repack_qe_test" = true ]]; then
        log_job "Skipping staged Reference cleanup in repack mode."
      else
        clean_reference_staging
      fi
    fi
  else
    failures=$((failures + 1))
  fi
fi

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures tar creation warning/failure(s)."
  exit 1
fi

log_job "Completed successfully."
