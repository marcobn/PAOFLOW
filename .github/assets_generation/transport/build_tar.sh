#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] build_tar.sh [options] [example01 example02 ...]

Options:
  --transport            Create transport_test_assets.tar.gz from example *.save folders
                         plus staged PAOFLOW test Reference outputs.
  --all                  Create transport_test_assets.tar.gz.
                         If no mode option is passed, the script runs --transport.
  --clean-paoflow-test-staging
                         Remove staged PAOFLOW test outputs for the selected examples
                         after the tarball is created successfully.
  --assets-out PATH      Output path for the combined transport asset tar.gz.
                         Default: TRANSPORT_ASSETS_OUT or .github/assets_generation/transport/_assets/transport_test_assets.tar.gz.
  --paoflow-test-staging-dir PATH
                         Directory containing staged PAOFLOW test outputs.
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

clean_reference_staging() {
  local ex cleaned
  cleaned=0

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

build_transport_assets_tar() {
  local examples_csv

  mkdir -p "$(dirname "$TRANSPORT_ASSETS_OUT")"
  examples_csv="$(IFS=,; printf '%s' "${examples[*]}")"

  if ! PYTHONPATH="$REPO_ROOT" "$PYTHON_EXEC" "$REPO_ROOT/.github/assets_generation/transport/build_assets.py" \
    --transport-root "$EXAMPLES_ROOT" \
    --reference-root "$PAOFLOW_TEST_STAGING_DIR" \
    --examples "$examples_csv" \
    --out "$TRANSPORT_ASSETS_OUT"; then
    log_job "WARN: transport assets tar was not created. No staged Reference folders or *.save folders were found."
    return 1
  fi

  log_job "Built transport assets: $TRANSPORT_ASSETS_OUT"
}

run_transport=false
clean_paoflow_test_staging=false
examples_arg=""
assets_out_arg=""
paoflow_test_staging_dir_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --transport) run_transport=true ;;
    --all) run_transport=true ;;
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

if [[ "$run_transport" = false ]]; then
  run_transport=true
fi

EXAMPLES_ROOT="$(resolve_examples_root)"
TESTS_ROOT="$(resolve_tests_root)"
REPO_ROOT="$(cd "$EXAMPLES_ROOT/../.." && pwd)"
LOG_DIR="$EXAMPLES_ROOT/logs"
mkdir -p "$LOG_DIR"

JOB_LOG="$LOG_DIR/build_tar.log"
: > "$JOB_LOG"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
TRANSPORT_ASSETS_OUT="${assets_out_arg:-${TRANSPORT_ASSETS_OUT:-$REPO_ROOT/.github/assets_generation/transport/_assets/transport_test_assets.tar.gz}}"
PAOFLOW_TEST_STAGING_DIR="${paoflow_test_staging_dir_arg:-${PAOFLOW_TEST_STAGING_DIR:-$TESTS_ROOT/_assets/staging}}"

declare -a examples=()
if [[ -n "$examples_arg" ]]; then
  IFS=',' read -r -a selected_examples <<< "$examples_arg"
  mapfile -t examples < <(expand_examples_from_selectors "${selected_examples[@]}")
elif [[ ${#extra_examples[@]} -gt 0 ]]; then
  mapfile -t examples < <(expand_examples_from_selectors "${extra_examples[@]}")
else
  mapfile -t examples < <(find "$EXAMPLES_ROOT" -maxdepth 1 -type d -name 'example*' -printf '%f\n' | sort)
fi

status=0

if [[ "$run_transport" = true ]]; then
  build_transport_assets_tar || status=$?
fi

if [[ $status -eq 0 && "$clean_paoflow_test_staging" = true ]]; then
  clean_reference_staging
fi

exit $status
