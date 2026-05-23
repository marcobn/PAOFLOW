#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

usage() {
  cat <<'EOF'
Usage: [environment variables] job.sh [options] [example01 example02 ...]

Options:
  --qe                   Run QE on examples and create qe_assets.tar.gz.
  --paoflow-examples     Run PAOFLOW from examples and write Reference folders.
  --paoflow-test         Run PAOFLOW from tests and create paoflow_assets.tar.gz.
  --all                  Run --qe + --paoflow-test.
  --skip-qe-if-save-exists
                         Skip QE in a job directory when '*.save' already exists.
  --qe-assets-out PATH   Output path for qe assets tar.gz.
  --paoflow-assets-out PATH
                         Output path for paoflow assets tar.gz.
  --examples LIST        Comma-separated list of example selectors (example name,
                         nested path, or glob under examples/qe_examples).
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

infer_outputdir_from_main() {
  local main_py="$1"
  "$PYTHON_EXEC" - <<'PY' "$main_py"
import re, sys
text = open(sys.argv[1], "r", encoding="utf-8").read()
m = re.search(r"outputdir\s*=\s*['\"]([^'\"]+)['\"]", text)
print(m.group(1) if m else "output")
PY
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

should_preserve_wfc_from_main() {
  local main_py="$1"
  [[ -f "$main_py" ]] || return 1

  "$PYTHON_EXEC" - <<'PY' "$main_py"
import ast
import sys

path = sys.argv[1]
try:
    tree = ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
except Exception:
    print("false")
    raise SystemExit(0)


def is_true_literal(node):
    if isinstance(node, ast.Constant):
        return node.value is True
    return isinstance(node, ast.NameConstant) and node.value is True


preserve = False
for node in ast.walk(tree):
    if isinstance(node, ast.keyword) and node.arg == "internal" and is_true_literal(node.value):
        preserve = True
        break
    if isinstance(node, ast.Assign) and is_true_literal(node.value):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "internal":
                preserve = True
                break
        if preserve:
            break

print("true" if preserve else "false")
PY
}

collect_example_jobs() {
  local base_dir="$1"
  local -a dirs=()
  if find "$base_dir" -maxdepth 1 -type f \( -name '*.in' -o -name 'main.py' \) | grep -q .; then
    dirs+=("$base_dir")
  fi
  while IFS= read -r -d '' d; do dirs+=("$d"); done < <(find "$base_dir" -type f -name '*.in' -printf '%h\0' | sort -zu)
  while IFS= read -r -d '' d; do dirs+=("$d"); done < <(find "$base_dir" -type f -name 'main.py' -printf '%h\0' | sort -zu)
  printf '%s\n' "${dirs[@]}" | awk '!seen[$0]++'
}

collect_test_jobs() {
  local base_dir="$1"
  find "$base_dir" -type f -name 'main.py' -printf '%h\n' | sort -u
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

run_qe_dir() {
  local jobdir="$1" label="$2"
  local keep_wfc="false"
  if [[ "$skip_qe_if_save_exists" = true ]] && find "$jobdir" -maxdepth 1 -type d -name '*.save' | grep -q .; then
    log_job "QE: $label - skipped (existing .save detected)"
    return 0
  fi

  mapfile -t inputs < <(find "$jobdir" -maxdepth 1 -type f -name '*.in' -printf '%f\n' | sort)
  if [[ ${#inputs[@]} -eq 0 ]]; then
    log_job "QE: $label - skipped"
    return 0
  fi
  for input in scf.in nscf.in proj.in eps.in; do
    if [[ -f "$jobdir/$input" ]]; then
      local cmd="$PW_EXEC"
      [[ "$input" == *proj* ]] && cmd="$PP_EXEC"
      [[ "$input" == "eps.in" ]] && cmd="$EPSILON_EXEC"
      if ! (cd "$jobdir" && run_qe_exec "$cmd" "$input") >> "$QE_LOG" 2>&1; then
        log_job "QE: $label - FAILED on $input"
        return 1
      fi
    fi
  done

  keep_wfc="$(should_preserve_wfc_from_main "$jobdir/main.py")"
  while IFS= read -r -d '' savedir; do
    if [[ "$keep_wfc" = "true" ]]; then
      # Preserve QE wavefunction artifacts only inside *.save when internal=True.
      find "$savedir" -type f ! -name '*.xml' ! -name '*.UPF' ! -iname '*wfc*' -delete
    else
      find "$savedir" -type f ! -name '*.xml' ! -name '*.UPF' -delete
    fi
  done < <(find "$jobdir" -maxdepth 1 -type d -name '*.save' -print0)
  find "$jobdir" -type f -name '.hub*' -delete
  find "$jobdir" -maxdepth 1 -type f \( -name '*pdos_*' -o -name '*.wfc*' -o \( -name '*.xml' ! -name 'inputfile.xml' \) \) -delete

  if [[ "$keep_wfc" = "true" ]]; then
    log_job "QE: $label - preserving *wfc* only inside *.save (internal=True in main.py)"
  fi

  log_job "QE: $label - OK"
}

run_paoflow_example_dir() {
  local jobdir="$1" label="$2"
  local main_py="$jobdir/main.py"
  [[ -f "$main_py" ]] || { log_job "PAOFLOW(examples): $label - skipped"; return 0; }
  local outputdir output_path
  outputdir="$(infer_outputdir_from_main "$main_py")"
  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then output_path="$outputdir"; else output_path="$jobdir/$outputdir"; fi
  if ! (cd "$jobdir" && "$PYTHON_EXEC" main.py) >> "$PAOFLOW_EXAMPLES_LOG" 2>&1; then
    log_job "PAOFLOW(examples): $label - FAILED"
    return 1
  fi
  if [[ -d "$output_path" ]]; then
    rm -rf "$jobdir/Reference"
    mv "$output_path" "$jobdir/Reference"
  fi
  log_job "PAOFLOW(examples): $label - OK"
}

overlay_qe_save_dirs() {
  local source_jobdir="$1" test_jobdir="$2" copied_list_file="$3"
  [[ -d "$source_jobdir" ]] || return 0
  while IFS= read -r -d '' savedir; do
    local dst="$test_jobdir/$(basename "$savedir")"
    [[ -e "$dst" ]] && continue
    cp -a "$savedir" "$dst"
    printf '%s\n' "$dst" >> "$copied_list_file"
  done < <(find "$source_jobdir" -maxdepth 1 -type d -name '*.save' -print0)
}

run_paoflow_test_dir() {
  local jobdir="$1" label="$2" source_jobdir="$3" staging_dir="$4"
  local copied_list main_py outputdir output_path
  main_py="$jobdir/main.py"
  [[ -f "$main_py" ]] || { log_job "PAOFLOW(test): $label - skipped"; return 0; }
  copied_list="$(mktemp)"
  overlay_qe_save_dirs "$source_jobdir" "$jobdir" "$copied_list"
  outputdir="$(infer_outputdir_from_main "$main_py")"
  outputdir="${outputdir%/}"
  if [[ "$outputdir" = /* ]]; then output_path="$outputdir"; else output_path="$jobdir/$outputdir"; fi
  if ! (cd "$jobdir" && "$PYTHON_EXEC" main.py) >> "$PAOFLOW_TEST_LOG" 2>&1; then
    while IFS= read -r p; do [[ -n "$p" ]] && rm -rf "$p"; done < "$copied_list"
    rm -f "$copied_list"
    log_job "PAOFLOW(test): $label - FAILED"
    return 1
  fi
  if [[ -d "$output_path" ]]; then
    mkdir -p "$staging_dir/$label/Reference"
    cp -a "$output_path/." "$staging_dir/$label/Reference/"
    rm -rf "$output_path"
  fi
  while IFS= read -r p; do [[ -n "$p" ]] && rm -rf "$p"; done < "$copied_list"
  rm -f "$copied_list"
  log_job "PAOFLOW(test): $label - OK"
}

build_qe_assets_tar() {
  mkdir -p "$(dirname "$QE_ASSETS_OUT")"
  (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" "$PYTHON_EXEC" "$REPO_ROOT/.github/assets_generation/qe/build_assets.py" --qe-root "$EXAMPLES_ROOT" --out "$QE_ASSETS_OUT")
  log_job "Built QE assets: $QE_ASSETS_OUT"
}

build_paoflow_assets_tar() {
  local staging_dir="$1"
  mkdir -p "$(dirname "$PAOFLOW_ASSETS_OUT")"
  tar -C "$staging_dir" -czf "$PAOFLOW_ASSETS_OUT" .
  log_job "Built PAOFLOW assets: $PAOFLOW_ASSETS_OUT"
}

run_qe=false
run_paoflow_examples=false
run_paoflow_test=false
skip_qe_if_save_exists=false
examples_arg=""
qe_assets_out_arg=""
paoflow_assets_out_arg=""
extra_examples=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qe) run_qe=true ;;
    --paoflow-examples) run_paoflow_examples=true ;;
    --paoflow-test) run_paoflow_test=true ;;
    --all) run_qe=true; run_paoflow_test=true ;;
    --skip-qe-if-save-exists) skip_qe_if_save_exists=true ;;
    --qe-assets-out) qe_assets_out_arg="$2"; shift ;;
    --qe-assets-out=*) qe_assets_out_arg="${1#*=}" ;;
    --paoflow-assets-out) paoflow_assets_out_arg="$2"; shift ;;
    --paoflow-assets-out=*) paoflow_assets_out_arg="${1#*=}" ;;
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
QE_ASSETS_OUT="${qe_assets_out_arg:-${QE_ASSETS_OUT:-$EXAMPLES_ROOT/_assets/qe_assets.tar.gz}}"
PAOFLOW_ASSETS_OUT="${paoflow_assets_out_arg:-${PAOFLOW_ASSETS_OUT:-$TESTS_ROOT/_assets/paoflow_assets.tar.gz}}"

if [[ "$run_qe" = true ]]; then
  PW_EXEC="$(resolve_exec PW_EXEC pw.x)"
  PP_EXEC="$(resolve_exec PP_EXEC projwfc.x)"
  EPSILON_EXEC="$(resolve_exec EPSILON_EXEC epsilon.x)"
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
log_job "Skip QE on existing .save: $skip_qe_if_save_exists"

failures=0

for ex in "${examples[@]}"; do
  exdir="$EXAMPLES_ROOT/$ex"
  test_exdir="$TESTS_ROOT/$ex"

  if [[ ! -d "$exdir" ]]; then
    log_job "WARN: example directory not found: $exdir"
    failures=$((failures + 1))
    continue
  fi

  mapfile -t example_jobs < <(collect_example_jobs "$exdir")
  if [[ ${#example_jobs[@]} -eq 0 ]]; then
    log_job "WARN: no runnable jobs discovered under $exdir"
    continue
  fi

  example_labels=()
  for jobdir in "${example_jobs[@]}"; do
    example_labels+=("${jobdir#"$EXAMPLES_ROOT"/}")
  done
  log_job "Discovered jobs for $ex: ${example_labels[*]}"

  if [[ "$run_qe" = true ]]; then
    for jobdir in "${example_jobs[@]}"; do
      label="${jobdir#"$EXAMPLES_ROOT"/}"
      run_qe_dir "$jobdir" "$label" || failures=$((failures + 1))
    done
  fi

  if [[ "$run_paoflow_examples" = true ]]; then
    for jobdir in "${example_jobs[@]}"; do
      label="${jobdir#"$EXAMPLES_ROOT"/}"
      run_paoflow_example_dir "$jobdir" "$label" || failures=$((failures + 1))
    done
  fi

  if [[ "$run_paoflow_test" = true && -d "$test_exdir" ]]; then
    if [[ -z "${PAOFLOW_TEST_STAGING_DIR:-}" ]]; then
      PAOFLOW_TEST_STAGING_DIR="$(mktemp -d)"
      trap 'rm -rf "$PAOFLOW_TEST_STAGING_DIR"' EXIT
    fi
    while IFS= read -r test_jobdir; do
      label="${test_jobdir#"$TESTS_ROOT"/}"
      source_jobdir="$EXAMPLES_ROOT/$label"
      run_paoflow_test_dir "$test_jobdir" "$label" "$source_jobdir" "$PAOFLOW_TEST_STAGING_DIR" || failures=$((failures + 1))
    done < <(collect_test_jobs "$test_exdir")
  fi
done

if [[ "$run_qe" = true && $failures -eq 0 ]]; then
  build_qe_assets_tar
fi

if [[ "$run_paoflow_test" = true && $failures -eq 0 && -n "${PAOFLOW_TEST_STAGING_DIR:-}" ]]; then
  build_paoflow_assets_tar "$PAOFLOW_TEST_STAGING_DIR"
fi

if [[ $failures -gt 0 ]]; then
  log_job "Completed with $failures failure(s)."
  exit 1
fi

log_job "Completed successfully."
