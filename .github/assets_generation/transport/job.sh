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
  #!/usr/bin/env bash
  set -euo pipefail

  usage() {
    cat <<'EOF'
  Usage: [environment variables] job.sh [options] [example01 example02 ...]

  Convenience wrapper that runs create_assets.sh and then build_tar.sh.

  Common options:
    --qe
    --paoflow-examples
    --paoflow-test
    --all
    --skip-qe-if-save-exists
    --paoflow-test-staging-dir PATH
    --assets-out PATH
    --clean-paoflow-test-staging
    --examples LIST
    -h, --help
  EOF
  }

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

  create_args=()
  build_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --qe|--paoflow-examples|--paoflow-test|--all|--skip-qe-if-save-exists)
        create_args+=("$1")
        shift
        ;;
      --transport|--clean-paoflow-test-staging)
        build_args+=("$1")
        shift
        ;;
      --paoflow-test-staging-dir|--examples)
        create_args+=("$1" "$2")
        build_args+=("$1" "$2")
        shift 2
        ;;
      --paoflow-test-staging-dir=*|--examples=*)
        create_args+=("$1")
        build_args+=("$1")
        shift
        ;;
      --assets-out|--assets-out=*)
        build_args+=("$1")
        if [[ "$1" == "--assets-out" ]]; then
          build_args+=("$2")
          shift 2
        else
          shift
        fi
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        create_args+=("$1")
        build_args+=("$1")
        shift
        ;;
    esac
  done

  create_status=0
  build_status=0

  "$script_dir/create_assets.sh" "${create_args[@]}" || create_status=$?
  "$script_dir/build_tar.sh" "${build_args[@]}" || build_status=$?

  if [[ $create_status -ne 0 || $build_status -ne 0 ]]; then
    exit 1
  fi
    return 0
