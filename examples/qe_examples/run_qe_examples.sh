#!/usr/bin/env bash
#
# run_qe_examples.sh
#
# Walk the directory tree rooted at the script's location and run every
# scf.in and nscf.in file found with pw.x (Quantum ESPRESSO).
#
# Usage:
#   ./run_qe_examples.sh [options]
#
# Options:
#   -e PW_EXEC   Path to pw.x executable (default: pw.x from PATH)
#   -n NP        Number of MPI processes (default: 1, uses mpirun when >1)
#   -r ROOT_DIR  Root directory to search (default: directory of this script)
#   -d           Dry run — print commands without executing them
#   -h           Show this help message
#

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
PW_EXEC="pw.x"
NP=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
DRY_RUN=false

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

while getopts ":e:n:r:dh" opt; do
    case "${opt}" in
        e) PW_EXEC="${OPTARG}" ;;
        n) NP="${OPTARG}" ;;
        r) ROOT_DIR="${OPTARG}" ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        :) echo "Option -${OPTARG} requires an argument." >&2; exit 1 ;;
        \?) echo "Unknown option: -${OPTARG}" >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Resolve MPI prefix
# --------------------------------------------------------------------------- #
if [[ "${NP}" -gt 1 ]]; then
    if ! command -v mpirun &>/dev/null; then
        echo "WARNING: mpirun not found; falling back to serial execution." >&2
        NP=1
        MPI_PREFIX=""
    else
        MPI_PREFIX="mpirun -np ${NP}"
    fi
else
    MPI_PREFIX=""
fi

# Build the pw.x command prefix
if [[ -n "${MPI_PREFIX}" ]]; then
    PW_CMD="${MPI_PREFIX} ${PW_EXEC}"
else
    PW_CMD="${PW_EXEC}"
fi

# --------------------------------------------------------------------------- #
# Validate pw.x availability
# --------------------------------------------------------------------------- #
if [[ "${DRY_RUN}" == false ]]; then
    if ! command -v "${PW_EXEC}" &>/dev/null && [[ ! -x "${PW_EXEC}" ]]; then
        echo "ERROR: pw.x not found. Set PW_EXEC with -e or add it to PATH." >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------------- #
# Helper: run one input file
# --------------------------------------------------------------------------- #
run_input() {
    local input_file="$1"
    local work_dir
    work_dir="$(dirname "${input_file}")"
    local base
    base="$(basename "${input_file}")"
    local output_file="${work_dir}/${base%.in}.out"

    echo ""
    echo "=== Running: ${input_file} ==="
    echo "    Output : ${output_file}"

    if [[ "${DRY_RUN}" == true ]]; then
        echo "    [DRY RUN] cd ${work_dir} && ${PW_CMD} -in ${base} > ${base%.in}.out 2>&1"
        return
    fi

    (
        cd "${work_dir}"
        ${PW_CMD} -in "${base}" > "${base%.in}.out" 2>&1
    )

    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        echo "    WARNING: pw.x exited with code ${exit_code} for ${input_file}" >&2
    else
        echo "    Done."
    fi
}

# --------------------------------------------------------------------------- #
# Main loop — find and run scf.in first, then nscf.in
# --------------------------------------------------------------------------- #
echo "Root directory : ${ROOT_DIR}"
echo "pw.x command   : ${PW_CMD}"
[[ "${DRY_RUN}" == true ]] && echo "Mode           : DRY RUN"
echo ""

# Collect directories that contain at least one target input file
SCF_FILES=()
while IFS= read -r line; do SCF_FILES+=("${line}"); done < <(find "${ROOT_DIR}" -name "scf.in" | sort)
NSCF_FILES=()
while IFS= read -r line; do NSCF_FILES+=("${line}"); done < <(find "${ROOT_DIR}" -name "nscf.in" | sort)

total=$(( ${#SCF_FILES[@]} + ${#NSCF_FILES[@]} ))
if [[ ${total} -eq 0 ]]; then
    echo "No scf.in or nscf.in files found under ${ROOT_DIR}."
    exit 0
fi

echo "Found ${#SCF_FILES[@]} scf.in and ${#NSCF_FILES[@]} nscf.in file(s)."

# Run SCF calculations first (nscf depends on scf output)
for f in "${SCF_FILES[@]}"; do
    run_input "${f}"
done

# Then run NSCF calculations
for f in "${NSCF_FILES[@]}"; do
    run_input "${f}"
done

echo ""
echo "All calculations finished."
