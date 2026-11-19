#!/usr/bin/env bash
# Build projection_test_assets.tar.gz: the QE saves of tests/integration/projections/<system>.
#
# For every system folder (scf.in, nscf.in, proj.in; pseudopotentials from PSEUDOS/library) this runs
# pw.x (scf, nscf) and projwfc.x in a scratch copy, keeps what PAOFLOW reads (data-file-schema.xml,
# the wavefunctions of the internal projections, atomic_proj.xml of projwfc.x, the UPFs) and packs
# <system>/<prefix>.save into the tarball. The tests need no reference outputs: they compare the PAO
# bands with the DFT eigenvalues of the saves.
#
# usage: [QE_BIN=...] [MPI="mpirun -np 4"] build_assets.sh [OUT.tar.gz] [system ...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
source_root="$repo/tests/integration/projections"
out="${1:-$here/_assets/projection_test_assets.tar.gz}"
shift || true
systems=("$@")
if [[ ${#systems[@]} -eq 0 ]]; then
  for d in "$source_root"/*/; do [[ -f "$d/scf.in" ]] && systems+=("$(basename "$d")"); done
fi
mpi=${MPI:-"mpirun -np 4"}
pw="${QE_BIN:+$QE_BIN/}pw.x"
projwfc="${QE_BIN:+$QE_BIN/}projwfc.x"
work="$(mktemp -d)"
stage="$work/stage"
mkdir -p "$stage" "$(dirname "$out")"
for system in "${systems[@]}"; do
  echo "== $system"
  job="$work/run/$system"
  mkdir -p "$job"
  for f in scf.in nscf.in proj.in; do
    # inputs read the library relative to their folder: point them at it from the scratch copy
    sed "s#pseudo_dir *= *'[^']*'#pseudo_dir = '$repo/PSEUDOS/library/'#" "$source_root/$system/$f" > "$job/$f"
  done
  (cd "$job" && $mpi "$pw" -in scf.in > scf.out && $mpi "$pw" -in nscf.in > nscf.out \
    && $mpi "$projwfc" -in proj.in > proj.out)
  save="$(cd "$job" && ls -d *.save)"
  find "$job/$save" -type f ! -name 'data-file-schema.xml' ! -name 'atomic_proj.xml' \
    ! -name 'wfc*.dat' ! -iname '*.upf' -delete
  mkdir -p "$stage/$system"
  cp -R "$job/$save" "$stage/$system/"
  du -sh "$stage/$system/$save"
done
tar -C "$stage" -czf "$out" .
shasum -a 256 "$out"
rm -rf "$work"
