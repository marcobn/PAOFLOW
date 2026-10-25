#!/bin/bash
# Build elphon_test_assets.tar.gz for tests/integration/elphon from an EPW tutorial-06 style Si run.
#
#   build_tar.sh <nscf save dir> <ph.x pp.py save dir> <epw dir with si.epb* and si.ukk> <out dir> [<ibz nscf save dir>]
#
# The QE steps (QE 7.5 / EPW 6.0): pw.x scf (Si_r.upf, 20 Ry, 12 bands) -> ph.x 3^3 with fildvscf ->
# EPW/bin/pp.py -> pw.x nscf on the full 6^3 K_POINTS crystal list (12 bands) -> epw.x epbwrite with
# dis_win_max = 30 (all bands kept).  The reference subset holds epmatq at Gamma and at one rotated
# star member (all k, all bands), 6 MB, instead of the 1.5 GB of .epb files.
# The optional fifth argument is a second nscf of the same scf with "K_POINTS automatic 6 6 6 0 0 0"
# (16 irreducible points, 12 bands), stored as si/si_ibz.save for tests/integration/elphon/test_irreducible_wedge.py.
set -euo pipefail
SAVE=$1; PH=$2; EPW=$3; OUT=$4; IBZ=${5:-}
PY=${PY:-python}
mkdir -p "$OUT/si/reference"
cp -r "$SAVE" "$OUT/si/si.save"
rm -f "$OUT"/si/si.save/*.hdf5 2>/dev/null || true
if [ -n "$IBZ" ]; then
  mkdir -p "$OUT/si/si_ibz.save"
  cp "$IBZ"/data-file-schema.xml "$IBZ"/wfc*.dat "$IBZ"/*.upf "$IBZ"/*.UPF "$OUT/si/si_ibz.save/" 2>/dev/null || true
fi
mkdir -p "$OUT/si/save/si.phsave"
cp "$PH"/si.dvscf_q* "$PH"/si.dyn_q* "$OUT/si/save/"
cp "$PH"/si.phsave/patterns.*.xml "$OUT/si/save/si.phsave/"
$PY - "$EPW" "$OUT/si" <<'PYEOF'
import os, sys
import numpy as np
from PAOFLOW.elphon.do_pao_eph import load_epw_coupling
from PAOFLOW.elphon.dfpt_vertex import CoarseVertexSetup
epw_dir, out = sys.argv[1], sys.argv[2]
setup = CoarseVertexSetup(os.path.join(out, 'si.save'), os.path.join(out, 'save'), (3, 3, 3))
epw = load_epw_coupling(epw_dir, setup.nbnd, setup.nk, setup.nat, setup.bg)
labels = {tuple(np.rint(q * 3).astype(int) % 3): i for i, q in enumerate(epw['q_cryst'])}
gamma = next(m for m in setup.members if np.abs(m.q_cryst).max() < 1e-9)
rotated = next(m for m in setup.members if m.iq_irr == 1 and m.isym != 0 and np.abs(m.umklapp).max() > 0)
qs, eps, tol = [], [], []
for m, t in ((gamma, 1e-10), (rotated, 1e-6)):
    qs.append(m.q_cryst); eps.append(epw['epmatq'][..., labels[tuple(np.rint(m.q_cryst * 3).astype(int) % 3)]]); tol.append(t)
np.savez_compressed(os.path.join(out, 'reference', 'epmatq_subset.npz'), q_cryst=np.array(qs), epmatq=np.array(eps),
                    tolerance=np.array(tol), qgrid=np.array([3, 3, 3]), hermiticity_tolerance=np.array(1e-5))
print('wrote reference for q =', [np.round(q, 3).tolist() for q in qs])
PYEOF
tar -C "$OUT" -czf "$OUT/elphon_test_assets.tar.gz" si
shasum -a 256 "$OUT/elphon_test_assets.tar.gz"
