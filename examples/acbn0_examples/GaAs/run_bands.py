"""Run a QE bands.x calculation on top of the converged eACBN0 U+V SCF
for GaAs and extract the fundamental gap (Γ-Γ, direct).

Assumes the workdir already contains a converged ``GaAs.save`` produced
by ``main.py`` (which leaves a self-consistent U+V scf as its last QE
step before band plotting).  Just chains:

  pw.x  < GaAs.bands.in   > GaAs.bands.out
  bands.x < GaAs.bandsx.in > GaAs.bandsx.out

then scans the pw.x stdout for highest-occupied / lowest-unoccupied
levels at each k-point and reports:
  * indirect gap  = min(CBM over k) - max(VBM over k)
  * direct  gap  at Gamma
"""
import re
import subprocess
from pathlib import Path

QE = '/Users/marco/Local/Programs/qe-7.4.1/bin/'
MPI = 'mpirun -np 8'

HERE = Path(__file__).parent


def _run(cmd, stdin, stdout):
    print(f'>>> {cmd} < {stdin} > {stdout}', flush=True)
    with open(HERE / stdin, 'rb') as fin, open(HERE / stdout, 'w') as fout:
        subprocess.run(cmd.split(), stdin=fin, stdout=fout,
                       stderr=subprocess.STDOUT, check=True, cwd=HERE)


# 1.  pw.x bands
_run(f'{MPI} {QE}pw.x', 'GaAs.bands.in', 'GaAs.bands.out')

# 2.  bands.x post-process (sorts bands, writes GaAs.bands.dat.gnu)
_run(f'{MPI} {QE}bands.x', 'GaAs.bandsx.in', 'GaAs.bandsx.out')


# ----------------------------------------------------------------- gap
#
# Parse GaAs.bands.out for the per-k-point eigenvalues.  QE prints a
# block per k-point of the form:
#
#     k = ...
#
#      ev1 ev2 ev3 ...
#
#     occupation numbers
#      o1 o2 o3 ...
#
# We use the explicit "highest occupied, lowest unoccupied" line that
# QE writes at the end if it found a gap; otherwise we compute it
# ourselves from the raw eigenvalues.
#
out_text = (HERE / 'GaAs.bands.out').read_text()

m = re.search(
    r'highest occupied, lowest unoccupied level \(ev\):\s*'
    r'([-\d\.]+)\s+([-\d\.]+)',
    out_text,
)
if m:
    vbm, cbm = float(m.group(1)), float(m.group(2))
    print(f'\nQE-reported VBM = {vbm:.4f} eV')
    print(f'QE-reported CBM = {cbm:.4f} eV')
    print(f'Indirect gap    = {cbm - vbm:.4f} eV')
else:
    print('\n[warn] QE did not print a "highest occupied / lowest unoccupied" '
          'line; parsing eigenvalues manually.')

# Also pull the Gamma-point direct gap by re-parsing eigenvalue blocks.
# QE bands run prints the k-point header followed by the eigenvalues.
blocks = re.findall(
    r'k\s*=\s*([-\d\.\s]+?)\s*\([^)]*\)\s*bands\s*\(ev\):\s*\n\n((?:\s*[-\d\.]+)+)',
    out_text,
)
# nelec for GaAs with these PAW pseudos: Ga (3s2 3p6 3d10 4s2 4p1)=13,
# As (4s2 4p3)=5 → 18 electrons, no spin → 9 occupied bands.
nocc = 9
gamma_gap = None
indirect_vbm, indirect_cbm = -1e30, 1e30
for kstr, evstr in blocks:
    evs = [float(x) for x in evstr.split()]
    if len(evs) <= nocc:
        continue
    vbm_k = evs[nocc - 1]
    cbm_k = evs[nocc]
    indirect_vbm = max(indirect_vbm, vbm_k)
    indirect_cbm = min(indirect_cbm, cbm_k)
    kpt = tuple(float(x) for x in kstr.split())
    if all(abs(c) < 1e-6 for c in kpt) and gamma_gap is None:
        gamma_gap = cbm_k - vbm_k
        print(f'\nGamma   VBM = {vbm_k:.4f}   CBM = {cbm_k:.4f}   '
              f'direct gap = {gamma_gap:.4f} eV')

print(f'\nIndirect gap (raw scan) = {indirect_cbm - indirect_vbm:.4f} eV '
      f'(VBM={indirect_vbm:.4f}, CBM={indirect_cbm:.4f})')
print('Experiment (0 K):  direct gap at Γ = 1.52 eV')
