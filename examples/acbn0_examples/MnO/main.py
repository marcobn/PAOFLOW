"""ACBN0 self-consistent U on antiferromagnetic MnO.

AFM-II rocksalt MnO requires two Mn sublattice species (MnA, MnB)
so that QE can initialise opposite starting_magnetization values.
Both sublattices receive the same self-consistent U.

Note: MnO templates do not include a projwfc.in; run_bare_dft therefore
only performs scf and nscf.
"""

import shutil
import subprocess

from PAOFLOW.ACBN0 import ACBN0

PREFIX = 'MnO'
OUT = './tmp/'

MPI_QE = 'srun'
MPI_PY = 'srun'
MPI_HARTREE = 'srun'

QE_PATH = ''  # empty → QE binaries must be on PATH
PY_PATH = '/usr/bin/'


def _shell(cmd, stdin=None, stdout=None):
    """Run a shell command, optionally redirecting stdin/stdout to files."""
    msg = f'>>> {cmd}'
    if stdin:
        msg += f' < {stdin}'
    if stdout:
        msg += f' > {stdout}'
    print(msg, flush=True)
    fin = open(stdin, 'rb') if stdin else None
    fout = open(stdout, 'w') if stdout else None
    try:
        subprocess.run(cmd.split(), stdin=fin, stdout=fout, stderr=subprocess.STDOUT, check=True)
    finally:
        if fin is not None:
            fin.close()
        if fout is not None:
            fout.close()


def run_bare_dft():
    """Run pw.x scf+nscf on the unmodified templates.

    MnO does not supply a projwfc.in template; projwfc is handled
    internally by the ACBN0 driver during the self-consistent loop.
    """
    for c in ('scf', 'nscf'):
        shutil.copy(f'{PREFIX}.{c}.in', f'{c}.in')
    _shell(f'{MPI_QE} {QE_PATH}pw.x', stdin='scf.in', stdout='scf.out')
    _shell(f'{MPI_QE} {QE_PATH}pw.x', stdin='nscf.in', stdout='nscf.out')


# ---------------------------------------------------------------------- #
# Stage 1 — bare DFT                                                     #
# ---------------------------------------------------------------------- #
print('\n=== Stage 1: bare DFT ===\n', flush=True)
run_bare_dft()

# ---------------------------------------------------------------------- #
# Stage 2 — ACBN0 (U only)                                               #
# ---------------------------------------------------------------------- #
print('\n=== Stage 2: ACBN0 self-consistent U ===\n', flush=True)
a = ACBN0(
    PREFIX,
    workdir='./',
    mpi_qe=MPI_QE,
    mpi_python=MPI_PY,
    mpi_hartree=MPI_HARTREE,
    qe_options='-npool 4',
    qe_path=QE_PATH,
    python_path=PY_PATH,
    outputdir=OUT,
    projection='ortho-atomic',
)

# Three equivalent ways to declare Hubbard-active orbitals:
#
#  1) List of labels — U values default to 0.01 eV
hubbard = ['MnA-3d', 'MnB-3d', 'O-2p']
#
#  2) Dict with custom seed U (eV)
# hubbard = {'MnA-3d': 5.0, 'MnB-3d': 5.0, 'O-2p': 2.0}
#
#  3) Dict with (initial_U, occupation) tuples to also fix hubbard_occ
# hubbard = {'MnA-3d': 5.0, 'MnB-3d': 5.0, 'O-2p': (2.0, 4.0)}

a.set_hubbard_parameters(hubbard)
a.optimize_hubbard_U(convergence_threshold=0.01)

print('\nFinal U values:')
for k, v in a.uVals.items():
    print(f'  {k} : {v:.4f} eV')
