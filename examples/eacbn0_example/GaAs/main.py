"""eACBN0 joint U+V SCF run on zincblende GaAs.

Mirrors the Si example: runs three calculations in sequence on the same
GaAs primitive cell and compares their band structures on a single plot.

  1. **bare**   — plain DFT, no Hubbard correction.
  2. **U-only** — ACBN0 self-consistent U on Ga-4s, Ga-4p, As-4s, As-4p.
  3. **U+V**    — eACBN0 self-consistent U on the same orbitals plus
                  intersite V on every Ga-As bond within 2.6 Å
                  (4 nearest neighbours at d = a*sqrt(3)/4 ~ 2.45 Å).

For each stage the PAO band structure is dumped to
``tmp/bands_<stage>_0.dat`` and finally all three are plotted together
with :class:`PAOFLOW.GPAO.GPAO`.

Notes
-----
- ``Ga.pbe-dn-kjpaw_psl.1.0.0.UPF`` includes Ga-3d in valence.  This
  semicore d-shell could also be made Hubbard-active (it improves the
  gap further), but is omitted here for parity with the Si example.
- After the U+V loop converges, on-site U on both s manifolds is
  zeroed out (Lee-Son: "on-site interactions for s orbitals were
  neglected") and DFT is re-run once before the final band plot.
"""

import shutil
import subprocess
from os.path import join

from PAOFLOW.ACBN0 import ACBN0
from PAOFLOW.eACBN0 import eACBN0
from PAOFLOW import GPAO


PREFIX = 'GaAs'
OUT = './tmp/'

MPI_QE = 'mpirun -np 8'
MPI_PY = 'mpirun -np 1'
MPI_HARTREE = 'mpirun -np 8'

QE_PATH = '/Users/marco/Local/Programs/qe-7.4.1/bin/'
PY_PATH = '/Users/marco/anaconda3/envs/work/bin/'

# Conventional FCC band path, k-point density.
IBRAV = 2
NK = 400


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
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
        subprocess.run(
            cmd.split(),
            stdin=fin,
            stdout=fout,
            stderr=subprocess.STDOUT,
            check=True,
        )
    finally:
        if fin is not None:
            fin.close()
        if fout is not None:
            fout.close()


def run_bare_dft():
    """Run pw.x scf+nscf and projwfc.x on the *unmodified* templates,
    i.e. without injecting any HUBBARD card."""
    for c in ('scf', 'nscf', 'projwfc'):
        shutil.copy(f'{PREFIX}.{c}.in', f'{c}.in')
    _shell(f'{MPI_QE} {QE_PATH}pw.x', stdin='scf.in', stdout='scf.out')
    _shell(f'{MPI_QE} {QE_PATH}pw.x', stdin='nscf.in', stdout='nscf.out')
    _shell(f'{MPI_QE} {QE_PATH}projwfc.x -nd 1',
           stdin='projwfc.in', stdout='projwfc.out')


def compute_bands(label):
    """Build the PAO Hamiltonian from the current ``GaAs.save`` and dump
    the band structure to ``<OUT>/bands_<label>_0.dat``."""
    script = (
        'from PAOFLOW import PAOFLOW\n'
        f"p = PAOFLOW.PAOFLOW(outputdir='{OUT}', savedir='{PREFIX}.save',\n"
        "                    smearing='gauss', npool=1, verbose=False)\n"
        'p.read_atomic_proj_QE()\n'
        'p.projectability(pthr=0.95)\n'
        'p.pao_hamiltonian()\n'
        f"p.bands(ibrav={IBRAV}, nk={NK}, fname='bands_{label}')\n"
        'p.finish_execution()\n'
    )
    with open('bands_run.py', 'w') as f:
        f.write(script)
    _shell(f'{MPI_PY} {PY_PATH}python bands_run.py')


# ---------------------------------------------------------------------- #
# Stage 1 — bare DFT                                                     #
# ---------------------------------------------------------------------- #
print('\n=== Stage 1: bare DFT (no Hubbard) ===\n', flush=True)
run_bare_dft()
compute_bands('bare')

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
    qe_options='',
    qe_path=QE_PATH,
    python_path=PY_PATH,
    outputdir=OUT,
    projection='ortho-atomic',
)
# Both s and p on each species must be declared so that the V_ss, V_sp,
# V_pp channels exist in the eACBN0 stage; ACBN0 will fit U for all.
a.set_hubbard_parameters({
    'Ga-4s': 0.5,
    'Ga-4p': 0.5,
    'As-4s': 0.5,
    'As-4p': 0.5,
})
a.optimize_hubbard_U(convergence_threshold=0.05)
compute_bands('U')

converged_U = dict(a.uVals)
print('\nConverged U values:')
for k, v in converged_U.items():
    print(f'  {k} : {v:.4f} eV')

# ---------------------------------------------------------------------- #
# Stage 3 — eACBN0 (U + intersite V)                                     #
# ---------------------------------------------------------------------- #
print('\n=== Stage 3: eACBN0 joint U+V loop ===\n', flush=True)
e = eACBN0(
    PREFIX,
    workdir='./',
    mpi_qe=MPI_QE,
    mpi_python=MPI_PY,
    mpi_hartree=MPI_HARTREE,
    qe_options='',
    qe_path=QE_PATH,
    python_path=PY_PATH,
    outputdir=OUT,
    projection='ortho-atomic',
)
# Seed the joint loop with the ACBN0-converged on-site U.
e.set_hubbard_parameters(converged_U)
e.set_intersite_pairs(cutoff=2.6, V_init=0.5)
e.print_intersite_pairs()
e.optimize_hubbard_UV(
    convergence_threshold=0.05,
    max_iter=25,
    mixing=0.7,
)

# Lee-Son PRR 2020: "on-site interactions for s orbitals were neglected."
# Keep the s manifolds active so the V_ss / V_sp channels survive, but
# set their U to zero.  Re-run DFT once with the updated HUBBARD card
# before bands.
e.uVals['Ga-4s'] = 0.0
e.uVals['As-4s'] = 0.0
e.run_dft(PREFIX, e.uspecies, e.uVals)
compute_bands('UV')

print('\nFinal U values:')
for k, v in e.uVals.items():
    print(f'  {k} : {v:.4f} eV')
print('\nFinal V values:')
for k, v in e.vVals.items():
    print(f'  {k} : {v:.4f} eV')

# ---------------------------------------------------------------------- #
# Comparison plot                                                        #
# ---------------------------------------------------------------------- #
print('\n=== Plotting band-structure comparison ===\n', flush=True)
g = GPAO.GPAO()
g.plot_bands(
    [
        join(OUT, 'bands_bare_0.dat'),
        join(OUT, 'bands_U_0.dat'),
        join(OUT, 'bands_UV_0.dat'),
    ],
    sym_points=join(OUT, 'kpath_points.txt'),
    labels=['bare DFT', 'DFT+U', 'DFT+U+V'],
    cols=['k', 'tab:blue', 'tab:red'],
    title='GaAs band structure',
    y_lim=(-13.0, 6.0),
)
