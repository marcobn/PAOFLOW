"""Si (ULTRASOFT, scalar-relativistic) — minimal vs standard vs extended presets.

Scalar-relativistic counterpart to ``Si_ONCV/`` using the
``Si.pbe-n-rrkjus_psl.1.0.0.UPF`` ultrasoft pseudopotential.  This
example exercises the augmented branch of
:mod:`PAOFLOW.basis_gen`: the radial Schroedinger equation is solved
as the generalized eigenproblem ``H u = eps S u`` with the
augmentation overlap ``S = I + sum_ij q_ij |beta_i><beta_j|`` built
from ``PP_AUGMENTATION/PP_Q``.  NLCC is also active for this pseudo
and is added to the frozen valence density when evaluating ``V_xc``.

Workflow (run from this directory):

    pw.x < scf.in  > scf.out
    pw.x < nscf.in > nscf.out
    python main.py

For each preset (``minimal`` / ``standard`` / ``extended``) the script
runs ``projections -> projectability -> pao_hamiltonian -> bands`` and
writes ``output_<preset>/bands_0.dat``.  If matplotlib is available,
an overlay ``bands_minimal_vs_standard_vs_extended.png`` is produced.

Compare against ``Si_ONCV/`` (same atom, NC pseudo) to verify that
the augmented USPP solver reproduces the NC bands on a controlled
benchmark.
"""

import os
import sys

import numpy as np

from PAOFLOW import PAOFLOW
from PAOFLOW.basis_gen import generate_basis_for_pseudo
from PAOFLOW.basis_gen.driver import _default_shells
from PAOFLOW.inputs.read_upf import UPF as _UPFParser

try:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    RANK = 0

HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, 'silicon.save')
UPF = os.path.join(HERE, 'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
BASISPATH = os.path.join(HERE, 'BASIS_PS') + os.sep

IBRAV = 2  # fcc
NK = 400


def _run(preset):
    outdir = f'output_{preset}'
    paoflow = PAOFLOW.PAOFLOW(
        workpath=HERE,
        outputdir=outdir,
        savedir=SAVEDIR,
        smearing=None,
        npool=1,
        verbose=False,
    )
    arry, attr = paoflow.data_controller.data_dicts()

    paoflow.projections(basispath=BASISPATH, configuration=preset)
    paoflow.projectability(pthr=0.95)
    nawf = attr['nawf']
    nbnd = attr['bnd']

    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=IBRAV, nk=NK, fname='bands')
    bands_path = os.path.join(HERE, outdir, 'bands_0.dat')

    paoflow.finish_execution()
    return nawf, nbnd, bands_path


def _maybe_plot(results):
    if os.environ.get('PAOFLOW_SKIP_PLOT'):
        return
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {'minimal': 'tab:blue', 'standard': 'tab:green', 'extended': 'tab:red'}
    styles = {'minimal': '-', 'standard': '-.', 'extended': '--'}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for preset, (nawf, nbnd, path) in results.items():
        if not os.path.exists(path):
            continue
        data = np.loadtxt(path)
        ik = data[:, 0]
        bands = data[:, 1:]
        ax.plot(
            ik,
            bands[:, 0],
            color=colors[preset],
            linestyle=styles[preset],
            linewidth=0.9,
            label=f'{preset} (nawf={nawf}, Pn>0.95: {nbnd})',
        )
        if bands.shape[1] > 1:
            ax.plot(ik, bands[:, 1:], color=colors[preset], linestyle=styles[preset], linewidth=0.9)

    ax.set_xlabel('k-point index')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Si USPP — minimal vs standard vs extended PAO bands')
    ax.set_ylim(-13, 20)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, 'bands_minimal_vs_standard_vs_extended.png')
    fig.savefig(out, dpi=150)
    print(f'Wrote {out}')


def main():
    if not os.path.isdir(SAVEDIR):
        print(f'silicon.save not found at {SAVEDIR}.')
        print('Run scf.in then nscf.in with pw.x in this directory first.')
        sys.exit(1)

    if RANK == 0:
        _upf = _UPFParser(UPF)
        elem_dir = os.path.join(BASISPATH, _upf.element.strip())
        expected = _default_shells(_upf, preset='extended')
        missing = [s for s in expected if not os.path.exists(os.path.join(elem_dir, f'{s}.dat'))]
        if missing:
            print(f'Generating pseudo-atom basis under {BASISPATH} ...')
            print('  (USPP path: solving generalized H u = eps S u with augmentation overlap)')
            if os.path.isdir(elem_dir) and missing != expected:
                print(f'  (regenerating: missing shells {missing})')
            generate_basis_for_pseudo(UPF, BASISPATH.rstrip(os.sep),
                                      preset='extended', verbose=True)
        else:
            print(f'Using existing pseudo-atom basis under {BASISPATH}')
    if 'MPI' in globals():
        MPI.COMM_WORLD.Barrier()

    print('--- Si USPP: minimal vs standard vs extended ---')
    results = {}
    for preset in ('minimal', 'standard', 'extended'):
        nawf, nbnd, path = _run(preset)
        print(f'  [{preset:8s}] nawf = {nawf:3d}   Pn>0.95 bands = {nbnd:3d}')
        results[preset] = (nawf, nbnd, path)

    if RANK == 0:
        _maybe_plot(results)


if __name__ == '__main__':
    main()
