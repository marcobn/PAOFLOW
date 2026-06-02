"""Pt (fully-relativistic ULTRASOFT) — j-resolved PAO basis from a USPP UPF.

This example showcases the augmented (USPP / PAW) branch of
:mod:`PAOFLOW.basis_gen`: the radial Schroedinger equation is solved as
the generalized eigenproblem ``H u = eps S u`` where the augmentation
overlap ``S = I + sum_ij q_ij |beta_i><beta_j|`` is built from the
UPF's ``PP_AUGMENTATION/PP_Q`` block.  Because the pseudopotential is
fully relativistic (``Pt.rel-pz-n-rrkjus_psl.0.1.UPF``,
``has_spinorbit = True``), the solver produces *j-resolved* radials for
every ``l >= 1`` channel (``6P_j1.dat`` for j=1/2, ``6P_j3.dat`` for
j=3/2, etc., plus a degeneracy-weighted j-average ``6P.dat`` used as
scalar fallback).  PAOFLOW's ``build_aewfc_basis`` then picks up the
j-resolved files automatically when the QE wavefunctions are
noncollinear + spin-orbit, so the resulting Hamiltonian sees the
correct ``(l, j, m_j)`` basis.

Pre-requisite: ``pw.x < scf.in > scf.out`` then ``pw.x < nscf.in >
nscf.out`` in this directory.

For each preset (``minimal`` / ``standard`` / ``extended``) the script
runs the standard PAOFLOW pipeline

    projections -> projectability -> pao_hamiltonian -> bands

and writes ``output_<preset>/bands_0.dat``.  If matplotlib is
available, an overlay
``bands_minimal_vs_standard_vs_extended.png`` is also produced.

Contrast with the ``Pt_REL/`` subdirectory: that one uses an
*all-norm-conserving* ONCV pseudopotential, so the augmentation
overlap is the identity and the solver collapses to the ordinary
``numpy.linalg.eigh`` path.  Comparing the two folders' bands at
similar nawf is the cleanest way to demonstrate that the new USPP/PAW
solver reproduces the NC result on a controlled benchmark while now
also being able to handle ultrasoft and PAW pseudos.
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
SAVEDIR = os.path.join(HERE, 'pt.save')
UPF = os.path.join(HERE, 'Pt.rel-pz-n-rrkjus_psl.0.1.UPF')
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
    ax.set_title('Pt (USPP+SO) — j-resolved PAO bands (minimal vs standard vs extended)')
    ax.set_ylim(-11, 14)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, 'bands_minimal_vs_standard_vs_extended.png')
    fig.savefig(out, dpi=150)
    print(f'Wrote {out}')


def _report_basis_j_resolved():
    """Pretty-print the j-resolved files just generated.

    Each ``<n><L>_j<2j>.dat`` represents the eigenstate of the
    generalized augmented radial problem for the specific (l, j)
    channel; the bare ``<n><L>.dat`` is a degeneracy-weighted average
    used as scalar fallback.
    """
    elem_dir = os.path.join(BASISPATH, 'Pt')
    if not os.path.isdir(elem_dir):
        return
    j_files = sorted(f for f in os.listdir(elem_dir) if '_j' in f)
    if j_files:
        print('  generated j-resolved radial files:')
        for f in j_files:
            print(f'    {f}')


def main():
    if not os.path.isdir(SAVEDIR):
        print(f'pt.save not found at {SAVEDIR}.')
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
            generate_basis_for_pseudo(
                UPF, BASISPATH.rstrip(os.sep), preset='extended', verbose=True
            )
        else:
            print(f'Using existing pseudo-atom basis under {BASISPATH}')
        _report_basis_j_resolved()
    if 'MPI' in globals():
        MPI.COMM_WORLD.Barrier()

    print('--- Pt (USPP+SO): minimal vs standard vs extended ---')
    results = {}
    for preset in ('minimal', 'standard', 'extended'):
        nawf, nbnd, path = _run(preset)
        print(f'  [{preset:8s}] nawf = {nawf:3d}   Pn>0.95 bands = {nbnd:3d}')
        results[preset] = (nawf, nbnd, path)

    if RANK == 0:
        _maybe_plot(results)


if __name__ == '__main__':
    main()
