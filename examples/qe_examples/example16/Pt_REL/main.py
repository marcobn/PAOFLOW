"""Pt (fully-relativistic PSL US) — ``minimal`` vs ``extended``
projection presets.

Self-contained driver: assumes ``pw.x`` has already produced
``pt.save/`` in this directory (run ``scf.in`` then ``nscf.in``).
The pseudopotential is noncollinear + spin-orbit
(``Pt.rel-pz-n-rrkjus_psl.0.1.UPF``), so PAOFLOW takes the SO branch
of the AE-basis builder.

For each preset it runs

    projections -> projectability -> pao_hamiltonian -> bands

and writes ``output_<preset>/bands_0.dat``.  If matplotlib is
available, an overlay ``bands_minimal_vs_extended.png`` is also
produced.

The path to the AE radial database ``BASIS/`` is auto-resolved from
the repository layout; export ``PAOFLOW_BASISPATH`` to override.

Note on SO + ``extended``: the AE radials in ``BASIS/<elem>/`` are
scalar-relativistic.  For an ``l >= 1`` channel the SO branch
currently duplicates the scalar radial across ``j = l +/- 1/2``;
expect spurious / poorly-projected bands in the upper conduction
window with the extended preset.  Valence bands are reproduced
correctly.  See ``TODOs/basis_presets_extended.md`` for details.
"""

import os
import sys

import numpy as np

from PAOFLOW import PAOFLOW

try:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    RANK = 0

HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, 'pt.save')
BASISPATH = (
    os.environ.get(
        'PAOFLOW_BASISPATH',
        os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', 'BASIS')),
    )
    + os.sep
)

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

    colors = {'minimal': 'tab:blue', 'extended': 'tab:red'}
    styles = {'minimal': '-', 'extended': '--'}

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
    ax.set_title('Pt (SO) — minimal vs extended PAO bands')
    ax.set_ylim(-11, 14)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, 'bands_minimal_vs_extended.png')
    fig.savefig(out, dpi=150)
    print(f'Wrote {out}')


def main():
    if not os.path.isdir(SAVEDIR):
        print(f'pt.save not found at {SAVEDIR}.')
        print('Run scf.in then nscf.in with pw.x in this directory first.')
        sys.exit(1)

    print('--- Pt (SO): minimal vs extended ---')
    results = {}
    for preset in ('minimal', 'extended'):
        nawf, nbnd, path = _run(preset)
        print(f'  [{preset:8s}] nawf = {nawf:3d}   Pn>0.95 bands = {nbnd:3d}')
        results[preset] = (nawf, nbnd, path)

    if RANK == 0:
        _maybe_plot(results)


if __name__ == '__main__':
    main()
