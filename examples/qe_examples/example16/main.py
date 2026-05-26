"""Auto-augmented internal-basis demo (preset configuration).

Mirrors ``examples/qe_examples/example10`` for the GaAs ``.save`` data,
but compares three different ways to choose the projection basis:

1. ``manual``   — the legacy hand-written 11-shell list per atom.
2. ``minimal``  — pseudo-atomic wavefunctions from the UPF (pure
   pseudo basis).  Resolved automatically by
   :func:`PAOFLOW.defs.basis_presets.resolve_configuration`.
3. ``extended`` — mixed scheme: pseudo wavefunctions from the UPF
   *plus* a small set of all-electron polarization shells from
   ``basispath``.

For each scheme the script:

* runs ``projections`` (+ ``projectability``) and reports ``nawf`` and
  the number of well-projected bands;
* builds the PAO Hamiltonian and computes the band structure along the
  default ``ibrav=2`` (fcc) high-symmetry path;
* writes the bands to ``output_<scheme>/bands_0.dat``.

If matplotlib is available, a comparison plot
``bands_comparison.png`` is generated at the end.  Set the
environment variable ``PAOFLOW_SKIP_PLOT=1`` to skip plotting.
"""

import os
import sys

import numpy as np

from PAOFLOW import PAOFLOW

HERE = os.path.dirname(os.path.abspath(__file__))

# Default to example10's QE save directory and pseudopotentials.
SAVEDIR = os.environ.get(
    'PAOFLOW_GAAS_SAVEDIR',
    os.path.normpath(os.path.join(HERE, '..', 'example10', 'GaAs.save')),
)
BASISPATH = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'BASIS')) + os.sep

HAND_WRITTEN = {
    'Ga': ['3S', '3P', '4S', '4P', '3D', '4D', '5S', '5P', '5D', '6S', '6P'],
    'As': ['3S', '4S', '4P', '3D', '3P', '4D', '5S', '5P', '5D', '6S', '6P'],
}

# GaAs is zincblende -> fcc.  Use the default ibrav=2 high-symmetry
# path baked into do_bands.
NK = 400
IBRAV = 2


def _run(configuration, label):
    """Run projections + bands for one scheme.

    Returns
    -------
    (nawf, nbnd, bands_path) : tuple
    """
    outdir_name = f'output_{label}'
    paoflow = PAOFLOW.PAOFLOW(
        workpath=HERE,
        outputdir=outdir_name,
        savedir=SAVEDIR,
        smearing=None,
        npool=1,
        verbose=False,
    )
    arry, attr = paoflow.data_controller.data_dicts()

    # 'manual' uses the legacy AE-only path (internal=True + dict).
    # The presets ignore ``internal`` and dispatch internally.
    paoflow.projections(internal=True, basispath=BASISPATH, configuration=configuration)
    paoflow.projectability(pthr=0.95)

    nawf = attr['nawf']
    nbnd = attr['bnd']
    resolved = dict(arry.get('configuration') or {})

    if paoflow.rank == 0:
        print(
            '  [%-8s] nawf = %3d   bands with Pn>0.95 = %3d   shells = %s'
            % (label, nawf, nbnd, resolved)
        )

    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=IBRAV, nk=NK, fname='bands')

    bands_path = os.path.join(HERE, outdir_name, 'bands_0.dat')
    paoflow.finish_execution()
    return nawf, nbnd, bands_path


def _maybe_plot(results):
    """Overlay bands from each scheme.  Skipped if matplotlib missing
    or ``PAOFLOW_SKIP_PLOT`` is set."""
    if os.environ.get('PAOFLOW_SKIP_PLOT'):
        return
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping comparison plot.')
        return

    colors = {'manual': 'k', 'minimal': 'tab:blue', 'extended': 'tab:red'}
    styles = {'manual': '-', 'minimal': '--', 'extended': ':'}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, (nawf, nbnd, path) in results.items():
        if not os.path.exists(path):
            print(f'  warning: {path} not found; skipping')
            continue
        data = np.loadtxt(path)
        ik = data[:, 0]
        bands = data[:, 1:]
        # Single legend entry per scheme: label the first band only.
        ax.plot(
            ik,
            bands[:, 0],
            color=colors.get(label),
            linestyle=styles.get(label, '-'),
            linewidth=0.9,
            label=f'{label} (nawf={nawf}, Pn>0.95: {nbnd})',
        )
        if bands.shape[1] > 1:
            ax.plot(
                ik,
                bands[:, 1:],
                color=colors.get(label),
                linestyle=styles.get(label, '-'),
                linewidth=0.9,
            )

    ax.set_xlabel('k-point index')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('GaAs PAO bands — projection-scheme comparison')
    ax.set_ylim(-15, 10)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()

    out = os.path.join(HERE, 'bands_comparison.png')
    fig.savefig(out, dpi=150)
    print(f'Wrote comparison plot: {out}')


def main():
    if not os.path.isdir(SAVEDIR):
        print('GaAs.save not found at %s.' % SAVEDIR)
        print('Run examples/qe_examples/example10 first to generate it.')
        sys.exit(1)

    print('--- Projection-scheme comparison on GaAs ---')
    results = {}
    for cfg, label in (
        (HAND_WRITTEN, 'manual'),
        ('minimal', 'minimal'),
        ('extended', 'extended'),
    ):
        results[label] = _run(cfg, label)

    print('\nBand files written:')
    for label, (_, _, path) in results.items():
        print(f'  {label:8s} -> {path}')

    _maybe_plot(results)


if __name__ == '__main__':
    main()
