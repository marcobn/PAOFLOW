"""SnTe — mirror Chern number (``minimal`` projection preset).

Self-contained driver: assumes ``pw.x`` has already produced
``SnTe.save/`` in this directory (run ``scf.in``).

Runs

    projections -> projectability -> pao_hamiltonian -> mirror_chern_number

using the ``minimal`` projection preset (pseudo-atomic wavefunctions
shipped in the Sn/Te UPFs) and reports the mirror Chern number C_M,
the Z2 index nu and the gap.  The mirror Chern step requires the
optional ``z2pack`` and ``tbmodels`` packages.

The pseudopotentials are fully relativistic PAW: the projections, the
overlap of the atomic orbitals and the orthonormality of the stored
spinor bands are taken in the augmented spinor metric
S = 1 + sum q^{s1 s2}_{kl} |beta_k s1><beta_l s2| (QE's ``qq_so``),
built automatically by ``projections()``; see
docs/internals/internal-basis.md.  With verbose=True the run prints the
deviation of <psi|S|psi> from the identity (6e-6 here) and the number of
projectors (68).

Reference output (QE 7.5, the scf.in of this folder, October 2026):

    [minimal] nawf =  36   Pn>0.95 bands =  32   C_M = +2.000   nu = +0.000   gap = +0.083 eV

with a mirror-symmetry residual [H, M_z] of 2e-6 over a 36 eV bandwidth.
"""

import os
import sys

from PAOFLOW import PAOFLOW

HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, 'SnTe.save')


def _fmt(value):
    return 'None' if value is None else f'{value:+.3f}'


def main():
    if not os.path.isdir(SAVEDIR):
        print(f'SnTe.save not found at {SAVEDIR}.')
        print('Run scf.in with pw.x in this directory first.')
        sys.exit(1)

    paoflow = PAOFLOW.PAOFLOW(
        workpath=HERE,
        outputdir='output',
        savedir=SAVEDIR,
        smearing=None,
        npool=1,
        verbose=False,
    )
    arry, attr = paoflow.data_controller.data_dicts()

    paoflow.projections(configuration='minimal')
    paoflow.projectability(pthr=0.95)
    nawf = attr['nawf']
    nbnd = attr['bnd']

    paoflow.pao_hamiltonian()
    result = paoflow.mirror_chern_number(nbnd_occ='auto', z2pack=True)

    paoflow.finish_execution()

    if result is None:
        print(f'  [minimal] nawf = {nawf:3d}   Pn>0.95 bands = {nbnd:3d}   '
              'mirror Chern not computed')
        return

    print(f'  [minimal] nawf = {nawf:3d}   Pn>0.95 bands = {nbnd:3d}   '
          f'C_M = {_fmt(result.get("C_M"))}   nu = {_fmt(result.get("nu"))}   '
          f'nu_z2 = {_fmt(result.get("nu_z2"))}   gap = {_fmt(result.get("gap"))} eV')


if __name__ == '__main__':
    main()

