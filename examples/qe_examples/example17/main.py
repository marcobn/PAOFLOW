"""SnTe — mirror Chern number (``minimal`` projection preset).

Self-contained driver: assumes ``pw.x`` has already produced
``SnTe.save/`` in this directory (run ``scf.in``).

Runs

    projections -> projectability -> pao_hamiltonian -> mirror_chern_number

using the ``minimal`` projection preset (pseudo-atomic wavefunctions
shipped in the Sn/Te UPFs) and reports the mirror Chern number C_M,
the Z2 index nu and the gap.  The mirror Chern step requires the
optional ``z2pack`` and ``tbmodels`` packages.
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

