#!/usr/bin/env python3
"""Direct dielectric function of Si on the extended PAO basis (PAOFLOW optical route).

    python optics_eps.py        # after the pw.x nscf in optics/ (nbnd = 60)

Writes optics/output/epsr_<c>.dat and epsi_<c>.dat, read by
``main.py --emissivity --direct-dielectric optics/output``.  The extended basis
(44 orbitals for Si) needs more nscf bands than the electron-phonon run, so it
uses its own nscf; EPW is not rerun.
"""

import os

from PAOFLOW import PAOFLOW

HERE = os.path.dirname(os.path.abspath(__file__))
OPTICS = os.path.join(HERE, 'optics')
BASISDIR = os.path.join(HERE, 'BASIS_PS')  # paoflow-genbasis-ps --pseudo Si.upf --out BASIS_PS


def main():
    pf = PAOFLOW.PAOFLOW(
        workpath=OPTICS, outputdir=os.path.join(OPTICS, 'output'),
        savedir=os.path.join(OPTICS, 'Si2.save'), save_overlaps=False, verbose=False,
    )  # fmt: skip
    pf.projections(basispath=BASISDIR, configuration='extended')
    pf.projectability(pthr=0.95)
    pf.pao_hamiltonian()
    pf.interpolated_hamiltonian(nfft1=24, nfft2=24, nfft3=24)
    pf.pao_eigh()
    pf.gradient_and_momenta(nonlocal_velocity=True)
    pf.dielectric_tensor(emin=0.01, emax=10.0, ne=1000, d_tensor='diag', delta=0.1)
    pf.finish_execution()


if __name__ == '__main__':
    main()
