#!/usr/bin/env python3
"""Interband dielectric function of Pb on the extended PAO basis (no Drude term).

    python optics_eps.py        # after the pw.x nscf in optics/ (nbnd = 40)

Writes optics/output/epsr_<c>.dat and epsi_<c>.dat with intraband=False, read by
``transport.py --interband-dielectric optics/output``; the intraband part is
the extended Drude term of transport.py (alpha^2F_tr, plasma frequency).
"""

import os

from PAOFLOW import PAOFLOW

HERE = os.path.dirname(os.path.abspath(__file__))
OPTICS = os.path.join(HERE, 'optics')
BASISDIR = os.path.join(HERE, 'BASIS_PS')  # paoflow-genbasis-ps --pseudo Pb.upf --out BASIS_PS


def main():
    pf = PAOFLOW.PAOFLOW(
        workpath=OPTICS, outputdir=os.path.join(OPTICS, 'output'),
        savedir=os.path.join(OPTICS, 'pb.save'), smearing='gauss', save_overlaps=False,
        verbose=False,
    )  # fmt: skip
    pf.projections(basispath=BASISDIR, configuration='extended')
    pf.projectability(pthr=0.95)
    pf.pao_hamiltonian()
    pf.interpolated_hamiltonian(nfft1=36, nfft2=36, nfft3=36)
    pf.pao_eigh()
    pf.gradient_and_momenta(position_dipoles=True)
    pf.dielectric_tensor(emin=0.002, emax=6.0, ne=3000, d_tensor='diag', delta=0.1, intraband=False)
    pf.finish_execution()


if __name__ == '__main__':
    main()
