#!/usr/bin/env python3
"""Direct dielectric function of Si on the extended PAO basis (PAOFLOW optical route).

    python optics_eps.py                                   # after the pw.x nscf in optics/ (nbnd = 60)
    python optics_eps.py --savedir acbn0/optics/Si2.save --outdir acbn0/optics/output   # eACBN0 save

Writes <outdir>/epsr_<c>.dat and epsi_<c>.dat, read by
``main.py --emissivity --direct-dielectric <outdir>``.  The extended basis
(44 orbitals for Si) needs more nscf bands than the electron-phonon run, so it
uses its own nscf; EPW is not rerun.  With ``main.py --electrons-save`` (eACBN0
electrons) run it on the eACBN0 optics save, so that the direct dielectric
function and the band energies come from the same Hamiltonian.
"""

import argparse
import os

from PAOFLOW import PAOFLOW

HERE = os.path.dirname(os.path.abspath(__file__))
OPTICS = os.path.join(HERE, 'optics')
BASISDIR = os.path.join(HERE, 'BASIS_PS')  # paoflow-genbasis-ps --pseudo Si.upf --out BASIS_PS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--savedir', default=os.path.join(OPTICS, 'Si2.save'),
                        help='QE save of the optics nscf (default optics/Si2.save)')
    parser.add_argument('--outdir', default=os.path.join(OPTICS, 'output'),
                        help='output directory (default optics/output)')
    parser.add_argument('--nfft', type=int, default=24, help='interpolated k-grid per axis')
    args = parser.parse_args()
    savedir, outdir = os.path.abspath(args.savedir), os.path.abspath(args.outdir)
    pf = PAOFLOW.PAOFLOW(
        workpath=os.path.dirname(outdir), outputdir=outdir, savedir=savedir,
        save_overlaps=False, verbose=False,
    )  # fmt: skip
    pf.projections(basispath=BASISDIR, configuration='extended')
    pf.projectability(pthr=0.95)
    pf.pao_hamiltonian()
    pf.interpolated_hamiltonian(nfft1=args.nfft, nfft2=args.nfft, nfft3=args.nfft)
    pf.pao_eigh()
    pf.gradient_and_momenta(position_dipoles=True)
    pf.dielectric_tensor(emin=0.01, emax=10.0, ne=1000, d_tensor='diag', delta=0.1)
    pf.finish_execution()


if __name__ == '__main__':
    main()
