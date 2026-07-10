#!/usr/bin/env python
"""Example — AO route (Agapito-Bernardi): interpolate QE's DFPT coupling in PAO.

This is the recommended electron-phonon route.  It does **not** reconstruct the
DFPT perturbation from ``dvscf``.  Instead it reads Quantum ESPRESSO's *full*
coarse-grid coupling ``el_ph_mat`` (which already contains the bare local, bare
nonlocal and induced parts, plus any NLCC / ultrasoft augmentation), rotates it
into the PAOFLOW atomic-orbital (PAO) gauge, and Wigner-Seitz interpolates the
electrons and the vertex to a dense grid to evaluate the isotropic Eliashberg
properties (``alpha^2F``, ``lambda``, ``omega_log``, ``Tc``).

Reference: L. A. Agapito and M. Bernardi, Phys. Rev. B 97, 235146 (2018).

Required inputs (coarse ``pw.x`` nscf + ``ph.x`` DFPT on the SAME k-grid):
  * ``<save>/``                 -- nscf on the FULL k-grid (nosym, noinv), nbnd>nawf
  * ``elph_dir/elphmat.<iq>.dat`` -- QE ``el_ph_mat`` dumps (patched ``ph.x``)
  * ``<prefix>.dyn{1..nq}``     -- dynamical matrices (frequencies + eigenvectors)

The ``elphmat.<iq>.dat`` dumps are written by the PAOFLOW-patched
``PHonon/PH/elphon.f90`` (``elphsum``); run ``ph.x`` with
``electron_phonon='interpolated'`` and ``PAOFLOW_DUMP_ONLY=1``.

Edit the CONFIG block, then::

    conda run -n work python example_ao_from_qe_coupling.py
"""

import os

from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.do_ao_eph import eliashberg_from_qe_coupling
from PAOFLOW.elphon.elph_bloch import read_nscf

# --------------------------------------------------------------------------- #
# CONFIG -- edit for your system (paths are for the Pb 9^3 tutorial run).
# --------------------------------------------------------------------------- #
BASE = os.environ.get('ELPH_BASE', './exercise1')  # dir with lead.save, elph_dir, *.dyn*
BASIS = os.environ.get('PAOFLOW_BASIS', '../../BASIS')  # PAOFLOW atomic-orbital basis
SAVEDIR = 'lead.save'  # nscf save directory (inside BASE)
# Coupling source:
#   'ahc'     -> unpatched QE AHC dumps (ahc_dir/ahc_gkk_iq<iq>.bin); NC pseudos.
#   'elphmat' -> patched-QE el_ph_mat dumps (elph_dir/elphmat.<iq>.dat); any pseudo.
SOURCE = os.environ.get('ELPH_SOURCE', 'ahc')
COUPLING_DIR = 'ahc_dir' if SOURCE == 'ahc' else 'elph_dir'
DYNPREFIX = 'lead'  # dynamical-matrix prefix (<DYNPREFIX>.dyn*)
NG = (9, 9, 9)  # coarse coupling k-grid == pw.x SCF k-grid
# Star sizes of the q-points.  For the full AHC grid (all q written) use all 1s;
# for the 4 irreducible q of the patched dump use the 3^3 star sizes [1,8,6,12].
Q_WEIGHTS = [1] * 27 if SOURCE == 'ahc' else [1, 8, 6, 12]
MASS_AMU = [207.2]  # atomic masses (amu), one per atom in the cell
NELEC = 14  # valence electrons (dense E_F recompute)
NK_DENSE = 18  # dense interpolation grid
SIGMA_RY = 0.02  # Fermi-surface smearing (Ry)
MU_STAR = 0.10  # Coulomb pseudopotential for Tc


def main():
    save = os.path.join(BASE, SAVEDIR)

    # --- electronic structure: PAO Hamiltonian HRs + projections A_k ---------
    pf = PAOFLOW.PAOFLOW(workpath=BASE, outputdir='paoflow_out', savedir=SAVEDIR, verbose=False)
    pf.projections(configuration='standard', basispath=BASIS)
    pf.projectability(pthr=0.90)
    A = pf.data_controller.data_arrays['U'][:, :, :, 0].copy()  # grab before pao_hamiltonian
    pf.pao_hamiltonian()
    HRs = pf.data_controller.data_arrays['HRs']

    info = read_nscf(save)
    nq = len(Q_WEIGHTS)
    dyn_paths = [os.path.join(BASE, '%s.dyn%d' % (DYNPREFIX, i + 1)) for i in range(nq)]

    out = eliashberg_from_qe_coupling(
        A,
        HRs,
        info['kpts_cryst'],
        info['bg'],
        info['at'],
        os.path.join(BASE, COUPLING_DIR),
        Q_WEIGHTS,
        NG,
        dyn_paths,
        source=SOURCE,
        masses_amu=MASS_AMU,
        nk_dense=NK_DENSE,
        sigmas_ry=[SIGMA_RY],
        nelec=NELEC,
        mu_star=MU_STAR,
    )

    kB = 8.617333262e-5  # eV/K
    print('\nAO-route Eliashberg (Pb, 9^3 coarse -> %d^3 dense, source=%s):' % (NK_DENSE, SOURCE))
    print('  N(E_F)        = %.3f states/spin/Ry' % out['dos_ef'].mean())
    print('  lambda        = %.3f' % out['lambda'])
    print('  omega_log     = %.1f K' % (out['omega_log'] / kB))
    print('  Tc (Allen-Dynes) = %.2f K' % out['Tc_allen_dynes'])
    print('\n  per-q  sum_nu lambda_qv:')
    for iq in range(nq):
        print('    q%d (w=%2d):  %.4f' % (iq + 1, Q_WEIGHTS[iq], out['lambda_qv'][iq].sum()))


if __name__ == '__main__':
    main()
