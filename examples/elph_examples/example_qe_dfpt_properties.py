#!/usr/bin/env python
"""Example — Route 1 (WORKING): QE DFPT data -> PAOFLOW properties.

Read Quantum ESPRESSO's already-computed coarse-grid electron-phonon coupling
(the ``*.fc`` phonon force constants and the ``a2Fmatdyn.NN`` coupling force
constants), interpolate them to a dense q-grid with PAOFLOW's Wigner-Seitz
generalized-Fourier interpolation (the "Wannier replacement" of EPW), and
evaluate the Eliashberg spectral function alpha^2F, the coupling constant
lambda, omega_log and Tc.

This route reproduces QE ``matdyn``/``lambda`` essentially exactly and is the
validated benchmark for the property engine.

Required QE inputs (produced by the standard ph.x + q2r.x + matdyn(la2F) run):
  * <prefix>.fc          -- phonon interatomic force constants (q2r.x output)
  * a2Fmatdyn.NN         -- el-ph coupling force constants, one per smearing NN
                            (matdyn.x with la2F=.true.)

Edit the CONFIG block below to point at your files, then::

    conda run -n work python example_qe_dfpt_properties.py

For FCC Pb (EPW-School-2018 tutorial, degauss 0.02 == smearing index 4) this
prints lambda ~ 1.34 and omega_log ~ 65.6 K, matching QE.
"""

import os

import numpy as np

from PAOFLOW.elphon import (
    eliashberg_from_modes,
    interpolate_coupling,
    read_a2f_ifc,
    read_qe_ifc,
)

# --------------------------------------------------------------------------- #
# CONFIG -- edit these paths / parameters for your system.
# --------------------------------------------------------------------------- #
DATA = os.environ.get('ELPH_DATA', './data_files')
FC_FILE = os.path.join(DATA, 'Pb333.fc')  # phonon force constants
A2F_FILE = os.path.join(DATA, 'a2Fmatdyn.04')  # coupling FC at the chosen smearing
NAT = 1  # atoms per cell (Pb: 1)
NK = 16  # dense q-grid (NK x NK x NK)
MU_STAR = 0.10  # Coulomb pseudopotential for Tc


def main():
    # 1) read the QE real-space force constants (phonons + coupling).
    ifc_ph = read_qe_ifc(FC_FILE)
    ifc_a2f = read_a2f_ifc(A2F_FILE, nat=NAT)
    print('phonon FC grid :', tuple(ifc_ph['nr']))
    print('coupling FC grid:', tuple(ifc_a2f['nr']))
    print('N(E_F)          : %.4f states/spin/Ry' % ifc_a2f['dos_ef'])

    # 2) Wigner-Seitz generalized-Fourier interpolation to a dense q-grid.
    #    Returns per-mode lambda_{q nu} and omega_{q nu} on the NK^3 grid.
    interp = interpolate_coupling(ifc_ph, ifc_a2f, NK, asr=True, ws=True)
    lambda_qv = interp['lambda_qv']  # (nq, nmode)
    omega_thz = interp['omega_thz']  # (nq, nmode)

    # 3) property engine: alpha^2F, lambda, omega_log, Tc.
    res = eliashberg_from_modes(
        lambda_qv,
        omega_thz,
        mu_star=MU_STAR,
        sigma_w_frac=0.015,
        nomega=600,
    )

    kB_ev = 8.617333262e-5  # Boltzmann constant, eV/K (omega_log/omega_2 are in eV)
    print('\n--- results (dense %d^3 q-grid) ---' % NK)
    print('lambda          : %.4f' % res['lambda'])
    print('omega_log       : %.2f K' % (res['omega_log'] / kB_ev))
    print('omega_2         : %.2f K' % (res['omega_2'] / kB_ev))
    print('Tc (McMillan)   : %.3f K' % res['Tc_mcmillan'])
    print('Tc (Allen-Dynes): %.3f K' % res['Tc_allen_dynes'])
    print('max phonon freq : %.3f THz' % np.abs(omega_thz).max())

    # 4) optionally write alpha^2F(omega) to a file (omega in eV -> meV column).
    out = os.path.join(DATA, 'paoflow_a2F.dat')
    omega_mev = res['omega'] * 1.0e3
    np.savetxt(
        out,
        np.column_stack([omega_mev, res['a2F']]),
        header='omega[meV]   alpha^2F(omega)',
    )
    print('\nwrote alpha^2F to', out)


if __name__ == '__main__':
    main()
