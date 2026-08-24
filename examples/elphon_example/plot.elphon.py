#!/usr/bin/env python3
"""Plot the Eliashberg alpha^2F(omega) and cumulative lambda(omega).

    python plot.elphon.py

Reads OUTPUTDIR/eliashberg.npz written by main.elphon.py (analyse).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTDIR = 'output'
NPZ = os.path.join(HERE, OUTPUTDIR, 'eliashberg.npz')


def main():
    if not os.path.isfile(NPZ):
        raise SystemExit('%s not found; run main.elphon.py analyse first.' % NPZ)
    d = np.load(NPZ)
    omega = d['omega'] * 1e3   # eV -> meV
    a2F = d['a2F']
    lam = float(d['lambda'])
    tc_ad = float(d['Tc_allen_dynes']) if 'Tc_allen_dynes' in d else None
    tc_mcm = float(d['Tc_mcmillan']) if 'Tc_mcmillan' in d else None
    mu = float(d['mu_star']) if 'mu_star' in d else None
    # Cumulative lambda(omega) = 2 * integral_0^omega a2F(w)/w dw.
    w = d['omega']
    with np.errstate(divide='ignore', invalid='ignore'):
        integrand = np.where(w > 0, 2.0 * a2F / w, 0.0)
    lam_cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(w))])

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(omega, a2F, color='C0', label=r'$\alpha^2F(\omega)$')
    ax1.set_xlabel(r'$\omega$ (meV)')
    ax1.set_ylabel(r'$\alpha^2F(\omega)$', color='C0')
    ax1.set_xlim(left=0.0)
    ax1.set_ylim(bottom=0.0)
    ax2 = ax1.twinx()
    ax2.plot(omega, lam_cum, color='C3', label=r'$\lambda(\omega)$')
    ax2.set_ylabel(r'$\lambda(\omega)$', color='C3')
    ax2.set_ylim(bottom=0.0)
    title = r'Eliashberg spectral function ($\lambda = %.3f$)' % lam
    if tc_mcm is not None and tc_ad is not None:
        title += '\n' + r'$T_c^{McM} = %.2f$ K,  $T_c^{AD} = %.2f$ K ($\mu^* = %.2f$)' % (tc_mcm, tc_ad, mu)
    ax1.set_title(title)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
