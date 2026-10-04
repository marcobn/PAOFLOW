#!/usr/bin/env python3
"""Plot the Eliashberg alpha^2F(omega) and cumulative lambda(omega).

    python plot.elphon.py

Reads OUTPUTDIR/eliashberg.npz written by main.elphon.py (analyse).  When
EPW_A2F points to an existing EPW ``<prefix>.a2f`` file (EPW run with
``a2f_iso``/``eliashberg``), EPW's alpha^2F and cumulative lambda are overlaid
(dashed) for comparison.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTDIR = 'output'
NPZ = os.path.join(HERE, OUTPUTDIR, 'eliashberg.npz')
EPW_A2F = os.path.join(HERE, '.', 'pb.a2f')   # EPW's own alpha^2F file, or None


def read_epw_a2f(path):
    """EPW ``<prefix>.a2f`` -> (omega in meV, alpha^2F, cumulative lambda) for the first smearing."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 3:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue
    data = np.array(rows)
    return data[:, 0], data[:, 1], data[:, 2]


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
    if EPW_A2F and os.path.isfile(EPW_A2F):
        w_epw, a2f_epw, lam_epw = read_epw_a2f(EPW_A2F)
        ax1.plot(w_epw, a2f_epw, color='C0', ls='--', lw=1.0)
        ax2.plot(w_epw, lam_epw, color='C3', ls='--', lw=1.0)
        title += r'  [EPW, dashed: $\lambda = %.3f$]' % lam_epw[-1]
        for ax in (ax1, ax2):  # rescale so the EPW curves are not clipped
            ax.relim()
            ax.autoscale(axis='y')
            ax.set_ylim(bottom=0.0)
    if tc_mcm is not None and tc_ad is not None:
        title += '\n' + r'$T_c^{McM} = %.2f$ K,  $T_c^{AD} = %.2f$ K ($\mu^* = %.2f$)' % (tc_mcm, tc_ad, mu)
    ax1.set_title(title)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
