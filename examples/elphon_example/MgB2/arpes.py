#!/usr/bin/env python3
"""ARPES of MgB2 with the anisotropic electron-phonon self-energy: the sigma-band kink.

    python arpes.py [--temp 20] [--direction M|K] [--npts 81]

Uses the Fermi-surface pair coupling of main.py (output/fs_coupling.npz, the anisotropic
Migdal-Eliashberg input): the state-resolved Eliashberg function alpha^2F_nk(w) of every
irreducible Fermi-surface state gives the Engelsberg-Schrieffer self-energy Sigma_nk(w, T)
(PAOFLOW.elphon.self_energy.eliashberg_self_energy), strongly anisotropic between the sigma
(B p_xy) and pi (B p_z) sheets.  The ARPES band map of a segment of Gamma-M (or Gamma-K) in
the k_z = 0 plane across the sigma-band crossing (PAOFLOW.arpes) shows the E2g kink at
~70 meV, with the self-energy of the nearest irreducible state of each band and k-point.
Outputs: output/arpes/{sigma_states.dat, arpes_bare.npz, arpes_aniso.npz, arpes_iso.npz, kink.png}.
"""

import argparse
import os

import numpy as np

import main as _main
from PAOFLOW.arpes import (
    ArpesExperiment,
    BlochHamiltonian,
    GridSelfEnergy,
    PhotoemissionBasis,
    simulate,
)
from PAOFLOW.arpes.radial import BOHR_ANGSTROM
from PAOFLOW.elphon.anisotropic_eliashberg import coupling_strength
from PAOFLOW.elphon.fermi_surface_coupling import read_fs_coupling
from PAOFLOW.elphon.pao_electrons import pao_hamiltonian_from_save
from PAOFLOW.elphon.self_energy import eliashberg_self_energy, mass_enhancement
from PAOFLOW.elphon.wavefunction_symmetry import stores_irreducible_wedge

HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = getattr(_main, 'SAVEDIR', os.path.join(HERE, 'mgb2.save'))
BASISDIR = getattr(_main, 'BASISDIR', os.path.join(HERE, 'BASIS_PS'))
OUTPUTDIR = getattr(_main, 'OUTPUTDIR', os.path.join(HERE, 'output'))
PTHR = getattr(_main, 'PTHR', 0.95)
ARPES_DIR = os.path.join(OUTPUTDIR, 'arpes')
PHOTON_EV = 28.0
WORK_FUNCTION_EV = 4.5
WINDOW = (-0.25, 0.03, 561)
SEGMENT_EV = 0.4  # the segment across k_F spans about +- this energy
DIRECTIONS = {'M': np.array([0.5, 0.0, 0.0]), 'K': np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])}


def state_self_energies(coupling, w, temperature):
    """alpha^2F_a(w) and Sigma_a(w, T) of every irreducible Fermi-surface state."""
    pair = coupling['coupling']  # (na, nb, nw): integrated over each frequency step
    omega = np.asarray(coupling['freq_ev'], dtype=float)
    spacing = omega[0]
    a2f_state = 0.5 * omega[None, :] * pair.sum(axis=1) / spacing  # (na, nw)
    sigma = eliashberg_self_energy(omega, a2f_state, w, temperature)  # (na, nw)
    return a2f_state, sigma


def nearest_state(coupling, rots, k_cryst, band):
    """Index of the irreducible state of ``band`` whose star is closest to ``k_cryst``."""
    states = np.nonzero(coupling['band'] == band)[0]
    if states.size == 0:
        return None
    reps = coupling['k_cryst'][states]  # (ns, 3)
    best, best_d = None, np.inf
    for rot in rots:
        kr = rot @ k_cryst
        d = np.abs(reps - kr[None, :])
        d = np.minimum(d, 1.0 - d).max(axis=1)  # periodic distance (crystal coordinates)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best, best_d = states[i], d[i]
    return best


class AnisotropicSelfEnergy:
    """Self-energy of the nearest irreducible Fermi-surface state (the ARPES protocol)."""

    def __init__(self, coupling, rots, bloch, sigma_states, w_grid):
        self.coupling, self.rots, self.bloch = coupling, rots, bloch
        self.sigma_states, self.w_grid = sigma_states, w_grid

    def __call__(self, w, energies=None, kpoints=None, bands=None):
        k_cryst = self.bloch.crystal_coordinates(kpoints)
        out = np.zeros((k_cryst.shape[0], len(bands), w.size), dtype=complex)
        for ik, k in enumerate(k_cryst):
            for ib, band in enumerate(bands):
                a = nearest_state(self.coupling, self.rots, k, int(band))
                if a is not None:
                    s = self.sigma_states[a]
                    out[ik, ib] = np.interp(w, self.w_grid, s.real) + 1j * np.interp(w, self.w_grid, s.imag)
        return out


def plot(w, sigma_sigma, sigma_pi, sigma_iso, maps, path, lam):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1 + len(maps), figsize=(3.9 * (1 + len(maps)), 3.8), constrained_layout=True)
    ax = axes[0]
    for sig, color, label in ((sigma_sigma, '#0072B2', '$\\sigma$ state, $\\lambda$ = %.2f' % lam[0]),
                              (sigma_pi, '#D55E00', '$\\pi$ state, $\\lambda$ = %.2f' % lam[1]),
                              (sigma_iso, '0.4', 'isotropic, $\\lambda$ = %.2f' % lam[2])):
        ax.plot(w * 1e3, sig.real * 1e3, color=color, label='Re, ' + label)
        ax.plot(w * 1e3, sig.imag * 1e3, color=color, ls='--')
    ax.set_xlabel('$\\omega$ (meV)')
    ax.set_ylabel('$\\Sigma$ (meV)')
    ax.legend(frameon=False, fontsize=7)
    ax.set_title('MgB$_2$: electron-phonon self-energy', fontsize=10)
    for ax, (label, amap) in zip(axes[1:], maps):
        distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(amap.kpoints, axis=0), axis=1))])
        ax.pcolormesh(distance, amap.w * 1e3, (amap.intensity / amap.intensity.max()).T ** 0.5, cmap='Greys_r', shading='auto')
        for n in range(amap.energies.shape[1]):
            ax.plot(distance, amap.energies[:, n] * 1e3, color='#0072B2', lw=0.5, alpha=0.6)
        ax.axhline(0.0, color='w', lw=0.5, ls='--')
        ax.set_ylim(amap.w[0] * 1e3, amap.w[-1] * 1e3)
        ax.set_xlabel('k from the segment start (1/Å)')
        ax.set_ylabel('E - E$_F$ (meV)')
        ax.set_title('ARPES (I$^{0.5}$), %s' % label, fontsize=10)
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temp', type=float, default=20.0, help='temperature (K)')
    parser.add_argument('--direction', default='M', choices=sorted(DIRECTIONS), help='Gamma-M or Gamma-K')
    parser.add_argument('--npts', type=int, default=81, help='points of the segment')
    args = parser.parse_args()

    el = pao_hamiltonian_from_save(
        SAVEDIR, BASISDIR, 'standard', PTHR, expand_wedge=False,
        unfold_kpoints=stores_irreducible_wedge(SAVEDIR), workpath=HERE, outputdir='output',
    )  # fmt: skip
    nscf = el.nscf
    bloch = BlochHamiltonian.from_data_controller(el.data_controller)
    coupling = read_fs_coupling(os.path.join(OUTPUTDIR, 'fs_coupling.npz'))
    strength = coupling_strength(coupling)
    w = np.linspace(*WINDOW)
    a2f_state, sigma_states = state_self_energies(coupling, w, args.temp)
    sigma_iso = eliashberg_self_energy(strength['omega'], strength['a2F'], w, args.temp)
    rots = np.asarray(nscf['s_cryst'], dtype=float)

    # the sigma band crossing along the chosen direction in the k_z = 0 plane
    direction = DIRECTIONS[args.direction]
    fractions = np.linspace(0.0, 1.0, 2001)
    scan = fractions[:, None] * direction[None, :]
    energies, _ = bloch.eigenstates(scan)
    crossings = []
    for n in range(energies.shape[1]):
        e = energies[:, n]
        idx = np.nonzero(np.diff(np.sign(e)))[0]
        for i in idx:
            crossings.append((fractions[i], n, abs(e[i + 1] - e[i]) / (fractions[1] - fractions[0])))
    if not crossings:
        raise SystemExit('no band crosses E_F along Gamma-%s' % args.direction)
    x_f, band, slope = sorted(crossings)[0]  # the first crossing from Gamma: a sigma band
    half = SEGMENT_EV / slope
    segment = (max(x_f - half, 0.0), min(x_f + half, 1.0))
    k_cryst = np.linspace(segment[0], segment[1], args.npts)[:, None] * direction[None, :]
    k_cart = k_cryst @ nscf['bg'] * 2.0 * np.pi / (nscf['alat'] * BOHR_ANGSTROM)

    os.makedirs(ARPES_DIR, exist_ok=True)
    a_sigma = nearest_state(coupling, rots, x_f * direction, band)
    pi_states = [a for a in range(coupling['band'].size) if strength['lambda_nk'][a] < 0.6]
    a_pi = pi_states[int(np.argmax(coupling['weight'][pi_states]))] if pi_states else a_sigma
    lam_sigma, lam_pi = strength['lambda_nk'][a_sigma], strength['lambda_nk'][a_pi]
    print('MgB2 along Gamma-%s: band %d crosses E_F at %.4f; segment %.4f-%.4f (%d points)'
          % (args.direction, band, x_f, segment[0], segment[1], args.npts))
    print('  Fermi-surface coupling: %d irreducible states, lambda = %.3f (iso), lambda_nk %.2f-%.2f, N_F %.3f'
          % (coupling['band'].size, strength['lambda'], strength['lambda_nk'].min(), strength['lambda_nk'].max(), strength['dos_ef']))
    i0 = int(np.argmin(np.abs(w)))
    for name, a in (('sigma', a_sigma), ('pi', a_pi)):
        s = sigma_states[a]
        print('  %s state (band %d, k %s): lambda_nk %.3f, slope lambda %.3f, -Im Sigma(-150 meV) %.1f meV'
              % (name, coupling['band'][a], np.round(coupling['k_cryst'][a], 3), strength['lambda_nk'][a],
                 mass_enhancement(w, s)[0], -1e3 * s[int(np.argmin(np.abs(w + 0.15)))].imag))
    np.savetxt(os.path.join(ARPES_DIR, 'sigma_states.dat'),
               np.column_stack([w * 1e3, sigma_states[a_sigma].real * 1e3, sigma_states[a_sigma].imag * 1e3,
                                sigma_states[a_pi].real * 1e3, sigma_states[a_pi].imag * 1e3, sigma_iso.real * 1e3, sigma_iso.imag * 1e3]),
               header='omega (meV)  Re/Im Sigma sigma-state  Re/Im pi-state  Re/Im isotropic (meV), T = %.0f K' % args.temp)

    basis = PhotoemissionBasis.from_data_controller(el.data_controller)
    experiment = ArpesExperiment(
        photon_ev=PHOTON_EV, work_function_ev=WORK_FUNCTION_EV, temperature_k=args.temp,
        polarization=(1.0, 0.0, 1.0), energy=WINDOW, kpoints=k_cart, resolution=(0.005, 0.0),
    )  # fmt: skip
    bare = simulate(experiment, basis, bloch)
    aniso = simulate(experiment, basis, bloch, AnisotropicSelfEnergy(coupling, rots, bloch, sigma_states, w))
    iso = simulate(experiment, basis, bloch, GridSelfEnergy(w, sigma_iso))
    for name, amap in (('bare', bare), ('aniso', aniso), ('iso', iso)):
        amap.save(os.path.join(ARPES_DIR, 'arpes_%s.npz' % name))
    print('  ARPES maps: %d k-points, window bands %s' % (bare.intensity.shape[0], bare.bands.tolist()))
    plot(w, sigma_states[a_sigma], sigma_states[a_pi], sigma_iso,
         [('bare bands', bare), ('anisotropic $\\Sigma_{n\\mathbf{k}}$', aniso), ('isotropic $\\Sigma$', iso)],
         os.path.join(ARPES_DIR, 'kink.png'), (lam_sigma, lam_pi, strength['lambda']))


if __name__ == '__main__':
    main()
