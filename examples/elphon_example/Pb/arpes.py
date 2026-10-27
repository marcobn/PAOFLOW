#!/usr/bin/env python3
"""ARPES of Pb with the electron-phonon self-energy: the kink of the band crossing E_F.

    mpirun -np N python arpes.py [--temp 10] [--fan-migdal] [--nq 18] [--adaptive-smearing 0.7]

1. The isotropic Engelsberg-Schrieffer self-energy Sigma(w, T) from alpha^2F of main.py
   (output/eliashberg.npz): the kink, the mass enhancement lambda and Z.
2. The ARPES band map of a short segment of Gamma-X across k_F (He I, 21.2 eV, s+p
   polarisation; PAOFLOW.arpes with the matrix elements of the PAO orbitals), bare and with
   Sigma(w, T): the dispersion flattens by 1/(1 + lambda) within the phonon energies.
3. With --fan-migdal, the band-resolved Sigma_nk(w, T) on the same segment from the dense-q
   vertex (PAOFLOW.elphon.self_energy.fan_migdal_self_energy) and its band map.  For a metal
   this needs adaptive widths (--adaptive-smearing): a fixed width far below the level
   spacing |v_F| dq of the q-grid gives a noisy self-energy, and the adaptive widths are then
   much larger than the phonon energies of Pb, so the kink is resolved by the isotropic form
   (step 1) rather than by the state-resolved one at affordable q-grids.
Outputs: output/arpes/{sigma_isotropic.dat, arpes_bare.npz, arpes_isotropic.npz,
[self_energy.npz, arpes_fan_migdal.npz], kink.png}.
"""

import argparse
import os

import numpy as np
from mpi4py import MPI

from main import (
    COARSE_GRID,
    COUPLING_DIR,
    COUPLING_SOURCE,
    MASSES_AMU,
    NELEC,
    OUTPUTDIR,
    SAVEDIR,
    pao_electronic_structure,
)
from PAOFLOW.arpes import (
    ArpesExperiment,
    BlochHamiltonian,
    GridSelfEnergy,
    PhotoemissionBasis,
    simulate,
)
from PAOFLOW.arpes.radial import BOHR_ANGSTROM
from PAOFLOW.elphon.elph_bloch import atom_masses
from PAOFLOW.elphon.self_energy import (
    TabulatedSelfEnergy,
    eliashberg_self_energy,
    fan_migdal_self_energy,
    mass_enhancement,
)

ARPES_DIR = os.path.join(OUTPUTDIR, 'arpes')
PHOTON_EV = 21.2  # He I
WORK_FUNCTION_EV = 4.25
WINDOW = (-0.08, 0.02, 401)
SEGMENT_EV = 0.12  # the segment across k_F spans about +- this energy


def gamma_x_path(nscf, npts, fractions=(0.0, 1.0)):
    """A segment of Gamma -> X (crystal coordinates) and the same points in Cartesian 1/A."""
    k_cryst = np.linspace(fractions[0], fractions[1], npts)[:, None] * np.array([0.5, 0.0, 0.5])[None, :]
    k_cart = k_cryst @ nscf['bg'] * 2.0 * np.pi / (nscf['alat'] * BOHR_ANGSTROM)
    return k_cryst, k_cart


def fermi_crossing_segment(bloch, nscf):
    """The first band crossing E_F along Gamma-X: its index, the crossing and a segment around it."""
    fractions = np.linspace(0.0, 1.0, 4001)
    k_cryst, _ = gamma_x_path(nscf, fractions.size)
    energies, _ = bloch.eigenstates(k_cryst)
    for n in range(energies.shape[1]):
        e = energies[:, n]
        crossings = np.nonzero(np.diff(np.sign(e)))[0]
        if crossings.size:
            i = crossings[0]
            x_f = fractions[i]
            slope = abs(e[i + 1] - e[i]) / (fractions[1] - fractions[0])  # eV per fraction
            half = SEGMENT_EV / slope
            return n, x_f, (max(x_f - half, 0.0), min(x_f + half, 1.0))
    raise SystemExit('no band crosses E_F along Gamma-X')


def plot(w, sigma_iso, sigma_fm, maps, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1 + len(maps), figsize=(3.9 * (1 + len(maps)), 3.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(w * 1e3, sigma_iso.real * 1e3, color='#D55E00', label='Re, isotropic $\\alpha^2F$')
    ax.plot(w * 1e3, sigma_iso.imag * 1e3, color='#D55E00', ls='--', label='Im, isotropic')
    if sigma_fm is not None:
        ax.plot(w * 1e3, sigma_fm.real * 1e3, color='#0072B2', label='Re, Fan-Migdal at k$_F$')
        ax.plot(w * 1e3, sigma_fm.imag * 1e3, color='#0072B2', ls='--', label='Im, Fan-Migdal')
    ax.set_xlabel('$\\omega$ (meV)')
    ax.set_ylabel('$\\Sigma$ (meV)')
    ax.legend(frameon=False, fontsize=7)
    ax.set_title('Pb: electron-phonon self-energy', fontsize=10)
    for ax, (label, amap) in zip(axes[1:], maps):
        distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(amap.kpoints, axis=0), axis=1))])
        ax.pcolormesh(distance * 1e3, amap.w * 1e3, amap.intensity.T, cmap='Greys_r', shading='auto')
        ax.plot(distance * 1e3, amap.energies[:, 0] * 1e3, color='#0072B2', lw=0.6, alpha=0.7)
        ax.axhline(0.0, color='w', lw=0.5, ls='--')
        ax.set_ylim(amap.w[0] * 1e3, amap.w[-1] * 1e3)
        ax.set_xlabel('k along $\\Gamma$-X from the segment start (10$^{-3}$/Å)')
        ax.set_ylabel('E - E$_F$ (meV)')
        ax.set_title('ARPES, %s' % label, fontsize=10)
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--temp', type=float, default=10.0, help='temperature (K)')
    parser.add_argument('--npts', type=int, default=81, help='points of the segment')
    parser.add_argument('--fan-migdal', action='store_true', help='also the state-resolved self-energy')
    parser.add_argument('--nq', type=int, default=18, help='dense q-grid of the Fan-Migdal run')
    parser.add_argument('--eta-ev', type=float, default=0.005, help='Lorentzian width / floor (eV)')
    parser.add_argument('--adaptive-smearing', type=float, default=0.7, metavar='A',
                        help='adaptive widths a |v| dq of the Fan-Migdal run (0 = fixed eta)')
    args = parser.parse_args()
    rank = MPI.COMM_WORLD.Get_rank()

    projections, HRs, nscf, data_controller = pao_electronic_structure()
    bloch = BlochHamiltonian.from_data_controller(data_controller)
    band, x_f, segment = fermi_crossing_segment(bloch, nscf)
    k_cryst, k_cart = gamma_x_path(nscf, args.npts, segment)
    w = np.linspace(*WINDOW)

    result = None
    if args.fan_migdal:
        result = fan_migdal_self_energy(
            projections, HRs, nscf['kpts_cryst'], nscf['bg'], nscf['at'], nscf['alat'], nscf['omega'],
            COUPLING_DIR, COARSE_GRID, COARSE_GRID,
            masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']), nelec=NELEC,
            k_cryst=k_cryst, w_ev=w, temps_k=[args.temp], nq_dense=args.nq, eta_ev=args.eta_ev,
            adaptive_smearing=args.adaptive_smearing or None,
            coupling_source=COUPLING_SOURCE, save_dir=SAVEDIR, tau_cryst=nscf['tau_cryst'],
        )  # fmt: skip
    if rank != 0:
        return
    os.makedirs(ARPES_DIR, exist_ok=True)
    print('Pb along Gamma-X: band %d crosses E_F at %.4f of Gamma-X; segment %.4f-%.4f (%d points)'
          % (band, x_f, segment[0], segment[1], args.npts))

    eliashberg_file = os.path.join(OUTPUTDIR, 'eliashberg.npz')
    if not os.path.isfile(eliashberg_file):
        raise SystemExit('run main.py first (output/eliashberg.npz with alpha^2F)')
    data = np.load(eliashberg_file)
    sigma_iso = eliashberg_self_energy(data['omega'], data['a2F'], w, args.temp)
    lam_iso, z_iso = mass_enhancement(w, sigma_iso)
    i0 = int(np.argmin(np.abs(w)))
    print('  isotropic self-energy at %.0f K from alpha^2F (lambda = %.3f): slope lambda = %.3f, '
          'Z = %.3f, -Im Sigma(0) = %.2f meV, -Im Sigma(-50 meV) = %.2f meV'
          % (args.temp, float(data['lambda']), lam_iso, z_iso, -1e3 * sigma_iso[i0].imag,
             -1e3 * sigma_iso[int(np.argmin(np.abs(w + 0.05)))].imag))
    np.savetxt(os.path.join(ARPES_DIR, 'sigma_isotropic.dat'),
               np.column_stack([w * 1e3, sigma_iso.real * 1e3, sigma_iso.imag * 1e3]),
               header='omega (meV)  Re Sigma (meV)  Im Sigma (meV)   T = %.0f K' % args.temp)

    basis = PhotoemissionBasis.from_data_controller(data_controller)
    experiment = ArpesExperiment(
        photon_ev=PHOTON_EV, work_function_ev=WORK_FUNCTION_EV, temperature_k=args.temp,
        polarization=(1.0, 0.0, 1.0), energy=WINDOW, kpoints=k_cart, resolution=(0.002, 0.0),
    )  # fmt: skip
    bare = simulate(experiment, basis, bloch)
    dressed = simulate(experiment, basis, bloch, GridSelfEnergy(w, sigma_iso))
    bare.save(os.path.join(ARPES_DIR, 'arpes_bare.npz'))
    dressed.save(os.path.join(ARPES_DIR, 'arpes_isotropic.npz'))
    maps = [('bare band', bare), ('isotropic $\\Sigma(\\omega)$', dressed)]
    sigma_fm = None
    if result is not None:
        np.savez(os.path.join(ARPES_DIR, 'self_energy.npz'), k_cart=k_cart, **result)
        energies = result['energies_ev']
        ib = list(result['bands']).index(band)
        k_index = int(np.argmin(np.abs(energies[:, ib])))
        sigma_fm = result['sigma_ev'][0, k_index, ib]
        lam_fm, z_fm = mass_enhancement(w, sigma_fm)
        print('  Fan-Migdal (%d^3 q, %s): at k_F slope lambda = %.3f, Z = %.3f, -Im Sigma(0) = %.2f meV'
              % (args.nq, 'adaptive a = %.2f, floor %.0f meV' % (args.adaptive_smearing, 1e3 * args.eta_ev)
                 if args.adaptive_smearing else 'eta %.0f meV' % (1e3 * args.eta_ev),
                 lam_fm, z_fm, -1e3 * sigma_fm[i0].imag))
        fm_map = simulate(experiment, basis, bloch, TabulatedSelfEnergy.from_result(result, k_cart))
        fm_map.save(os.path.join(ARPES_DIR, 'arpes_fan_migdal.npz'))
        maps.append(('Fan-Migdal $\\Sigma_{n\\mathbf{k}}(\\omega)$', fm_map))
    print('  ARPES maps: %d k-points, window band %d' % (bare.intensity.shape[0], band))
    plot(w, sigma_iso, sigma_fm, maps, os.path.join(ARPES_DIR, 'kink.png'))


if __name__ == '__main__':
    main()
