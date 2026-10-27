#!/usr/bin/env python3
"""ARPES of Si with the electron-phonon self-energy: phonon-broadened bands versus temperature.

    mpirun -np N python arpes.py [--nq 12] [--temps 10 300 600] [--eta-ev 0.02] [--npts 41]

1. The band-resolved Fan-Migdal self-energy Sigma_nk(w, T) of the valence and lowest conduction
   bands along Gamma-X from the dense-q vertex of phonon/ (PAOFLOW.elphon.self_energy).
2. The linewidths -Im Sigma_nk(e_nk) at Gamma and X for every temperature (the Fan term alone;
   the Debye-Waller term needed for the band-edge shifts is not included).
3. The ARPES band map of the path (He I, 21.2 eV) at every temperature with the self-energy,
   and the bare one (PAOFLOW.arpes, matrix elements of the PAO orbitals).
Outputs: output/arpes/{self_energy.npz, arpes_bare.npz, arpes_<T>K.npz, arpes.png}.
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
    QGRID,
    SAVEDIR,
    long_range_model,
    pao_electronic_structure,
)
from PAOFLOW.arpes import ArpesExperiment, BlochHamiltonian, PhotoemissionBasis, simulate
from PAOFLOW.arpes.radial import BOHR_ANGSTROM
from PAOFLOW.elphon.elph_bloch import atom_masses
from PAOFLOW.elphon.self_energy import TabulatedSelfEnergy, fan_migdal_self_energy

ARPES_DIR = os.path.join(OUTPUTDIR, 'arpes')
PHOTON_EV = 21.2  # He I
WORK_FUNCTION_EV = 4.8
WINDOW = (-13.0, 1.5, 581)


def gamma_x_path(nscf, npts):
    """Gamma -> X (crystal coordinates) and the same points in Cartesian 1/A."""
    k_cryst = np.linspace(0.0, 1.0, npts)[:, None] * np.array([0.5, 0.0, 0.5])[None, :]
    k_cart = k_cryst @ nscf['bg'] * 2.0 * np.pi / (nscf['alat'] * BOHR_ANGSTROM)
    return k_cryst, k_cart


def linewidth_table(result):
    """-Im Sigma and Re Sigma at the band energy of every window state at Gamma and X."""
    w, e, sigma = result['w_ev'], result['energies_ev'], result['sigma_ev']
    rows = []
    for ib, band in enumerate(result['bands']):
        for ik, label in ((0, 'Gamma'), (e.shape[0] - 1, 'X')):
            iw = int(np.argmin(np.abs(w - e[ik, ib])))
            rows.append((int(band), label, e[ik, ib], -1e3 * sigma[:, ik, ib, iw].imag, 1e3 * sigma[:, ik, ib, iw].real))
    return rows


def plot(result, maps, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    temps = result['temps_k']
    w, e, sigma = result['w_ev'], result['energies_ev'], result['sigma_ev']
    fig, axes = plt.subplots(1, 1 + len(maps), figsize=(3.9 * (1 + len(maps)), 3.8), constrained_layout=True)
    ax = axes[0]
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for it, temp in enumerate(temps):
        width = np.array([[-sigma[it, ik, ib, int(np.argmin(np.abs(w - e[ik, ib])))].imag for ib in range(e.shape[1])] for ik in range(e.shape[0])])
        ax.scatter(e.ravel(), 1e3 * width.ravel(), s=6, color=colors[it % 4], label='%.0f K' % temp)
    ax.set_xlabel('E - E$_F$ (eV)')
    ax.set_ylabel('-Im $\\Sigma_{n\\mathbf{k}}(\\varepsilon_{n\\mathbf{k}})$ (meV)')
    ax.set_title('Si: phonon linewidths along $\\Gamma$-X', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for ax, (label, amap) in zip(axes[1:], maps):
        distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(amap.kpoints, axis=0), axis=1))])
        # a power scale shows the weaker p-like bands next to the strong s-like one
        ax.pcolormesh(distance, amap.w, (amap.intensity / amap.intensity.max()).T ** 0.3, cmap='Greys_r', shading='auto')
        ax.axhline(0.0, color='w', lw=0.5, ls='--')
        ax.set_xlabel('|k| along $\\Gamma$-X (1/Å)')
        ax.set_ylabel('E - E$_F$ (eV)')
        ax.set_title('ARPES band map (I$^{0.3}$), %s' % label, fontsize=10)
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--nq', type=int, default=12, help='dense q-grid (default %(default)s)')
    parser.add_argument('--temps', type=float, nargs='+', default=[10.0, 300.0, 600.0], help='temperatures (K)')
    parser.add_argument('--eta-ev', type=float, default=0.02, help='Lorentzian width (eV)')
    parser.add_argument('--npts', type=int, default=41, help='points along Gamma-X')
    args = parser.parse_args()
    rank = MPI.COMM_WORLD.Get_rank()

    projections, HRs, nscf, data_controller, _ = pao_electronic_structure()
    k_cryst, k_cart = gamma_x_path(nscf, args.npts)
    w = np.linspace(*WINDOW)
    result = fan_migdal_self_energy(
        projections, HRs, nscf['kpts_cryst'], nscf['bg'], nscf['at'], nscf['alat'], nscf['omega'],
        COUPLING_DIR, QGRID, COARSE_GRID,
        masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']), nelec=NELEC,
        k_cryst=k_cryst, w_ev=w, temps_k=args.temps, nq_dense=args.nq, eta_ev=args.eta_ev,
        coupling_source=COUPLING_SOURCE, save_dir=SAVEDIR, long_range=long_range_model(),
        tau_cryst=nscf['tau_cryst'], orbital_positions=nscf['orbital_positions'],
    )  # fmt: skip
    if rank != 0:
        return
    os.makedirs(ARPES_DIR, exist_ok=True)
    np.savez(os.path.join(ARPES_DIR, 'self_energy.npz'), k_cart=k_cart, **result)
    print('Fan-Migdal self-energy of Si along Gamma-X (%d^3 q, eta %.0f meV, bands %s):'
          % (args.nq, 1e3 * args.eta_ev, result['bands'].tolist()))
    print('  band  k      E (eV)   -Im Sigma(E) (meV) at %s K   |  Re Sigma(E) (meV)'
          % ' / '.join('%.0f' % t for t in args.temps))
    for band, label, energy, width, shift in linewidth_table(result):
        print('  %4d  %-5s %8.3f   %s   |  %s' % (band, label, energy,
              ' / '.join('%6.1f' % x for x in width), ' / '.join('%6.1f' % x for x in shift)))
    print('  (Fan term only: the band-edge shifts need the Debye-Waller term as well)')

    basis = PhotoemissionBasis.from_data_controller(data_controller)
    bloch = BlochHamiltonian.from_data_controller(data_controller)
    maps = []
    for it, temp in enumerate(args.temps):
        experiment = ArpesExperiment(
            photon_ev=PHOTON_EV, work_function_ev=WORK_FUNCTION_EV, temperature_k=temp,
            polarization=(1.0, 0.0, 1.0), energy=WINDOW, kpoints=k_cart, resolution=(0.05, 0.0),
        )  # fmt: skip
        if it == 0:
            bare = simulate(experiment, basis, bloch)
            bare.save(os.path.join(ARPES_DIR, 'arpes_bare.npz'))
            maps.append(('bare bands', bare))
        dressed = simulate(experiment, basis, bloch, TabulatedSelfEnergy.from_result(result, k_cart, it))
        dressed.save(os.path.join(ARPES_DIR, 'arpes_%.0fK.npz' % temp))
        if it == len(args.temps) - 1:
            maps.append(('with $\\Sigma_{n\\mathbf{k}}$ at %.0f K' % temp, dressed))
    print('  ARPES maps: %d k-points, %d window bands' % (bare.intensity.shape[0], bare.bands.size))
    plot(result, maps, os.path.join(ARPES_DIR, 'arpes.png'))


if __name__ == '__main__':
    main()
