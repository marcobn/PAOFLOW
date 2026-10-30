#!/usr/bin/env python3
"""ARPES band maps with the electron-phonon self-energy (kinks, phonon linewidths versus temperature).

    mpirun -np N python arpes.py                                   # settings of system.py (ARPES)
    mpirun -np N python arpes.py --self-energy isotropic fan-migdal --temps 10
    mpirun -np N python arpes.py --self-energy fan-migdal --full-path --temps 10 300 600 --nq 12
    python arpes.py --self-energy anisotropic isotropic --temps 20

The photoemission matrix elements of the PAO orbitals (PAOFLOW.arpes, chinook formalism) give
the bare band map of a segment of Gamma -> direction (across the Fermi crossing of a metal,
--segment-ev, or the full line, --full-path).  The spectral function is then dressed with one
or more electron-phonon self-energies (PAOFLOW.elphon.self_energy):

  isotropic    the Engelsberg-Schrieffer Sigma(w, T) from the alpha^2F of
               ../superconductivity/output/eliashberg.npz: the kink, lambda and Z;
  anisotropic  Sigma_nk(w, T) of the nearest irreducible Fermi-surface state from the pair
               coupling of ../superconductivity/output/fs_coupling.npz (two-gap metals);
  fan-migdal   the band-resolved Fan-Migdal Sigma_nk(w, T) on the path from the dense-q
               vertex (semiconductors: linewidths; metals need --adaptive-smearing, and the
               kink is better resolved by the isotropic form at affordable q-grids).

Outputs (output/): arpes_bare.npz, arpes_<kind>_<T>K.npz, sigma_<kind>_<T>K.dat,
self_energy_fan_migdal.npz and arpes.png.
"""

import argparse
import os

import numpy as np

from PAOFLOW.arpes import ArpesExperiment, GridSelfEnergy, simulate
from PAOFLOW.elphon.self_energy import TabulatedSelfEnergy
from PAOFLOW.elphon.workflows import is_root, load_system
from PAOFLOW.elphon.workflows.arpes import (
    anisotropic_self_energies,
    fan_migdal,
    fermi_crossing,
    isotropic_self_energy,
    linewidth_table,
    mass_enhancement,
    nearest_state,
    on_shell_widths,
    path_points,
    photoemission,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = CONFIG.ARPES
NAME = getattr(CONFIG, 'NAME', SYSTEM.prefix)
SUPERCONDUCTIVITY = os.path.join(os.path.dirname(HERE), 'superconductivity', 'output')
KINDS = ('isotropic', 'anisotropic', 'fan-migdal')
LABELS = {
    'isotropic': 'isotropic $\\Sigma(\\omega)$',
    'anisotropic': 'anisotropic $\\Sigma_{n\\mathbf{k}}$',
    'fan-migdal': 'Fan-Migdal $\\Sigma_{n\\mathbf{k}}(\\omega)$',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--self-energy', nargs='+', choices=KINDS, default=DEFAULTS.get('self_energy', ['isotropic']),
        help='self-energies to apply (default %(default)s)',
    )  # fmt: skip
    parser.add_argument(
        '--temps', type=float, nargs='+', default=DEFAULTS.get('temps_k', [10.0]),
        help='temperatures (K)',
    )  # fmt: skip
    parser.add_argument(
        '--direction', type=float, nargs=3, default=DEFAULTS.get('direction', (0.5, 0.0, 0.5)),
        help='end point of the line from Gamma (crystal coordinates)',
    )  # fmt: skip
    parser.add_argument(
        '--segment-ev', type=float, default=DEFAULTS.get('segment_ev'),
        help='segment of about +- this energy across the Fermi crossing (metals)',
    )  # fmt: skip
    parser.add_argument(
        '--full-path', action='store_true', help='the whole line Gamma -> direction'
    )
    parser.add_argument(
        '--npts', type=int, default=DEFAULTS.get('npts', 81), help='points of the path'
    )
    parser.add_argument(
        '--nq', type=int, default=DEFAULTS.get('nq', 12), help='dense q-grid of the Fan-Migdal run'
    )
    parser.add_argument(
        '--eta-ev', type=float, default=DEFAULTS.get('eta_ev', 0.01),
        help='Lorentzian width / floor (eV) of the Fan-Migdal run',
    )  # fmt: skip
    parser.add_argument(
        '--adaptive-smearing', type=float, default=DEFAULTS.get('adaptive_smearing'), metavar='A',
        help='adaptive widths a |v| dq of the Fan-Migdal run (metals; 0 = fixed eta)',
    )  # fmt: skip
    parser.add_argument('--no-plot', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    kinds = list(dict.fromkeys(args.self_energy))
    label = DEFAULTS.get('direction_label', str(tuple(args.direction)))
    SYSTEM.check_inputs()
    electrons = SYSTEM.electrons(outputdir=OUTPUT)
    nscf = electrons.nscf
    bloch, basis = photoemission(electrons)
    crossing = None
    if args.segment_ev and not args.full_path:
        crossing = fermi_crossing(bloch, args.direction, segment_ev=args.segment_ev)
        band, x_f, segment = crossing
        k_cryst, k_cart = path_points(nscf, args.direction, args.npts, segment)
    else:
        k_cryst, k_cart = path_points(nscf, args.direction, args.npts)
    w = np.linspace(*DEFAULTS.get('window', (-0.1, 0.02, 401)))
    result = None
    if 'fan-migdal' in kinds:  # all ranks: the dense-q loop is MPI parallel
        result = fan_migdal(
            SYSTEM, electrons, k_cryst=k_cryst, w_ev=w, temps_k=args.temps, nq=args.nq,
            eta_ev=args.eta_ev, adaptive_smearing=args.adaptive_smearing or None,
        )  # fmt: skip
    if not is_root():
        return
    os.makedirs(OUTPUT, exist_ok=True)
    if crossing is not None:
        print(
            '%s along Gamma-%s: band %d crosses E_F at %.4f of the line; segment %.4f-%.4f (%d points)'
            % (NAME, label, band, x_f, segment[0], segment[1], args.npts)
        )
    else:
        print('%s along Gamma-%s: %d points' % (NAME, label, args.npts))
    i0 = int(np.argmin(np.abs(w)))
    curves = {}  # kind -> representative Sigma(w) at the first temperature (for the figure)
    dressed = {kind: {} for kind in kinds}  # kind -> {T: self-energy callable}
    for it, temp in enumerate(args.temps):
        if 'isotropic' in kinds:
            sigma, lam = isotropic_self_energy(
                os.path.join(SUPERCONDUCTIVITY, 'eliashberg.npz'), w, temp
            )
            lam_slope, z = mass_enhancement(w, sigma)
            print(
                '  isotropic self-energy at %.0f K from alpha^2F (lambda = %.3f): slope lambda = '
                '%.3f, Z = %.3f, -Im Sigma(0) = %.2f meV'
                % (temp, lam, lam_slope, z, -1e3 * sigma[i0].imag)
            )
            np.savetxt(
                os.path.join(OUTPUT, 'sigma_isotropic_%.0fK.dat' % temp),
                np.column_stack([w * 1e3, sigma.real * 1e3, sigma.imag * 1e3]),
                header='omega (meV)  Re Sigma (meV)  Im Sigma (meV)   T = %.0f K' % temp,
            )
            dressed['isotropic'][temp] = GridSelfEnergy(w, sigma)
            curves.setdefault('isotropic', sigma)
        if 'anisotropic' in kinds:
            aniso = anisotropic_self_energies(
                os.path.join(SUPERCONDUCTIVITY, 'fs_coupling.npz'), nscf, bloch, w, temp
            )
            coupling, strength, states = aniso['coupling'], aniso['strength'], aniso['sigma_states']
            lam_nk = strength['lambda_nk']
            picks = {'weakest': int(np.argmin(lam_nk)), 'strongest': int(np.argmax(lam_nk))}
            if crossing is not None:
                a = nearest_state(coupling, nscf['s_cryst'], x_f * np.asarray(args.direction), band)
                if a is not None:
                    picks = {'crossing': a, **picks}
            if it == 0:
                print(
                    '  Fermi-surface coupling: %d irreducible states, lambda = %.3f (iso), '
                    'lambda_nk %.2f-%.2f, N_F %.3f'
                    % (coupling['band'].size, strength['lambda'], lam_nk.min(), lam_nk.max(),
                       strength['dos_ef'])
                )  # fmt: skip
            columns, header = [w * 1e3], ['omega (meV)']
            for name, a in picks.items():
                s = states[a]
                print(
                    '  %s state at %.0f K (band %d, k %s): lambda_nk %.3f, slope lambda %.3f, '
                    '-Im Sigma(0) %.1f meV'
                    % (name, temp, coupling['band'][a], np.round(coupling['k_cryst'][a], 3),
                       lam_nk[a], mass_enhancement(w, s)[0], -1e3 * s[i0].imag)
                )  # fmt: skip
                columns += [s.real * 1e3, s.imag * 1e3]
                header.append('Re/Im Sigma %s state' % name)
            columns += [aniso['sigma_iso'].real * 1e3, aniso['sigma_iso'].imag * 1e3]
            header.append('Re/Im isotropic (meV), T = %.0f K' % temp)
            np.savetxt(
                os.path.join(OUTPUT, 'sigma_anisotropic_%.0fK.dat' % temp),
                np.column_stack(columns), header='  '.join(header),
            )  # fmt: skip
            dressed['anisotropic'][temp] = aniso['self_energy']
            curves.setdefault('anisotropic', states[next(iter(picks.values()))])
        if 'fan-migdal' in kinds:
            dressed['fan-migdal'][temp] = TabulatedSelfEnergy.from_result(result, k_cart, it)
    if result is not None:
        np.savez(os.path.join(OUTPUT, 'self_energy_fan_migdal.npz'), k_cart=k_cart, **result)
        widths = (
            'adaptive a = %.2f, floor %.0f meV' % (args.adaptive_smearing, 1e3 * args.eta_ev)
            if args.adaptive_smearing
            else 'eta %.0f meV' % (1e3 * args.eta_ev)
        )
        print(
            '  Fan-Migdal self-energy (%d^3 q, %s, bands %s):'
            % (args.nq, widths, result['bands'].tolist())
        )
        if crossing is not None:
            ib = list(result['bands']).index(band)
            k_index = int(np.argmin(np.abs(result['energies_ev'][:, ib])))
            for it, temp in enumerate(args.temps):
                sigma = result['sigma_ev'][it, k_index, ib]
                lam_fm, z_fm = mass_enhancement(w, sigma)
                print(
                    '    at k_F, %.0f K: slope lambda = %.3f, Z = %.3f, -Im Sigma(0) = %.2f meV'
                    % (temp, lam_fm, z_fm, -1e3 * sigma[i0].imag)
                )
            curves.setdefault('fan-migdal', result['sigma_ev'][0, k_index, ib])
        else:
            print(
                '    band  k      E (eV)   -Im Sigma(E) (meV) at %s K   |  Re Sigma(E) (meV)'
                % ' / '.join('%.0f' % t for t in args.temps)
            )
            for b, where, energy, width, shift in linewidth_table(result, ('Gamma', label)):
                print(
                    '    %4d  %-5s %8.3f   %s   |  %s'
                    % (b, where, energy, ' / '.join('%6.1f' % x for x in width),
                       ' / '.join('%6.1f' % x for x in shift))
                )  # fmt: skip
            print('    (Fan term only: the band-edge shifts need the Debye-Waller term as well)')
    maps = []
    for it, temp in enumerate(args.temps):
        experiment = ArpesExperiment(
            photon_ev=DEFAULTS['photon_ev'], work_function_ev=DEFAULTS.get('work_function_ev', 4.5),
            temperature_k=temp, polarization=DEFAULTS.get('polarization', (1.0, 0.0, 1.0)),
            energy=(w[0], w[-1], w.size), kpoints=k_cart,
            resolution=DEFAULTS.get('resolution', (0.005, 0.0)),
        )  # fmt: skip
        if it == 0:
            bare = simulate(experiment, basis, bloch)
            bare.save(os.path.join(OUTPUT, 'arpes_bare.npz'))
            maps.append(('bare bands', bare))
        for kind in kinds:
            amap = simulate(experiment, basis, bloch, dressed[kind][temp])
            amap.save(os.path.join(OUTPUT, 'arpes_%s_%.0fK.npz' % (kind.replace('-', '_'), temp)))
            if it == len(args.temps) - 1:
                maps.append(('%s, %.0f K' % (LABELS[kind], temp), amap))
    print(
        '  ARPES maps: %d k-points, window bands %s'
        % (bare.intensity.shape[0], bare.bands.tolist())
    )
    if not args.no_plot:
        plot(w, curves, result, maps, crossing is None, args.temps)


def plot(w, curves, result, maps, full_path, temps):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'isotropic': '#D55E00', 'anisotropic': '#009E73', 'fan-migdal': '#0072B2'}
    fig, axes = plt.subplots(
        1, 1 + len(maps), figsize=(3.9 * (1 + len(maps)), 3.8), constrained_layout=True
    )
    ax = axes[0]
    if full_path and result is not None:
        e = result['energies_ev']
        palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
        for it, temp in enumerate(temps):
            width = on_shell_widths(result, it)
            ax.scatter(e.ravel(), 1e3 * width.ravel(), s=6, color=palette[it % 4],
                       label='%.0f K' % temp)  # fmt: skip
        ax.set_xlabel('E - E$_F$ (eV)')
        ax.set_ylabel('-Im $\\Sigma_{n\\mathbf{k}}(\\varepsilon_{n\\mathbf{k}})$ (meV)')
        ax.set_title('%s: phonon linewidths along the path' % NAME, fontsize=10)
    else:
        for kind, sigma in curves.items():
            ax.plot(w * 1e3, sigma.real * 1e3, color=colors[kind], label='Re, %s' % kind)
            ax.plot(w * 1e3, sigma.imag * 1e3, color=colors[kind], ls='--', label='Im, %s' % kind)
        ax.set_xlabel('$\\omega$ (meV)')
        ax.set_ylabel('$\\Sigma$ (meV)')
        ax.set_title('%s: electron-phonon self-energy, %.0f K' % (NAME, temps[0]), fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    power = DEFAULTS.get('intensity_power', 1.0)
    for ax, (label, amap) in zip(axes[1:], maps):
        distance = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(amap.kpoints, axis=0), axis=1))]
        )
        scale = 1e3 if not full_path else 1.0
        ax.pcolormesh(
            distance * scale, amap.w * 1e3, (amap.intensity / amap.intensity.max()).T ** power,
            cmap='Greys_r', shading='auto',
        )  # fmt: skip
        if not full_path:
            for n in range(amap.energies.shape[1]):
                ax.plot(distance * scale, amap.energies[:, n] * 1e3, color='#0072B2', lw=0.5,
                        alpha=0.6)  # fmt: skip
        ax.axhline(0.0, color='w', lw=0.5, ls='--')
        ax.set_ylim(amap.w[0] * 1e3, amap.w[-1] * 1e3)
        ax.set_xlabel('k from the path start (%s/Å)' % ('10$^{-3}$' if not full_path else '1'))
        ax.set_ylabel('E - E$_F$ (meV)')
        title = 'ARPES%s, %s' % (' (I$^{%.1f}$)' % power if power != 1.0 else '', label)
        ax.set_title(title, fontsize=10)
    path = os.path.join(OUTPUT, 'arpes.png')
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


if __name__ == '__main__':
    main()
