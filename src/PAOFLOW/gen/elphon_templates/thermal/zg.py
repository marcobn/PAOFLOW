#!/usr/bin/env python3
"""Thermal band structures from ZG supercells: the special displacement method (Zacharias-Giustino).

    python zg.py --write                    # ZG supercell inputs + zg/run_zg.sh (THERMAL of system.py)
    (cd zg && NP=8 bash run_zg.sh)          # pw.x scf of every cell
    mpirun -np N python zg.py --spectral    # unfolded A(k, E; T): band shifts and widths vs T
    mpirun -np N python zg.py --dielectric  # dielectric function of every ZG cell
    python zg.py --plot                     # figures from the output files
    python zg.py --write --temps 0 300 --supercell 4 4 4 --seeds 0 1 2

One supercell per temperature carries the Williams-Lax thermal average of the clamped-ion
properties: its atoms are displaced along the harmonic phonons of ../phonons/output/<prefix>.fc
with the ZG amplitudes and signs (PAOFLOW.special_displacement), and the scf of each cell is a
plain pw.x run.  The PAO Hamiltonian of every save is then unfolded onto the primitive zone
(PAOFLOW.elphon.workflows.thermal_spectral_function): the band shifts and the inhomogeneous
widths of the static disorder, which for a metal the adiabatic Eliashberg plateau
pi * int alpha^2F(w) (2n(w) + 1) dw of ../superconductivity/output/eliashberg.npz reproduces.
Several seeds of the sign search give an ensemble of cells per temperature, averaged here; the
undisplaced supercell (written with the first seed) measures the offset of the supercell PAO
bands from the primitive ones.

Outputs: zg/seed<s>/<prefix>_zg_T<T>K.{in,npz} with zg.json and zg/run_zg.sh; output/
thermal_spectral_T<T>K_seed<s>.npz per cell, thermal_spectral_T<T>K.npz (ensemble mean),
thermal_spectral_reference[_seed<s>].npz, thermal_summary.{json,dat}, spectral_<label>.png,
thermal_summary.png and dielectric_T<T>K_seed<s>/ (epsr_*.dat, epsi_*.dat).
"""

import argparse
import glob
import os

import numpy as np

from PAOFLOW.elphon.workflows import (
    is_root,
    load_system,
    peak_width_at_crossing,
    phonon_dispersion,
    plot_thermal_spectral_function,
    thermal_dielectric,
    thermal_spectral_function,
)
from PAOFLOW.elphon.workflows.thermal import (
    band_statistics,
    ensemble_spectral_function,
    load_thermal_spectral,
    load_thermal_summary,
    plot_thermal_summary,
    save_thermal_spectral,
    static_linewidth,
    symmetry_points,
    write_thermal_summary,
    write_zg_run_script,
    zg_cells,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'output')
ZG = os.path.join(HERE, 'zg')
CONFIG = load_system(HERE)
SYSTEM = CONFIG.SYSTEM
DEFAULTS = getattr(CONFIG, 'THERMAL', {})
NAME = getattr(CONFIG, 'NAME', SYSTEM.prefix)
PHONONS_OUTPUT = os.path.join(os.path.dirname(HERE), 'phonons', 'output')
ELIASHBERG = os.path.join(os.path.dirname(HERE), 'superconductivity', 'output', 'eliashberg.npz')
SUMMARY = os.path.join(OUTPUT, 'thermal_summary.json')


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--write', action='store_true', help='ZG supercell inputs and zg/run_zg.sh')
    parser.add_argument(
        '--spectral', action='store_true', help='unfolded spectral functions of the ZG saves'
    )
    parser.add_argument(
        '--dielectric', action='store_true', help='dielectric function of the ZG saves'
    )
    parser.add_argument('--plot', action='store_true', help='figures from the output files')
    parser.add_argument(
        '--temps', type=float, nargs='+', default=DEFAULTS.get('temps_k', [0.0, 300.0]),
        help='temperatures (K)',
    )  # fmt: skip
    parser.add_argument(
        '--supercell', type=int, nargs=3, default=DEFAULTS.get('supercell', (3, 3, 3)),
        help='ZG supercell (diagonal, in primitive cells)',
    )  # fmt: skip
    parser.add_argument(
        '--kgrid', type=int, nargs=3, default=DEFAULTS.get('kgrid'),
        help='k-grid of the supercell scf (default: the coarse grid divided by the supercell)',
    )  # fmt: skip
    parser.add_argument(
        '--nbnd-per-atom', type=int, default=DEFAULTS.get('nbnd_per_atom', 12),
        help='bands per atom of the supercell scf',
    )  # fmt: skip
    parser.add_argument(
        '--force-constants', default=DEFAULTS.get('force_constants'),
        help='q2r.x force constants (default ../phonons/output/<prefix>.fc)',
    )  # fmt: skip
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=DEFAULTS.get('seeds', [0]),
        help='seeds of the sign search: one cell per seed and temperature',
    )  # fmt: skip
    parser.add_argument(
        '--error-thresh', type=float, default=DEFAULTS.get('error_thresh', 0.05),
        help='accepted relative error of the mean-square displacements',
    )  # fmt: skip
    parser.add_argument(
        '--no-reference', action='store_true', help='skip the undisplaced supercell'
    )
    parser.add_argument(
        '--path', default=DEFAULTS.get('path', 'G-X'), help='k-path of the spectral function'
    )
    parser.add_argument(
        '--nk-per-seg', type=int, default=DEFAULTS.get('nk_per_seg', 100),
        help='k-points per path segment',
    )  # fmt: skip
    parser.add_argument(
        '--window', type=float, nargs=3, default=DEFAULTS.get('energy_window', (-0.3, 0.3, 601)),
        metavar=('EMIN', 'EMAX', 'N'), help='energy window (eV, relative to E_F) and points',
    )  # fmt: skip
    parser.add_argument(
        '--broadening-ev', type=float, default=DEFAULTS.get('broadening_ev', 0.005),
        help='Gaussian width of the spectral function (eV)',
    )  # fmt: skip
    parser.add_argument(
        '--crossing-ev', type=float, default=DEFAULTS.get('crossing_ev', 0.0),
        help='energy (eV) at which the width of the crossing band is measured',
    )  # fmt: skip
    parser.add_argument(
        '--half-window-ev', type=float, default=DEFAULTS.get('half_window_ev', 0.25),
        help='half width (eV) of the window of the peak and moment analysis',
    )  # fmt: skip
    return parser.parse_args()


def cell_label(cell):
    """Output label of a ZG cell: ``T<T>K_seed<s>`` or ``reference_seed<s>``."""
    if cell['temperature_k'] is None:
        return 'reference_seed%d' % cell['seed']
    return 'T%gK_seed%d' % (cell['temperature_k'], cell['seed'])


def write_cells(args):
    """ZG supercell inputs of every seed and temperature (+ the reference cell) and run_zg.sh."""
    from PAOFLOW.PAOFLOW import PAOFLOW

    SYSTEM.check_inputs()
    fc = SYSTEM.path(args.force_constants or os.path.join(PHONONS_OUTPUT, SYSTEM.prefix + '.fc'))
    if not os.path.isfile(fc):
        if is_root():
            print('%s not found: q2r of the ph.x dynamical matrices first (phonons.py)' % fc)
        phonon_dispersion(SYSTEM, PHONONS_OUTPUT)
    pp_dir = SYSTEM.path(DEFAULTS.get('pp_dir') or SYSTEM.qe_dir)
    reference = DEFAULTS.get('reference', True) and not args.no_reference
    dims = tuple(int(n) for n in args.supercell)
    for seed in args.seeds:
        pf = PAOFLOW(savedir=SYSTEM.save_path, outputdir=ZG, verbose=False)
        _, attr = pf.data_controller.data_dicts()
        nbnd = int(args.nbnd_per_atom * attr['natoms'] * np.prod(dims))
        pf.special_displacement(
            args.temps, supercell_matrix=dims, force_constants_file=fc, zg_dir='seed%d' % seed,
            pp_dir=pp_dir, prefix=SYSTEM.prefix, kgrid=args.kgrid, nbnd=nbnd,
            error_thresh=args.error_thresh, seed=seed,
            reference=reference and seed == args.seeds[0],
        )  # fmt: skip
    if is_root():
        script = write_zg_run_script(ZG)
        print(
            '%s: %d^3 ZG supercell, %d seeds x %d temperatures%s; next: (cd zg && NP=8 bash %s)'
            % (NAME, dims[0], len(args.seeds), len(args.temps),
               ' + the reference cell' if reference else '', os.path.basename(script))
        )  # fmt: skip


def spectral(args):
    """Unfolded spectral function of every ZG save, the ensemble means and the summary table."""
    cells = zg_cells(ZG)
    if not cells:
        raise SystemExit('no ZG cells under %s: run zg.py --write and zg/run_zg.sh first' % ZG)
    todo = [c for c in cells if c['save']]
    missing = [c['prefix'] for c in cells if not c['save']]
    if is_root():
        print(
            '%s: %d ZG cells (%s supercell, %d atoms), %d with a save%s'
            % (NAME, len(cells), 'x'.join(str(n) for n in cells[0]['dims']), cells[0]['natoms'],
               len(todo), '; not run yet: ' + ', '.join(missing) if missing else '')
        )  # fmt: skip
    points = symmetry_points(SYSTEM, DEFAULTS.get('sym_points'))
    window = (float(args.window[0]), float(args.window[1]), int(args.window[2]))
    groups = {}
    for cell in todo:
        label = cell_label(cell)
        path = os.path.join(OUTPUT, 'thermal_spectral_%s.npz' % label)
        if os.path.isfile(path):
            result = load_thermal_spectral(path)
        else:
            result = thermal_spectral_function(
                SYSTEM, cell['save'], args.path, points, temperature_k=cell['temperature_k'],
                nk_per_seg=args.nk_per_seg, energy_window=window,
                broadening_ev=args.broadening_ev, configuration=SYSTEM.configuration,
                pthr=SYSTEM.pthr, outdir=OUTPUT, label=label,
            )  # fmt: skip
        groups.setdefault(cell['temperature_k'], []).append(result)
    if not is_root():
        return
    temps = sorted(t for t in groups if t is not None)
    gamma = static_linewidth(ELIASHBERG, temps) if temps and os.path.isfile(ELIASHBERG) else None
    rows = []
    for temp in ([None] if None in groups else []) + temps:
        group = groups[temp]
        label = 'reference' if temp is None else 'T%gK' % temp
        mean = ensemble_spectral_function(group, label=label)
        save_thermal_spectral(mean, os.path.join(OUTPUT, 'thermal_spectral_%s.npz' % label))
        peak = peak_width_at_crossing(
            mean, energy_ev=args.crossing_ev, half_window_ev=args.half_window_ev
        )
        stats = band_statistics(mean, band=peak['band'], half_window_ev=args.half_window_ev)
        rows.append(
            dict(
                temperature_k=temp, label=label, configurations=len(group), band=peak['band'],
                kdist=peak['kdist'], E_pc_ev=peak['E_pc_ev'], peak_ev=peak['peak_ev'],
                peak_shift_ev=peak['shift_ev'], fwhm_ev=peak['fwhm_ev'],
                weight=peak['weight'], shift_ev=stats['shift_mean_ev'],
                shift_rms_ev=stats['shift_rms_ev'], spread_ev=stats['spread_mean_ev'],
                spread_intrinsic_ev=stats['spread_intrinsic_ev'],
                sum_rule=float(np.mean(mean['sum_rule'])),
                gamma_static_ev=None if gamma is None or temp is None else float(gamma[temps.index(temp)]),
            )
        )  # fmt: skip
    reference = next((r for r in rows if r['temperature_k'] is None), None)

    def versus_reference(row, key):  # thermal shifts referred to the undisplaced supercell
        if reference is None or reference[key] is None or row[key] is None:
            return None
        return row[key] - reference[key]

    for row in rows:
        row['peak_vs_reference_ev'] = versus_reference(row, 'peak_shift_ev')
        row['shift_vs_reference_ev'] = versus_reference(row, 'shift_ev')
    write_thermal_summary(rows, SUMMARY)
    print(
        '  unfolded along %s (%d points per segment, %d bands, broadening %.0f meV, '
        'window +-%.0f meV at the crossing%s)'
        % (args.path, args.nk_per_seg, mean['E_pc'].shape[1], 1e3 * args.broadening_ev,
           1e3 * args.half_window_ev,
           '; Gamma_static = pi int alpha^2F (2n+1) from %s' % os.path.relpath(ELIASHBERG, HERE)
           if gamma is not None else '')
    )  # fmt: skip
    print(
        '    T (K)   cells  band  crossing state - E_pc (meV)  vs ref (meV)  <shift> along path (meV)'
        '  vs ref (meV)  FWHM (meV)  spread (meV)  weight  sum rule'
        + ('  2 Gamma_static (meV)' if gamma is not None else '')
    )
    for row in rows:
        fmt = lambda x, s='%7.1f': '    n/a' if x is None else s % (1e3 * x)  # noqa: E731
        print(
            '    %-7s %5d  %4s  %s  %s  %s  %s  %s  %s  %6.2f  %8.3f%s'
            % ('ref' if row['temperature_k'] is None else '%g' % row['temperature_k'],
               row['configurations'], '-' if row['band'] is None else row['band'],
               fmt(row['peak_shift_ev'], '%27.1f'), fmt(row['peak_vs_reference_ev'], '%12.1f'),
               fmt(row['shift_ev'], '%23.1f'), fmt(row['shift_vs_reference_ev'], '%12.1f'),
               fmt(row['fwhm_ev'], '%10.1f'), fmt(row['spread_ev'], '%12.1f'),
               row['weight'] or 0.0, row['sum_rule'],
               '' if row['gamma_static_ev'] is None else '  %19.1f' % (2e3 * row['gamma_static_ev']))
        )  # fmt: skip
    print('  wrote %s' % SUMMARY)


def dielectric(args):
    """Dielectric function of every ZG cell (the Williams-Lax epsilon(omega; T))."""
    cells = [c for c in zg_cells(ZG) if c['save'] and c['temperature_k'] is not None]
    if not cells:
        raise SystemExit('no ZG saves under %s' % ZG)
    settings = dict(DEFAULTS.get('dielectric') or {})
    for cell in cells:
        outdir = os.path.join(OUTPUT, 'dielectric_%s' % cell_label(cell))
        if is_root():
            print('%s: dielectric function of %s -> %s' % (NAME, cell['prefix'], outdir))
        thermal_dielectric(
            SYSTEM, cell['save'], outdir, configuration=SYSTEM.configuration, **settings
        )


def plot(args):
    """spectral_<label>.png of every ensemble mean (and the reference) and thermal_summary.png."""
    files = sorted(glob.glob(os.path.join(OUTPUT, 'thermal_spectral_T*K.npz')))
    reference = os.path.join(OUTPUT, 'thermal_spectral_reference.npz')
    if os.path.isfile(reference):
        files.insert(0, reference)
    for path in files:
        result = load_thermal_spectral(path)
        figure = os.path.join(OUTPUT, 'spectral_%s.png' % result['label'])
        plot_thermal_spectral_function(result, figure)
        print('  wrote %s' % figure)
    if os.path.isfile(SUMMARY):
        figure = plot_thermal_summary(
            load_thermal_summary(SUMMARY), os.path.join(OUTPUT, 'thermal_summary.png'), name=NAME
        )
        print('  wrote %s' % figure)


def main():
    args = parse_arguments()
    actions = [a for a in ('write', 'spectral', 'dielectric', 'plot') if getattr(args, a)]
    if not actions:
        written = glob.glob(os.path.join(ZG, 'seed*', 'zg.json'))
        actions = ['spectral', 'plot'] if written else ['write']
    if 'write' in actions:
        write_cells(args)
    if 'spectral' in actions:
        spectral(args)
    if 'dielectric' in actions:
        dielectric(args)
    if 'plot' in actions and is_root():
        plot(args)


if __name__ == '__main__':
    main()
