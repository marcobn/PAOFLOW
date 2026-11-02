#!/usr/bin/env python3
"""Thermal band structures from ZG supercells: the special displacement method (Zacharias-Giustino).

    python zg.py --write                    # ZG supercell inputs + zg/run_zg.sh (THERMAL of system.py)
    (cd zg && NP=8 bash run_zg.sh)          # pw.x scf of every cell
    mpirun -np N python zg.py --spectral    # unfolded A(k, E; T): band shifts and widths vs T
    mpirun -np N python zg.py --dielectric  # dielectric function of every ZG cell
    python zg.py --plot                     # figures from the output files
    python zg.py --write --temps 0 300 --supercell 4 4 4 --seeds 0 1 2
    python zg.py --displace                 # displaced runs of every ZG cell (its own unit cell) + run_fd.sh
    python zg.py --iterate                  # thermal force constants of the cells -> the cells of the next iteration
    python zg.py --lambda                   # alpha^2F, lambda, Tc of every ZG cell ('fd' source; single rank)
    python zg.py --spectral --iteration 1   # any phase on the cells of iteration 1 (zg_iter1/, output_iter1/)

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

The anharmonic self-consistent loop (A-SDM): --displace writes, for every finished ZG cell, the
random (or Cartesian) displaced runs of the cell as its own unit cell (PAOFLOW.electron_phonon with
supercell_matrix=1, <cell>_fd/run_fd.sh); --iterate reads their forces, averages the local curvature
over the sub-cell translations (thermal force constants, output/thermal_fc_<label>.npz) and writes the
ZG cells of the next iteration under zg_iter<n>/; --lambda feeds the same displaced runs to the
dense-k Eliashberg driver as the finite-difference coupling source of the cell (Gamma only, the cell's
own modes): lambda(T), omega_log(T), Tc(T) in output/lambda_<label>/ and lambda_summary.dat.

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
    eliashberg,
    is_root,
    load_system,
    peak_width_at_crossing,
    phonon_dispersion,
    plot_thermal_spectral_function,
    thermal_dielectric,
    thermal_spectral_function,
    write_eliashberg,
)
from PAOFLOW.elphon.workflows.thermal import (
    band_statistics,
    cell_displacements,
    cell_force_constants,
    ensemble_spectral_function,
    fd_dir_of_cell,
    fd_runs_done,
    fd_system_for_cell,
    harmonic_frequencies,
    load_thermal_spectral,
    load_thermal_summary,
    plot_thermal_summary,
    save_thermal_spectral,
    static_linewidth,
    symmetry_points,
    unit_cell_of,
    write_fd_displacements,
    write_fd_run_script,
    write_next_zg_cell,
    write_thermal_summary,
    write_zg_run_script,
    zg_cells,
    zg_iteration_dir,
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
        '--displace', action='store_true',
        help='displaced runs of every finished ZG cell (the cell as its own unit cell) + run_fd.sh',
    )  # fmt: skip
    parser.add_argument(
        '--iterate', action='store_true',
        help='thermal force constants from the displaced runs -> the ZG cells of the next iteration',
    )  # fmt: skip
    parser.add_argument(
        '--lambda', dest='lambda_', action='store_true',
        help='alpha^2F, lambda and Tc of every ZG cell from its displaced runs (fd coupling source)',
    )  # fmt: skip
    parser.add_argument(
        '--iteration', type=int, default=0,
        help='cells of this A-SDM iteration (0: zg/, n: zg_iter<n>/ with output_iter<n>/)',
    )  # fmt: skip
    parser.add_argument(
        '--fd-mode', default=DEFAULTS.get('fd_mode', 'random'), choices=('random', 'cartesian'),
        help='displaced runs of a cell: random all-atom patterns (2 nconfig) or Cartesian (6 N)',
    )  # fmt: skip
    parser.add_argument(
        '--fd-nconfig', type=int, default=DEFAULTS.get('fd_nconfig', 60),
        help='random configurations per cell',
    )  # fmt: skip
    parser.add_argument(
        '--fd-distance', type=float, default=DEFAULTS.get('fd_distance', 0.03),
        help='displacement amplitude (bohr) of the displaced runs',
    )  # fmt: skip
    parser.add_argument(
        '--lambda-nk', type=int, default=DEFAULTS.get('lambda_nk', 12),
        help='dense k per axis of the supercell for lambda(T)',
    )  # fmt: skip
    parser.add_argument(
        '--lambda-sigma-ev', type=float, default=DEFAULTS.get('lambda_sigma_ev', 0.05),
        help='Fermi-surface smearing (eV) of lambda(T)',
    )  # fmt: skip
    parser.add_argument(
        '--mu-star',
        type=float,
        default=DEFAULTS.get('mu_star', 0.1),
        help='Coulomb pseudopotential',
    )
    parser.add_argument(
        '--ridge', type=float, default=DEFAULTS.get('ridge', 1.0e-4),
        help='ridge of the least-squares fits (random mode)',
    )  # fmt: skip
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


def thermal_cells(need_fd=False):
    """Finished thermal cells of the current iteration (optionally with finished displaced runs)."""
    cells = [c for c in zg_cells(ZG) if c['save'] and c['temperature_k'] is not None]
    if not cells:
        raise SystemExit('no finished ZG cells under %s: zg.py --write and zg/run_zg.sh first' % ZG)
    if need_fd:
        ready = []
        for cell in cells:
            done, total = fd_runs_done(fd_dir_of_cell(cell))
            if total and done == total:
                ready.append(cell)
            elif is_root():
                print('  %s: %d / %d displaced runs done (skipped)' % (cell['prefix'], done, total))
        if not ready:
            raise SystemExit(
                'no cell with finished displaced runs: zg.py --displace and run_fd.sh first'
            )
        cells = ready
    return cells


def displace(args):
    """Displaced runs of every finished ZG cell (the cell as its own unit cell) and run_fd.sh."""
    pp_dir = SYSTEM.path(DEFAULTS.get('pp_dir') or SYSTEM.qe_dir)
    for cell in thermal_cells():
        fd_dir = write_fd_displacements(
            cell, distance=args.fd_distance, mode=args.fd_mode, nconfig=args.fd_nconfig,
            seed=cell['seed'], pp_dir=pp_dir, configuration=SYSTEM.configuration,
        )  # fmt: skip
        if is_root():
            script = write_fd_run_script(fd_dir)
            done, total = fd_runs_done(fd_dir)
            print(
                '%s: %s -> %d displaced cells (%s) + reference under %s; next: (cd %s && NP=8 bash %s)'
                % (NAME, cell['prefix'], total - 1, args.fd_mode, os.path.relpath(fd_dir, HERE),
                   os.path.relpath(fd_dir, HERE), os.path.basename(script))
            )  # fmt: skip


def iterate(args):
    """Thermal force constants of the cells with finished displaced runs; the next ZG cells."""
    cells = thermal_cells(need_fd=True)
    if not is_root():
        return
    os.makedirs(OUTPUT, exist_ok=True)
    unitcell = unit_cell_of(SYSTEM)
    fc_file = SYSTEM.path(
        args.force_constants or os.path.join(PHONONS_OUTPUT, SYSTEM.prefix + '.fc')
    )
    harmonic = (
        harmonic_frequencies(unitcell, fc_file, cells[0]['dims'])
        if os.path.isfile(fc_file)
        else None
    )
    pp_dir = SYSTEM.path(DEFAULTS.get('pp_dir') or SYSTEM.qe_dir)
    next_dir = zg_iteration_dir(HERE, args.iteration + 1)
    for cell in cells:
        label = cell_label(cell)
        result = cell_force_constants(cell, unitcell, ridge=args.ridge)
        np.savez(os.path.join(OUTPUT, 'thermal_fc_%s.npz' % label), **result)
        freq = result['frequencies_thz']
        note = ''
        if harmonic is not None:
            keep = np.abs(harmonic) > 0.05
            ratio = freq[keep] / harmonic[keep] - 1.0
            note = '; frequencies vs harmonic: mean %+.1f %%, max %+.1f %%' % (
                100 * ratio.mean(),
                100 * ratio[np.argmax(np.abs(ratio))],
            )
        written = write_next_zg_cell(
            SYSTEM, cell, result['force_constants'], next_dir, pp_dir=pp_dir,
            error_thresh=args.error_thresh,
        )  # fmt: skip
        old = cell_displacements(cell)
        new_cell = next(
            (
                c
                for c in zg_cells(next_dir)
                if c['seed'] == cell['seed'] and c['temperature_k'] == cell['temperature_k']
            ),
            None,
        )
        new = cell_displacements(new_cell) if new_cell else None
        rms = lambda u: float(np.sqrt((u**2).sum(axis=1).mean()))  # noqa: E731
        change = (
            ''
            if old is None or new is None
            else '; rms displacement %.4f -> %.4f bohr (%+.1f %%)'
            % (rms(old), rms(new), 100 * (rms(new) / rms(old) - 1))
        )
        print(
            '%s: %s thermal force constants%s%s -> %s'
            % (NAME, label, note, change, ', '.join(os.path.relpath(w, HERE) for w in written))
        )
    script = write_zg_run_script(next_dir)
    print('iteration %d cells under %s; next: (cd %s && NP=8 bash %s), then zg.py --iteration %d ...'
          % (args.iteration + 1, os.path.relpath(next_dir, HERE), os.path.relpath(next_dir, HERE),
             os.path.basename(script), args.iteration + 1))  # fmt: skip


def lambda_cells(args):
    """alpha^2F, lambda, omega_log and Tc of every cell with finished displaced runs."""
    cells = thermal_cells(need_fd=True)
    rows = []
    for cell in cells:
        label = cell_label(cell)
        outdir = os.path.join(OUTPUT, 'lambda_%s' % label)
        fd_system = fd_system_for_cell(SYSTEM, cell, fd_options=dict(ridge=args.ridge))
        fd_system.check_inputs()
        electrons = fd_system.electrons(outputdir=outdir)
        out = eliashberg(
            fd_system, electrons, nk=args.lambda_nk, nq=1, sigma_ev=args.lambda_sigma_ev,
            mu_star=args.mu_star,
        )  # fmt: skip
        if is_root():
            write_eliashberg(
                out, outdir, mu_star=args.mu_star,
                label='%s, k %d^3 (supercell), q Gamma, sigma %g eV' % (label, args.lambda_nk, args.lambda_sigma_ev),
            )  # fmt: skip
            rows.append(
                (
                    cell['temperature_k'],
                    cell['seed'],
                    out['lambda'],
                    out['omega_log'],
                    out['Tc_allen_dynes'],
                )
            )
    if is_root() and rows:
        path = os.path.join(OUTPUT, 'lambda_summary.dat')
        with open(path, 'w') as handle:
            handle.write(
                '# T(K)  seed  lambda  omega_log(meV)  Tc_AD(K)   [%s, k %d^3, sigma %g eV]\n'
                % (NAME, args.lambda_nk, args.lambda_sigma_ev)
            )
            for T, seed, lam, wlog, tc in rows:
                handle.write('%8.2f  %4d  %8.4f  %10.3f  %8.3f\n' % (T, seed, lam, 1e3 * wlog, tc))
        print('%s: lambda(T) of %d cells -> %s' % (NAME, len(rows), os.path.relpath(path, HERE)))
        for T, seed, lam, wlog, tc in rows:
            print(
                '  T %7.2f K seed %d: lambda %.4f, omega_log %.3f meV, Tc(AD) %.3f K'
                % (T, seed, lam, 1e3 * wlog, tc)
            )


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
    # The electrons of a thermal cell are at its lattice temperature: Fermi-Dirac occupations
    # at T (k_B T = 0.39 eV at 4500 K) unless electronic_temperature=False in the settings.
    electronic = settings.pop('electronic_temperature', True)
    for cell in cells:
        outdir = os.path.join(OUTPUT, 'dielectric_%s' % cell_label(cell))
        if is_root():
            print('%s: dielectric function of %s -> %s' % (NAME, cell['prefix'], outdir))
        options = dict(settings)
        if electronic and 'temperature_k' not in options:
            options['temperature_k'] = float(cell['temperature_k'])
        thermal_dielectric(
            SYSTEM, cell['save'], outdir, configuration=SYSTEM.configuration, **options
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
    global ZG, OUTPUT, SUMMARY
    args = parse_arguments()
    if args.iteration > 0:  # the cells of a later A-SDM iteration and their own output folder
        ZG = zg_iteration_dir(HERE, args.iteration)
        OUTPUT = os.path.join(HERE, 'output_iter%d' % args.iteration)
        SUMMARY = os.path.join(OUTPUT, 'thermal_summary.json')
    actions = [
        a
        for a in ('write', 'displace', 'iterate', 'lambda_', 'spectral', 'dielectric', 'plot')
        if getattr(args, a)
    ]
    if not actions:
        written = glob.glob(os.path.join(ZG, 'seed*', 'zg.json'))
        actions = ['spectral', 'plot'] if written else ['write']
    if 'write' in actions:
        write_cells(args)
    if 'displace' in actions:
        displace(args)
    if 'iterate' in actions:
        iterate(args)
    if 'lambda_' in actions:
        lambda_cells(args)
    if 'spectral' in actions:
        spectral(args)
    if 'dielectric' in actions:
        dielectric(args)
    if 'plot' in actions and is_root():
        plot(args)


if __name__ == '__main__':
    main()
