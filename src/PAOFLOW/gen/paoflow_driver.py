#!/usr/bin/env python3
"""Interactive generator for PAOFLOW ``main.py`` driver scripts.

Given the output of a Quantum ESPRESSO run (a ``<prefix>.save`` directory) and
the pseudopotential used for it, this CLI asks a few questions and writes a
minimal, clearly-commented ``main.py`` that runs PAOFLOW.

Two workflows are supported:

1. **ACBN0 / eACBN0** self-consistent Hubbard U (and intersite V) using the
   standard basis for the PAO projections.
2. **Full property run** from the QE output, where the user picks the
   properties to compute from a menu.  Standard properties use the *standard*
   basis; optical properties (dielectric tensor) need the *extended* basis and
   the non-local velocity correction, so they are emitted as a **separate**
   PAOFLOW run in the same script.

Two further workflows generate multi-phase driver scripts:

3. **Harmonic phonons** (``main.phonon.py``) -- finite-displacement phonons via
   phonopy (generate supercells -> run pw.x -> analyse forces).
4. **Electron-phonon** (``main.elphon.py``) -- the pseudo-atomic-orbital (Agapito &
   Bernardi) interpolation of QE's DFPT coupling into ``alpha^2F`` / ``lambda`` /
   ``Tc`` (write the ph.x phonon + AHC inputs -> run QE -> analyse).

The generated script is static and heavily commented so it is easy to tweak
afterwards.
"""

import argparse
import glob
import os
import re
import sys


# --------------------------------------------------------------------------- #
# Interactive prompt helpers
# --------------------------------------------------------------------------- #
def _input(prompt):
    """input() that returns an empty string on EOF (non-interactive use)."""
    try:
        return input(prompt)
    except EOFError:
        return ''


def ask(prompt, default=None):
    """Ask for a free-text value with an optional default."""
    suffix = ' [{}]'.format(default) if default is not None else ''
    ans = _input('{}{}: '.format(prompt, suffix)).strip()
    return ans if ans else (default if default is not None else '')


def ask_yes_no(prompt, default=False):
    """Ask a yes/no question."""
    d = 'Y/n' if default else 'y/N'
    ans = _input('{} [{}]: '.format(prompt, d)).strip().lower()
    if not ans:
        return default
    return ans.startswith('y')


def ask_int(prompt, default):
    """Ask for an integer value with a default."""
    ans = _input('{} [{}]: '.format(prompt, default)).strip()
    if not ans:
        return default
    try:
        return int(ans)
    except ValueError:
        print('  (not an integer, using {})'.format(default))
        return default


def ask_float(prompt, default):
    """Ask for a floating-point value with a default."""
    ans = _input('{} [{}]: '.format(prompt, default)).strip()
    if not ans:
        return default
    try:
        return float(ans)
    except ValueError:
        print('  (not a number, using {})'.format(default))
        return default


def ask_choice(prompt, choices, default):
    """Ask the user to pick one item from *choices* (list of strings)."""
    while True:
        print(prompt)
        for i, choice in enumerate(choices, 1):
            mark = ' (default)' if choice == default else ''
            print('  {}: {}{}'.format(i, choice, mark))
        ans = _input('Choice [{}]: '.format(default)).strip()
        if not ans:
            return default
        if ans in choices:
            return ans
        try:
            idx = int(ans)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except ValueError:
            pass
        print('  (invalid choice, try again)')


def parse_laser_list(spec):
    """Parse a laser-wavelength specification into a list of floats (nm).

    Accepts a comma- or space-separated list (``'488, 514.5, 532'``) or a
    restricted Python expression that evaluates to an iterable of numbers, e.g.
    ``'[n for n in range(450, 650, 5)]'``.  The expression is evaluated with no
    builtins and only ``range`` exposed, so it cannot import modules or call
    arbitrary functions.
    """
    spec = (spec or '').strip()
    if not spec:
        return []
    if any(ch in spec for ch in '[]()') or 'range' in spec or ' for ' in spec:
        try:
            value = eval(spec, {'__builtins__': {}}, {'range': range})  # noqa: S307
        except Exception as exc:  # pragma: no cover - user input error path
            raise ValueError('Could not parse laser list %r: %s' % (spec, exc))
        values = [float(x) for x in value]
    else:
        values = [float(x) for x in spec.replace(',', ' ').split() if x.strip()]
    if not values:
        raise ValueError('No laser wavelengths parsed from %r' % spec)
    return values


# --------------------------------------------------------------------------- #
# Auto-detection of QE artifacts in the working directory
# --------------------------------------------------------------------------- #
def detect_savedir(workdir):
    matches = sorted(d for d in glob.glob(os.path.join(workdir, '*.save')) if os.path.isdir(d))
    return os.path.basename(matches[0]) if matches else None


def detect_upfs(workdir, savedir=None):
    # Per the QE standard, pseudopotentials are copied into the
    # ``<prefix>.save`` directory; look there first, then fall back to workdir.
    # Returns the list of every pseudopotential found (one per species).
    if savedir:
        save_path = os.path.join(workdir, savedir)
        save_matches = sorted(
            glob.glob(os.path.join(save_path, '*.UPF'))
            + glob.glob(os.path.join(save_path, '*.upf'))
        )
        if save_matches:
            return [os.path.join(savedir, os.path.basename(m)) for m in save_matches]
    matches = sorted(
        glob.glob(os.path.join(workdir, '*.UPF')) + glob.glob(os.path.join(workdir, '*.upf'))
    )
    return [os.path.basename(m) for m in matches]


def detect_prefix(workdir, savedir):
    if savedir:
        return os.path.splitext(os.path.basename(savedir))[0]
    for name in ('scf.in', 'nscf.in'):
        path = os.path.join(workdir, name)
        if os.path.isfile(path):
            prefix = _read_prefix_from_input(path)
            if prefix:
                return prefix
    return 'pwscf'


def _read_prefix_from_input(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                low = line.strip().lower()
                if low.startswith('prefix'):
                    _, _, val = line.partition('=')
                    return val.strip().strip(',').strip().strip('\'"')
    except OSError:
        return None
    return None


def detect_v_cutoff(workdir):
    """Read the suggested eACBN0 intersite-V cutoff from a QE input header.

    ``paoflow-gen-qe`` writes ``! Suggested eACBN0 intersite V cutoff: <x>
    Angstrom`` into the header of the ``<compound>.scf.in`` it generates.
    Returns that cutoff (in Angstrom) from the first matching input file, or
    ``None`` if no input file or comment is found.
    """
    candidates = sorted(glob.glob(os.path.join(workdir, '*.scf.in')))
    candidates += [os.path.join(workdir, n) for n in ('scf.in', 'nscf.in')]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                for line in handle:
                    if 'intersite v cutoff' in line.lower():
                        match = re.search(r'([0-9]+\.?[0-9]*)\s*Angstrom', line, re.IGNORECASE)
                        if match:
                            return float(match.group(1))
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Shared script fragments
# --------------------------------------------------------------------------- #
HEADER = '''#!/usr/bin/env python3
"""PAOFLOW driver script (generated by paoflow_gen.py).

Edit the constants below and the property calls in the run function(s) to
customize the calculation.  Run with:

    python main.py
    # or, in parallel:
    mpirun -np <N> python main.py
"""

import os
import sys

from PAOFLOW import PAOFLOW
from PAOFLOW.basis_gen import generate_basis_for_pseudo
from PAOFLOW.basis_gen.driver import _default_shells
from PAOFLOW.inputs.read_upf import UPF as _UPFParser

try:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    RANK = 0
'''

BASIS_GEN_BLOCK = '''
def ensure_basis(preset="extended"):
    """Generate the pseudo-atom basis under BASISPATH for every species.

    The 'extended' preset is a superset of 'standard' and 'minimal', so
    generating it once is enough for every configuration used below.
    """
    if RANK != 0:
        return
    for upf_path in UPFS:
        upf = _UPFParser(upf_path)
        element = upf.element.strip()
        elem_dir = os.path.join(BASISPATH, element)
        expected = _default_shells(upf, preset=preset)
        missing = [
            s for s in expected
            if not os.path.exists(os.path.join(elem_dir, "{}.dat".format(s)))
        ]
        if missing:
            print("Generating pseudo-atom basis for {} under {} ...".format(
                element, BASISPATH))
            generate_basis_for_pseudo(
                upf_path, BASISPATH.rstrip(os.sep), preset=preset, verbose=True
            )
        else:
            print("Using existing pseudo-atom basis for {} under {}".format(
                element, BASISPATH))
'''

ENERGY_RANGE_BLOCK = '''
def suggest_energy_window(p):
    """Print the full PAO band range (eV, rel. to E_F) as a suggested window.

    Must be called after p.pao_eigh().  The eigenvalues ('E_k') are distributed
    across MPI ranks, so the local extrema are reduced to give every process the
    same global range.  This only prints a hint; the calculation uses the
    user-defined EMIN / EMAX / NE constants above.
    """
    import numpy as np

    E_k = p.data_controller.data_arrays['E_k']
    emin = float(np.amin(E_k))
    emax = float(np.amax(E_k))
    if "MPI" in globals():
        comm = MPI.COMM_WORLD
        emin = comm.allreduce(emin, op=MPI.MIN)
        emax = comm.allreduce(emax, op=MPI.MAX)
    if RANK == 0:
        print('Suggested energy window from PAO bands: '
              'EMIN={:.4f}, EMAX={:.4f} eV'.format(emin, emax))
        print('  (currently using EMIN={}, EMAX={}, NE={})'.format(EMIN, EMAX, NE))
    return emin, emax
'''


# --------------------------------------------------------------------------- #
# Property registry for the full run
# --------------------------------------------------------------------------- #
# Each property: menu label and a flag for whether it needs the spin operator.
PROPERTY_MENU = [
    ('bands', 'Band structure'),
    ('dos', 'DOS / projected DOS'),
    ('transport', 'Boltzmann transport (conductivity, Seebeck)'),
    ('fermi_surface', 'Fermi surface'),
    ('spin_texture', 'Spin texture'),
    ('spin_Hall', 'Spin Hall conductivity'),
    ('anomalous_Hall', 'Anomalous Hall / Berry curvature'),
    ('topology', 'Band topology (Berry, effective mass)'),
    ('optical', 'Optical / dielectric tensor (separate extended-basis run)'),
    ('emissivity', 'Emissivity (Fresnel/Kirchhoff; implies the optical run)'),
]
SPIN_PROPERTIES = {'spin_texture', 'spin_Hall'}
# Properties that read an explicit (emin, emax) energy window.
ENERGY_WINDOW_PROPERTIES = {'dos', 'transport', 'spin_Hall', 'anomalous_Hall'}


def select_properties():
    """Show the property menu and return the ordered set of chosen keys."""
    print('\nAvailable properties:')
    for i, (_key, label) in enumerate(PROPERTY_MENU, 1):
        print('  {}: {}'.format(i, label))
    raw = _input("Enter a comma/space separated list (e.g. '1 2 8'): ").strip()
    tokens = raw.replace(',', ' ').split()
    chosen = []
    for tok in tokens:
        try:
            idx = int(tok)
        except ValueError:
            continue
        if 1 <= idx <= len(PROPERTY_MENU):
            key = PROPERTY_MENU[idx - 1][0]
            if key not in chosen:
                chosen.append(key)
    return chosen


# --------------------------------------------------------------------------- #
# Script builders
# --------------------------------------------------------------------------- #
def _format_upfs_line(upfs):
    """Render the ``UPFS = [...]`` constant listing one pseudo per species."""
    items = ', '.join('os.path.join(HERE, {!r})'.format(u) for u in upfs)
    return 'UPFS = [{}]'.format(items)


# In-plane high-symmetry band paths for 2D systems (vacuum along c, kz = 0).
# Keyed by the in-plane Bravais lattice (the QE ibrav used by detect_ibrav_2d):
# 4 hexagonal, 6 square (tetragonal), 8 rectangular (orthorhombic),
# 9 centred-rectangular, 12 oblique (monoclinic).  Only the kz = 0 points of
# the corresponding 3D path are retained.
_BAND_PATH_2D = {
    4: (
        'gG-M-K-gG',
        {'gG': (0.0, 0.0, 0.0), 'M': (0.5, 0.0, 0.0), 'K': (1.0 / 3.0, 1.0 / 3.0, 0.0)},
    ),
    6: ('gG-X-M-gG', {'gG': (0.0, 0.0, 0.0), 'X': (0.0, 0.5, 0.0), 'M': (0.5, 0.5, 0.0)}),
    8: (
        'gG-X-S-Y-gG',
        {'gG': (0.0, 0.0, 0.0), 'X': (0.5, 0.0, 0.0), 'S': (0.5, 0.5, 0.0), 'Y': (0.0, 0.5, 0.0)},
    ),
    9: (
        'gG-X-S-Y-gG',
        {'gG': (0.0, 0.0, 0.0), 'X': (0.5, 0.0, 0.0), 'S': (0.5, 0.5, 0.0), 'Y': (0.0, 0.5, 0.0)},
    ),
    12: ('X-gG-Y', {'gG': (0.0, 0.0, 0.0), 'X': (0.5, 0.0, 0.0), 'Y': (0.0, 0.5, 0.0)}),
}


def _band_path_2d(ibrav):
    """Return ``(band_path, high_sym_points)`` for a 2D in-plane path.

    Returns ``(None, None)`` when there is no built-in 2D path for *ibrav*.
    """
    try:
        return _BAND_PATH_2D[int(ibrav)]
    except (KeyError, ValueError, TypeError):
        return None, None


def _format_high_sym(high_sym):
    """Render a HIGH_SYM dict literal for the generated script."""
    items = ', '.join('{!r}: ({}, {}, {})'.format(k, *v) for k, v in high_sym.items())
    return '{' + items + '}'


def _wants_explicit_band_path(cfg):
    """True when the script must pass an explicit band_path/high_sym to bands()."""
    return cfg['ibrav'] == 0 or bool(cfg.get('is_2d'))


def _band_path_constant_lines(cfg):
    """Source lines defining BAND_PATH / HIGH_SYM (empty list when not needed)."""
    ibrav = cfg['ibrav']
    if ibrav == 0:
        return [
            '# ibrav=0: PAOFLOW needs an explicit band path. Fill these in:',
            "# TODO: list the high-symmetry labels along the path, e.g. 'gG-X-W-K-gG-L'.",
            'BAND_PATH = None',
            "# TODO: map each label to its crystal-coordinate k-point, e.g. {'gG': [0, 0, 0], ...}.",
            'HIGH_SYM = None',
        ]
    if cfg.get('is_2d'):
        band_path, high_sym = _band_path_2d(ibrav)
        if band_path is None:
            return [
                '# 2D system: restrict the band path to the in-plane (kz=0) points.',
                '# No built-in 2D path for ibrav={}; fill these in:'.format(ibrav),
                'BAND_PATH = None',
                'HIGH_SYM = None',
            ]
        return [
            '# 2D system: in-plane band path only (vacuum along c, so kz=0).',
            'BAND_PATH = {!r}'.format(band_path),
            'HIGH_SYM = {}'.format(_format_high_sym(high_sym)),
        ]
    return []


def build_run_script(cfg):
    """Assemble a full property-run main.py from the collected config."""
    props = cfg['properties']
    standard_props = [p for p in props if p not in ('optical', 'emissivity')]
    has_optical = 'optical' in props or 'emissivity' in props
    has_emissivity = 'emissivity' in props
    needs_spin = any(p in SPIN_PROPERTIES for p in props)
    needs_energy_window = any(p in ENERGY_WINDOW_PROPERTIES for p in props)

    lines = [HEADER]

    # ---- editable constants ------------------------------------------- #
    lines.append('')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('# Configuration  (edit freely)                                            #')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('HERE = os.path.dirname(os.path.abspath(__file__))')
    lines.append('SAVEDIR = os.path.join(HERE, {!r})'.format(cfg['savedir']))
    lines.append(_format_upfs_line(cfg['upfs']))
    lines.append('BASISPATH = os.path.join(HERE, {!r}) + os.sep'.format(cfg['basisdir']))
    lines.append('OUTPUTDIR = {!r}'.format(cfg['outputdir']))
    lines.append('')
    lines.append('NPOOL = {}'.format(cfg['npool']))
    lines.append('SMEARING = {!r}'.format(cfg['smearing']))
    lines.append(
        'SPIN_ORBIT = {}   # set True for spin-orbit (noncollinear) runs'.format(cfg['spin_orbit'])
    )
    lines.append(
        'STD_BASIS = {!r}   # basis configuration for standard properties'.format(cfg['std_basis'])
    )
    lines.append('PTHR = 0.95   # projectability threshold')
    lines.append('')
    lines.append('IBRAV = {}'.format(cfg['ibrav']))
    lines.append('NK = {}   # k-points along the band path'.format(cfg['nk']))
    lines.extend(_band_path_constant_lines(cfg))
    lines.append('')
    lines.append('# Energy grid for DOS / transport / Hall properties (eV, relative to E_F).')
    lines.append('# The full PAO band range is printed as a suggestion at run time')
    lines.append('# (see suggest_energy_window); edit these values to match your needs.')
    lines.append('EMIN, EMAX, NE = {}, {}, {}'.format(cfg['emin'], cfg['emax'], cfg['ne']))
    if 'dos' in props:
        lines.append('DO_PDOS = {}'.format(cfg['do_pdos']))
    if 'transport' in props:
        lines.append('TMIN, TMAX, NT = 300.0, 300.0, 1   # temperature grid (K) for transport')
    lines.append('')
    lines.append('# Double-grid interpolation (denser FFT grid). Set to None to skip.')
    if cfg['interpolate']:
        lines.append('NFFT = ({n}, {n}, {n})'.format(n=cfg['nfft']))
    else:
        lines.append('NFFT = None')
    lines.append('')

    # ---- basis generation --------------------------------------------- #
    lines.append(BASIS_GEN_BLOCK)
    if needs_energy_window:
        lines.append(ENERGY_RANGE_BLOCK)

    # ---- standard property run ---------------------------------------- #
    if standard_props:
        lines.append('')
        lines.append('def run_properties():')
        lines.append('    """Standard properties on the {} basis."""'.format(cfg['std_basis']))
        lines.append('    p = PAOFLOW.PAOFLOW(')
        lines.append('        workpath=HERE,')
        lines.append('        outputdir=OUTPUTDIR,')
        lines.append('        savedir=SAVEDIR,')
        lines.append('        smearing=SMEARING,')
        lines.append('        npool=NPOOL,')
        lines.append('        verbose=False,')
        lines.append('    )')
        lines.append('')
        lines.append('    p.projections(basispath=BASISPATH, configuration=STD_BASIS)')
        lines.append('    p.projectability(pthr=PTHR)')
        lines.append('    p.pao_hamiltonian()')
        lines.append('')
        lines.extend(_emit_standard_properties(standard_props, needs_spin, cfg))
        lines.append('')
        lines.append('    p.finish_execution()')
        lines.append('')

    # ---- optical run (separate, extended basis) ----------------------- #
    if has_optical:
        lines.append('')
        lines.append('def run_optical():')
        lines.append('    """Optical properties: extended basis + non-local velocity.')
        lines.append('')
        lines.append('    The dielectric tensor requires the non-local velocity correction,')
        lines.append('    so it runs as a separate PAOFLOW instance on the extended basis.')
        if has_emissivity:
            lines.append('')
            lines.append('    Emissivity (Fresnel directional reflectivity, spectral and total')
            lines.append('    hemispherical emissivity) is computed from the diagonal dielectric')
            lines.append('    function in the same call.')
        lines.append('    """')
        lines.append('    p = PAOFLOW.PAOFLOW(')
        lines.append('        workpath=HERE,')
        lines.append('        outputdir=OUTPUTDIR,')
        lines.append('        savedir=SAVEDIR,')
        lines.append('        smearing=SMEARING,')
        lines.append('        npool=NPOOL,')
        lines.append('        verbose=False,')
        lines.append('    )')
        lines.append('')
        lines.append("    p.projections(basispath=BASISPATH, configuration='extended')")
        lines.append('    p.projectability(pthr=PTHR)')
        lines.append('    p.pao_hamiltonian()')
        lines.append('    if NFFT is not None:')
        lines.append(
            '        p.interpolated_hamiltonian(nfft1=NFFT[0], nfft2=NFFT[1], nfft3=NFFT[2])'
        )
        lines.append('    p.pao_eigh()')
        lines.append('')
        lines.append('    # Non-local velocity is essential for the optical matrix elements.')
        lines.append('    p.gradient_and_momenta(nonlocal_velocity=True)')
        lines.append('    p.adaptive_smearing()')
        if has_emissivity:
            lines.append("    p.dielectric_tensor(emax=10.0, ne=801, d_tensor='diag', delta=0.1,")
            lines.append('                        emissivity=True, emis_angles=(0.0, 30.0, 60.0),')
            lines.append('                        emis_ntheta=90, emis_temperature=300.0)')
        else:
            lines.append("    p.dielectric_tensor(emax=10.0, ne=801, d_tensor='diag', delta=0.1)")
        lines.append('')
        lines.append('    p.finish_execution()')
        lines.append('')

    # ---- main --------------------------------------------------------- #
    lines.append('')
    lines.append('def main():')
    lines.append('    if not os.path.isdir(SAVEDIR):')
    lines.append('        print("{} not found. Run pw.x (scf then nscf) first.".format(SAVEDIR))')
    lines.append('        sys.exit(1)')
    lines.append('')
    lines.append("    ensure_basis(preset='extended')")
    lines.append('    if "MPI" in globals():')
    lines.append('        MPI.COMM_WORLD.Barrier()')
    lines.append('')
    if standard_props:
        lines.append('    run_properties()')
    if has_optical:
        lines.append('    run_optical()')
    if not standard_props and not has_optical:
        lines.append('    pass  # no properties selected')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')
    lines.append('')

    return '\n'.join(lines)


def _emit_standard_properties(props, needs_spin, cfg):
    """Emit the ordered PAOFLOW calls for the chosen standard properties."""
    body = []
    selected = set(props)

    # Band structure (and topology) come right after the Hamiltonian.
    if 'bands' in selected:
        if _wants_explicit_band_path(cfg):
            body.append('    p.bands(ibrav=IBRAV, nk=NK, band_path=BAND_PATH,')
            body.append("            high_sym_points=HIGH_SYM, fname='bands')")
        else:
            body.append("    p.bands(ibrav=IBRAV, nk=NK, fname='bands')")

    if 'topology' in selected:
        body.append('    p.topology(Berry=True, eff_mass=True, spol=2, ipol=0, jpol=1)')

    body.append('')
    body.append('    if NFFT is not None:')
    body.append('        p.interpolated_hamiltonian(nfft1=NFFT[0], nfft2=NFFT[1], nfft3=NFFT[2])')
    body.append('    p.pao_eigh()')
    body.append('')

    if selected & ENERGY_WINDOW_PROPERTIES:
        body.append('    # Print the full PAO band range as a suggested energy window.')
        body.append('    suggest_energy_window(p)')
        body.append('')

    if 'fermi_surface' in selected:
        body.append('    p.fermi_surface()')
    if 'spin_texture' in selected:
        # spin_texture needs the spin operator cached beforehand.
        body.append('    p.spin_operator(spin_orbit=SPIN_ORBIT)')
        body.append('    p.spin_texture()')

    body.append('    p.gradient_and_momenta()')
    body.append('    p.adaptive_smearing()')

    if 'dos' in selected:
        body.append('    p.dos(do_dos=True, do_pdos=DO_PDOS, emin=EMIN, emax=EMAX, ne=NE)')
    if 'spin_Hall' in selected:
        # Do not pre-build the spin operator here: spin_Hall builds it
        # internally with the correct do_spin_orbit flag from the DFT data.
        body.append('    p.spin_Hall(emin=EMIN, emax=EMAX, s_tensor=[[0, 1, 2]])')
    if 'anomalous_Hall' in selected:
        body.append('    p.anomalous_Hall(do_ac=True, emin=EMIN, emax=EMAX, a_tensor=[[0, 1]])')
    if 'transport' in selected:
        body.append('    p.transport(tmin=TMIN, tmax=TMAX, nt=NT, emin=EMIN, emax=EMAX,')
        body.append('                ne=NE, write_to_file=True)')

    return body


def build_acbn0_script(cfg):
    """Assemble an ACBN0 / eACBN0 main.py from the collected config."""
    use_v = cfg['use_intersite_v']
    hubbard = cfg['hubbard']

    lines = [HEADER]
    lines.append('')
    lines.append('from PAOFLOW import GPAO')
    if use_v:
        lines.append('from PAOFLOW.ACBN0 import ACBN0, eACBN0')
    else:
        lines.append('from PAOFLOW.ACBN0 import ACBN0')
    lines.append('')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('# Configuration  (edit freely)                                            #')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('HERE = os.path.dirname(os.path.abspath(__file__))')
    lines.append('PREFIX = {!r}'.format(cfg['prefix']))
    lines.append(_format_upfs_line(cfg['upfs']))
    lines.append('BASISPATH = os.path.join(HERE, {!r}) + os.sep'.format(cfg['basisdir']))
    lines.append('OUT = {!r}'.format(cfg['outputdir']))
    lines.append('')
    lines.append('# Parallel launch commands and executable paths (edit to your machine).')
    lines.append('MPI_QE = {!r}'.format(cfg['mpi_qe']))
    lines.append('MPI_PY = {!r}'.format(cfg['mpi_py']))
    lines.append('MPI_HARTREE = {!r}'.format(cfg['mpi_hartree']))
    lines.append('QE_PATH = {!r}'.format(cfg['qe_path']))
    lines.append('PY_PATH = {!r}'.format(cfg['py_path']))
    lines.append('')
    lines.append(
        'PROJECTION = {!r}   # ortho-atomic projections fit the Hubbard U'.format(cfg['projection'])
    )
    lines.append('CONV_THR = {}'.format(cfg['conv_thr']))
    lines.append('IBRAV = {}'.format(cfg['ibrav']))
    lines.append('NK = {}'.format(cfg['nk']))
    lines.extend(_band_path_constant_lines(cfg))
    lines.append('')
    lines.append('# Residual tolerance for the Gaussian fit of the pseudo wavefunctions.')
    lines.append('# Loosen (e.g. 0.05 - 0.1) if ACBN0 raises "Could not optimize the wfcs";')
    lines.append('# ONCV / fully-relativistic pseudos often need a looser value than 0.01.')
    lines.append('GAUSSIAN_THRESHOLD = {}'.format(cfg['gaussian_threshold']))
    lines.append('')
    lines.append('# Hubbard manifolds and their initial U (eV).')
    lines.append('HUBBARD_INIT = {')
    for orb, val in hubbard:
        lines.append('    {!r}: {},'.format(orb, val))
    lines.append('}')
    if use_v:
        lines.append('')
        lines.append(
            'V_CUTOFF = {}   # intersite neighbour cutoff (Angstrom)'.format(cfg['v_cutoff'])
        )
        lines.append('V_INIT = {}     # initial intersite V (eV)'.format(cfg['v_init']))
    lines.append('')
    lines.append(BASIS_GEN_BLOCK)
    lines.append('')
    lines.append('def compute_bands(label):')
    lines.append('    """Reconstruct the PAO band structure from the current <PREFIX>.save')
    lines.append('    using the minimal basis and dump it to OUT/bands_<label>_0.dat.')
    lines.append('    """')
    lines.append('    p = PAOFLOW.PAOFLOW(')
    lines.append('        workpath=HERE,')
    lines.append('        outputdir=OUT,')
    lines.append("        savedir='{}.save'.format(PREFIX),")
    lines.append("        smearing='gauss',")
    lines.append('        npool=1,')
    lines.append('        verbose=False,')
    lines.append('    )')
    lines.append('    # Standard basis for the projections, as required for ACBN0.')
    lines.append("    p.projections(basispath=BASISPATH, configuration='standard')")
    lines.append('    p.projectability(pthr=0.95)')
    lines.append('    p.pao_hamiltonian()')
    if _wants_explicit_band_path(cfg):
        lines.append('    p.bands(ibrav=IBRAV, nk=NK, band_path=BAND_PATH,')
        lines.append("            high_sym_points=HIGH_SYM, fname='bands_{}'.format(label))")
    else:
        lines.append("    p.bands(ibrav=IBRAV, nk=NK, fname='bands_{}'.format(label))")
    lines.append('    p.finish_execution()')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append("    ensure_basis(preset='extended')")
    lines.append('    if "MPI" in globals():')
    lines.append('        MPI.COMM_WORLD.Barrier()')
    lines.append('')
    lines.append('    # ------------------------------------------------------------------ #')
    lines.append('    # ACBN0: self-consistent on-site U                                    #')
    lines.append('    # ------------------------------------------------------------------ #')
    lines.append('    a = ACBN0(')
    lines.append('        PREFIX,')
    lines.append("        workdir='./',")
    lines.append('        mpi_qe=MPI_QE,')
    lines.append('        mpi_python=MPI_PY,')
    lines.append('        mpi_hartree=MPI_HARTREE,')
    lines.append("        qe_options='',")
    lines.append('        qe_path=QE_PATH,')
    lines.append('        python_path=PY_PATH,')
    lines.append('        outputdir=OUT,')
    lines.append('        projection=PROJECTION,')
    lines.append('        use_local_basis=True,')
    lines.append('        basispath=BASISPATH,')
    lines.append("        configuration='standard',")
    lines.append('        gaussian_threshold=GAUSSIAN_THRESHOLD,')
    lines.append('    )')
    lines.append('    a.set_hubbard_parameters(dict(HUBBARD_INIT))')
    lines.append('    a.optimize_hubbard_U(convergence_threshold=CONV_THR)')
    lines.append("    compute_bands('U')")
    lines.append('    converged_U = dict(a.uVals)')
    lines.append("    print('\\nConverged U values:')")
    lines.append('    for k, v in converged_U.items():')
    lines.append("        print('  {} : {:.4f} eV'.format(k, v))")
    if use_v:
        lines.append('')
        lines.append('    # ------------------------------------------------------------------ #')
        lines.append('    # eACBN0: joint on-site U + intersite V                               #')
        lines.append('    # ------------------------------------------------------------------ #')
        lines.append('    e = eACBN0(')
        lines.append('        PREFIX,')
        lines.append("        workdir='./',")
        lines.append('        mpi_qe=MPI_QE,')
        lines.append('        mpi_python=MPI_PY,')
        lines.append('        mpi_hartree=MPI_HARTREE,')
        lines.append("        qe_options='',")
        lines.append('        qe_path=QE_PATH,')
        lines.append('        python_path=PY_PATH,')
        lines.append('        outputdir=OUT,')
        lines.append('        projection=PROJECTION,')
        lines.append('        use_local_basis=True,')
        lines.append('        basispath=BASISPATH,')
        lines.append("        configuration='standard',")
        lines.append('        gaussian_threshold=GAUSSIAN_THRESHOLD,')
        lines.append('    )')
        lines.append('    e.set_hubbard_parameters(converged_U)')
        lines.append('    e.set_intersite_pairs(cutoff=V_CUTOFF, V_init=V_INIT)')
        lines.append('    e.print_intersite_pairs()')
        lines.append(
            '    e.optimize_hubbard_UV(convergence_threshold=CONV_THR, max_iter=25, mixing=0.7)'
        )
        lines.append('    e.run_dft(PREFIX, e.uspecies, e.uVals)')
        lines.append("    compute_bands('UV')")
        lines.append("    print('\\nFinal U values:')")
        lines.append('    for k, v in e.uVals.items():')
        lines.append("        print('  {} : {:.4f} eV'.format(k, v))")
        lines.append("    print('\\nFinal V values:')")
        lines.append('    for k, v in e.vVals.items():')
        lines.append("        print('  {} : {:.4f} eV'.format(k, v))")
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')
    lines.append('')

    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Phonon workflow builder
# --------------------------------------------------------------------------- #
def _pp_dir_line(cfg):
    """Render the ``PP_DIR = ...`` constant for the phonon script."""
    ppd = cfg.get('pp_dir', 'HERE')
    if ppd in ('', 'HERE', '.'):
        return 'PP_DIR = HERE   # directory containing the UPF pseudopotentials'
    if os.path.isabs(ppd):
        return 'PP_DIR = {!r}'.format(ppd)
    return 'PP_DIR = os.path.join(HERE, {!r})'.format(ppd)


def _hubbard_file_line(cfg):
    """Render the ``HUBBARD_FILE = ...`` constant for the phonon script."""
    hf = (cfg.get('hubbard_file', '') or '').strip()
    if not hf:
        return 'HUBBARD_FILE = None   # pw.x input with a HUBBARD card (on-site U injected)'
    if os.path.isabs(hf):
        return 'HUBBARD_FILE = {!r}'.format(hf)
    return 'HUBBARD_FILE = os.path.join(HERE, {!r})'.format(hf)


def _derive_ph_command(pw_command):
    """Derive a ``ph.x`` command from the user's ``pw.x`` command.

    Keeps any parallelisation flags (e.g. ``-npool 4``, which ``ph.x`` also
    accepts) by swapping only the executable token; falls back to ``ph.x`` when
    no ``pw.x`` token is present.
    """
    pw_command = (pw_command or 'pw.x').strip()
    if 'pw.x' in pw_command:
        return pw_command.replace('pw.x', 'ph.x')
    return 'ph.x'


def build_phonon_script(cfg):
    """Assemble a main.phonon.py harmonic-phonon workflow from the config.

    The generated script drives the finite-displacement phonon workflow in
    three phases (generate displaced supercells -> run pw.x -> analyse forces),
    which can be run together or one at a time (handy on HPC schedulers).
    """
    sc = cfg['supercell']
    mesh = cfg['mesh']

    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""PAOFLOW harmonic-phonon workflow (generated by paoflow_gen.py).')
    lines.append('')
    lines.append('Finite-displacement phonons via phonopy.  The workflow runs in three')
    lines.append('phases, which can be invoked separately:')
    lines.append('')
    lines.append('    python main.phonon.py generate   # write displaced-supercell QE inputs')
    lines.append('    python main.phonon.py run        # run pw.x on every displaced supercell')
    lines.append('    python main.phonon.py analyse    # forces -> fc2 -> bands / DOS / thermal')
    lines.append('')
    lines.append('When BORN is enabled the same three phases also write, run and harvest the')
    lines.append('Born-charge / dielectric calculation (ph.x DFPT or lelfield pw.x), so the')
    lines.append('phonon bands include the LO-TO splitting (non-analytical correction).')
    lines.append('')
    lines.append('``all`` (the default) runs the three phases in sequence:')
    lines.append('')
    lines.append('    python main.phonon.py')
    lines.append('')
    lines.append('On HPC, run the phases separately so pw.x can go through the scheduler.')
    lines.append('All lengths follow the phonopy QE convention (Bohr).')
    lines.append('"""')
    lines.append('')
    lines.append('import argparse')
    lines.append('import glob')
    lines.append('import os')
    lines.append('import subprocess')
    lines.append('import sys')
    lines.append('')
    lines.append('from PAOFLOW import PAOFLOW')
    lines.append('')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('# Configuration  (edit freely)                                            #')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('HERE = os.path.dirname(os.path.abspath(__file__))')
    lines.append('SAVEDIR = os.path.join(HERE, {!r})'.format(cfg['savedir']))
    lines.append('OUTPUTDIR = {!r}'.format(cfg['outputdir']))
    lines.append("PHONON_DIR = 'phonon'   # sub-directory (under OUTPUTDIR) for the supercells")
    lines.append('PREFIX = {!r}'.format(cfg['prefix']))
    lines.append(_pp_dir_line(cfg))
    lines.append('')
    lines.append('# Finite-displacement settings.')
    lines.append('SUPERCELL_MATRIX = {}   # scalar, length-3 or 3x3'.format(sc))
    lines.append('DISPLACEMENT = {}   # displacement amplitude (Bohr)'.format(cfg['displacement']))
    lines.append('MESH = [{}, {}, {}]   # q-mesh for DOS / thermal properties'.format(*mesh))
    lines.append('UNITS = {!r}   # frequency units for the outputs'.format(cfg['units']))
    lines.append(
        'DO_THERMAL = {}   # free energy, entropy and heat capacity'.format(cfg['do_thermal'])
    )
    lines.append('')
    lines.append('# Born charges & high-frequency dielectric tensor (LO-TO / NAC).')
    lines.append(
        'BORN = {}   # compute Z* + epsilon_inf and apply the LO-TO correction'.format(
            bool(cfg.get('born', False))
        )
    )
    lines.append(
        "BORN_METHOD = {!r}   # 'dfpt' (ph.x) or 'field' (lelfield pw.x)".format(
            cfg.get('born_method', 'dfpt')
        )
    )
    lines.append("FIELD_STRENGTH = 0.001   # efield (a.u.) for BORN_METHOD='field'")
    lines.append("NBERRYCYC = 3            # Berry-phase cycles for BORN_METHOD='field'")
    lines.append(
        "BORN_FILE = os.path.join(HERE, OUTPUTDIR, PHONON_DIR, 'BORN')   # written by the analyse phase"
    )
    lines.append('')
    lines.append('# Vibrational (ionic) dielectric function eps(w) = eps_inf + ionic')
    lines.append('# (lattice resonances -> reststrahlen band).  Needs the Born charges.')
    lines.append(
        'VIBDIELECTRIC = {}   # compute the ionic dielectric eps(w)'.format(
            bool(cfg.get('vibdielectric', False))
        )
    )
    lines.append(
        'VIBDIELECTRIC_GAMMA = {}   # phonon linewidth (in UNITS) damping eps(w)'.format(
            cfg.get('vibdielectric_gamma', 4.0)
        )
    )
    lines.append(
        "VIBDIELECTRIC_DIR = 'vibdielectric'   # sub-dir (under OUTPUTDIR) for the eps files"
    )
    lines.append(
        'VIBDIELECTRIC_EMISSIVITY = {}   # reststrahlen emissivity (1 - R) from eps(w)'.format(
            bool(cfg.get('vibdielectric_emissivity', False))
        )
    )
    lines.append(
        'VIBDIELECTRIC_EMIS_TEMP = {!r}   # temperature(s) (K) for total hemispherical emissivity'.format(
            cfg.get('vibdielectric_emis_temp', [300.0])
        )
    )
    lines.append('')
    lines.append('# Quasi-harmonic approximation (QHA): isotropic volume scan combining the')
    lines.append('# static E(V) with the harmonic F_vib(T, V) -> V(T), thermal expansion,')
    lines.append('# bulk modulus, Gibbs energy, Cp and the Gruneisen parameter.')
    lines.append(
        'QHA = {}   # run the quasi-harmonic volume scan'.format(bool(cfg.get('qha', False)))
    )
    lines.append(
        'QHA_NVOLUMES = {}   # sampled volumes: 5 (full EOS fit) or 3 (parabolic fit)'.format(
            int(cfg.get('qha_nvolumes', 5))
        )
    )
    lines.append(
        'QHA_STRAIN = {}   # maximum linear strain of the scan'.format(cfg.get('qha_strain', 0.02))
    )
    lines.append(
        "QHA_EOS = {!r}   # 'vinet', 'birch_murnaghan' or 'murnaghan' (QHA_NVOLUMES >= 4)".format(
            cfg.get('qha_eos', 'vinet')
        )
    )
    lines.append('QHA_TMIN = {}   # minimum temperature (K)'.format(cfg.get('qha_tmin', 0.0)))
    lines.append('QHA_TMAX = {}   # maximum temperature (K)'.format(cfg.get('qha_tmax', 1000.0)))
    lines.append('QHA_TSTEP = {}   # temperature step (K)'.format(cfg.get('qha_tstep', 10.0)))
    lines.append(
        'QHA_PRESSURE = {}   # external pressure (GPa) added as a pV term'.format(
            cfg.get('qha_pressure', 0.0)
        )
    )
    lines.append("QHA_DIR = 'qha'   # sub-directory (under OUTPUTDIR) for the volume scan")
    lines.append('')
    lines.append('# HUBBARD card injected into the force (and lelfield) inputs.  Only on-site')
    lines.append('# U parameters are kept; intersite V lines are dropped (their atom indices')
    lines.append('# are not valid for the supercell).')
    lines.append(_hubbard_file_line(cfg))
    lines.append('')
    lines.append('# Dispersion path in fractional reciprocal coordinates.  None derives a')
    lines.append('# high-symmetry path automatically from IBRAV below (falling back to the')
    lines.append('# optional seekpath package when IBRAV is 0).  To override, give a list of')
    lines.append('# segments and matching tick labels, e.g.')
    lines.append('#   Q_PATH = [[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],')
    lines.append('#             [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]]')
    lines.append("#   Q_LABELS = ['L', '$\\\\Gamma$', 'X']")
    lines.append(
        'IBRAV = {}   # QE Bravais lattice index for the default q-path'.format(cfg['ibrav'])
    )
    lines.append('Q_PATH = None')
    lines.append('Q_LABELS = None')
    lines.append('Q_NPOINTS = 101   # q-points per path segment')
    lines.append('')
    lines.append('# pw.x launch settings for the supercell SCF runs (edit to your machine).')
    lines.append('MPI_QE = {!r}'.format(cfg['mpi_qe']))
    lines.append(
        "QE_PW = {!r}   # pw.x command (may include flags, e.g. 'pw.x -npool 4')".format(
            cfg.get('qe_pw', 'pw.x')
        )
    )
    lines.append(
        'QE_PH = {!r}   # ph.x command for the Born DFPT run (may include flags)'.format(
            _derive_ph_command(cfg.get('qe_pw', 'pw.x'))
        )
    )
    lines.append('')
    lines.append('')
    lines.append('def _make_paoflow():')
    lines.append('    """Fresh PAOFLOW instance reading the unit-cell structure from SAVEDIR."""')
    lines.append('    return PAOFLOW.PAOFLOW(')
    lines.append('        workpath=HERE,')
    lines.append('        outputdir=OUTPUTDIR,')
    lines.append('        savedir=SAVEDIR,')
    lines.append('        verbose=True,')
    lines.append('    )')
    lines.append('')
    lines.append('')
    lines.append('def generate():')
    lines.append('    """Phase 1: write a QE SCF input for every displaced supercell."""')
    lines.append('    p = _make_paoflow()')
    lines.append('    p.phonons(')
    lines.append('        supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('        displacement_distance=DISPLACEMENT,')
    lines.append('        phonon_dir=PHONON_DIR,')
    lines.append('        pp_dir=PP_DIR,')
    lines.append('        prefix=PREFIX,')
    lines.append('        hubbard_file=HUBBARD_FILE,')
    lines.append('        forces=None,')
    lines.append('    )')
    lines.append('    if BORN:')
    lines.append('        p.born_charges(')
    lines.append('            supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('            method=BORN_METHOD,')
    lines.append('            phonon_dir=PHONON_DIR,')
    lines.append('            pp_dir=PP_DIR,')
    lines.append('            prefix=PREFIX,')
    lines.append('            field_strength=FIELD_STRENGTH,')
    lines.append('            nberrycyc=NBERRYCYC,')
    lines.append('            hubbard_file=HUBBARD_FILE,')
    lines.append('            forces=None,')
    lines.append('        )')
    lines.append('    if QHA:')
    lines.append('        p.quasi_harmonic(')
    lines.append('            supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('            displacement_distance=DISPLACEMENT,')
    lines.append('            nvolumes=QHA_NVOLUMES,')
    lines.append('            strain=QHA_STRAIN,')
    lines.append('            qha_dir=QHA_DIR,')
    lines.append('            pp_dir=PP_DIR,')
    lines.append('            prefix=PREFIX,')
    lines.append('            hubbard_file=HUBBARD_FILE,')
    lines.append('            forces=None,')
    lines.append('        )')
    lines.append('')
    lines.append('')
    lines.append('def _job_done(path):')
    lines.append('    """True when a pw.x output exists and reached JOB DONE."""')
    lines.append('    try:')
    lines.append('        with open(path) as fh:')
    lines.append("            return 'JOB DONE' in fh.read()")
    lines.append('    except OSError:')
    lines.append('        return False')
    lines.append('')
    lines.append('')
    lines.append('def run_supercells():')
    lines.append('    """Phase 2: run pw.x on every displaced-supercell input."""')
    lines.append('    indir = os.path.join(HERE, OUTPUTDIR, PHONON_DIR)')
    lines.append("    inputs = sorted(glob.glob(os.path.join(indir, 'supercell-*.in')))")
    lines.append('    if not inputs:')
    lines.append("        print('No supercell inputs in {}; run the generate phase first.'.format(")
    lines.append('            indir))')
    lines.append('        sys.exit(1)')
    lines.append('    pwx = QE_PW')
    lines.append('    for fin in inputs:')
    lines.append("        fout = fin[:-3] + '.out'")
    lines.append('        if _job_done(fout):')
    lines.append("            print('  skip (already done):', os.path.basename(fout))")
    lines.append('            continue')
    lines.append("        cmd = '{mpi} {pwx} -in {inp} > {out}'.format(")
    lines.append('            mpi=MPI_QE, pwx=pwx,')
    lines.append('            inp=os.path.basename(fin), out=os.path.basename(fout))')
    lines.append("        print('  running:', os.path.basename(fin))")
    lines.append('        subprocess.run(cmd, shell=True, check=True, cwd=indir)')
    lines.append('')
    lines.append('')
    lines.append('def run_born():')
    lines.append('    """Phase 2 (Born): run the ph.x (DFPT) or lelfield pw.x field inputs."""')
    lines.append('    indir = os.path.join(HERE, OUTPUTDIR, PHONON_DIR)')
    lines.append("    if BORN_METHOD == 'dfpt':")
    lines.append('        phx = QE_PH')
    lines.append("        fin = os.path.join(indir, 'ph_epsil.in')")
    lines.append('        if not os.path.isfile(fin):')
    lines.append("            print('No ph_epsil.in in {}; run the generate phase first.'.format(")
    lines.append('                indir))')
    lines.append('            sys.exit(1)')
    lines.append("        fout = os.path.join(indir, 'ph_epsil.out')")
    lines.append('        if _job_done(fout):')
    lines.append("            print('  skip (already done):', os.path.basename(fout))")
    lines.append('            return')
    lines.append("        cmd = '{mpi} {phx} -in ph_epsil.in > ph_epsil.out'.format(")
    lines.append('            mpi=MPI_QE, phx=phx)')
    lines.append("        print('  running: ph_epsil.in')")
    lines.append('        subprocess.run(cmd, shell=True, check=True, cwd=indir)')
    lines.append('    else:')
    lines.append("        inputs = sorted(glob.glob(os.path.join(indir, 'field-*.in')))")
    lines.append('        if not inputs:')
    lines.append("            print('No field-*.in in {}; run the generate phase first.'.format(")
    lines.append('                indir))')
    lines.append('            sys.exit(1)')
    lines.append('        pwx = QE_PW')
    lines.append('        for fin in inputs:')
    lines.append("            fout = fin[:-3] + '.out'")
    lines.append('            if _job_done(fout):')
    lines.append("                print('  skip (already done):', os.path.basename(fout))")
    lines.append('                continue')
    lines.append("            cmd = '{mpi} {pwx} -in {inp} > {out}'.format(")
    lines.append('                mpi=MPI_QE, pwx=pwx,')
    lines.append('                inp=os.path.basename(fin), out=os.path.basename(fout))')
    lines.append("            print('  running:', os.path.basename(fin))")
    lines.append('            subprocess.run(cmd, shell=True, check=True, cwd=indir)')
    lines.append('')
    lines.append('')
    lines.append('def run_qha():')
    lines.append(
        '    """Phase 2 (QHA): run pw.x on every volume\'s scf and displaced supercells."""'
    )
    lines.append('    base = os.path.join(HERE, OUTPUTDIR, QHA_DIR)')
    lines.append("    vol_dirs = sorted(glob.glob(os.path.join(base, 'vol-*')))")
    lines.append('    if not vol_dirs:')
    lines.append(
        "        print('No vol-* directories in {}; run the generate phase first.'.format("
    )
    lines.append('            base))')
    lines.append('        sys.exit(1)')
    lines.append('    pwx = QE_PW')
    lines.append('    for vdir in vol_dirs:')
    lines.append("        inputs = sorted(glob.glob(os.path.join(vdir, 'scf.in')))")
    lines.append("        inputs += sorted(glob.glob(os.path.join(vdir, 'supercell-*.in')))")
    lines.append('        for fin in inputs:')
    lines.append("            fout = fin[:-3] + '.out'")
    lines.append('            if _job_done(fout):')
    lines.append("                print('  skip (already done):', os.path.relpath(fout, base))")
    lines.append('                continue')
    lines.append("            cmd = '{mpi} {pwx} -in {inp} > {out}'.format(")
    lines.append('                mpi=MPI_QE, pwx=pwx,')
    lines.append('                inp=os.path.basename(fin), out=os.path.basename(fout))')
    lines.append("            print('  running:', os.path.relpath(fin, base))")
    lines.append('            subprocess.run(cmd, shell=True, check=True, cwd=vdir)')
    lines.append('')
    lines.append('')
    lines.append('def analyse():')
    lines.append('    """Phase 3: harvest forces, build fc2, write bands / DOS / thermal props."""')
    lines.append('    p = _make_paoflow()')
    lines.append('')
    lines.append('    # Forces: ingest a pre-existing FORCE_SETS when present, otherwise')
    lines.append('    # harvest the QE displaced-supercell outputs.')
    lines.append("    force_sets = os.path.join(HERE, OUTPUTDIR, PHONON_DIR, 'FORCE_SETS')")
    lines.append("    forces = force_sets if os.path.isfile(force_sets) else 'qe'")
    lines.append('')
    lines.append('    # Born charges: reuse an existing BORN file when present; otherwise')
    lines.append('    # (when BORN is enabled) harvest the QE DFPT / field output to create it.')
    lines.append('    have_born = os.path.isfile(BORN_FILE)')
    lines.append('    if BORN and not have_born:')
    lines.append('        p.born_charges(')
    lines.append('            supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('            method=BORN_METHOD,')
    lines.append("            forces='qe',")
    lines.append('            phonon_dir=PHONON_DIR,')
    lines.append('            prefix=PREFIX,')
    lines.append('        )')
    lines.append('        have_born = os.path.isfile(BORN_FILE)')
    lines.append('    nac = BORN and have_born')
    lines.append('')
    lines.append('    p.phonons(')
    lines.append('        supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('        displacement_distance=DISPLACEMENT,')
    lines.append('        forces=forces,')
    lines.append('        phonon_dir=PHONON_DIR,')
    lines.append('        pp_dir=PP_DIR,')
    lines.append('        prefix=PREFIX,')
    lines.append('        ibrav=IBRAV,')
    lines.append('        nac=nac,')
    lines.append('        born_file=(BORN_FILE if nac else None),')
    lines.append('        q_path=Q_PATH,')
    lines.append('        q_labels=Q_LABELS,')
    lines.append('        q_npoints=Q_NPOINTS,')
    lines.append('        mesh=MESH,')
    lines.append('        do_bands=True,')
    lines.append('        do_dos=True,')
    lines.append('        do_thermal=DO_THERMAL,')
    lines.append('        units=UNITS,')
    lines.append("        fname='phonon',")
    lines.append('    )')
    lines.append('')
    lines.append('    # Infrared spectrum (mode effective charges; requires Born charges).')
    lines.append('    if nac:')
    lines.append('        p.ir_spectrum(')
    lines.append('            supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('            forces=forces,')
    lines.append('            phonon_dir=PHONON_DIR,')
    lines.append('            born_file=BORN_FILE,')
    lines.append('            units=UNITS,')
    lines.append("            fname='phonon',")
    lines.append('        )')
    lines.append('')
    lines.append(
        '    # Vibrational (ionic) dielectric eps(w) (reststrahlen; requires Born charges).'
    )
    lines.append('    if nac and VIBDIELECTRIC:')
    lines.append('        p.vibrational_dielectric(')
    lines.append('            supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('            forces=forces,')
    lines.append('            phonon_dir=PHONON_DIR,')
    lines.append('            born_file=BORN_FILE,')
    lines.append('            gamma=VIBDIELECTRIC_GAMMA,')
    lines.append('            units=UNITS,')
    lines.append('            outdir=VIBDIELECTRIC_DIR,')
    lines.append('            emissivity=VIBDIELECTRIC_EMISSIVITY,')
    lines.append('            emis_temperature=VIBDIELECTRIC_EMIS_TEMP,')
    lines.append("            fname='phonon',")
    lines.append('        )')
    lines.append('')
    lines.append('    # Quasi-harmonic approximation: harvest the volume scan and write V(T),')
    lines.append('    # thermal expansion, bulk modulus, Gibbs energy, Cp and Gruneisen.')
    lines.append('    if QHA:')
    lines.append('        p.quasi_harmonic(')
    lines.append('            supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('            displacement_distance=DISPLACEMENT,')
    lines.append('            nvolumes=QHA_NVOLUMES,')
    lines.append('            strain=QHA_STRAIN,')
    lines.append("            forces='qe',")
    lines.append('            qha_dir=QHA_DIR,')
    lines.append('            mesh=MESH,')
    lines.append('            t_min=QHA_TMIN,')
    lines.append('            t_max=QHA_TMAX,')
    lines.append('            t_step=QHA_TSTEP,')
    lines.append('            eos=QHA_EOS,')
    lines.append('            pressure=QHA_PRESSURE,')
    lines.append('            ibrav=IBRAV,')
    lines.append('            q_path=Q_PATH,')
    lines.append('            q_labels=Q_LABELS,')
    lines.append('            q_npoints=Q_NPOINTS,')
    lines.append('            units=UNITS,')
    lines.append("            fname='qha',")
    lines.append('        )')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append('    parser = argparse.ArgumentParser(')
    lines.append("        description='PAOFLOW harmonic-phonon workflow.')")
    lines.append("    parser.add_argument('phase', nargs='?',")
    lines.append("                        choices=['generate', 'run', 'analyse', 'all'],")
    lines.append("                        default='all',")
    lines.append("                        help='Workflow phase to run (default: all).')")
    lines.append('    args = parser.parse_args()')
    lines.append('')
    lines.append('    if not os.path.isdir(SAVEDIR):')
    lines.append('        print("{} not found. Run pw.x (scf) first.".format(SAVEDIR))')
    lines.append('        sys.exit(1)')
    lines.append('')
    lines.append("    if args.phase in ('generate', 'all'):")
    lines.append('        generate()')
    lines.append("    if args.phase in ('run', 'all'):")
    lines.append('        run_supercells()')
    lines.append('        if BORN:')
    lines.append('            run_born()')
    lines.append('        if QHA:')
    lines.append('            run_qha()')
    lines.append("    if args.phase in ('analyse', 'all'):")
    lines.append('        analyse()')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')
    lines.append('')

    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Electron-phonon (PAO route) workflow builder
# --------------------------------------------------------------------------- #
ELPHON_TEMPLATE = r'''#!/usr/bin/env python3
"""PAOFLOW electron-phonon (Eliashberg) workflow -- PAO route (generated by paoflow_gen.py).

Pseudo-atomic-orbital (Agapito & Bernardi, Phys. Rev. B 97, 235146 (2018)) interpolation
of Quantum ESPRESSO's DFPT electron-phonon coupling.  PAOFLOW reads QE's *full*
coarse-grid coupling (no potential reconstruction), rotates it into the PAO gauge
and Wigner-Seitz interpolates electrons + vertex to a dense grid for alpha^2F,
lambda, omega_log and Tc.

Two phases:

    python main.elphon.py inputs    # write the ph.x phonon + AHC input templates
    python main.elphon.py analyse   # PAO interpolation -> alpha^2F, lambda, Tc

The ``analyse`` phase parallelises the per-q interpolation over MPI ranks, so
on a cluster launch it with mpirun for an (up to nq-fold) speedup::

    mpirun -np N python main.elphon.py analyse

The dense electron diagonalisation is done redundantly on every rank, so it is
usually best to keep N at or below the number of q-points and let BLAS threads
(OMP_NUM_THREADS) fill the remaining cores.

Between them, run the two QE ph.x steps in the same outdir (typically on HPC):

    1. phonon:  ph.x < <prefix>.ph.in    # full DFPT dvscf on the q-grid
    2. ahc:     ph.x < <prefix>.ahc.in   # electron_phonon='ahc' -> ahc_dir/

``fildvscf`` and ``fildyn`` MUST match between the two ph.x steps.  The AHC path
(SOURCE='ahc') is for norm-conserving pseudopotentials; for ultrasoft / PAW use
the patched-QE ``el_ph_mat`` dump (SOURCE='elphmat').
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.do_pao_eph import eliashberg_from_qe_coupling
from PAOFLOW.elphon.elph_bloch import read_nscf


# ----------------------------------------------------------------------- #
# Configuration  (edit freely -- masses / NELEC / NBND are system-specific) #
# ----------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, __SAVEDIR__)
OUTPUTDIR = __OUTPUTDIR__
PREFIX = __PREFIX__
BASISDIR = os.path.join(HERE, __BASISDIR__)

# Coupling source:
#   'ahc'     -> unpatched QE AHC dumps (ahc_dir/ahc_gkk_iq<iq>.bin); NC pseudos.
#   'elphmat' -> patched-QE el_ph_mat dumps (elph_dir/elphmat.<iq>.dat); any pseudo.
SOURCE = __SOURCE__
COUPLING_DIR = os.path.join(HERE, __COUPLING_DIR__)

KGRID = __KGRID__          # SCF / coupling k-grid (== pw.x K_POINTS)
QGRID = __QGRID__          # phonon q-grid (nq1, nq2, nq3)
NBND = __NBND__            # bands in the nscf / AHC run (nbnd = ahc_nbnd)
MASSES_AMU = __MASSES__    # atomic masses (amu), one per atom in the cell
NELEC = __NELEC__          # valence electrons (dense E_F recompute)
NK_DENSE = __NK_DENSE__    # dense interpolation grid
SIGMA_RY = __SIGMA__       # Fermi-surface smearing (Ry)
MU_STAR = __MU_STAR__      # Coulomb pseudopotential for Tc
PTHR = __PTHR__            # projectability threshold
DYNPREFIX = PREFIX         # <DYNPREFIX>.dyn<iq> phonon files

# For SOURCE='elphmat' (patched, symmetry-reduced) set the irreducible-q star
# weights here; leave empty (or use SOURCE='ahc') to sum the full q-grid with
# unit weights.
Q_WEIGHTS = __QWEIGHTS__


def _phonon_input():
    return '\n'.join([
        '&inputph',
        "  prefix='%s'," % PREFIX,
        "  outdir='./',",
        "  fildyn='%s.dyn'," % PREFIX,
        "  fildvscf='dvscf',",
        '  tr2_ph=1.0d-12,',
        '  ldisp=.true., nq1=%d, nq2=%d, nq3=%d,' % (QGRID[0], QGRID[1], QGRID[2]),
        '/',
        '',
    ])


def _ahc_input():
    return '\n'.join([
        '&inputph',
        "  prefix='%s'," % PREFIX,
        "  outdir='./',",
        "  fildyn='%s.dyn'," % PREFIX,
        "  fildvscf='dvscf',",
        "  electron_phonon='ahc',",
        '  ahc_nbnd=%d, ahc_nbndskip=0, skip_upperfan=.true.,' % NBND,
        "  ahc_dir='ahc_dir/',",
        '  ldisp=.true., nq1=%d, nq2=%d, nq3=%d,' % (QGRID[0], QGRID[1], QGRID[2]),
        '/',
        '',
    ])


def inputs():
    """Phase 1: write the ph.x phonon and AHC input templates."""
    ph = os.path.join(HERE, PREFIX + '.ph.in')
    ahc = os.path.join(HERE, PREFIX + '.ahc.in')
    with open(ph, 'w') as fh:
        fh.write(_phonon_input())
    with open(ahc, 'w') as fh:
        fh.write(_ahc_input())
    print('Wrote %s' % ph)
    print('Wrote %s' % ahc)
    print('Run (same outdir), then `analyse`:')
    print('  1) ph.x < %s' % os.path.basename(ph))
    print('  2) ph.x < %s' % os.path.basename(ahc))


def analyse():
    """Phase 2: PAO interpolation of the QE coupling -> alpha^2F, lambda, Tc."""
    if not os.path.isdir(SAVEDIR):
        sys.exit('%s not found. Run pw.x (nscf) first.' % SAVEDIR)
    if not os.path.isdir(COUPLING_DIR):
        sys.exit('%s not found. Run the QE phonon + AHC steps first (phase: inputs).' % COUPLING_DIR)

    pf = PAOFLOW.PAOFLOW(workpath=HERE, outputdir=OUTPUTDIR, savedir=SAVEDIR, verbose=False)
    pf.projections(configuration='standard', basispath=BASISDIR)
    pf.projectability(pthr=PTHR)
    # Grab the projection matrices A_k BEFORE pao_hamiltonian (which deletes them).
    A = pf.data_controller.data_arrays['U'][:, :, :, 0].copy()
    pf.pao_hamiltonian()
    HRs = pf.data_controller.data_arrays['HRs']
    info = read_nscf(SAVEDIR)

    if SOURCE == 'ahc' or not Q_WEIGHTS:
        nq = QGRID[0] * QGRID[1] * QGRID[2]
        q_weights = [1.0] * nq
    else:
        q_weights = list(Q_WEIGHTS)
        nq = len(q_weights)
    dyn_paths = [os.path.join(HERE, '%s.dyn%d' % (DYNPREFIX, i + 1)) for i in range(nq)]

    out = eliashberg_from_qe_coupling(
        A, HRs, info['kpts_cryst'], info['bg'], info['at'],
        COUPLING_DIR, q_weights, tuple(KGRID), dyn_paths,
        source=SOURCE, masses_amu=MASSES_AMU, nk_dense=NK_DENSE,
        sigmas_ry=[SIGMA_RY], nelec=NELEC, mu_star=MU_STAR,
    )

    # The Eliashberg result is identical on every rank (Allreduce inside the
    # driver); only rank 0 reports and writes the output files.
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    kB = 8.617333262e-5  # eV/K
    print('PAO-route Eliashberg (%s, source=%s, Nk=%d):' % (PREFIX, SOURCE, NK_DENSE))
    print('  N(E_F)   = %.3f states/spin/Ry' % out['dos_ef'].mean())
    print('  lambda   = %.4f' % out['lambda'])
    print('  <w_log>  = %.1f K' % (out['omega_log'] / kB))
    print('  Tc (McM) = %.2f K  (mu* = %.3f)' % (out['Tc_mcmillan'], MU_STAR))
    print('  Tc (AD)  = %.2f K  (mu* = %.3f)' % (out['Tc_allen_dynes'], MU_STAR))

    outdir = os.path.join(HERE, OUTPUTDIR)
    os.makedirs(outdir, exist_ok=True)
    a2f = os.path.join(outdir, 'alpha2F.dat')
    np.savetxt(a2f, np.column_stack([out['omega'] * 1.0e3, out['a2F']]),
               header='omega(meV)   alpha^2F(omega)   '
                      '(lambda=%.4f, omega_log=%.2fK, Tc_McM=%.3fK, Tc_AD=%.3fK, mu*=%.3f)'
                      % (out['lambda'], out['omega_log'] / kB, out['Tc_mcmillan'],
                         out['Tc_allen_dynes'], MU_STAR))
    npz = os.path.join(outdir, 'eliashberg.npz')
    np.savez(npz, **out)
    print('  wrote %s' % a2f)
    print('  wrote %s' % npz)


def main():
    parser = argparse.ArgumentParser(description='PAOFLOW electron-phonon (PAO) workflow.')
    parser.add_argument('phase', nargs='?', choices=['inputs', 'analyse', 'all'],
                        default='all', help='Workflow phase to run (default: all).')
    args = parser.parse_args()
    if args.phase in ('inputs', 'all'):
        inputs()
    if args.phase in ('analyse', 'all'):
        analyse()


if __name__ == '__main__':
    main()
'''


def build_elphon_script(cfg):
    """Assemble a main.elphon.py electron-phonon (PAO route) workflow from the config."""
    subs = {
        '__SAVEDIR__': repr(cfg['savedir']),
        '__OUTPUTDIR__': repr(cfg['outputdir']),
        '__PREFIX__': repr(cfg['prefix']),
        '__BASISDIR__': repr(cfg['basisdir']),
        '__SOURCE__': repr(cfg['source']),
        '__COUPLING_DIR__': repr(cfg['coupling_dir']),
        '__KGRID__': repr(tuple(cfg['kgrid'])),
        '__QGRID__': repr(tuple(cfg['qgrid'])),
        '__NBND__': str(int(cfg['nbnd'])),
        '__MASSES__': repr([float(m) for m in cfg['masses_amu']]),
        '__NELEC__': str(int(cfg['nelec'])),
        '__NK_DENSE__': str(int(cfg['nk_dense'])),
        '__SIGMA__': repr(float(cfg['sigma_ry'])),
        '__MU_STAR__': repr(float(cfg['mu_star'])),
        '__PTHR__': repr(float(cfg['pthr'])),
        '__QWEIGHTS__': repr([float(w) for w in cfg['q_weights']]),
    }
    content = ELPHON_TEMPLATE
    for token, value in subs.items():
        content = content.replace(token, value)
    return content


# --------------------------------------------------------------------------- #
# Electron-phonon plotting-script builder
# --------------------------------------------------------------------------- #
ELPHON_PLOT_TEMPLATE = r'''#!/usr/bin/env python3
"""Plot the Eliashberg alpha^2F(omega) and cumulative lambda(omega).

    python plot.elphon.py

Reads OUTPUTDIR/eliashberg.npz written by main.elphon.py (analyse).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTDIR = __OUTPUTDIR__
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
'''


def build_elphon_plot_script(cfg):
    """Assemble a plot.elphon.py for the Eliashberg alpha^2F / lambda output."""
    content = ELPHON_PLOT_TEMPLATE
    content = content.replace('__OUTPUTDIR__', repr(cfg['outputdir']))
    return content


# --------------------------------------------------------------------------- #
# Plotting-script builder
# --------------------------------------------------------------------------- #
PLOT_HEADER = r'''#!/usr/bin/env python3
"""Plotting helper for the PAOFLOW run (generated by paoflow_gen.py).

This script mirrors the property selection made when the PAOFLOW driver was
generated.  Run it after the PAOFLOW calculation has produced its output files
and pick what to plot from the menu:

    python plot.py

The energy window and property y-axis limits can be set on the command line:

    python plot.py --emin -5 --emax 5 --ymin 0 --ymax 100

Pass ``--all`` to plot every available quantity without any prompting (the
menu and axis-limit prompts are skipped and the generated defaults are used):

    python plot.py --all

and are also asked for interactively when the script runs (press Enter to keep
the default / command-line value).  EMIN/EMAX set the energy axis (vertical for
band plots, horizontal for the energy-resolved property plots) and default to
the window identified when the driver was generated.  YMIN/YMAX set the property
y-axis and default to automatic scaling.  Optical spectra always start at 0 on
the energy axis.

The plots use the helpers in PAOFLOW.GPAO.
"""

import argparse
import glob
import os
import sys

from PAOFLOW import GPAO


HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTDIR = os.path.join(HERE, __OUTPUTDIR__)

pplt = GPAO.GPAO()

# Default energy window (eV, relative to E_F) identified when the driver was
# generated.  Overridable on the command line via --emin/--emax.  YMIN/YMAX
# default to None (automatic axis limits) and can be set via --ymin/--ymax.
EMIN = __EMIN__
EMAX = __EMAX__
YMIN = None
YMAX = None


def _ewin():
    """Energy-axis window (EMIN, EMAX) used for band / energy-resolved plots."""
    return (EMIN, EMAX)


def _ewin_optical():
    """Energy-axis window for optical spectra (always starts at 0)."""
    return (0.0, EMAX)


def _ylim():
    """Property-axis limits (YMIN, YMAX), or None for automatic scaling."""
    if YMIN is None and YMAX is None:
        return None
    return (YMIN, YMAX)


def _dos_mag_lim(fnames):
    """Magnitude-axis limits for DOS/PDOS plots.

    When the user sets YMIN/YMAX those win.  Otherwise the magnitude axis is
    rescaled to the data that falls inside the chosen energy window [EMIN, EMAX]
    so that limiting the energy range actually zooms the curve (important for
    metals, whose largest DOS peaks may sit far outside the window).
    """
    if YMIN is not None or YMAX is not None:
        return _ylim()
    if isinstance(fnames, str):
        fnames = [fnames]
    import numpy as np

    peak = 0.0
    for fn in fnames:
        if not fn:
            continue
        try:
            data = np.loadtxt(fn)
        except (OSError, ValueError):
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            continue
        es = data[:, 0]
        mag = data[:, 1:]
        mask = (es >= EMIN) & (es <= EMAX)
        if not mask.any():
            continue
        peak = max(peak, float(np.nanmax(np.abs(mag[mask]))))
    if peak <= 0.0:
        return None
    return (0.0, 1.1 * peak)



def _ask_float(prompt, default):
    """Prompt for a float, accepting Enter to keep *default* (may be None)."""
    shown = 'auto' if default is None else default
    try:
        raw = input('{} [{}]: '.format(prompt, shown)).strip()
    except EOFError:
        return default
    if raw == '':
        return default
    if raw.lower() in ('auto', 'none'):
        return None
    try:
        return float(raw)
    except ValueError:
        print('  (invalid number, keeping {})'.format(shown))
        return default



def _one(pattern):
    """Return the first OUTPUTDIR file matching *pattern* (or None).

    Patterns are written with an optional '<prefix>.' in front (e.g.
    '*.bands_0.dat').  PAOFLOW may write the files either with that prefix
    ('Si.bands_0.dat') or without it ('bands_0.dat'), so a '*.' pattern
    falls back to the prefix-less form.
    """
    hits = sorted(glob.glob(os.path.join(OUTPUTDIR, pattern)))
    if not hits and pattern.startswith('*.'):
        hits = sorted(glob.glob(os.path.join(OUTPUTDIR, pattern[2:])))
    return hits[0] if hits else None


def _many(pattern):
    """Return all OUTPUTDIR files matching *pattern* (sorted).

    As with _one, a '*.' pattern also matches prefix-less files.
    """
    hits = sorted(glob.glob(os.path.join(OUTPUTDIR, pattern)))
    if not hits and pattern.startswith('*.'):
        hits = sorted(glob.glob(os.path.join(OUTPUTDIR, pattern[2:])))
    return hits


def _missing(*names):
    """Warn about missing output files; return True if anything is missing."""
    gone = [n for n in names if n is None]
    if gone:
        print("  (output file(s) not found in {}; run PAOFLOW first)".format(OUTPUTDIR))
        return True
    return False


def _file_energy_range(fname):
    """Return (min, max) of the energy column of *fname*, or None."""
    if not fname:
        return None
    import numpy as np

    try:
        data = np.loadtxt(fname)
    except (OSError, ValueError):
        return None
    if data.ndim != 2 or data.shape[0] == 0:
        return None
    es = data[:, 0]
    return (float(np.nanmin(es)), float(np.nanmax(es)))


def _default_energy_window():
    """Default energy window for the prompts.

    When a DOS file is present its energy grid is used so that the DOS fills
    the side-by-side panel (the DOS is usually computed on a narrower grid than
    the bands).  Otherwise the window identified when the driver was generated
    is kept.
    """
    return _file_energy_range(_one('*.dosdk_0.dat'))


_SPIN_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']


def _spin_channels(pattern):
    """Return ([files], [labels]) for the spin channels of *pattern*.

    The generated patterns target the first spin channel through a '_0' tag.
    A spin-polarized (nspin=2) run also writes the '_1' channel; when present
    both files are returned so the two channels can be overlaid in one figure.
    """
    f0 = _one(pattern)
    if f0 is None:
        return [], []
    files, labels = [f0], ['spin up']
    f1 = _one(pattern.replace('_0.', '_1.'))
    if f1 is not None:
        files.append(f1)
        labels.append('spin down')
    return files, labels


def _overlay_transport(files, labels, title, y_label, scale=1.0, min_zero=False):
    """Overlay the diagonal-averaged transport tensor of several files.

    Used for spin-polarized conductivity / Seebeck output, where each spin
    channel is written to its own file: both channels are drawn on a single
    figure (one line per spin and temperature).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PAOFLOW.inputs.read_pao_output import read_transport_PAO

    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    ic = 0
    ymax = 0.0
    for fn, base in zip(files, labels):
        enes, temps, tensors = read_transport_PAO(fn)
        for it, temp in enumerate(temps):
            trace = scale * np.einsum('eii->e', tensors[it]) / 3.0
            lbl = base if len(temps) == 1 else '{}, T={:g}'.format(base, temp)
            ax.plot(enes, trace, color=_SPIN_COLORS[ic % len(_SPIN_COLORS)], label=lbl)
            ymax = max(ymax, float(np.nanmax(np.abs(trace))))
            ic += 1
    ax.set_xlim(*_ewin())
    yl = _ylim()
    if yl is not None:
        ax.set_ylim(*yl)
    elif min_zero and ymax > 0.0:
        ax.set_ylim(0.0, 1.1 * ymax)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()


def _overlay_bands_dos(band_files, dos_files, labels, sym_file, title):
    """Bands beside DOS with both spin channels overlaid on one figure.

    Reproduces plot_dos_beside_bands but loops over the spin channels so
    that the two channels share a single bands panel and a single DOS panel.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from PAOFLOW.inputs.read_pao_output import (
        read_bands_PAO,
        read_dos_PAO,
        read_band_path_PAO,
    )

    sym_points = read_band_path_PAO(sym_file) if sym_file else None
    y_lim = _ewin()

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[5, 1])
    fig.suptitle(title)
    ax_b = fig.add_subplot(spec[0])
    ax_d = fig.add_subplot(spec[1])

    nk = 1
    dos_peak = 0.0
    for i, (fb, fd, lbl) in enumerate(zip(band_files, dos_files, labels)):
        col = _SPIN_COLORS[i % len(_SPIN_COLORS)]
        bands = read_bands_PAO(fb)
        nk = bands.shape[1]
        for j, b in enumerate(bands):
            ax_b.plot(b, color=col, label=lbl if j == 0 else None)
        es, dos = read_dos_PAO(fd)
        ax_d.plot(dos, es, color=col, label=lbl)
        mask = (es >= y_lim[0]) & (es <= y_lim[1])
        if mask.any():
            dos_peak = max(dos_peak, float(np.nanmax(np.abs(dos[mask]))))

    ax_b.set_xlim(0, nk - 1)
    ax_b.set_ylim(*y_lim)
    if sym_points is None:
        ax_b.xaxis.set_visible(False)
    else:
        ax_b.set_xticks(sym_points[0])
        ax_b.set_xticklabels(sym_points[1])
        ax_b.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')
    ax_b.set_ylabel(r'$\epsilon$($\mathbf{k}$) (eV)', fontsize=12)
    ax_b.legend()

    xl = _ylim()
    if xl is not None:
        ax_d.set_xlim(*xl)
    elif dos_peak > 0.0:
        ax_d.set_xlim(0.0, 1.1 * dos_peak)
    ax_d.set_ylim(*y_lim)
    ax_d.yaxis.set_visible(False)
    ax_d.set_xlabel('DOS')

    plt.tight_layout()
    plt.show()
'''


# Plot-function bodies keyed by an internal id.  Each value is a list of source
# lines for a ``plot_<id>()`` function in the generated script.
def _plot_func(name, body_lines):
    return ['def {}():'.format(name)] + ['    ' + ln for ln in body_lines] + ['']


def build_plot_script(cfg):
    """Assemble a plot.py that mimics the property selection in *cfg*."""
    props = cfg['properties']
    has_bands = 'bands' in props
    has_dos = 'dos' in props
    has_optical = 'optical' in props or 'emissivity' in props
    has_emissivity = 'emissivity' in props
    do_pdos = cfg.get('do_pdos', False)

    funcs = []  # source for each plot_<id> function
    menu = []  # ordered list of (label, func_name)

    if has_bands:
        funcs += _plot_func(
            'plot_band_structure',
            [
                "files, labels = _spin_channels('*.bands_0.dat')",
                "sp = _one('*.kpath_points.txt')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'if len(files) > 1:',
                "    pplt.plot_bands(files, sym_points=sp, title='Band structure',",
                '                    y_lim=_ewin(), labels=labels, cols=_SPIN_COLORS[:len(files)])',
                'else:',
                "    pplt.plot_bands(files[0], sym_points=sp, title='Band structure', y_lim=_ewin())",
            ],
        )
        menu.append(('Band structure', 'plot_band_structure'))

    if has_dos:
        dos_body = [
            "files, labels = _spin_channels('*.dosdk_0.dat')",
            'if not files:',
            '    _missing(None)',
            '    return',
            'if len(files) > 1:',
            "    pplt.plot_pdos(files, title='Density of states', x_lim=_ewin(),",
            '                   y_lim=_dos_mag_lim(files), labels=labels)',
            'else:',
            "    pplt.plot_dos(files[0], title='Density of states', x_lim=_ewin(), y_lim=_dos_mag_lim(files[0]))",
        ]
        if do_pdos:
            dos_body += [
                "pdos = _many('*.pdosdk*')",
                'if pdos:',
                "    pplt.plot_pdos(pdos, title='Projected DOS', x_lim=_ewin(), y_lim=_dos_mag_lim(pdos))",
            ]
        funcs += _plot_func('plot_density_of_states', dos_body)
        menu.append(('DOS / projected DOS', 'plot_density_of_states'))

    # Per the workflow: when both bands and DOS are present, offer the combined
    # side-by-side plot via plot_dos_beside_bands.
    if has_bands and has_dos:
        funcs += _plot_func(
            'plot_bands_and_dos',
            [
                "bfiles, blabels = _spin_channels('*.bands_0.dat')",
                "dfiles, _ = _spin_channels('*.dosdk_0.dat')",
                "sp = _one('*.kpath_points.txt')",
                'if not bfiles or not dfiles:',
                '    _missing(None)',
                '    return',
                'if len(bfiles) > 1 and len(dfiles) > 1:',
                '    _overlay_bands_dos(bfiles, dfiles, blabels, sp,',
                "                       title='Bands and DOS')",
                'else:',
                '    pplt.plot_dos_beside_bands(dfiles[0], bfiles[0], sym_points=sp, dos_ticks=True,',
                "                               title='Bands and DOS', x_lim=_dos_mag_lim(dfiles[0]), y_lim=_ewin())",
            ],
        )
        menu.append(('Bands + DOS (side by side)', 'plot_bands_and_dos'))

    if 'transport' in props:
        funcs += _plot_func(
            'plot_conductivity',
            [
                "files, labels = _spin_channels('*sigma*_0.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'if len(files) > 1:',
                "    _overlay_transport(files, labels, 'Electrical conductivity',",
                "                       r'Conductivity $(\\Omega\\, m\\, s)^{-1}$', min_zero=True)",
                'else:',
                "    pplt.plot_electrical_conductivity(files[0], title='Electrical conductivity', x_lim=_ewin(), y_lim=_ylim())",
            ],
        )
        menu.append(('Electrical conductivity', 'plot_conductivity'))
        funcs += _plot_func(
            'plot_seebeck',
            [
                "files, labels = _spin_channels('*Seebeck*_0.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'if len(files) > 1:',
                "    _overlay_transport(files, labels, 'Seebeck coefficient',",
                "                       r'Seebeck ($\\mu$V/K)', scale=1e6)",
                'else:',
                "    pplt.plot_seebeck(files[0], title='Seebeck coefficient', x_lim=_ewin(), y_lim=_ylim())",
            ],
        )
        menu.append(('Seebeck coefficient', 'plot_seebeck'))
        funcs += _plot_func(
            'plot_thermal_conductivity',
            [
                "files, labels = _spin_channels('*kappa*_0.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                "_overlay_transport(files, labels, 'Electron thermal conductivity',",
                "                   r'$\\kappa$ (W m$^{-1}$ K$^{-1}$)', min_zero=True)",
            ],
        )
        menu.append(('Thermal conductivity', 'plot_thermal_conductivity'))
        funcs += _plot_func(
            'plot_power_factor',
            [
                "files, labels = _spin_channels('*PF*_0.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                "_overlay_transport(files, labels, 'Power factor',",
                "                   r'$S^2\\sigma$ (W m$^{-1}$ K$^{-2}$)', min_zero=True)",
            ],
        )
        menu.append(('Power factor', 'plot_power_factor'))
        funcs += _plot_func(
            'plot_hall_coefficient',
            [
                '# Trace of the Hall coefficient tensor (only present when the run',
                '# enabled do_hall); each file is energy vs a single value.',
                "files, _ = _spin_channels('hall_trace_0.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'pplt.plot_shc(files if len(files) > 1 else files[0],',
                "              title='Hall coefficient', x_lim=_ewin(), y_lim=_ylim())",
            ],
        )
        menu.append(('Hall coefficient', 'plot_hall_coefficient'))

    if 'spin_Hall' in props:
        funcs += _plot_func(
            'plot_spin_hall',
            [
                "files = _many('*.shcEf*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                "pplt.plot_shc(files if len(files) > 1 else files[0], title='Spin Hall conductivity', x_lim=_ewin(), y_lim=_ylim())",
            ],
        )
        menu.append(('Spin Hall conductivity', 'plot_spin_hall'))

    if 'anomalous_Hall' in props:
        funcs += _plot_func(
            'plot_anomalous_hall',
            [
                "files = _many('*.ahcEf*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                "pplt.plot_ahc(files if len(files) > 1 else files[0], title='Anomalous Hall conductivity', x_lim=_ewin(), y_lim=_ylim())",
            ],
        )
        menu.append(('Anomalous Hall conductivity', 'plot_anomalous_hall'))

    if 'topology' in props:
        berry_body = [
            "fb = _one('*.Omega_*.dat')",
            'if _missing(fb):',
            '    return',
            "sp = _one('*.kpath_points.txt')",
            "pplt.plot_berry(fb, sym_points=sp, title='Berry curvature', y_lim=_ylim())",
        ]
        if has_bands:
            berry_body += [
                "fband = _one('*.bands_0.dat')",
                'if fband is not None:',
                '    pplt.plot_berry_under_bands(fb, fband, sym_points=sp, dos_ticks=True,',
                "                                title='Berry curvature under bands', y_lim=_ewin())",
            ]
        funcs += _plot_func('plot_berry_curvature', berry_body)
        menu.append(('Berry curvature', 'plot_berry_curvature'))

    if has_optical:
        # Each optical quantity written by dielectric_tensor() gets its own
        # top-level menu entry: (menu label, function name, [file globs],
        # y-axis limits). Directional spectra (those with '_th' in the name)
        # are handled separately by the emissivity menu.
        optical_groups = [
            (
                'Dielectric function',
                'plot_dielectric_function',
                ['epsi_*.dat', 'epsr_*.dat'],
                '_ylim()',
            ),
            (
                'Refractive index (n, kappa)',
                'plot_refractive_index',
                ['nref_*.dat', 'kref_*.dat'],
                '_ylim()',
            ),
            ('EELS', 'plot_eels', ['eels_*.dat'], '_ylim()'),
            ('Absorption coefficient', 'plot_absorption', ['alpha_*.dat'], '_ylim()'),
            ('Reflectivity', 'plot_reflectivity', ['refl_*.dat'], '(0.0, 1.0)'),
            (
                'Optical conductivity',
                'plot_optical_conductivity',
                ['sigmar_*.dat', 'sigmai_*.dat'],
                '_ylim()',
            ),
        ]
        for label, fname, globs, ylim in optical_groups:
            glob_calls = ' + '.join('_many({!r})'.format(gl) for gl in globs)
            funcs += _plot_func(
                fname,
                [
                    'files = {}'.format(glob_calls),
                    "files = [f for f in files if '_th' not in os.path.basename(f)]",
                    'if not files:',
                    '    _missing(None)',
                    '    return',
                    'pplt.plot_dielectric(files if len(files) > 1 else files[0],',
                    '                     title={!r}, x_lim=_ewin_optical(), y_lim={})'.format(
                        label, ylim
                    ),
                ],
            )
            menu.append((label, fname))

        # Perceived visible color (sRGB) derived from the reflectivity spectrum.
        funcs += _plot_func(
            'plot_visible_color',
            [
                "files = [f for f in _many('refl_*.dat') if '_th' not in os.path.basename(f)]",
                'if not files:',
                '    _missing(None)',
                '    return',
                'pplt.optical_color(path=OUTPUTDIR, component="avg", illuminant="E",',
                "                   title='Perceived color')",
            ],
        )
        menu.append(('Visible color (sRGB swatch)', 'plot_visible_color'))

    if has_emissivity:
        funcs += _plot_func(
            'plot_emissivity',
            [
                '# Spectral hemispherical and directional emissivity (energy axis).',
                "files = _many('emish_*.dat') + _many('emis_th*_*.dat')",
                'if not files:',
                '    _missing(None)',
                '    return',
                'pplt.plot_dielectric(files if len(files) > 1 else files[0],',
                "                     title='Emissivity', x_lim=_ewin_optical(), y_lim=(0.0, 1.0),",
                '                     legend_outside=True)',
                '# Total hemispherical emissivity vs temperature: overlay the',
                '# x/y/z tensor components (per spin channel) on one figure.',
                'by_spin = {}',
                "for tf in _many('emist_*.dat'):",
                '    parts = os.path.basename(tf)[:-4].split("_")  # emist_<comp>[_<spin>]',
                '    comp = parts[1]',
                '    sp = int(parts[2]) if len(parts) > 2 else None',
                '    by_spin.setdefault(sp, []).append(comp)',
                'for sp, comps in by_spin.items():',
                "    pplt.plot_optical(['emist'], path=OUTPUTDIR, component=sorted(set(comps)),",
                "                      spin=sp, title='Total hemispherical emissivity', y_lim=(0.0, 1.0))",
            ],
        )
        menu.append(('Emissivity (spectral & directional)', 'plot_emissivity'))

    if 'fermi_surface' in props:
        funcs += _plot_func(
            'plot_fermi_surface',
            [
                "files = _many('FermiSurf_*.bxsf')",
                'if not files:',
                '    _missing(None)',
                '    return',
                "print('Fermi surface BXSF file(s) written; open in XCrySDen:')",
                'for f in files:',
                "    print('  xcrysden --bxsf {}'.format(f))",
            ],
        )
        menu.append(('Fermi surface (XCrySDen)', 'plot_fermi_surface'))

    if 'spin_texture' in props:
        funcs += _plot_func(
            'plot_spin_texture',
            [
                "f = _one('spin-texture-bands.dat')",
                'if _missing(f):',
                '    return',
                "print('Spin-texture data written to {}.'.format(f))",
                "print('Visualize the k-resolved spin vectors with your preferred tool.')",
            ],
        )
        menu.append(('Spin texture (data file)', 'plot_spin_texture'))

    header = PLOT_HEADER.replace('__OUTPUTDIR__', repr(cfg['outputdir']))
    header = header.replace('__EMIN__', repr(float(cfg.get('emin', -8.0))))
    header = header.replace('__EMAX__', repr(float(cfg.get('emax', 4.0))))
    lines = [header]
    lines.append('')
    lines.extend(funcs)

    # Menu registry and interactive driver.
    lines.append('# Ordered menu of available plots: (label, function).')
    lines.append('PLOTS = [')
    for label, fname in menu:
        lines.append('    ({!r}, {}),'.format(label, fname))
    lines.append(']')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append('    global EMIN, EMAX, YMIN, YMAX')
    lines.append('    parser = argparse.ArgumentParser(')
    lines.append("        description='Plot PAOFLOW results with optional axis limits.')")
    lines.append("    parser.add_argument('--emin', type=float, default=None,")
    lines.append("                        help='Lower energy-axis limit (eV, rel. E_F).')")
    lines.append("    parser.add_argument('--emax', type=float, default=None,")
    lines.append("                        help='Upper energy-axis limit (eV, rel. E_F).')")
    lines.append("    parser.add_argument('--ymin', type=float, default=None,")
    lines.append("                        help='Lower property y-axis limit (default: automatic)')")
    lines.append("    parser.add_argument('--ymax', type=float, default=None,")
    lines.append("                        help='Upper property y-axis limit (default: automatic)')")
    lines.append("    parser.add_argument('--all', action='store_true',")
    lines.append("                        help='Plot every available quantity without prompting.')")
    lines.append('    args = parser.parse_args()')
    lines.append('    # Default the energy window to the DOS data range (so the DOS fills the')
    lines.append('    # side-by-side panel) unless it was overridden on the command line.')
    lines.append('    _win = _default_energy_window() or (EMIN, EMAX)')
    lines.append('    EMIN = args.emin if args.emin is not None else _win[0]')
    lines.append('    EMAX = args.emax if args.emax is not None else _win[1]')
    lines.append('    YMIN, YMAX = args.ymin, args.ymax')
    lines.append('    # Ask for the axis limits interactively (Enter keeps the shown default;')
    lines.append('    # any value passed on the command line becomes that default).')
    lines.append('    if not args.all:')
    lines.append("        print('Axis limits (press Enter to accept the default):')")
    lines.append("        EMIN = _ask_float('  Energy axis EMIN (eV, rel. E_F)', EMIN)")
    lines.append("        EMAX = _ask_float('  Energy axis EMAX (eV, rel. E_F)', EMAX)")
    lines.append("        YMIN = _ask_float('  Property axis YMIN (auto for automatic)', YMIN)")
    lines.append("        YMAX = _ask_float('  Property axis YMAX (auto for automatic)', YMAX)")
    lines.append('    if args.all:')
    lines.append('        chosen = list(range(1, len(PLOTS) + 1))')
    lines.append('    else:')
    lines.append("        print('Available plots:')")
    lines.append('        for i, (label, _fn) in enumerate(PLOTS, 1):')
    lines.append("            print('  {}: {}'.format(i, label))")
    lines.append('        try:')
    lines.append(
        '            raw = input("Enter a comma/space separated list (or \'all\'): ").strip()'
    )
    lines.append('        except EOFError:')
    lines.append("            raw = ''")
    lines.append("        if raw.lower() == 'all':")
    lines.append('            chosen = list(range(1, len(PLOTS) + 1))')
    lines.append('        else:')
    lines.append('            chosen = []')
    lines.append("            for tok in raw.replace(',', ' ').split():")
    lines.append('                try:')
    lines.append('                    idx = int(tok)')
    lines.append('                except ValueError:')
    lines.append('                    continue')
    lines.append('                if 1 <= idx <= len(PLOTS) and idx not in chosen:')
    lines.append('                    chosen.append(idx)')
    lines.append('    if not chosen:')
    lines.append("        print('Nothing selected.')")
    lines.append('        return')
    lines.append('    for idx in chosen:')
    lines.append('        label, fn = PLOTS[idx - 1]')
    lines.append("        print('\\n== {} =='.format(label))")
    lines.append('        fn()')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')
    lines.append('')

    return '\n'.join(lines)


def build_acbn0_plot_script(cfg):
    """Assemble a plot.acbn0.py that overlays the ACBN0 / eACBN0 band structures.

    The ACBN0 driver writes one band-structure file per converged case
    (``bands_U`` for the on-site-U solution and, for eACBN0, ``bands_UV`` for
    the joint U+V solution).  This script overlays whichever of those cases are
    present so they can be compared directly on a single figure.
    """
    body = [
        'import numpy as np  # noqa: F401',
        'import matplotlib.pyplot as plt',
        'from PAOFLOW.inputs.read_pao_output import read_bands_PAO, read_band_path_PAO',
        '',
        "files = _many('*.bands_*_0.dat')",
        'if not files:',
        '    _missing(None)',
        '    return',
        '# Parse the case label embedded in each file name (bands_<label>_0.dat).',
        'cases = []',
        'for fn in files:',
        "    stem = os.path.basename(fn)[:-4]  # drop '.dat'",
        "    stem = stem[stem.find('bands_') + len('bands_'):]  # drop prefix + 'bands_'",
        "    if stem.endswith('_0'):",
        '        stem = stem[:-2]',
        '    cases.append((stem, fn))',
        '# Show the on-site-U case before the longer U+V label.',
        'cases.sort(key=lambda kv: (len(kv[0]), kv[0]))',
        "sym = _one('*.kpath_points.txt')",
        'sym_points = read_band_path_PAO(sym) if sym else None',
        'y_lim = _ewin()',
        'fig = plt.figure()',
        'ax = fig.add_subplot(111)',
        'nk = 1',
        'for i, (label, fn) in enumerate(cases):',
        '    col = _SPIN_COLORS[i % len(_SPIN_COLORS)]',
        '    bands = read_bands_PAO(fn)',
        '    nk = bands.shape[1]',
        '    nice = _CASE_LABELS.get(label, label)',
        '    for j, b in enumerate(bands):',
        '        ax.plot(b, color=col, label=nice if j == 0 else None)',
        'ax.set_xlim(0, nk - 1)',
        'ax.set_ylim(*y_lim)',
        'if sym_points is None:',
        '    ax.xaxis.set_visible(False)',
        'else:',
        '    ax.set_xticks(sym_points[0])',
        '    ax.set_xticklabels(sym_points[1])',
        "    ax.vlines(sym_points[0], y_lim[0], y_lim[1], color='gray')",
        "ax.set_ylabel(r'$\\epsilon$($\\mathbf{k}$) (eV)', fontsize=12)",
        "ax.set_title('ACBN0 band-structure comparison')",
        'ax.legend()',
        'plt.tight_layout()',
        'plt.show()',
    ]

    header = PLOT_HEADER.replace('__OUTPUTDIR__', repr(cfg['outputdir']))
    header = header.replace('__EMIN__', repr(float(cfg.get('emin', -8.0))))
    header = header.replace('__EMAX__', repr(float(cfg.get('emax', 4.0))))
    lines = [header]
    lines.append('')
    lines.append('# Human-readable names for the ACBN0 band-structure cases.')
    lines.append('_CASE_LABELS = {')
    lines.append("    'U': 'DFT+U (ACBN0)',")
    lines.append("    'UV': 'DFT+U+V (eACBN0)',")
    lines.append('}')
    lines.append('')
    lines.append('')
    lines.extend(_plot_func('plot_acbn0_bands', body))
    lines.append('')
    lines.append('def main():')
    lines.append('    global EMIN, EMAX')
    lines.append('    parser = argparse.ArgumentParser(')
    lines.append("        description='Compare ACBN0 / eACBN0 band structures.')")
    lines.append("    parser.add_argument('--emin', type=float, default=None,")
    lines.append("                        help='Lower energy-axis limit (eV, rel. E_F).')")
    lines.append("    parser.add_argument('--emax', type=float, default=None,")
    lines.append("                        help='Upper energy-axis limit (eV, rel. E_F).')")
    lines.append('    args = parser.parse_args()')
    lines.append('    EMIN = args.emin if args.emin is not None else EMIN')
    lines.append('    EMAX = args.emax if args.emax is not None else EMAX')
    lines.append("    print('Axis limits (press Enter to accept the default):')")
    lines.append("    EMIN = _ask_float('  Energy axis EMIN (eV, rel. E_F)', EMIN)")
    lines.append("    EMAX = _ask_float('  Energy axis EMAX (eV, rel. E_F)', EMAX)")
    lines.append('    plot_acbn0_bands()')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')
    lines.append('')
    return '\n'.join(lines)


def build_phonon_plot_script(cfg):
    """Assemble a plot.phonon.py that plots the dispersion, DOS and thermals.

    Reads the ``phonon_band.dat`` / ``phonon_dos.dat`` / ``phonon_band.labels``
    / ``phonon_thermal.dat`` files written by ``main.phonon.py`` and renders
    them through :class:`PAOFLOW.GPAO.GPAO`.
    """
    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""Plot the PAOFLOW phonon results (generated by paoflow_gen.py).')
    lines.append('')
    lines.append('Plots the phonon dispersion and DOS and, when available, the harmonic')
    lines.append('thermal properties (free energy, entropy, heat capacity).')
    lines.append('')
    lines.append('Run it after main.phonon.py has produced its output files:')
    lines.append('')
    lines.append('    python plot.phonon.py')
    lines.append('')
    lines.append('Restrict the frequency axis or save to a file with:')
    lines.append('')
    lines.append('    python plot.phonon.py --ymin 0 --ymax 550 --save phonon.png')
    lines.append('"""')
    lines.append('')
    lines.append('import argparse')
    lines.append('import os')
    lines.append('')
    lines.append('from PAOFLOW import GPAO')
    lines.append('')
    lines.append('HERE = os.path.dirname(os.path.abspath(__file__))')
    lines.append('OUTPUTDIR = os.path.join(HERE, {!r})'.format(cfg['outputdir']))
    lines.append("FNAME = 'phonon'   # output filename prefix used by main.phonon.py")
    lines.append('UNITS = {!r}'.format(cfg.get('units', 'cm-1')))
    lines.append('')
    lines.append('pplt = GPAO.GPAO()')
    lines.append('')
    lines.append('')
    lines.append('def _path(suffix):')
    lines.append('    return os.path.join(OUTPUTDIR, FNAME + suffix)')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append('    parser = argparse.ArgumentParser(')
    lines.append("        description='Plot the phonon dispersion, DOS and thermal properties.')")
    lines.append("    parser.add_argument('--ymin', type=float, default=None,")
    lines.append("                        help='Lower frequency-axis limit.')")
    lines.append("    parser.add_argument('--ymax', type=float, default=None,")
    lines.append("                        help='Upper frequency-axis limit.')")
    lines.append("    parser.add_argument('--save', default=None,")
    lines.append(
        "                        help='Save the figure to this file instead of showing it.')"
    )
    lines.append('    args = parser.parse_args()')
    lines.append('')
    lines.append("    band = _path('_band.dat')")
    lines.append('    if not os.path.isfile(band):')
    lines.append('        print("{} not found; run main.phonon.py first.".format(band))')
    lines.append('        return')
    lines.append("    dos = _path('_dos.dat')")
    lines.append("    labels = _path('_band.labels')")
    lines.append('')
    lines.append('    y_lim = None')
    lines.append('    if args.ymin is not None or args.ymax is not None:')
    lines.append('        y_lim = (args.ymin, args.ymax)')
    lines.append('')
    lines.append('    pplt.plot_phonons(')
    lines.append('        band,')
    lines.append('        dos_file=dos if os.path.isfile(dos) else None,')
    lines.append('        labels_file=labels if os.path.isfile(labels) else None,')
    lines.append("        title='Phonon dispersion',")
    lines.append('        units=UNITS,')
    lines.append('        y_lim=y_lim,')
    lines.append('        filename=args.save,')
    lines.append('    )')
    lines.append('')
    lines.append("    thermal = _path('_thermal.dat')")
    lines.append('    if os.path.isfile(thermal):')
    lines.append('        save_thermal = None')
    lines.append('        if args.save:')
    lines.append('            stem, ext = os.path.splitext(args.save)')
    lines.append("            save_thermal = stem + '_thermal' + (ext or '.png')")
    lines.append('        pplt.plot_phonon_thermal(')
    lines.append('            thermal,')
    lines.append("            title='Thermal properties',")
    lines.append('            filename=save_thermal,')
    lines.append('        )')
    lines.append('')
    lines.append('    # Quasi-harmonic quantities: V(T), thermal expansion, bulk modulus, Cp and')
    lines.append('    # the Gruneisen parameter, plus the static E-V curve.')
    lines.append("    qha_volume = os.path.join(OUTPUTDIR, 'qha_volume.dat')")
    lines.append('    if os.path.isfile(qha_volume):')
    lines.append('        save_qha = None')
    lines.append('        if args.save:')
    lines.append('            stem, ext = os.path.splitext(args.save)')
    lines.append("            save_qha = stem + '_qha' + (ext or '.png')")
    lines.append('        pplt.plot_qha(')
    lines.append('            volume_file=qha_volume,')
    lines.append(
        "            thermal_expansion_file=os.path.join(OUTPUTDIR, 'qha_thermal_expansion.dat'),"
    )
    lines.append("            bulk_modulus_file=os.path.join(OUTPUTDIR, 'qha_bulk_modulus.dat'),")
    lines.append("            heat_capacity_file=os.path.join(OUTPUTDIR, 'qha_heat_capacity.dat'),")
    lines.append("            gruneisen_file=os.path.join(OUTPUTDIR, 'qha_gruneisen.dat'),")
    lines.append("            ev_file=os.path.join(OUTPUTDIR, 'qha_ev.dat'),")
    lines.append("            title='Quasi-harmonic approximation',")
    lines.append('            filename=save_qha,')
    lines.append('        )')
    lines.append('')
    lines.append('    # Mode Gruneisen parameters along the q-path (dispersion style).')
    lines.append("    qha_gband = os.path.join(OUTPUTDIR, 'qha_gruneisen_band.dat')")
    lines.append('    if os.path.isfile(qha_gband):')
    lines.append('        save_gband = None')
    lines.append('        if args.save:')
    lines.append('            stem, ext = os.path.splitext(args.save)')
    lines.append("            save_gband = stem + '_gruneisen_band' + (ext or '.png')")
    lines.append("        gband_labels = os.path.join(OUTPUTDIR, 'qha_gruneisen_band.labels')")
    lines.append('        pplt.plot_gruneisen_band(')
    lines.append('            qha_gband,')
    lines.append('            labels_file=gband_labels if os.path.isfile(gband_labels) else None,')
    lines.append("            title='Mode Gruneisen parameters',")
    lines.append('            filename=save_gband,')
    lines.append('        )')
    lines.append('')
    lines.append("    ir = _path('_ir_spectrum.dat')")
    lines.append('    if os.path.isfile(ir):')
    lines.append('        save_ir = None')
    lines.append('        if args.save:')
    lines.append('            stem, ext = os.path.splitext(args.save)')
    lines.append("            save_ir = stem + '_ir' + (ext or '.png')")
    lines.append('        pplt.plot_ir_spectrum(')
    lines.append('            ir,')
    lines.append("            modes_file=_path('_ir_modes.dat'),")
    lines.append("            title='Infrared spectrum',")
    lines.append('            units=UNITS,')
    lines.append('            filename=save_ir,')
    lines.append('        )')
    lines.append('')
    lines.append('    # Vibrational (ionic) dielectric eps(w): overlay Re/Im eps to show the')
    lines.append('    # reststrahlen band (Re eps < 0 between the TO and LO frequencies).')
    lines.append("    vibdir = os.path.join(OUTPUTDIR, 'vibdielectric')")
    lines.append("    if os.path.isfile(os.path.join(vibdir, 'epsr_xx.dat')):")
    lines.append('        pplt.plot_optical(')
    lines.append("            ['epsr', 'epsi'],")
    lines.append('            path=vibdir,')
    lines.append("            component='xx',")
    lines.append("            title='Vibrational dielectric function',")
    lines.append('        )')
    lines.append('')
    lines.append('    # Reststrahlen (phonon) emissivity (1 - R), when computed.')
    lines.append("    if os.path.isfile(os.path.join(vibdir, 'emish_xx.dat')):")
    lines.append('        pplt.plot_optical(')
    lines.append("            ['emish'],")
    lines.append('            path=vibdir,')
    lines.append("            component='xx',")
    lines.append("            title='Vibrational (reststrahlen) emissivity',")
    lines.append('        )')
    lines.append('')
    lines.append('')
    lines.append('if __name__ == "__main__":')
    lines.append('    main()')
    lines.append('')
    return '\n'.join(lines)


def build_raman_script(cfg):
    """Assemble a main.raman.py non-resonant (Placzek) Raman workflow.

    The generated script displaces the primitive cell by ``+/-delta`` along
    every optical zone-centre eigenvector (read from the harmonic-phonon
    ``FORCE_SETS``), writes an SCF-only QE input per displacement, and -- in the
    analyse phase -- runs the PAOFLOW internal-projection optical pipeline on
    each displaced cell to finite-difference the Raman tensor.
    """
    nbnd = int(cfg.get('raman_nbnd', 0) or 0)
    nfft = int(cfg.get('raman_nfft', 0) or 0)
    method = cfg.get('raman_method', 'static')
    title = {
        'static': 'non-resonant (Placzek)',
        'resonance': 'resonance',
        'all': 'static + resonance',
    }.get(method, 'non-resonant (Placzek)')
    laser = cfg.get('raman_laser_nm', None)
    if laser is None:
        laser_repr = 'None'
    elif isinstance(laser, (list, tuple)):
        laser_repr = repr([float(x) for x in laser])
    else:
        laser_repr = repr(float(laser))

    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""PAOFLOW {} Raman workflow (generated by paoflow_gen.py).'.format(title))
    lines.append('')
    lines.append('Finite-difference Raman spectrum: the primitive cell is displaced by')
    lines.append('``+/-delta`` along every optical zone-centre eigenvector and the static')
    lines.append('dielectric tensor of each displaced cell is computed with PAOFLOW internal')
    lines.append('projections (a fast SCF-only optical run -- no NSCF, no projwfc).  The Raman')
    lines.append('tensor of each mode follows from a central difference of the dielectric tensor.')
    lines.append('')
    lines.append('Three phases, which can be invoked separately:')
    lines.append('')
    lines.append('    python main.raman.py generate   # write displaced primitive-cell SCF inputs')
    lines.append('    python main.raman.py run        # run pw.x (SCF) on every displaced cell')
    lines.append('    python main.raman.py analyse    # PAOFLOW epsilon per cell -> Raman spectrum')
    lines.append('')
    lines.append('``all`` (the default) runs the three phases in sequence:')
    lines.append('')
    lines.append('    python main.raman.py')
    lines.append('')
    lines.append('The harmonic force constants (zone-centre eigenvectors) are read from the')
    lines.append('``FORCE_SETS`` produced by the harmonic-phonon workflow (main.phonon.py), so')
    lines.append('run that first (at least its ``analyse`` phase).  All lengths follow the')
    lines.append('phonopy QE convention (Bohr).')
    lines.append('"""')
    lines.append('')
    lines.append('import argparse')
    lines.append('import glob')
    lines.append('import os')
    lines.append('import subprocess')
    lines.append('import sys')
    lines.append('')
    lines.append('from PAOFLOW import PAOFLOW')
    lines.append('from PAOFLOW.basis_gen import generate_basis_for_pseudo')
    lines.append('from PAOFLOW.basis_gen.driver import _default_shells')
    lines.append('from PAOFLOW.inputs.read_upf import UPF as _UPFParser')
    lines.append('')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('# Configuration  (edit freely)                                            #')
    lines.append('# ----------------------------------------------------------------------- #')
    lines.append('HERE = os.path.dirname(os.path.abspath(__file__))')
    lines.append('SAVEDIR = os.path.join(HERE, {!r})'.format(cfg['savedir']))
    lines.append(_format_upfs_line(cfg['upfs']))
    lines.append('BASISPATH = os.path.join(HERE, {!r}) + os.sep'.format(cfg['basisdir']))
    lines.append('OUTPUTDIR = {!r}'.format(cfg['outputdir']))
    lines.append("PHONON_DIR = 'phonon'   # harmonic-phonon sub-directory (holds FORCE_SETS)")
    lines.append("RAMAN_DIR = 'raman'     # sub-directory (under OUTPUTDIR) for displaced cells")
    lines.append('PREFIX = {!r}'.format(cfg['prefix']))
    lines.append(_pp_dir_line(cfg))
    lines.append('')
    lines.append('# Finite-displacement settings.')
    lines.append(
        'SUPERCELL_MATRIX = {}   # must match the harmonic-phonon FORCE_SETS'.format(
            cfg['supercell']
        )
    )
    lines.append(
        'DELTA = {}   # mass-weighted normal-coordinate step (Bohr*sqrt(amu))'.format(
            cfg.get('raman_delta', 0.05)
        )
    )
    if nbnd > 0:
        lines.append(
            'NBND = {}   # bands for the displaced SCF (empty states for optics)'.format(nbnd)
        )
    else:
        lines.append('NBND = None   # bands for the displaced SCF (None = QE default)')
    lines.append(
        'UNITS = {!r}   # frequency units for the outputs'.format(cfg.get('units', 'cm-1'))
    )
    lines.append('')
    lines.append('# PAOFLOW optical-pipeline settings for the per-cell dielectric tensor.')
    lines.append('SMEARING = {!r}'.format(cfg.get('raman_smearing', 'gauss')))
    lines.append('NPOOL = {}'.format(cfg.get('raman_npool', 4)))
    lines.append('PTHR = {}   # projectability threshold'.format(cfg.get('raman_pthr', 0.95)))
    lines.append(
        'CONFIGURATION = {!r}   # PAO basis configuration for optics'.format(
            cfg.get('raman_configuration', 'extended')
        )
    )
    if nfft > 0:
        lines.append(
            'NFFT = ({n}, {n}, {n})   # double-grid interpolation; None to skip'.format(n=nfft)
        )
    else:
        lines.append('NFFT = None   # double-grid interpolation; None to skip')
    lines.append(
        'E_STATIC = {}   # eV; upper bound of the 2-point grid for epsilon(0)'.format(
            cfg.get('raman_e_static', 0.05)
        )
    )
    lines.append('')
    lines.append('# Raman flavour: "static" (non-resonant Placzek) or "resonance".')
    lines.append('# Resonance evaluates eps at the laser frequency and needs LASER_NM set.')
    lines.append('METHOD = {!r}'.format(cfg.get('raman_method', 'static')))
    lines.append(
        'LIFETIME = {}   # eV; Lorentzian broadening of eps(omega_L) (resonance)'.format(
            cfg.get('raman_lifetime', 0.1)
        )
    )
    lines.append(
        'E_WINDOW = None   # eV; upper bound of the per-cell eps grid (resonance; None=auto)'
    )
    lines.append('E_NE = None       # points on the per-cell eps grid (resonance; None=auto)')
    lines.append('')
    lines.append('# Raman cross-section options.')
    lines.append(
        'LASER_NM = {}   # excitation wavelength(s) in nm; None, a float, or a list.'.format(
            laser_repr
        )
    )
    lines.append(
        'TEMPERATURE = {}   # K, Bose (n+1) Stokes factor'.format(
            cfg.get('raman_temperature', 300.0)
        )
    )
    lines.append('GAMMA = {}   # Lorentzian FWHM (in UNITS)'.format(cfg.get('raman_gamma', 4.0)))
    lines.append('')
    lines.append('# Force constants for the zone-centre eigenvectors.')
    lines.append("FORCE_SETS = os.path.join(HERE, OUTPUTDIR, PHONON_DIR, 'FORCE_SETS')")
    lines.append('')
    lines.append('# pw.x launch settings for the displaced-cell SCF runs (edit to your machine).')
    lines.append('MPI_QE = {!r}'.format(cfg['mpi_qe']))
    lines.append(
        "QE_PW = {!r}   # pw.x command (may include flags, e.g. 'pw.x -npool 4')".format(
            cfg.get('qe_pw', 'pw.x')
        )
    )
    lines.append('')
    lines.append('')
    lines.append("def ensure_basis(preset='extended'):")
    lines.append('    """Generate the pseudo-atom basis under BASISPATH for every species."""')
    lines.append('    for upf_path in UPFS:')
    lines.append('        upf = _UPFParser(upf_path)')
    lines.append('        element = upf.element.strip()')
    lines.append('        elem_dir = os.path.join(BASISPATH, element)')
    lines.append('        expected = _default_shells(upf, preset=preset)')
    lines.append('        missing = [')
    lines.append('            s for s in expected')
    lines.append("            if not os.path.exists(os.path.join(elem_dir, '{}.dat'.format(s)))")
    lines.append('        ]')
    lines.append('        if missing:')
    lines.append("            print('Generating pseudo-atom basis for {} under {} ...'.format(")
    lines.append('                element, BASISPATH))')
    lines.append('            generate_basis_for_pseudo(')
    lines.append('                upf_path, BASISPATH.rstrip(os.sep), preset=preset, verbose=True')
    lines.append('            )')
    lines.append('')
    lines.append('')
    lines.append('def _make_paoflow():')
    lines.append('    """Fresh PAOFLOW instance reading the unit-cell structure from SAVEDIR."""')
    lines.append('    return PAOFLOW.PAOFLOW(')
    lines.append('        workpath=HERE,')
    lines.append('        outputdir=OUTPUTDIR,')
    lines.append('        savedir=SAVEDIR,')
    lines.append('        smearing=SMEARING,')
    lines.append('        npool=NPOOL,')
    lines.append('        verbose=True,')
    lines.append('    )')
    lines.append('')
    lines.append('')
    lines.append('def _forces():')
    lines.append('    """Force source: a pre-existing FORCE_SETS when present, else QE harvest."""')
    lines.append("    return FORCE_SETS if os.path.isfile(FORCE_SETS) else 'qe'")
    lines.append('')
    lines.append('')
    lines.append('def generate():')
    lines.append('    """Phase 1: write a QE SCF input for every +/- displaced primitive cell."""')
    lines.append('    p = _make_paoflow()')
    lines.append('    p.raman_spectrum(')
    lines.append('        supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('        forces=_forces(),')
    lines.append('        phonon_dir=PHONON_DIR,')
    lines.append('        raman_dir=RAMAN_DIR,')
    lines.append('        delta=DELTA,')
    lines.append('        nbnd=NBND,')
    lines.append('        units=UNITS,')
    lines.append('        generate=True,')
    lines.append('    )')
    lines.append('')
    lines.append('')
    lines.append('def _job_done(path):')
    lines.append('    """True when a pw.x output exists and reached JOB DONE."""')
    lines.append('    try:')
    lines.append('        with open(path) as fh:')
    lines.append("            return 'JOB DONE' in fh.read()")
    lines.append('    except OSError:')
    lines.append('        return False')
    lines.append('')
    lines.append('')
    lines.append('def run_cells():')
    lines.append('    """Phase 2: run pw.x (SCF) in every displaced-cell directory."""')
    lines.append('    base = os.path.join(HERE, OUTPUTDIR, RAMAN_DIR)')
    lines.append("    inputs = sorted(glob.glob(os.path.join(base, 'mode-*', PREFIX + '.scf.in')))")
    lines.append('    if not inputs:')
    lines.append(
        "        print('No displaced-cell inputs in {}; run the generate phase first.'.format(base))"
    )
    lines.append('        sys.exit(1)')
    lines.append('    pwx = QE_PW')
    lines.append('    for fin in inputs:')
    lines.append('        cell_dir = os.path.dirname(fin)')
    lines.append("        fout = os.path.join(cell_dir, PREFIX + '.scf.out')")
    lines.append('        if _job_done(fout):')
    lines.append("            print('  skip (already done):', os.path.relpath(fout, base))")
    lines.append('            continue')
    lines.append("        cmd = '{mpi} {pwx} -in {inp} > {out}'.format(")
    lines.append('            mpi=MPI_QE, pwx=pwx,')
    lines.append("            inp=PREFIX + '.scf.in', out=PREFIX + '.scf.out')")
    lines.append("        print('  running:', os.path.relpath(fin, base))")
    lines.append('        subprocess.run(cmd, shell=True, check=True, cwd=cell_dir)')
    lines.append('')
    lines.append('')
    lines.append('def analyse():')
    lines.append('    """Phase 3: PAOFLOW dielectric per cell -> Raman tensor -> spectrum."""')
    lines.append('    p = _make_paoflow()')
    lines.append('    p.raman_spectrum(')
    lines.append('        supercell_matrix=SUPERCELL_MATRIX,')
    lines.append('        forces=_forces(),')
    lines.append('        phonon_dir=PHONON_DIR,')
    lines.append('        raman_dir=RAMAN_DIR,')
    lines.append('        delta=DELTA,')
    lines.append('        basispath=BASISPATH,')
    lines.append('        configuration=CONFIGURATION,')
    lines.append('        pthr=PTHR,')
    lines.append('        nfft=NFFT,')
    lines.append('        e_static=E_STATIC,')
    lines.append('        method=METHOD,')
    lines.append('        lifetime=LIFETIME,')
    lines.append('        e_window=E_WINDOW,')
    lines.append('        e_ne=E_NE,')
    lines.append('        laser_nm=LASER_NM,')
    lines.append('        temperature=TEMPERATURE,')
    lines.append('        gamma=GAMMA,')
    lines.append('        units=UNITS,')
    lines.append("        fname='phonon',")
    lines.append('        generate=False,')
    lines.append('    )')
    lines.append('    p.finish_execution()')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append('    parser = argparse.ArgumentParser(')
    lines.append('        description=__doc__,')
    lines.append('        formatter_class=argparse.RawDescriptionHelpFormatter)')
    lines.append("    parser.add_argument('phase', nargs='?', default='all',")
    lines.append("                        choices=['generate', 'run', 'analyse', 'all'],")
    lines.append("                        help='workflow phase to execute (default: all)')")
    lines.append('    args = parser.parse_args()')
    lines.append('')
    lines.append('    if not os.path.isdir(SAVEDIR):')
    lines.append(
        "        print('{} not found. Run the SCF for the primitive cell first.'.format(SAVEDIR))"
    )
    lines.append('        sys.exit(1)')
    lines.append('')
    lines.append("    if args.phase in ('generate', 'analyse', 'all'):")
    lines.append("        ensure_basis(preset='extended')")
    lines.append('')
    lines.append("    if args.phase in ('generate', 'all'):")
    lines.append('        generate()')
    lines.append("    if args.phase in ('run', 'all'):")
    lines.append('        run_cells()')
    lines.append("    if args.phase in ('analyse', 'all'):")
    lines.append('        analyse()')
    lines.append('')
    lines.append('')
    lines.append("if __name__ == '__main__':")
    lines.append('    main()')
    lines.append('')
    return '\n'.join(lines)


def build_raman_plot_script(cfg):
    """Assemble a plot.raman.py that overlays the broadened Raman spectra."""
    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""Plot the PAOFLOW Raman spectrum (generated by paoflow_gen.py).')
    lines.append('')
    lines.append('Run it after main.raman.py has produced its output files:')
    lines.append('')
    lines.append('    python plot.raman.py')
    lines.append('')
    lines.append('Every spectrum written by main.raman.py is overlaid on a single plot, so a')
    lines.append("method='all' run or a laser excitation profile (laser_nm=[...]) appears as one")
    lines.append('figure with a legend rather than many separate windows.')
    lines.append('')
    lines.append('Restrict the frequency axis, normalise each curve to unit peak, draw the mode')
    lines.append('sticks, or save to a file with:')
    lines.append('')
    lines.append(
        '    python plot.raman.py --xmin 0 --xmax 600 --normalize --sticks --save raman.png'
    )
    lines.append('')
    lines.append('With several laser energies (resonance/all with laser_nm=[...]) pass')
    lines.append('--excitation to instead plot the Raman intensity of each active mode versus')
    lines.append('the laser energy (eV) -- the resonance excitation profile:')
    lines.append('')
    lines.append('    python plot.raman.py --excitation --save excitation.png')
    lines.append('"""')
    lines.append('')
    lines.append('import argparse')
    lines.append('import glob')
    lines.append('import os')
    lines.append('')
    lines.append('import matplotlib.pyplot as plt')
    lines.append('import numpy as np')
    lines.append('')
    lines.append('HERE = os.path.dirname(os.path.abspath(__file__))')
    lines.append('OUTPUTDIR = os.path.join(HERE, {!r})'.format(cfg['outputdir']))
    lines.append("FNAME = 'phonon'   # output filename prefix used by main.raman.py")
    lines.append('UNITS = {!r}'.format(cfg.get('units', 'cm-1')))
    lines.append('EV_NM = 1239.841984   # E(eV) = EV_NM / lambda(nm)')
    ylabel = "'Raman intensity (normalised)' if args.normalize else 'Raman intensity (arb. units)'"
    lines.append('')
    lines.append('')
    lines.append('def _label(path):')
    lines.append('    """Excitation label taken from the file name (static or e.g. 532nm)."""')
    lines.append('    base = os.path.basename(path)[len(FNAME):]')
    lines.append("    base = base[:-len('_raman_spectrum.dat')].strip('_')")
    lines.append("    return base or 'static'")
    lines.append('')
    lines.append('')
    lines.append('def _sort_key(path):')
    lines.append('    """static first, then by ascending laser wavelength (nm)."""')
    lines.append('    label = _label(path)')
    lines.append("    if label.endswith('nm'):")
    lines.append('        try:')
    lines.append('            return (1, float(label[:-2]))')
    lines.append('        except ValueError:')
    lines.append("            return (1, float('inf'))")
    lines.append('    return (0, 0.0)')
    lines.append('')
    lines.append('')
    lines.append('def _laser_ev(label):')
    lines.append('    """Laser energy (eV) from a label like \'532nm\'; None for \'static\'."""')
    lines.append("    if label.endswith('nm'):")
    lines.append('        try:')
    lines.append('            return EV_NM / float(label[:-2])')
    lines.append('        except ValueError:')
    lines.append('            return None')
    lines.append('    return None')
    lines.append('')
    lines.append('')
    lines.append('def plot_excitation(args):')
    lines.append('    """Resonance excitation profile: mode intensity vs laser energy (eV)."""')
    lines.append('    # Columns of <FNAME>*_raman_modes.dat: mode(0) freq(1) ... intensity(4)')
    lines.append('    # ... active(7); the trailing irrep column is a string so we use usecols.')
    lines.append("    files = glob.glob(os.path.join(OUTPUTDIR, FNAME + '*_raman_modes.dat'))")
    lines.append('    points = []   # (laser_eV, mode_index, frequency, intensity, active)')
    lines.append('    for path in files:')
    lines.append('        base = os.path.basename(path)[len(FNAME):]')
    lines.append("        label = base[:-len('_raman_modes.dat')].strip('_') or 'static'")
    lines.append('        ev = _laser_ev(label)')
    lines.append('        if ev is None:')
    lines.append('            continue   # static channel has no laser energy')
    lines.append('        rows = np.atleast_2d(np.loadtxt(path, usecols=(0, 1, 4, 7)))')
    lines.append('        for row in rows:')
    lines.append('            points.append((ev, int(row[0]), row[1], row[2], row[3]))')
    lines.append('    if not points:')
    lines.append(
        "        print('No resonance modes files (<FNAME>_<nm>nm_raman_modes.dat) found; '"
    )
    lines.append("              'run a resonance/all workflow with laser_nm=[...] first.')")
    lines.append('        return')
    lines.append('')
    lines.append('    fig, ax = plt.subplots()')
    lines.append("    fig.suptitle('Resonance Raman excitation profile')")
    lines.append('    plotted = False')
    lines.append('    for mi in sorted({p[1] for p in points}):')
    lines.append('        pts = sorted((p for p in points if p[1] == mi), key=lambda p: p[0])')
    lines.append('        if not any(p[4] > 0.5 for p in pts):')
    lines.append('            continue   # mode is Raman-silent at every laser energy')
    lines.append('        energies = np.array([p[0] for p in pts])')
    lines.append('        inten = np.array([p[3] for p in pts])')
    lines.append('        if args.normalize and inten.max() > 0.0:')
    lines.append('            inten = inten / inten.max()')
    lines.append(
        "        ax.plot(energies, inten, marker='o', label='%.1f %s' % (pts[0][2], UNITS))"
    )
    lines.append('        plotted = True')
    lines.append('    if not plotted:')
    lines.append("        print('No Raman-active modes to plot.')")
    lines.append('        return')
    lines.append('')
    lines.append("    ax.set_xlabel('Laser energy (eV)', fontsize=12)")
    lines.append('    ax.set_ylabel({}, fontsize=12)'.format(ylabel))
    lines.append('    ax.set_ylim(0.0, ax.get_ylim()[1])')
    lines.append('    ax.grid(alpha=0.3)')
    lines.append("    ax.legend(title='mode', fontsize=9)")
    lines.append('    if args.save is not None:')
    lines.append("        plt.savefig(args.save, dpi=300, bbox_inches='tight')")
    lines.append('    plt.show()')
    lines.append('')
    lines.append('')
    lines.append('def main():')
    lines.append("    parser = argparse.ArgumentParser(description='Plot the Raman spectrum.')")
    lines.append("    parser.add_argument('--xmin', type=float, default=None,")
    lines.append("                        help='Lower frequency-axis limit.')")
    lines.append("    parser.add_argument('--xmax', type=float, default=None,")
    lines.append("                        help='Upper frequency-axis limit.')")
    lines.append("    parser.add_argument('--normalize', action='store_true',")
    lines.append(
        "                        help='Scale each curve to unit peak (compare line shapes).')"
    )
    lines.append("    parser.add_argument('--sticks', action='store_true',")
    lines.append(
        "                        help='Also draw the discrete mode intensities as sticks.')"
    )
    lines.append("    parser.add_argument('--excitation', action='store_true',")
    lines.append(
        "                        help='Plot mode intensity vs laser energy (resonance profile).')"
    )
    lines.append("    parser.add_argument('--save', default=None,")
    lines.append(
        "                        help='Save the figure to this file instead of showing it.')"
    )
    lines.append('    args = parser.parse_args()')
    lines.append('')
    lines.append('    if args.excitation:')
    lines.append('        plot_excitation(args)')
    lines.append('        return')
    lines.append('')
    lines.append('    # Discover every spectrum written by main.raman.py: the plain')
    lines.append("    # '<FNAME>_raman_spectrum.dat' (static/single resonance) plus any")
    lines.append(
        "    # '<FNAME>_static_*' / '<FNAME>_<nm>nm_*' (method='all' / excitation profile)."
    )
    lines.append("    spectra = glob.glob(os.path.join(OUTPUTDIR, FNAME + '*_raman_spectrum.dat'))")
    lines.append('    if not spectra:')
    lines.append(
        "        print('No *_raman_spectrum.dat in {}; run main.raman.py first.'.format(OUTPUTDIR))"
    )
    lines.append('        return')
    lines.append('    spectra = sorted(spectra, key=_sort_key)')
    lines.append('    multi = len(spectra) > 1')
    lines.append('')
    lines.append('    fig, ax = plt.subplots()')
    lines.append("    fig.suptitle('Raman spectrum')")
    lines.append('    xmins, xmaxs = [], []')
    lines.append('    for spectrum in spectra:')
    lines.append('        data = np.loadtxt(spectrum)')
    lines.append('        freq, inten = data[:, 0], data[:, 1]')
    lines.append('        if args.normalize and inten.max() > 0.0:')
    lines.append('            inten = inten / inten.max()')
    lines.append('        label = _label(spectrum)')
    lines.append('        line, = ax.plot(freq, inten, label=label)')
    lines.append('        if args.sticks:')
    lines.append(
        "            modes_file = spectrum.replace('_raman_spectrum.dat', '_raman_modes.dat')"
    )
    lines.append('            if os.path.isfile(modes_file):')
    lines.append('                m = np.loadtxt(modes_file, usecols=(1, 4))')
    lines.append('                m = np.atleast_2d(m)')
    lines.append('                stick = m[:, 1]')
    lines.append('                if args.normalize and stick.max() > 0.0:')
    lines.append('                    stick = stick / stick.max()')
    lines.append('                ax.vlines(m[:, 0], 0.0, stick, color=line.get_color(),')
    lines.append('                          linewidth=1.0, alpha=0.5)')
    lines.append('        xmins.append(freq[0])')
    lines.append('        xmaxs.append(freq[-1])')
    lines.append('')
    lines.append('    x_lim = (min(xmins), max(xmaxs))')
    lines.append('    if args.xmin is not None:')
    lines.append('        x_lim = (args.xmin, x_lim[1])')
    lines.append('    if args.xmax is not None:')
    lines.append('        x_lim = (x_lim[0], args.xmax)')
    lines.append('    ax.set_xlim(*x_lim)')
    lines.append('    ax.set_ylim(0.0, ax.get_ylim()[1])')
    lines.append("    ax.set_xlabel('Frequency (%s)' % UNITS, fontsize=12)")
    lines.append('    ax.set_ylabel({}, fontsize=12)'.format(ylabel))
    lines.append('    ax.grid(alpha=0.3)')
    lines.append('    if multi:')
    lines.append("        ax.legend(title='excitation', fontsize=9)")
    lines.append('')
    lines.append('    if args.save is not None:')
    lines.append("        plt.savefig(args.save, dpi=300, bbox_inches='tight')")
    lines.append('    plt.show()')
    lines.append('')
    lines.append('')
    lines.append("if __name__ == '__main__':")
    lines.append('    main()')
    lines.append('')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Config collection
# --------------------------------------------------------------------------- #
def collect_common(args, workdir):
    """Prompt for configuration shared by both workflows."""
    det_save = detect_savedir(workdir)
    det_upfs = detect_upfs(workdir, det_save)
    det_prefix = detect_prefix(workdir, det_save)

    savedir = args.savedir or ask('Save directory (<prefix>.save)', det_save or 'pwscf.save')
    prefix = args.prefix or ask('Prefix', det_prefix)
    if args.upf:
        upfs = [u.strip() for u in args.upf.split(',') if u.strip()]
    else:
        default_upfs = ', '.join(det_upfs) if det_upfs else 'pseudo.UPF'
        answer = ask('Pseudopotential (UPF) file(s), comma-separated', default_upfs)
        upfs = [u.strip() for u in answer.split(',') if u.strip()]
    basisdir = ask('Basis directory name', 'BASIS_PS')
    ibrav = ask_int('ibrav (0 = read cell; needs band path for bands)', 0)
    is_2d = ask_yes_no('Is this a 2D system (slab with vacuum along c)?', False)
    return {
        'savedir': savedir,
        'prefix': prefix,
        'upfs': upfs,
        'basisdir': basisdir,
        'ibrav': ibrav,
        'is_2d': is_2d,
        'outputdir': 'output',
        'workdir': workdir,
    }


def collect_run(common):
    """Prompt for the full property-run configuration."""
    props = select_properties()
    if not props:
        print('No valid properties selected; nothing to generate.')
        return None

    std_basis = 'standard'
    if any(p != 'optical' for p in props):
        std_basis = ask_choice(
            '\nBasis configuration for the standard properties:',
            ['minimal', 'standard', 'extended'],
            'standard',
        )

    cfg = dict(common)
    cfg['properties'] = props
    cfg['std_basis'] = std_basis
    cfg['npool'] = ask_int('npool', 1)
    cfg['smearing'] = ask('Smearing (gauss / m-p / None)', 'gauss')
    if cfg['smearing'].lower() == 'none':
        cfg['smearing'] = None
    cfg['spin_orbit'] = ask_yes_no('Spin-orbit (noncollinear) calculation?', False)
    cfg['nk'] = ask_int('Band path k-points (NK)', 400) if 'bands' in props else 400
    if any(p in ENERGY_WINDOW_PROPERTIES for p in props):
        print('\nEnergy grid for DOS / transport / Hall properties (eV, relative to E_F).')
        print(
            '  The full PAO band range is computed at run time and printed as a '
            'suggestion;\n  the values below are what the calculation actually uses.'
        )
        cfg['emin'] = ask_float('  EMIN', -8.0)
        cfg['emax'] = ask_float('  EMAX', 4.0)
        cfg['ne'] = ask_int('  NE (number of energy points)', 1000)
    else:
        cfg['emin'] = -8.0
        cfg['emax'] = 4.0
        cfg['ne'] = 1000
    cfg['do_pdos'] = ask_yes_no('Projected DOS as well?', True) if 'dos' in props else False
    cfg['interpolate'] = ask_yes_no(
        'Use double-grid interpolation (denser FFT)?',
        'optical' in props,
    )
    cfg['nfft'] = ask_int('FFT grid size per direction', 24) if cfg['interpolate'] else 0
    cfg['plot'] = ask_yes_no('Also generate a plotting script (plot.py)?', True)
    return cfg


def collect_phonon(common):
    """Prompt for the harmonic-phonon (finite-displacement) configuration."""
    cfg = dict(common)
    cfg['supercell'] = ask_int('Supercell multiplicity (isotropic diagonal)', 2)
    cfg['displacement'] = ask_float('Displacement amplitude (Bohr)', 0.06)
    mesh = ask_int('q-mesh density per axis (DOS / thermal)', 12)
    cfg['mesh'] = [mesh, mesh, mesh]
    cfg['units'] = ask_choice('Frequency units', ['cm-1', 'THz'], 'cm-1')
    cfg['do_thermal'] = ask_yes_no('Compute thermal properties (F, S, Cv)?', True)
    cfg['pp_dir'] = ask('Pseudopotential directory (blank or "HERE" = script directory)', 'HERE')
    cfg['hubbard_file'] = ask(
        'pw.x input with a HUBBARD card to inject (on-site U only; blank = none)', ''
    )
    cfg['born'] = ask_yes_no(
        'Compute Born charges + dielectric tensor (NAC / LO-TO splitting)?', True
    )
    if cfg['born']:
        cfg['born_method'] = ask_choice(
            'Born/epsilon method (dfpt = ph.x, field = lelfield pw.x)',
            ['dfpt', 'field'],
            'dfpt',
        )
    else:
        cfg['born_method'] = 'dfpt'
    cfg['vibdielectric'] = cfg['born'] and ask_yes_no(
        'Also compute the vibrational (ionic) dielectric function eps(w) (reststrahlen band)?',
        True,
    )
    if cfg['vibdielectric']:
        cfg['vibdielectric_gamma'] = ask_float(
            '  Phonon linewidth gamma for eps(w) (in the chosen units)', 4.0
        )
        cfg['vibdielectric_emissivity'] = ask_yes_no(
            '  Also compute the reststrahlen (phonon) emissivity (1 - R)?', True
        )
        if cfg['vibdielectric_emissivity']:
            temp = ask_float('    Temperature for the total hemispherical emissivity (K)', 300.0)
            cfg['vibdielectric_emis_temp'] = [temp]
        else:
            cfg['vibdielectric_emis_temp'] = [300.0]
    else:
        cfg['vibdielectric_gamma'] = 4.0
        cfg['vibdielectric_emissivity'] = False
        cfg['vibdielectric_emis_temp'] = [300.0]
    cfg['qha'] = ask_yes_no(
        'Compute quasi-harmonic properties (thermal expansion, V(T), B(T), Gruneisen)?',
        False,
    )
    if cfg['qha']:
        cfg['qha_nvolumes'] = ask_choice(
            '  Number of sampled volumes (5 = full EOS fit; 3 = parabolic fit)',
            ['5', '3'],
            '5',
        )
        cfg['qha_nvolumes'] = int(cfg['qha_nvolumes'])
        cfg['qha_strain'] = ask_float('  Maximum linear strain of the volume scan', 0.02)
        if cfg['qha_nvolumes'] >= 4:
            cfg['qha_eos'] = ask_choice(
                '  Equation of state',
                ['vinet', 'birch_murnaghan', 'murnaghan'],
                'vinet',
            )
        else:
            cfg['qha_eos'] = 'vinet'
        cfg['qha_tmin'] = ask_float('  Minimum temperature (K)', 0.0)
        cfg['qha_tmax'] = ask_float('  Maximum temperature (K)', 1000.0)
        cfg['qha_tstep'] = ask_float('  Temperature step (K)', 10.0)
        cfg['qha_pressure'] = ask_float('  External pressure (GPa)', 0.0)
    else:
        cfg['qha_nvolumes'] = 5
        cfg['qha_strain'] = 0.02
        cfg['qha_eos'] = 'vinet'
        cfg['qha_tmin'] = 0.0
        cfg['qha_tmax'] = 1000.0
        cfg['qha_tstep'] = 10.0
        cfg['qha_pressure'] = 0.0
    print('\npw.x launch settings for the displaced-supercell SCF runs:')
    cfg['mpi_qe'] = ask('MPI command for pw.x', 'mpirun -np 4')
    cfg['qe_pw'] = ask(
        "pw.x executable/command (may include flags, e.g. 'pw.x', 'pw.x -npool 4', "
        "'/opt/qe/bin/pw.x')",
        'pw.x',
    )
    cfg['plot'] = ask_yes_no('Also generate a plotting script (plot.phonon.py)?', True)
    cfg['raman_method'] = ask_choice(
        'Also generate a Raman workflow (main.raman.py)?  Pick the flavour '
        '(none = skip; all = static + resonance from one run)',
        ['none', 'static', 'resonance', 'all'],
        'none',
    )
    cfg['raman'] = cfg['raman_method'] != 'none'
    if cfg['raman']:
        print('\nRaman finite-difference settings:')
        cfg['raman_delta'] = ask_float('  Mass-weighted displacement step (Bohr*sqrt(amu))', 0.05)
        cfg['raman_nbnd'] = ask_int('  Bands for the displaced SCF (0 = QE default)', 0)
        cfg['raman_npool'] = ask_int('  npool for the per-cell dielectric run', 4)
        cfg['raman_smearing'] = ask('  Smearing for the dielectric run', 'gauss')
        raman_nfft = ask_int('  Double-grid FFT size per direction (0 = skip)', 24)
        cfg['raman_nfft'] = raman_nfft
        cfg['raman_e_static'] = ask_float('  Static-epsilon upper energy (eV)', 0.05)
        cfg['raman_temperature'] = ask_float('  Temperature for the Bose factor (K)', 300.0)
        cfg['raman_gamma'] = ask_float('  Lorentzian FWHM (in the chosen units)', 4.0)
        if cfg['raman_method'] in ('resonance', 'all'):
            laser_str = ask(
                '  Laser wavelength(s) in nm (comma-separated, or a Python list '
                'expression like "[n for n in range(450, 650, 5)]")',
                '532',
            )
            lasers = parse_laser_list(laser_str)
            cfg['raman_laser_nm'] = lasers if len(lasers) > 1 else lasers[0]
            cfg['raman_lifetime'] = ask_float(
                '  Lifetime broadening of the dielectric tensor (eV)', 0.1
            )
        else:
            laser = ask_float('  Laser wavelength (nm; 0 = omit the (wL-wv)^4 factor)', 0.0)
            cfg['raman_laser_nm'] = laser if laser > 0 else None
            cfg['raman_lifetime'] = 0.1
        cfg['raman_pthr'] = 0.95
        cfg['raman_configuration'] = 'extended'
    return cfg


def collect_elphon(common):
    """Prompt for the electron-phonon (PAO route) configuration."""
    cfg = dict(common)
    cfg['source'] = ask_choice(
        'Coupling source (ahc = unpatched QE, norm-conserving; '
        'elphmat = patched QE, any pseudopotential)',
        ['ahc', 'elphmat'],
        'ahc',
    )
    cfg['coupling_dir'] = 'ahc_dir' if cfg['source'] == 'ahc' else 'elph_dir'
    kg = ask_int('SCF / coupling k-grid per axis (NK1 = NK2 = NK3)', 9)
    cfg['kgrid'] = [kg, kg, kg]
    qg = ask_int('Phonon q-grid per axis (nq1 = nq2 = nq3)', 3)
    cfg['qgrid'] = [qg, qg, qg]
    cfg['nbnd'] = ask_int('Bands in the nscf / AHC run (nbnd = ahc_nbnd)', 0)
    masses = ask('Atomic masses (amu), comma-separated, one per atom in the cell', '')
    cfg['masses_amu'] = [float(x) for x in masses.replace(',', ' ').split() if x.strip()]
    cfg['nelec'] = ask_int('Valence electrons (nelec)', 0)
    cfg['nk_dense'] = ask_int('Dense interpolation grid per axis (NK_DENSE)', 18)
    cfg['sigma_ry'] = ask_float('Fermi-surface smearing (Ry)', 0.02)
    cfg['mu_star'] = ask_float('Coulomb pseudopotential mu* (for Tc)', 0.10)
    cfg['pthr'] = ask_float('Projectability threshold (pthr)', 0.90)
    if cfg['source'] == 'elphmat':
        weights = ask(
            'Irreducible-q star weights, comma-separated (blank = full grid, unit weights)', ''
        )
        cfg['q_weights'] = [float(x) for x in weights.replace(',', ' ').split() if x.strip()]
    else:
        cfg['q_weights'] = []
    if cfg['nbnd'] <= 0 or not cfg['masses_amu'] or cfg['nelec'] <= 0:
        print(
            '\nNote: NBND, MASSES_AMU and NELEC are system-specific; edit the '
            'generated main.elphon.py before running the "analyse" phase.'
        )
    return cfg


def collect_acbn0(common):
    """Prompt for the ACBN0 / eACBN0 configuration."""
    cfg = dict(common)
    cfg['use_intersite_v'] = ask_yes_no(
        'Include intersite V (eACBN0)? (No = on-site U only, ACBN0)', False
    )
    cfg['projection'] = ask('Projection scheme', 'ortho-atomic')
    cfg['conv_thr'] = float(ask('U/V convergence threshold (eV)', '0.05'))
    cfg['nk'] = ask_int('Band path k-points (NK)', 400)
    cfg['gaussian_threshold'] = float(
        ask('Gaussian-fit tolerance (loosen for ONCV/SOC pseudos)', '0.05')
    )

    print("\nEnter the Hubbard manifolds, one per line, as '<species>-<nl> <initU>'")
    print("  e.g. 'Si-3p 0.5'.  Leave blank to finish.")
    hubbard = []
    while True:
        line = _input('  manifold: ').strip()
        if not line:
            break
        parts = line.split()
        orb = parts[0].strip().strip('\'"')
        try:
            val = float(parts[1]) if len(parts) > 1 else 0.5
        except ValueError:
            val = 0.5
        hubbard.append((orb, val))
    if not hubbard:
        print('No Hubbard manifolds entered; nothing to generate.')
        return None
    cfg['hubbard'] = hubbard

    if cfg['use_intersite_v']:
        v_default = detect_v_cutoff(common.get('workdir', '.')) or 2.6
        cfg['v_cutoff'] = float(ask('Intersite V neighbour cutoff (Angstrom)', repr(v_default)))
        cfg['v_init'] = float(ask('Initial intersite V (eV)', '0.5'))

    print('\nParallel launch commands (edit later in the script if unsure):')
    cfg['mpi_qe'] = ask('MPI command for pw.x', 'mpirun -np 4')
    cfg['mpi_py'] = ask('MPI command for python', 'mpirun -np 1')
    cfg['mpi_hartree'] = ask('MPI command for the Hartree step', 'mpirun -np 4')
    cfg['qe_path'] = ask('Quantum ESPRESSO bin path', '')
    cfg['py_path'] = ask('Python bin path', '')
    return cfg


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate a PAOFLOW main.py driver script interactively.'
    )
    parser.add_argument(
        '-d',
        '--workdir',
        default='.',
        help='Working directory to scan for QE artifacts (default: .)',
    )
    parser.add_argument('--prefix', default=None, help='QE prefix override')
    parser.add_argument('--savedir', default=None, help='<prefix>.save override')
    parser.add_argument(
        '--upf',
        default=None,
        help='Pseudopotential file override (comma-separated for multiple species)',
    )
    parser.add_argument(
        '-o',
        '--out',
        default=None,
        help='Output script path (default: main.py for the run workflow, '
        'main.acbn0.py for the acbn0 workflow)',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Force generation of a plotting script (run workflow only)',
    )
    parser.add_argument(
        '--plot-out',
        default=None,
        help='Plotting-script output path (default: plot.py for the run '
        'workflow, plot.acbn0.py for the acbn0 workflow)',
    )
    parser.add_argument(
        '-f', '--force', action='store_true', help='Overwrite an existing output file'
    )
    args = parser.parse_args(argv)

    workdir = os.path.abspath(os.path.expanduser(args.workdir))

    workflow = ask_choice(
        'Which workflow?',
        ['acbn0', 'run', 'phonon', 'elphon'],
        'run',
    )

    # Each non-default workflow gets its own main.<workflow>.py name so it is
    # not confused with a regular PAOFLOW main.py; an explicit -o/--out wins.
    _default_out = {
        'acbn0': 'main.acbn0.py',
        'phonon': 'main.phonon.py',
        'elphon': 'main.elphon.py',
    }
    out_path = args.out or _default_out.get(workflow, 'main.py')
    if os.path.exists(out_path) and not args.force:
        sys.stderr.write('Refusing to overwrite {} (use --force).\n'.format(out_path))
        return 1

    common = collect_common(args, workdir)
    if workflow == 'acbn0':
        cfg = collect_acbn0(common)
        if cfg is None:
            return 1
        content = build_acbn0_script(cfg)
    elif workflow == 'phonon':
        cfg = collect_phonon(common)
        if cfg is None:
            return 1
        content = build_phonon_script(cfg)
    elif workflow == 'elphon':
        cfg = collect_elphon(common)
        if cfg is None:
            return 1
        content = build_elphon_script(cfg)
    else:
        cfg = collect_run(common)
        if cfg is None:
            return 1
        content = build_run_script(cfg)

    with open(out_path, 'w', encoding='utf-8') as handle:
        handle.write(content)
    print('\nWrote {}'.format(os.path.abspath(out_path)))

    if workflow == 'acbn0':
        # Always pair the ACBN0 driver with a plot.acbn0.py that compares the
        # band structures of the cases it computes (DFT+U and, for eACBN0,
        # DFT+U+V).
        plot_path = args.plot_out or 'plot.acbn0.py'
        if os.path.exists(plot_path) and not args.force:
            sys.stderr.write('Refusing to overwrite {} (use --force).\n'.format(plot_path))
        else:
            plot_content = build_acbn0_plot_script(cfg)
            with open(plot_path, 'w', encoding='utf-8') as handle:
                handle.write(plot_content)
            print('Wrote {}'.format(os.path.abspath(plot_path)))
    elif workflow == 'phonon':
        if args.plot or cfg.get('plot'):
            plot_path = args.plot_out or 'plot.phonon.py'
            if os.path.exists(plot_path) and not args.force:
                sys.stderr.write('Refusing to overwrite {} (use --force).\n'.format(plot_path))
            else:
                plot_content = build_phonon_plot_script(cfg)
                with open(plot_path, 'w', encoding='utf-8') as handle:
                    handle.write(plot_content)
                print('Wrote {}'.format(os.path.abspath(plot_path)))
        if cfg.get('raman'):
            raman_path = 'main.raman.py'
            if os.path.exists(raman_path) and not args.force:
                sys.stderr.write('Refusing to overwrite {} (use --force).\n'.format(raman_path))
            else:
                raman_content = build_raman_script(cfg)
                with open(raman_path, 'w', encoding='utf-8') as handle:
                    handle.write(raman_content)
                print('Wrote {}'.format(os.path.abspath(raman_path)))
            if args.plot or cfg.get('plot'):
                raman_plot_path = 'plot.raman.py'
                if os.path.exists(raman_plot_path) and not args.force:
                    sys.stderr.write(
                        'Refusing to overwrite {} (use --force).\n'.format(raman_plot_path)
                    )
                else:
                    raman_plot_content = build_raman_plot_script(cfg)
                    with open(raman_plot_path, 'w', encoding='utf-8') as handle:
                        handle.write(raman_plot_content)
                    print('Wrote {}'.format(os.path.abspath(raman_plot_path)))
    elif workflow == 'elphon':
        # Always pair the elphon driver with a plot.elphon.py for alpha^2F / lambda.
        plot_path = args.plot_out or 'plot.elphon.py'
        if os.path.exists(plot_path) and not args.force:
            sys.stderr.write('Refusing to overwrite {} (use --force).\n'.format(plot_path))
        else:
            plot_content = build_elphon_plot_script(cfg)
            with open(plot_path, 'w', encoding='utf-8') as handle:
                handle.write(plot_content)
            print('Wrote {}'.format(os.path.abspath(plot_path)))
    elif args.plot or cfg.get('plot'):
        plot_path = args.plot_out or 'plot.py'
        if os.path.exists(plot_path) and not args.force:
            sys.stderr.write('Refusing to overwrite {} (use --force).\n'.format(plot_path))
        else:
            plot_content = build_plot_script(cfg)
            with open(plot_path, 'w', encoding='utf-8') as handle:
                handle.write(plot_content)
            print('Wrote {}'.format(os.path.abspath(plot_path)))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
