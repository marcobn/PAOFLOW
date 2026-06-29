"""I/O for the PAOFLOW finite-displacement phonon workflow.

Responsibilities (Stage 1):

* write complete, ready-to-run Quantum ESPRESSO ``pw.x`` SCF inputs for the
  phonopy-generated displaced supercells (phonopy naming ``supercell-NNN.in``);
* harvest atomic forces from the corresponding ``pw.x`` outputs;
* ingest externally produced ``FORCE_SETS`` files;
* persist a ``FORCE_SETS`` for provenance.

All structural quantities follow the phonopy QE convention (Bohr lengths,
Ry/au forces), matching ``Phonopy(..., calculator='qe')`` set up in
:mod:`PAOFLOW.phonon.do_phonopy`.
"""

import os

import numpy as np


def resolve_phonon_dir(data_controller, phonon_dir='phonon'):
    """Return (and create) the directory holding the displaced supercells.

    ``phonon_dir`` is resolved relative to the PAOFLOW output directory
    (``attr['opath']``) unless an absolute path is given.
    """
    _, attr = data_controller.data_dicts()
    base = attr.get('opath', '.')
    path = phonon_dir if os.path.isabs(phonon_dir) else os.path.join(base, phonon_dir)
    if attr.get('rank', 0) == 0 or 'rank' not in attr:
        os.makedirs(path, exist_ok=True)
    return path


def _pp_filenames(data_controller):
    """Build a ``{element_symbol: pseudopotential_filename}`` mapping."""
    from .structure import _element_symbol

    arry, _ = data_controller.data_dicts()
    species = arry.get('species', None)
    pp = {}
    if species is not None:
        for entry in species:
            label, pseudo = entry[0], entry[1]
            sym = _element_symbol(label)
            pp.setdefault(sym, os.path.basename(str(pseudo)))
    return pp


def _supercell_kgrid(data_controller, supercell_matrix):
    """Scale the unit-cell Monkhorst-Pack grid down to the supercell.

    For a diagonal supercell the per-axis k-density is divided by the diagonal
    multiplicity; for a general matrix the (cube-root of the) determinant is
    used as an isotropic fallback.
    """
    _, attr = data_controller.data_dicts()
    nk = np.array(
        [attr.get('nk1', 1) or 1, attr.get('nk2', 1) or 1, attr.get('nk3', 1) or 1],
        dtype=float,
    )
    sc = np.asarray(supercell_matrix, dtype=float)
    if np.count_nonzero(sc - np.diag(np.diagonal(sc))) == 0:
        diag = np.diagonal(sc)
    else:
        diag = np.full(3, round(abs(np.linalg.det(sc)) ** (1.0 / 3.0)))
    diag = np.where(diag == 0, 1, diag)
    kg = np.maximum(1, np.ceil(nk / diag)).astype(int)
    return kg


def _namelists(data_controller, supercell, prefix, pp_dir):
    """Assemble the ``&control/&system/&electrons`` namelist block."""
    _, attr = data_controller.data_dicts()

    nat = len(supercell)
    ntyp = len(set(map(str, supercell.symbols)))
    ecutwfc = float(attr.get('ecutwfc', 60.0))  # Ry
    ecutrho = float(attr.get('ecutrho', 4.0 * ecutwfc))  # Ry
    nspin = int(attr.get('nspin', 1))
    insulator = bool(attr.get('insulator', True))

    lines = []
    lines.append('&control')
    lines.append("    calculation = 'scf'")
    lines.append("    prefix = '%s'" % prefix)
    lines.append("    outdir = './tmp_%s'" % prefix)
    lines.append("    pseudo_dir = '%s'" % pp_dir)
    lines.append('    tprnfor = .true.')
    lines.append('    tstress = .false.')
    lines.append('/')
    lines.append('&system')
    lines.append('    ibrav = 0')
    lines.append('    nat = %d' % nat)
    lines.append('    ntyp = %d' % ntyp)
    lines.append('    ecutwfc = %.2f' % ecutwfc)
    lines.append('    ecutrho = %.2f' % ecutrho)
    if not insulator:
        lines.append("    occupations = 'smearing'")
        lines.append("    smearing = 'mp'")
        lines.append('    degauss = %.4f' % float(attr.get('degauss', 0.02)))
    if nspin == 2:
        lines.append('    nspin = 2')
        lines.append('    starting_magnetization(1) = 0.1')
    lines.append('/')
    lines.append('&electrons')
    lines.append('    conv_thr = 1.0d-8')
    lines.append('    mixing_beta = 0.7')
    lines.append('/')
    return '\n'.join(lines)


def _qe_input_text(
    data_controller,
    cell,
    supercell_matrix,
    prefix,
    pp_dir,
    pp_filenames,
    kgrid,
    hubbard_card=None,
):
    """Compose a complete ``pw.x`` SCF input for one (super)cell."""
    from phonopy.interface.qe import get_pwscf_structure

    header = _namelists(data_controller, cell, prefix, pp_dir)
    structure = get_pwscf_structure(cell, pp_filenames=pp_filenames)
    k1, k2, k3 = kgrid
    kpoints = 'K_POINTS automatic\n %d %d %d 0 0 0\n' % (int(k1), int(k2), int(k3))
    text = header + '\n' + structure + '\n' + kpoints
    if hubbard_card:
        text += '\n' + hubbard_card.rstrip('\n') + '\n'
    return text


# Recognised leading tokens of a new-style ``HUBBARD`` card body.  ``U`` and
# ``J``-type lines are manifold-based (``U Ga-4p <value>``) and therefore
# transfer verbatim to any supercell; ``V`` lines carry explicit atom indices
# that are only valid for the cell they were generated on.
_HUBBARD_KEYWORDS = ('U', 'V', 'J', 'J0', 'B', 'E2', 'E3')


def read_hubbard_card(path, include_v=False):
    """Extract the new-style ``HUBBARD`` card from a Quantum ESPRESSO input.

    Parameters
    ----------
    path : str
        Path to a ``pw.x`` input file containing a ``HUBBARD (...)`` card.
    include_v : bool
        Keep the intersite ``V`` lines.  Default ``False`` drops them (their
        explicit atom indices are not valid for a supercell); the on-site ``U``
        (and ``J``-type) manifold lines are always kept.

    Returns
    -------
    str or None
        The card text (header + kept parameter lines), or ``None`` when the
        file contains no ``HUBBARD`` card.
    """
    with open(path) as fh:
        lines = fh.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('HUBBARD'):
            start = i
            break
    if start is None:
        return None

    kept = [lines[start].rstrip('\n')]
    for line in lines[start + 1 :]:
        toks = line.split()
        if not toks:
            break
        key = toks[0].upper()
        if key not in _HUBBARD_KEYWORDS:
            break  # next card / namelist terminates the HUBBARD block
        if key == 'V' and not include_v:
            continue
        kept.append(line.rstrip('\n'))
    return '\n'.join(kept) + '\n'


def write_displaced_supercells(
    data_controller,
    phonon_dir='phonon',
    pp_dir=None,
    prefix=None,
    kgrid=None,
    hubbard_card=None,
):
    """Write QE SCF inputs for every phonopy-generated displaced supercell.

    Requires that displacements have already been generated on the stored
    :class:`phonopy.Phonopy` object (``arry['phonopy']``).

    Parameters
    ----------
    hubbard_card : str, optional
        New-style ``HUBBARD`` card text appended to every input (e.g. from
        :func:`read_hubbard_card`).  On-site ``U`` parameters transfer verbatim
        to the supercell, so the forces reflect the DFT+U electronic structure.

    Returns
    -------
    list[str]
        Absolute paths of the written ``supercell-NNN.in`` files (perfect
        supercell ``supercell.in`` excluded).
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)

    if prefix is None:
        savedir = attr.get('savedir', None)
        prefix = os.path.basename(str(savedir)).replace('.save', '') if savedir else 'supercell'
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')

    pp_filenames = _pp_filenames(data_controller)
    if kgrid is None:
        kgrid = _supercell_kgrid(data_controller, phonon.supercell_matrix)

    rank = getattr(data_controller, 'rank', 0)
    written = []

    supercells = phonon.supercells_with_displacements
    ndisp = len(supercells)
    width = max(3, len(str(ndisp)))

    if rank == 0:
        # Perfect supercell (reference / provenance).
        with open(os.path.join(out_dir, 'supercell.in'), 'w') as f:
            f.write(
                _qe_input_text(
                    data_controller,
                    phonon.supercell,
                    phonon.supercell_matrix,
                    prefix,
                    pp_dir,
                    pp_filenames,
                    kgrid,
                    hubbard_card=hubbard_card,
                )
            )
        for i, cell in enumerate(supercells, start=1):
            fname = os.path.join(out_dir, 'supercell-{0:0{w}}.in'.format(i, w=width))
            with open(fname, 'w') as f:
                f.write(
                    _qe_input_text(
                        data_controller,
                        cell,
                        phonon.supercell_matrix,
                        prefix,
                        pp_dir,
                        pp_filenames,
                        kgrid,
                        hubbard_card=hubbard_card,
                    )
                )
            written.append(fname)

        # Persist the displacement dataset for provenance / restart.
        phonon.save(os.path.join(out_dir, 'phonopy_disp.yaml'))

        if attr.get('verbose', False):
            print('Wrote %d displaced supercell inputs to %s' % (ndisp, out_dir))

    return written


def harvest_qe_forces(data_controller, phonon_dir='phonon'):
    """Parse forces from ``supercell-NNN.out`` files and set them on phonopy.

    Returns
    -------
    numpy.ndarray
        Force sets of shape ``(num_displacements, natoms_supercell, 3)`` in
        Ry/au.
    """
    from phonopy.interface.qe import parse_set_of_forces

    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    ndisp = len(phonon.supercells_with_displacements)
    width = max(3, len(str(ndisp)))

    filenames = []
    for i in range(1, ndisp + 1):
        fn = os.path.join(out_dir, 'supercell-{0:0{w}}.out'.format(i, w=width))
        if not os.path.isfile(fn):
            raise FileNotFoundError('Missing QE output for displacement %d: %s' % (i, fn))
        filenames.append(fn)

    natoms = len(phonon.supercell)
    force_sets = parse_set_of_forces(natoms, filenames, verbose=attr.get('verbose', False))
    force_sets = np.array(force_sets)

    phonon.forces = force_sets
    return force_sets


def ingest_force_sets(data_controller, force_sets_path):
    """Load an external ``FORCE_SETS`` onto the stored phonopy object.

    The displacement/forces dataset fully replaces the current dataset, so the
    supercell matrix used to build the phonopy object must match the one that
    produced the ``FORCE_SETS``.
    """
    from phonopy.file_IO import parse_FORCE_SETS

    arry, _ = data_controller.data_dicts()
    phonon = arry['phonopy']

    natoms = len(phonon.supercell)
    dataset = parse_FORCE_SETS(natom=natoms, filename=force_sets_path)
    phonon.dataset = dataset
    return dataset


def write_force_sets(data_controller, phonon_dir='phonon', filename='FORCE_SETS'):
    """Write the current displacement/forces dataset to ``FORCE_SETS``."""
    from phonopy.file_IO import write_FORCE_SETS

    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']

    rank = getattr(data_controller, 'rank', 0)
    if rank != 0:
        return None

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    path = os.path.join(out_dir, filename)
    write_FORCE_SETS(phonon.dataset, filename=path)
    return path


def write_born_file(data_controller, born, epsilon, phonon_dir='phonon', filename='BORN'):
    """Write a phonopy ``BORN`` file (dielectric tensor + Born charges).

    Parameters
    ----------
    born : array_like
        Born effective charges of shape ``(natom_prim, 3, 3)`` in units of the
        elementary charge, ordered to match the phonopy primitive cell.
    epsilon : array_like
        ``(3, 3)`` high-frequency dielectric tensor.

    The file follows the phonopy format (a unit-conversion factor line, the
    dielectric tensor, then one Born tensor per symmetry-inequivalent atom),
    written via :func:`phonopy.file_IO.write_BORN` using the stored phonopy
    primitive cell for the atom ordering and symmetry reduction.
    """
    from phonopy.file_IO import write_BORN

    arry, _ = data_controller.data_dicts()
    phonon = arry['phonopy']

    rank = getattr(data_controller, 'rank', 0)
    if rank != 0:
        return None

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    path = os.path.join(out_dir, filename)
    write_BORN(
        phonon.primitive,
        np.asarray(born, dtype=float),
        np.asarray(epsilon, dtype=float),
        filename=path,
    )
    return path


def read_born_file(data_controller, born_path):
    """Parse a phonopy ``BORN`` file into NAC parameters.

    Returns
    -------
    dict
        ``{'born', 'dielectric', 'factor'}`` suitable for assignment to
        ``phonopy.Phonopy.nac_params``.  Symmetry is used to expand the reduced
        tensors onto every primitive atom, matching the stored phonopy
        primitive cell.
    """
    from phonopy.file_IO import parse_BORN

    arry, _ = data_controller.data_dicts()
    phonon = arry['phonopy']

    nac = parse_BORN(phonon.primitive, filename=born_path)
    born = nac['born'] if isinstance(nac, dict) else nac.born
    dielectric = nac['dielectric'] if isinstance(nac, dict) else nac.dielectric
    if isinstance(nac, dict):
        factor = nac.get('factor', None)
    else:
        factor = getattr(nac, 'factor', None)
    if factor is None:
        # The BORN file omitted the explicit unit-conversion factor; fall back
        # to the phonopy Quantum ESPRESSO default.
        from phonopy.interface.calculator import get_calculator_physical_units

        factor = get_calculator_physical_units('qe').nac_factor
    return {
        'born': np.asarray(born, dtype=float),
        'dielectric': np.asarray(dielectric, dtype=float),
        'factor': float(factor),
    }


# ---------------------------------------------------------------------------
# Stage 2b: finite electric-field (QE ``lelfield``) inputs for Born effective
# charges and the high-frequency dielectric tensor, evaluated on the PRIMITIVE
# cell (the field runs must use the primitive cell, not the phonon supercell).
# ---------------------------------------------------------------------------

# Ordered list of finite-field perturbations: (label, cartesian field vector in
# units of the requested field strength).  The zero-field run provides the
# reference forces and polarization for the central differences.
_FIELD_LABELS = (
    ('field-0', (0.0, 0.0, 0.0)),
    ('field-x+', (1.0, 0.0, 0.0)),
    ('field-x-', (-1.0, 0.0, 0.0)),
    ('field-y+', (0.0, 1.0, 0.0)),
    ('field-y-', (0.0, -1.0, 0.0)),
    ('field-z+', (0.0, 0.0, 1.0)),
    ('field-z-', (0.0, 0.0, -1.0)),
)


def _field_namelists(data_controller, cell, prefix, pp_dir, efield_cart, nberrycyc):
    """Assemble the namelist block for a primitive-cell ``lelfield`` SCF run."""
    _, attr = data_controller.data_dicts()

    nat = len(cell)
    ntyp = len(set(map(str, cell.symbols)))
    ecutwfc = float(attr.get('ecutwfc', 60.0))  # Ry
    ecutrho = float(attr.get('ecutrho', 4.0 * ecutwfc))  # Ry
    nspin = int(attr.get('nspin', 1))

    ex, ey, ez = efield_cart

    lines = []
    lines.append('&control')
    lines.append("    calculation = 'scf'")
    lines.append("    prefix = '%s'" % prefix)
    lines.append("    outdir = './tmp_%s'" % prefix)
    lines.append("    pseudo_dir = '%s'" % pp_dir)
    lines.append('    tprnfor = .true.')
    lines.append('    tstress = .false.')
    lines.append('    lelfield = .true.')
    lines.append('    nberrycyc = %d' % int(nberrycyc))
    lines.append('/')
    lines.append('&system')
    lines.append('    ibrav = 0')
    lines.append('    nat = %d' % nat)
    lines.append('    ntyp = %d' % ntyp)
    lines.append('    ecutwfc = %.2f' % ecutwfc)
    lines.append('    ecutrho = %.2f' % ecutrho)
    # Born charges / epsilon_inf are defined for insulators only: enforce fixed
    # occupations regardless of the stored metallic/insulating flag.
    lines.append("    occupations = 'fixed'")
    if nspin == 2:
        lines.append('    nspin = 2')
        lines.append('    starting_magnetization(1) = 0.1')
    lines.append('/')
    lines.append('&electrons')
    lines.append('    conv_thr = 1.0d-9')
    lines.append('    mixing_beta = 0.5')
    lines.append('    efield_cart(1) = %.8f' % ex)
    lines.append('    efield_cart(2) = %.8f' % ey)
    lines.append('    efield_cart(3) = %.8f' % ez)
    lines.append('/')
    return '\n'.join(lines)


def write_field_inputs(
    data_controller,
    field_strength=0.001,
    phonon_dir='phonon',
    pp_dir=None,
    prefix=None,
    kgrid=None,
    nberrycyc=3,
    hubbard_card=None,
):
    """Write primitive-cell ``lelfield`` SCF inputs for Born charges + epsilon.

    Seven inputs are produced: a zero-field reference plus +/- field runs along
    each Cartesian axis.  The finite differences of the per-atom forces give the
    Born effective charges, and the finite differences of the macroscopic
    polarization give the high-frequency dielectric tensor.

    Parameters
    ----------
    field_strength : float
        Magnitude of ``efield_cart`` in Quantum ESPRESSO atomic units
        (1 a.u. = 36.3609e10 V/m).  Keep small (linear response).
    nberrycyc : int
        Number of Berry-phase self-consistency cycles per SCF step.
    hubbard_card : str, optional
        New-style ``HUBBARD`` card text appended to every input (e.g. from
        :func:`read_hubbard_card`).  The on-site ``U`` parameters transfer
        verbatim to the primitive cell.

    Returns
    -------
    list[str]
        Absolute paths of the written ``field-*.in`` files.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']
    cell = phonon.primitive

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)

    if prefix is None:
        savedir = attr.get('savedir', None)
        prefix = os.path.basename(str(savedir)).replace('.save', '') if savedir else 'field'
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')

    pp_filenames = _pp_filenames(data_controller)
    if kgrid is None:
        kgrid = np.array(
            [attr.get('nk1', 1) or 1, attr.get('nk2', 1) or 1, attr.get('nk3', 1) or 1],
            dtype=int,
        )
    k1, k2, k3 = (int(kgrid[0]), int(kgrid[1]), int(kgrid[2]))

    from phonopy.interface.qe import get_pwscf_structure

    structure = get_pwscf_structure(cell, pp_filenames=pp_filenames)
    kpoints = 'K_POINTS automatic\n %d %d %d 0 0 0\n' % (k1, k2, k3)

    rank = getattr(data_controller, 'rank', 0)
    written = []
    if rank == 0:
        for label, direction in _FIELD_LABELS:
            efield = tuple(field_strength * d for d in direction)
            header = _field_namelists(data_controller, cell, prefix, pp_dir, efield, nberrycyc)
            text = header + '\n' + structure + '\n' + kpoints
            if hubbard_card:
                text += '\n' + hubbard_card.rstrip('\n') + '\n'
            path = os.path.join(out_dir, label + '.in')
            with open(path, 'w') as f:
                f.write(text)
            written.append(path)
        if attr.get('verbose', False):
            print(
                'Wrote %d lelfield inputs (E = %.4g a.u.) to %s'
                % (len(written), field_strength, out_dir)
            )

    return written


# Physical constants (CODATA) for converting the finite-field finite differences
# into Born effective charges (units of e) and the dielectric tensor.
_RY_TO_J = 2.1798723611035e-18  # Rydberg -> Joule
_BOHR_TO_M = 0.529177210903e-10  # Bohr -> metre
_E_CHARGE = 1.602176634e-19  # elementary charge (C)
_EPS0 = 8.8541878128e-12  # vacuum permittivity (F/m)
# Quantum ESPRESSO ``efield_cart`` unit: 1 a.u. = 36.3609e10 V/m (QE manual).
_QE_EFIELD_AU_TO_V_PER_M = 36.3609e10
# Force conversion Ry/Bohr -> Newton.
_RY_BOHR_TO_N = _RY_TO_J / _BOHR_TO_M


def _parse_qe_forces(out_path, natom):
    """Return the per-atom forces (Ry/au) from the last force block of a run."""
    forces = None
    with open(out_path) as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if 'Forces acting on atoms' in line:
            block = []
            for sub in lines[i + 1 :]:
                if 'force =' in sub:
                    parts = sub.split('force =')[1].split()
                    block.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if len(block) == natom:
                        break
                elif 'Total force' in sub:
                    break
            if len(block) == natom:
                forces = np.array(block, dtype=float)
    if forces is None:
        raise ValueError('Could not parse forces from %s' % out_path)
    return forces


def _parse_qe_efield(in_path):
    """Return the ``efield_cart`` vector (a.u.) from a ``lelfield`` input."""
    efield = np.zeros(3)
    with open(in_path) as fh:
        for line in fh:
            low = line.strip().lower()
            for idx in (1, 2, 3):
                key = 'efield_cart(%d)' % idx
                if low.startswith(key):
                    efield[idx - 1] = float(low.split('=')[1])
    return efield


def _parse_qe_polarization(out_path):
    """Return the macroscopic polarization vector (C/m^2) from a lelfield run.

    Quantum ESPRESSO prints, for each Cartesian direction, a line of the form
    ``P =  <value>  (mod <value>) C/m^2`` inside the ``lelfield`` Berry-phase
    summary.  The last three direction blocks are collected in order.
    """
    import re

    pat = re.compile(r'P\s*=\s*([-+0-9.Ee]+)\s*\(mod\s*([-+0-9.Ee]+)\)\s*C/m\^2')
    found = []
    with open(out_path) as fh:
        for line in fh:
            m = pat.search(line)
            if m is not None:
                found.append(float(m.group(1)))
    if len(found) < 3:
        return None
    # The final converged SCF step prints the three cartesian components last.
    return np.array(found[-3:], dtype=float)


def harvest_field_results(data_controller, phonon_dir='phonon'):
    """Assemble Born charges and epsilon_inf from the ``field-*`` runs.

    Parses the per-atom forces and macroscopic polarization from the seven
    ``field-*.out`` files and forms the central differences along each Cartesian
    axis.

    Returns
    -------
    dict
        ``{'born', 'dielectric', 'forces', 'polarization'}``.
    """
    arry, attr = data_controller.data_dicts()
    phonon = arry['phonopy']
    natom = len(phonon.primitive)

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)

    def _out(label):
        path = os.path.join(out_dir, label + '.out')
        if not os.path.isfile(path):
            raise FileNotFoundError('Missing lelfield output: %s' % path)
        return path

    labels = [lbl for lbl, _ in _FIELD_LABELS]
    forces = {lbl: _parse_qe_forces(_out(lbl), natom) for lbl in labels}
    pols = {lbl: _parse_qe_polarization(_out(lbl)) for lbl in labels}

    axis_labels = (
        ('x', 'field-x+', 'field-x-'),
        ('y', 'field-y+', 'field-y-'),
        ('z', 'field-z+', 'field-z-'),
    )

    born = np.zeros((natom, 3, 3))
    has_pol = all(pols[lbl] is not None for lbl in labels)
    chi = np.zeros((3, 3)) if has_pol else None

    e_si_per_au = _E_CHARGE * _QE_EFIELD_AU_TO_V_PER_M  # C * (V/m) per a.u.

    for b, (_, plus, minus) in enumerate(axis_labels):
        efield = _parse_qe_efield(os.path.join(out_dir, plus + '.in'))
        e_mag = float(np.max(np.abs(efield)))  # a.u.
        if e_mag == 0.0:
            raise ValueError('Zero field magnitude parsed for %s' % plus)
        # Z*_{k,a,b} = (1/e) dF_{k,a}/dE_b  (central difference).
        dF = (forces[plus] - forces[minus]) / (2.0 * e_mag)  # Ry/bohr per a.u.
        born[:, :, b] = (dF * _RY_BOHR_TO_N) / e_si_per_au  # dimensionless (e)
        if has_pol:
            dP = (pols[plus] - pols[minus]) / (2.0 * e_mag)  # (C/m^2) per a.u.
            chi[:, b] = (dP / _QE_EFIELD_AU_TO_V_PER_M) / _EPS0  # dimensionless

    if has_pol:
        dielectric = np.eye(3) + chi
    else:
        epsr = arry.get('epsilon_inf', None)
        if epsr is not None:
            dielectric = np.asarray(epsr, dtype=float).reshape(3, 3)
        else:
            dielectric = np.eye(3)

    return {
        'born': born,
        'dielectric': dielectric,
        'forces': forces,
        'polarization': pols,
    }


# ---------------------------------------------------------------------------
# Stage 2b (DFPT route): density-functional perturbation theory via QE ``ph.x``
# at the Gamma point.  A single ``epsil=.true., trans=.false.`` run yields both
# the high-frequency dielectric tensor and the Born effective charges, far more
# cheaply than the finite-field route.  Not available for DFT+U / hybrid
# functionals (use the ``lelfield`` route there instead).
# ---------------------------------------------------------------------------


def _ph_prefix_outdir(data_controller, prefix=None, outdir=None):
    """Resolve the ``ph.x`` ``prefix``/``outdir`` from the stored DFT save dir.

    ``ph.x`` reads the self-consistent save produced by ``pw.x``; the PAOFLOW
    ``savedir`` (``<outdir>/<prefix>.save``) already contains the charge density
    and wavefunctions, so by default we point ``ph.x`` straight at it.
    """
    _, attr = data_controller.data_dicts()
    savedir = attr.get('savedir', None)
    if prefix is None:
        prefix = os.path.basename(str(savedir)).replace('.save', '') if savedir else 'pwscf'
    if outdir is None:
        outdir = os.path.dirname(str(savedir)) if savedir else '.'
    return prefix, outdir


def write_ph_epsil_input(
    data_controller,
    phonon_dir='phonon',
    prefix=None,
    outdir=None,
    fildyn=None,
    tr2_ph=1.0e-16,
    filename='ph_epsil.in',
):
    """Write a ``ph.x`` input for the Gamma-point dielectric + Born charges.

    The run uses ``epsil=.true., trans=.false.`` (electric-field perturbation
    only), which computes the high-frequency dielectric tensor and the Born
    effective charges (``d Force / dE``) without the full phonon dynamical
    matrix.

    Returns
    -------
    str
        Absolute path of the written ``ph.x`` input file.
    """
    arry, attr = data_controller.data_dicts()

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    prefix, outdir = _ph_prefix_outdir(data_controller, prefix, outdir)
    if fildyn is None:
        fildyn = prefix + '.dyn'

    lines = []
    lines.append('Gamma-point dielectric tensor and Born effective charges')
    lines.append('&inputph')
    lines.append("    prefix = '%s'" % prefix)
    lines.append("    outdir = '%s'" % outdir)
    lines.append("    fildyn = '%s'" % fildyn)
    lines.append('    tr2_ph = %s' % ('%.1e' % tr2_ph).replace('e', 'd'))
    lines.append('    epsil = .true.')
    lines.append('    trans = .false.')
    lines.append('    zeu = .true.')
    lines.append('/')
    lines.append('0.0 0.0 0.0')
    text = '\n'.join(lines) + '\n'

    rank = getattr(data_controller, 'rank', 0)
    path = os.path.join(out_dir, filename)
    if rank == 0:
        with open(path, 'w') as f:
            f.write(text)
        if attr.get('verbose', False):
            print('Wrote ph.x dielectric/Born input to %s' % path)
    return path


def _read_3x3_paren_block(lines, start):
    """Read three ``( a b c )`` rows of floats starting at/after ``start``."""
    rows = []
    for line in lines[start:]:
        if '(' in line and ')' in line:
            inner = line[line.index('(') + 1 : line.rindex(')')]
            vals = inner.split()
            if len(vals) == 3:
                rows.append([float(v) for v in vals])
                if len(rows) == 3:
                    break
    if len(rows) != 3:
        return None
    return np.array(rows, dtype=float)


def harvest_ph_results(data_controller, phonon_dir='phonon', filename='ph_epsil.out'):
    """Parse the dielectric tensor and Born charges from a ``ph.x`` output.

    Returns
    -------
    dict
        ``{'born', 'dielectric'}`` with Born charges of shape
        ``(natom_prim, 3, 3)`` (units of e) ordered to match the phonopy
        primitive cell, and the ``(3, 3)`` dielectric tensor.
    """
    arry, _ = data_controller.data_dicts()
    phonon = arry['phonopy']
    natom = len(phonon.primitive)

    out_dir = resolve_phonon_dir(data_controller, phonon_dir)
    path = os.path.join(out_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError('Missing ph.x output: %s' % path)

    with open(path) as fh:
        lines = fh.readlines()

    text = ''.join(lines)

    # ph.x DFPT does not support the electric-field response for intersite
    # DFT+U+V (or the 'atomic'/non-ortho Hubbard projectors): it aborts in
    # phq_readin and writes a CRASH file, leaving ph_epsil.out without the
    # dielectric / Born blocks.  Detect this and point at the field back-end.
    if 'Hubbard projectors type is not implemented' in text or os.path.isfile(
        os.path.join(out_dir, 'CRASH')
    ):
        raise RuntimeError(
            'ph.x (DFPT) crashed for %s.\n'
            'The DFPT electric-field response (epsil/zeu) is not implemented for '
            'this DFT+U setup (intersite U+V / this Hubbard projectors type) in '
            'Quantum ESPRESSO.  Use the finite-field back-end instead:\n'
            "    born_charges(method='field', ...)\n"
            'or recompute the SCF with on-site U only.  '
            "(See the 'CRASH' file in %s.)" % (path, out_dir)
        )

    if 'JOB DONE' not in text:
        raise RuntimeError(
            'ph.x output %s does not contain "JOB DONE"; the run did not finish '
            'successfully.  Re-run ph.x and check for a CRASH file.' % path
        )

    # Dielectric tensor.
    dielectric = None
    for i, line in enumerate(lines):
        if 'Dielectric constant in cartesian axis' in line:
            dielectric = _read_3x3_paren_block(lines, i + 1)
    if dielectric is None:
        raise ValueError('Could not parse the dielectric tensor from %s' % path)

    # Born effective charges (d Force / dE).  Use the last printed block; each
    # atom contributes three rows labelled Ex / Ey / Ez.
    start = None
    for i, line in enumerate(lines):
        if 'Effective charges (d Force / dE)' in line:
            start = i
    if start is None:
        raise ValueError('Could not find the Born-charge block in %s' % path)

    born = np.zeros((natom, 3, 3))
    k = -1
    row = 0
    for line in lines[start + 1 :]:
        if 'atom ' in line and '(' not in line:
            k += 1
            row = 0
            if k >= natom:
                break
            continue
        if k < 0 or k >= natom:
            continue
        if '(' in line and ')' in line:
            inner = line[line.index('(') + 1 : line.rindex(')')]
            vals = inner.split()
            if len(vals) == 3 and row < 3:
                born[k, row, :] = [float(v) for v in vals]
                row += 1

    return {'born': born, 'dielectric': dielectric}
