"""Shared Quantum ESPRESSO scf-input writer.

This module is database-agnostic: it takes a
:class:`PAOFLOW.gen.qe_input.record.MaterialRecord` (produced by any source
adapter) plus a pseudopotential folder and assembles a ready-to-run
``<compound>.scf.in`` file.  Only the Python standard library and numpy are
used.
"""

import math
import os
import re
import sys

# Threshold below which a band gap is treated as metallic (needs smearing).
EGAP_METAL_THRESHOLD = 1.0e-6
# Threshold above which a per-atom local moment marks the system as magnetic.
SPIND_THRESHOLD = 1.0e-3
# Default Gaussian smearing width (Ry) for metals.
DEFAULT_DEGAUSS = 0.05
# Initial guess for starting_magnetization on every magnetic species.
DEFAULT_START_MAG = 0.1
# Safety margin applied to the estimated PAO basis size when setting nbnd, so
# that the requested number of bands comfortably spans the extended basis.
NBND_MARGIN = 1.15
# Fallback Monkhorst-Pack densities used when the source provides no k-grid.
# Metals need a denser, shifted grid; both are intentionally over-dense and the
# user is warned to verify convergence.
DEFAULT_KGRID_INSULATOR = 12
DEFAULT_KGRID_METAL = 18
# Map of spectroscopic shell letters to angular momentum (fallback parsing).
L_OF_LETTER = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}


# --------------------------------------------------------------------------- #
# Pseudopotential folder helpers
# --------------------------------------------------------------------------- #
def find_pseudo_file(pseudo_dir, element):
    """Return the .upf filename in *pseudo_dir* matching *element* (case-insensitive)."""
    target = element.lower() + '.upf'
    matches = []
    for name in os.listdir(pseudo_dir):
        lower = name.lower()
        if not lower.endswith('.upf'):
            continue
        if lower == target:
            return name
        if lower.split('.')[0] == element.lower():
            matches.append(name)
    if matches:
        return matches[0]
    raise RuntimeError("No pseudopotential '{}.upf' found in {}".format(element, pseudo_dir))


def lmax_from_upf(upf_path):
    """Return the maximum angular momentum among the pseudo-wavefunctions.

    Reads the ``PP_CHI`` entries of a UPF file (``l="..."`` attributes, with a
    fallback to spectroscopic ``label`` letters). Returns ``None`` if no
    pseudo-wavefunction information is found.
    """
    try:
        with open(upf_path, 'r', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError:
        return None

    l_values = [int(v) for v in re.findall(r'\bl\s*=\s*"(\d+)"', text)]
    if l_values:
        return max(l_values)

    letters = re.findall(r'label\s*=\s*"\s*\d*\s*([A-Za-z])', text)
    found = [L_OF_LETTER[c.lower()] for c in letters if c.lower() in L_OF_LETTER]
    return max(found) if found else None


def is_fully_relativistic(upf_path):
    """True if the UPF is fully relativistic (``relativistic="full"``).

    Fully-relativistic pseudopotentials carry spin-orbit information, so a QE
    run using them must be noncollinear with spin-orbit coupling.  Returns
    ``False`` when the file cannot be read or the attribute is absent.
    """
    try:
        with open(upf_path, 'r', encoding='utf-8', errors='replace') as handle:
            text = handle.read()
    except OSError:
        return False
    match = re.search(r'relativistic\s*=\s*"?\s*([A-Za-z]+)', text)
    return bool(match) and match.group(1).lower() == 'full'


def extended_orbitals_per_atom(lmax):
    """Number of PAO orbitals per atom in the extended basis.

    The extended basis includes one shell for every angular momentum from 0 up
    to ``lmax + 1`` (the leading polarization channel), giving
    ``sum_{l=0}^{lmax+1} (2l + 1) = (lmax + 2) ** 2`` orbitals.
    """
    return (lmax + 2) ** 2


def load_atomic_masses(pseudo_dir):
    """Build a {symbol: atomic_mass} map from PeriodicTableJSON.json."""
    import json

    path = os.path.join(pseudo_dir, 'PeriodicTableJSON.json')
    if not os.path.isfile(path):
        raise RuntimeError('PeriodicTableJSON.json not found in {}'.format(pseudo_dir))
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    masses = {}
    elements = data.get('elements', data) if isinstance(data, dict) else data
    if isinstance(elements, list):
        for entry in elements:
            sym = entry.get('symbol')
            mass = entry.get('atomic_mass')
            if sym is not None and mass is not None:
                masses[sym] = mass
    elif isinstance(elements, dict):
        for sym, entry in elements.items():
            mass = entry.get('atomic_mass') if isinstance(entry, dict) else entry
            if mass is not None:
                masses[sym] = mass
    return masses


def load_reference_cutoffs(pseudo_dir):
    """Build a {element: ecutwfc_Ry} map from reference.json (hn in Hartree)."""
    import json

    path = os.path.join(pseudo_dir, 'reference.json')
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    cutoffs = {}
    for element, entry in data.items():
        if isinstance(entry, dict) and 'hn' in entry:
            try:
                cutoffs[element] = float(entry['hn']) * 2.0  # Hartree -> Rydberg
            except (TypeError, ValueError):
                continue
    return cutoffs


# --------------------------------------------------------------------------- #
# Bravais-lattice detection (QE ibrav + celldm) from the explicit cell
# --------------------------------------------------------------------------- #
# celldm slots (1-based) that QE reads for each ibrav.  Only these are emitted.
_CELLDM_SLOTS = {
    1: (1,),
    2: (1,),
    3: (1,),
    -3: (1,),
    4: (1, 3),
    5: (1, 4),
    -5: (1, 4),
    6: (1, 3),
    7: (1, 3),
    8: (1, 2, 3),
    9: (1, 2, 3),
    -9: (1, 2, 3),
    91: (1, 2, 3),
    10: (1, 2, 3),
    11: (1, 2, 3),
    12: (1, 2, 3, 4),
    -12: (1, 2, 3, 5),
    13: (1, 2, 3, 4),
    -13: (1, 2, 3, 5),
    14: (1, 2, 3, 4, 5, 6),
}


def format_celldm_lines(ibrav, celldm):
    """Return the ``celldm(i) = ...`` namelist lines QE needs for *ibrav* (Bohr)."""
    slots = _CELLDM_SLOTS[ibrav]
    return ['    celldm({}) = {:.10f},'.format(i, float(celldm[i - 1])) for i in slots]


def cell_rows_to_matrix(geometry):
    """Return the geometry cell as a (3,3) array in Bohr, or ``None`` if unsupported.

    Only ``angstrom`` and ``bohr`` cell units can be converted without an
    external ``alat``; other units cause a fall back to the verbatim ibrav=0
    path by returning ``None``.
    """
    import numpy as np

    from PAOFLOW.inputs.lattice_format import BOHR_RADIUS_ANGS

    rows = [[float(x) for x in row.split()[:3]] for row in geometry['cell_rows']]
    lat = np.array(rows, dtype=float)
    if lat.shape != (3, 3):
        return None
    unit = (geometry.get('cell_unit') or '').lower()
    if unit.startswith('angstrom') or unit in ('ang',):
        return lat / BOHR_RADIUS_ANGS
    if unit.startswith('bohr') or unit in ('au', 'a.u.'):
        return lat
    return None


def remap_atomic_positions(geometry, lat_bohr, m_matrix):
    """Remap the geometry positions to crystal coordinates of the QE primitive cell.

    ``f_qe = f_in @ inv(M)`` where ``M`` maps the input cell onto the QE cell.
    Each coordinate is wrapped to the minimum-image range ``[-0.5, 0.5)`` so
    that bonded neighbours fall inside the home unit cell (needed by the
    eACBN0 intersite-V neighbour search).  Returns the new
    ``ATOMIC_POSITIONS (crystal)`` data rows, or ``None`` if the position
    units cannot be resolved to fractional coordinates.
    """
    import numpy as np

    parsed = _parse_frac_positions(geometry, lat_bohr)
    if parsed is None:
        return None
    labels, frac_in, tails = parsed

    frac_qe = frac_in @ np.linalg.inv(m_matrix)
    # Wrap into the minimum-image cell [-0.5, 0.5): keeps every atom as close
    # to the origin as possible so intersite bonds stay within the home cell.
    frac_qe = frac_qe - np.floor(frac_qe + 0.5 + 1.0e-8)

    rows = []
    for label, frac, tail in zip(labels, frac_qe, tails):
        line = '  {:<3s} {:18.12f} {:18.12f} {:18.12f}'.format(label, frac[0], frac[1], frac[2])
        if tail:
            line += ' ' + ' '.join(tail)
        rows.append(line)
    return rows


def _parse_frac_positions(geometry, lat_bohr):
    """Return ``(labels, frac, tails)`` for the geometry positions.

    ``frac`` is an ``(nat, 3)`` array of fractional coordinates in the input
    cell ``lat_bohr`` (Bohr rows).  Returns ``None`` if the position units
    cannot be resolved to fractional coordinates.
    """
    import numpy as np

    from PAOFLOW.inputs.lattice_format import BOHR_RADIUS_ANGS

    labels, coords, tails = [], [], []
    for row in geometry['pos_rows']:
        parts = row.split()
        labels.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
        tails.append(parts[4:])
    coords = np.array(coords, dtype=float)

    unit = (geometry.get('pos_unit') or '').lower()
    if unit.startswith('crystal'):
        frac = coords
    elif unit.startswith('angstrom') or unit in ('ang',):
        frac = (coords / BOHR_RADIUS_ANGS) @ np.linalg.inv(lat_bohr)
    elif unit.startswith('bohr') or unit in ('au', 'a.u.'):
        frac = coords @ np.linalg.inv(lat_bohr)
    else:
        return None
    return labels, frac, tails


def suggest_intersite_cutoff(lat_bohr, frac):
    """Suggest an eACBN0 intersite-V cutoff (Å) from the cell and positions.

    Returns ``(d_nn_ang, cutoff_ang)``: the nearest-neighbour distance and a
    cutoff placed midway between the first and second neighbour shells, so the
    intersite-V search captures exactly the first shell.  Returns ``None`` when
    no neighbour shell can be resolved.
    """
    import numpy as np

    from PAOFLOW.inputs.lattice_format import BOHR_RADIUS_ANGS

    frac = np.asarray(frac, dtype=float)
    nat = frac.shape[0]
    if nat == 0:
        return None
    cart = frac @ lat_bohr  # (nat, 3) Bohr

    bound = 2
    rng = range(-bound, bound + 1)
    shifts = np.array([(na, nb, nc) for na in rng for nb in rng for nc in rng], dtype=float)
    trans = shifts @ lat_bohr  # (n_img, 3)

    collected = []
    for i in range(nat):
        diff = cart[None, :, :] + trans[:, None, :] - cart[i]  # (n_img, nat, 3)
        d2 = np.einsum('ijk,ijk->ij', diff, diff).ravel()
        d2 = d2[d2 > 1.0e-6]
        if d2.size:
            collected.append(np.sqrt(d2))
    if not collected:
        return None

    dvals = np.sort(np.concatenate(collected))
    # Collapse near-degenerate distances into discrete neighbour shells.
    shells = [dvals[0]]
    for d in dvals[1:]:
        if d - shells[-1] > 1.0e-2:  # Bohr; shells differ by far more
            shells.append(d)
            if len(shells) >= 2:
                break

    d_nn = shells[0] * BOHR_RADIUS_ANGS
    if len(shells) >= 2:
        cutoff = 0.5 * (shells[0] + shells[1]) * BOHR_RADIUS_ANGS
    else:
        cutoff = 1.2 * shells[0] * BOHR_RADIUS_ANGS
    return d_nn, cutoff


def detect_ibrav(record, symprec):
    """Detect QE ibrav/celldm and remap positions, or ``None`` to keep ibrav=0.

    Any failure (missing numpy, unsupported units, undetectable lattice) yields
    ``None`` so the caller falls back to the safe explicit-cell path.
    """
    try:
        import numpy as np

        from PAOFLOW.inputs.lattice_format import qe_ibrav_from_lattice

        geometry = record.geometry
        lat = cell_rows_to_matrix(geometry)
        if lat is None:
            return None
        res = qe_ibrav_from_lattice(
            lat,
            bravais_hint=record.bravais_hint,
            spacegroup=record.spacegroup,
            symprec=symprec,
        )
        if res['ibrav'] == 0:
            return None
        pos_rows = remap_atomic_positions(geometry, lat, np.asarray(res['M'], dtype=float))
        if pos_rows is None:
            return None
        return {'ibrav': res['ibrav'], 'celldm': res['celldm'], 'pos_rows': pos_rows}
    except Exception as exc:  # pragma: no cover - defensive fall back
        sys.stderr.write('Warning: ibrav detection failed ({}); using ibrav=0.\n'.format(exc))
        return None


# --------------------------------------------------------------------------- #
# 2D lattice detection: keep the in-plane symmetry, vacuum along c
# --------------------------------------------------------------------------- #
def detect_ibrav_2d(record, symprec):
    """Detect a proper QE ibrav for a 2D material, vacuum pinned to ``c``.

    A 2D cell with vacuum along the third lattice vector is still a valid 3D
    Bravais lattice.  Classifying the *in-plane* lattice and pinning the
    vacuum to ``c`` avoids the deprecated ``ibrav=0`` while keeping QE's
    convention ``v3 = (0, 0, c)``:

    ===================== ===== ===========================
    in-plane lattice      ibrav celldm
    ===================== ===== ===========================
    hexagonal (γ=120°)    4     (1, 3)
    square (a=b, γ=90°)   6     (1, 3)
    rectangular (γ=90°)   8     (1, 2, 3)
    centred rectangular   9     (1, 2, 3)
    oblique (general γ)   12    (1, 2, 3, 4)  unique axis c
    ===================== ===== ===========================

    Returns ``{'ibrav', 'celldm', 'pos_rows'}`` with positions written in
    crystal coordinates of the detected cell, or ``None`` to fall back to the
    explicit-cell path.
    """
    try:
        import numpy as np

        geometry = record.geometry
        lat = cell_rows_to_matrix(geometry)
        if lat is None:
            return None

        # Identify the vacuum axis: the lattice vector most aligned with the
        # cross product of the other two (the out-of-plane direction) and the
        # one with negligible in-plane projection.  C2DB always lists it as the
        # third vector, but we detect it robustly.
        vac_axis = _vacuum_axis(lat)
        if vac_axis is None:
            return None
        in_axes = [i for i in range(3) if i != vac_axis]

        a_vec = lat[in_axes[0]]
        b_vec = lat[in_axes[1]]
        c_vec = lat[vac_axis]

        # The in-plane vectors must be orthogonal to the vacuum vector for the
        # canonical c-along-z forms to apply.
        cn = c_vec / np.linalg.norm(c_vec)
        if abs(a_vec @ cn) > 10.0 * symprec or abs(b_vec @ cn) > 10.0 * symprec:
            return None

        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))
        if a <= 0 or b <= 0 or c <= 0:
            return None
        cos_g = float((a_vec @ b_vec) / (a * b))
        gamma = math.degrees(math.acos(max(-1.0, min(1.0, cos_g))))

        atol = max(symprec, 1.0e-4)
        ang_tol = max(math.degrees(symprec / max(a, 1.0)), 0.1)

        ibrav, celldm, lat_qe = _classify_2d(a, b, c, gamma, cos_g, atol, ang_tol, np)
        if ibrav is None:
            return None

        # Build the integer map M (input -> QE cell) and remap positions.
        m_matrix = _map_inplane_to_qe(a_vec, b_vec, c_vec, lat_qe, in_axes, vac_axis, atol, np)
        if m_matrix is None:
            return None
        pos_rows = remap_atomic_positions(geometry, lat, m_matrix)
        if pos_rows is None:
            return None
        return {'ibrav': ibrav, 'celldm': celldm, 'pos_rows': pos_rows}
    except Exception as exc:  # pragma: no cover - defensive fall back
        sys.stderr.write('Warning: 2D ibrav detection failed ({}); using ibrav=0.\n'.format(exc))
        return None


def _vacuum_axis(lat):
    """Return the index of the lattice vector spanning the vacuum, or ``None``.

    The vacuum axis is the one most parallel to the normal of the plane formed
    by the other two vectors.  Returns ``None`` when the cell is not clearly
    layered (all three vectors contribute in-plane).
    """
    import numpy as np

    best = None
    best_score = 0.0
    for k in range(3):
        i, j = [m for m in range(3) if m != k]
        normal = np.cross(lat[i], lat[j])
        nn = np.linalg.norm(normal)
        if nn < 1.0e-12:
            continue
        score = abs(lat[k] @ (normal / nn)) / max(np.linalg.norm(lat[k]), 1.0e-12)
        if score > best_score:
            best_score = score
            best = k
    # Require the candidate axis to be essentially perpendicular to the plane.
    return best if best_score > 0.95 else None


def _classify_2d(a, b, c, gamma, cos_g, atol, ang_tol, np):
    """Return ``(ibrav, celldm, lat_qe)`` for the in-plane lattice or Nones."""
    from PAOFLOW.inputs.lattice_format import lattice_format_QE

    celldm = np.zeros(6)
    ibrav = None

    same_ab = abs(a - b) <= atol * max(a, 1.0)
    is_90 = abs(gamma - 90.0) <= ang_tol
    is_120 = abs(gamma - 120.0) <= ang_tol
    is_60 = abs(gamma - 60.0) <= ang_tol

    if same_ab and is_120:
        ibrav = 4
        celldm[0], celldm[2] = a, c / a
    elif same_ab and is_60:
        # 60° hexagonal description -> use the 120° QE convention.
        ibrav = 4
        celldm[0], celldm[2] = a, c / a
    elif same_ab and is_90:
        ibrav = 6
        celldm[0], celldm[2] = a, c / a
    elif is_90:
        ibrav = 8
        celldm[0], celldm[1], celldm[2] = a, b / a, c / a
    elif same_ab:
        # a == b with a general angle: centred-rectangular (rhombic) lattice.
        # Conventional orthorhombic cell A=2a|cos(g/2)|, B=2a|sin(g/2)|.
        half = math.radians(gamma) / 2.0
        A = 2.0 * a * abs(math.cos(half))
        B = 2.0 * a * abs(math.sin(half))
        if A <= 0 or B <= 0:
            return None, None, None
        ibrav = 9
        celldm[0], celldm[1], celldm[2] = A, B / A, c / A
    else:
        # General oblique: monoclinic P, unique axis c (v3 along z).
        ibrav = 12
        celldm[0], celldm[1], celldm[2], celldm[3] = a, b / a, c / a, cos_g

    try:
        lat_qe = lattice_format_QE(ibrav, celldm)
    except ValueError:
        return None, None, None
    return ibrav, celldm, lat_qe


def _map_inplane_to_qe(a_vec, b_vec, c_vec, lat_qe, in_axes, vac_axis, atol, np):
    """Integer unimodular map ``M`` from the input cell to the QE 2D cell.

    Rows of ``M`` express each input lattice vector in the QE basis; positions
    transform as ``f_qe = f_in @ inv(M)``.  Returns ``None`` when no integer
    map reproduces the QE cell within tolerance.
    """
    lat_in = np.zeros((3, 3))
    lat_in[in_axes[0]] = a_vec
    lat_in[in_axes[1]] = b_vec
    lat_in[vac_axis] = c_vec

    inv_qe = np.linalg.inv(lat_qe)
    m_real = lat_in @ inv_qe
    m_int = np.rint(m_real)
    if np.max(np.abs(m_real - m_int)) > 1.0e-4:
        return None
    if abs(abs(np.linalg.det(m_int)) - 1.0) > 1.0e-6:
        return None
    return m_int


# --------------------------------------------------------------------------- #
# Input assembly
# --------------------------------------------------------------------------- #
def build_qe_input(
    record,
    pseudo_dir,
    soc=False,
    degauss=DEFAULT_DEGAUSS,
    nbnd_override=None,
    ibrav_mode='auto',
    symprec=1.0e-4,
    assume_isolated_2d=True,
):
    """Assemble the full QE scf input text from a :class:`MaterialRecord`."""
    geometry = record.geometry
    compound = record.compound or 'system'
    species = record.species
    ntyp = len(species)
    nat = record.natoms or len(geometry['atom_order'])
    metallic = record.metallic
    magnetic = record.magnetic

    masses = load_atomic_masses(pseudo_dir)
    cutoffs = load_reference_cutoffs(pseudo_dir)

    species_rows = []
    ecut_values = []
    orbitals_per_element = {}
    fully_relativistic = False
    for element, _count in species:
        upf = find_pseudo_file(pseudo_dir, element)
        mass = masses.get(element)
        if mass is None:
            raise RuntimeError("Atomic mass for '{}' not in PeriodicTableJSON.json".format(element))
        species_rows.append('  {:<3s} {:>10.4f}  {}'.format(element, float(mass), upf))
        if element in cutoffs:
            ecut_values.append(cutoffs[element])
        upf_path = os.path.join(pseudo_dir, upf)
        if is_fully_relativistic(upf_path):
            fully_relativistic = True
        lmax = lmax_from_upf(upf_path)
        if lmax is None:
            lmax = 2  # generous default (covers s/p/d extended basis)
            sys.stderr.write(
                "Warning: could not read pseudo-wavefunctions for '{}'; "
                'assuming l_max=2 for nbnd.\n'.format(element)
            )
        orbitals_per_element[element] = extended_orbitals_per_atom(lmax)

    # Fully-relativistic pseudopotentials carry spin-orbit coupling, so the run
    # must be noncollinear with lspinorb even though the source only reports a
    # collinear (spin-polarized) magnetization.
    if fully_relativistic and not soc:
        soc = True
        sys.stderr.write(
            'Detected fully-relativistic pseudopotential(s); enabling '
            'noncolin + lspinorb (spin-orbit) input.\n'
        )

    if ecut_values:
        ecutwfc = max(ecut_values)
    elif record.energy_cutoff is not None:
        ecutwfc = float(record.energy_cutoff)
        sys.stderr.write(
            'Warning: reference.json missing/incomplete; using source '
            'energy_cutoff={} Ry.\n'.format(ecutwfc)
        )
    else:
        raise RuntimeError('ecutwfc unavailable: provide reference.json in the pseudo folder.')

    # nbnd large enough to span the extended PAO basis used in PAOFLOW
    # projections: sum of per-atom extended-basis orbitals, doubled for
    # spin-orbit (spinor) calculations, plus a safety margin.
    if nbnd_override is not None:
        nbnd = nbnd_override
    else:
        nawf = sum(
            orbitals_per_element.get(el, extended_orbitals_per_atom(2))
            for el in geometry['atom_order']
        )
        if soc:
            nawf *= 2
        nbnd = max(nawf + 2, int(math.ceil(nawf * NBND_MARGIN)))

    # Lattice handling: 2D entries keep the in-plane symmetry with vacuum on c;
    # 3D entries use the generic Bravais classifier.
    detected = None
    if ibrav_mode == 'auto':
        if record.is_2d:
            detected = detect_ibrav_2d(record, symprec)
        else:
            detected = detect_ibrav(record, symprec)

    # Suggest an intersite-V neighbour cutoff (for a later eACBN0 / U+V run)
    # from the structure itself; emitted as a header comment and on stderr.
    header = []
    lat_bohr = cell_rows_to_matrix(geometry)
    if lat_bohr is not None:
        parsed = _parse_frac_positions(geometry, lat_bohr)
        if parsed is not None:
            suggestion = suggest_intersite_cutoff(lat_bohr, parsed[1])
            if suggestion is not None:
                d_nn, v_cutoff = suggestion
                sys.stderr.write(
                    'Suggested eACBN0 intersite V cutoff: {:.2f} Angstrom '
                    '(nearest-neighbour distance {:.2f} A).\n'.format(v_cutoff, d_nn)
                )
                header = [
                    '! Suggested eACBN0 intersite V cutoff: {:.2f} Angstrom'.format(v_cutoff),
                    '!   nearest-neighbour distance: {:.2f} Angstrom'.format(d_nn),
                    '!   covers the first neighbour shell; use as V_CUTOFF /',
                    '!   set_intersite_pairs(cutoff=...) in the PAOFLOW U+V driver.',
                ]

    # K_POINTS grid: honour a source-provided grid, otherwise fall back to a
    # safe (over-)dense default and warn the user to verify convergence.
    if record.kpoints:
        kpts = record.kpoints
        if record.is_2d:
            kpts = (kpts[0], kpts[1], 1)
        kshift = (0, 0, 0)
    else:
        density = DEFAULT_KGRID_METAL if metallic else DEFAULT_KGRID_INSULATOR
        kz = 1 if record.is_2d else density
        kpts = (density, density, kz)
        kshift = (1, 1, 0) if metallic else (0, 0, 0)
        caveat = (
            'No k-point grid provided by source ({}); using a default '
            '{}x{}x{} {} grid -- CHECK K-POINT CONVERGENCE before '
            'production runs.'.format(
                record.source or 'unknown',
                kpts[0],
                kpts[1],
                kpts[2],
                'metal' if metallic else 'insulator',
            )
        )
        sys.stderr.write('Warning: ' + caveat + '\n')
        header.append('! ' + caveat)

    out = list(header)
    # &control
    out.append(' &control')
    out.append("    calculation = 'scf'")
    out.append("    restart_mode = 'from_scratch',")
    out.append("    prefix = '{}',".format(compound))
    out.append("    pseudo_dir = '{}',".format(pseudo_dir))
    out.append("    outdir = './'")
    out.append(' /')

    # &system
    out.append(' &system')
    if detected is not None:
        out.append('    ibrav = {},'.format(detected['ibrav']))
        out.extend(format_celldm_lines(detected['ibrav'], detected['celldm']))
    else:
        out.append('    ibrav = 0,')
    out.append('    nat = {},'.format(nat))
    out.append('    ntyp = {},'.format(ntyp))
    out.append('    ecutwfc = {:.1f},'.format(ecutwfc))
    out.append('    nbnd = {},'.format(nbnd))
    if record.is_2d and assume_isolated_2d:
        out.append("    assume_isolated = '2D',")
    if metallic:
        out.append("    occupations = 'smearing',")
        out.append('    degauss = {},'.format(degauss))
    if soc:
        out.append('    noncolin = .true.,')
        out.append('    lspinorb = .true.,')
        for i in range(1, ntyp + 1):
            out.append('    starting_magnetization({}) = {},'.format(i, DEFAULT_START_MAG))
    elif magnetic:
        out.append('    nspin = 2,')
        for i in range(1, ntyp + 1):
            out.append('    starting_magnetization({}) = {},'.format(i, DEFAULT_START_MAG))
    out.append(' /')

    # &electrons
    out.append(' &electrons')
    out.append('    mixing_beta = {}'.format(0.2 if metallic else 0.7))
    out.append(' /')

    # ATOMIC_SPECIES
    out.append('ATOMIC_SPECIES')
    out.extend(species_rows)

    # CELL_PARAMETERS (only when no ibrav was detected)
    if detected is None:
        out.append(geometry['cell_header'])
        out.extend('  ' + row for row in geometry['cell_rows'])

    # ATOMIC_POSITIONS
    if detected is not None:
        out.append('ATOMIC_POSITIONS (crystal)')
        out.extend(detected['pos_rows'])
    else:
        out.append(geometry['pos_header'])
        out.extend('  ' + row for row in geometry['pos_rows'])

    # K_POINTS
    out.append('K_POINTS {automatic}')
    out.append('  {} {} {} {} {} {}'.format(*kpts, *kshift))

    return '\n'.join(out) + '\n'
