"""Reconstruct Bravais lattice vectors from Quantum ESPRESSO ``ibrav``.

This module implements the lattice-vector conventions of Quantum
ESPRESSO's ``PW/src/latgen.f90`` (documented in ``INPUT_PW``).  Given a
QE ``ibrav`` index and the six ``celldm`` parameters it returns the three
primitive lattice vectors (as the *rows* of a ``(3, 3)`` array) in Bohr —
the same layout and units used elsewhere in PAOFLOW
(:func:`PAOFLOW.inputs.file_io.struct_from_outputfile_QE`).

The per-``ibrav`` vector definitions are kept in a registry
(:data:`_IBRAV_BUILDERS`) so the canonical forms can be reused by a future
inverse map (explicit lattice / AFLOW ``ibrav=0`` → canonical QE
``ibrav`` + ``celldm``).

celldm convention
-----------------
Following QE, ``celldm`` is a length-6 array:

- ``celldm[0]`` — ``a`` (the lattice parameter) in **Bohr**.
- ``celldm[1]`` — ``b/a``.
- ``celldm[2]`` — ``c/a``.
- ``celldm[3]`` — first cosine (meaning depends on ``ibrav``).
- ``celldm[4]`` — second cosine (meaning depends on ``ibrav``).
- ``celldm[5]`` — third cosine (meaning depends on ``ibrav``).

Public API
----------
- :func:`lattice_format_QE` — ``(ibrav, celldm)`` → lattice vectors (Bohr).
- :func:`celldm_from_namelist` — assemble ``celldm`` from a parsed QE
  ``&system`` block, accepting either the ``celldm(i)`` convention or the
  ``A``/``B``/``C``/``cosAB``/``cosAC``/``cosBC`` convention.
"""

import numpy as np

BOHR_RADIUS_ANGS = 0.529177210903

__all__ = [
    'lattice_format_QE',
    'celldm_from_namelist',
    'cell_lengths_angles',
    'bravais_to_ibrav',
    'qe_ibrav_from_lattice',
]


# --------------------------------------------------------------------------- #
# Per-ibrav primitive lattice vectors (rows), in Bohr.                         #
#                                                                              #
# Each builder takes the length-6 ``celldm`` array (celldm[0] = a in Bohr)     #
# and returns a (3, 3) ndarray whose rows are the primitive vectors, matching  #
# QE PW/src/latgen.f90 exactly.                                                #
# --------------------------------------------------------------------------- #
def _ibrav_1(celldm):
    """Simple cubic."""
    a = celldm[0]
    return a * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _ibrav_2(celldm):
    """Face-centred cubic."""
    a = celldm[0]
    return (a / 2.0) * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])


def _ibrav_3(celldm):
    """Body-centred cubic."""
    a = celldm[0]
    return (a / 2.0) * np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]])


def _ibrav_m3(celldm):
    """Body-centred cubic, more symmetric axis choice (ibrav=-3)."""
    a = celldm[0]
    return (a / 2.0) * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])


def _ibrav_4(celldm):
    """Hexagonal / trigonal P."""
    a = celldm[0]
    coa = celldm[2]
    return a * np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, coa]])


def _ibrav_5(celldm):
    """Trigonal R, 3-fold axis along c."""
    a = celldm[0]
    cg = celldm[3]
    tx = np.sqrt((1.0 - cg) / 2.0)
    ty = np.sqrt((1.0 - cg) / 6.0)
    tz = np.sqrt((1.0 + 2.0 * cg) / 3.0)
    return a * np.array([[tx, -ty, tz], [0.0, 2.0 * ty, tz], [-tx, -ty, tz]])


def _ibrav_m5(celldm):
    """Trigonal R, 3-fold axis along <111> (ibrav=-5)."""
    a = celldm[0]
    cg = celldm[3]
    ty = np.sqrt((1.0 - cg) / 6.0)
    tz = np.sqrt((1.0 + 2.0 * cg) / 3.0)
    ap = a / np.sqrt(3.0)
    u = tz - 2.0 * np.sqrt(2.0) * ty
    v = tz + np.sqrt(2.0) * ty
    return ap * np.array([[u, v, v], [v, u, v], [v, v, u]])


def _ibrav_6(celldm):
    """Tetragonal P."""
    a = celldm[0]
    coa = celldm[2]
    return a * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, coa]])


def _ibrav_7(celldm):
    """Tetragonal I (body-centred)."""
    a = celldm[0]
    coa = celldm[2]
    return (a / 2.0) * np.array([[1.0, -1.0, coa], [1.0, 1.0, coa], [-1.0, -1.0, coa]])


def _ibrav_8(celldm):
    """Orthorhombic P."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    return np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])


def _ibrav_9(celldm):
    """Base-centred orthorhombic, C-type."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    return np.array([[a / 2.0, b / 2.0, 0.0], [-a / 2.0, b / 2.0, 0.0], [0.0, 0.0, c]])


def _ibrav_m9(celldm):
    """Base-centred orthorhombic, C-type, alternate description (ibrav=-9)."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    return np.array([[a / 2.0, -b / 2.0, 0.0], [a / 2.0, b / 2.0, 0.0], [0.0, 0.0, c]])


def _ibrav_91(celldm):
    """Base-centred orthorhombic, A-type (ibrav=91)."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    return np.array([[a, 0.0, 0.0], [0.0, b / 2.0, -c / 2.0], [0.0, b / 2.0, c / 2.0]])


def _ibrav_10(celldm):
    """Face-centred orthorhombic."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    return np.array([[a / 2.0, 0.0, c / 2.0], [a / 2.0, b / 2.0, 0.0], [0.0, b / 2.0, c / 2.0]])


def _ibrav_11(celldm):
    """Body-centred orthorhombic."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    return np.array(
        [
            [a / 2.0, b / 2.0, c / 2.0],
            [-a / 2.0, b / 2.0, c / 2.0],
            [-a / 2.0, -b / 2.0, c / 2.0],
        ]
    )


def _ibrav_12(celldm):
    """Monoclinic P, unique axis c (celldm[3] = cos(ab) = cos gamma)."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    cg = celldm[3]
    sg = np.sqrt(1.0 - cg * cg)
    return np.array([[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [0.0, 0.0, c]])


def _ibrav_m12(celldm):
    """Monoclinic P, unique axis b (celldm[4] = cos(ac) = cos beta)."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    cb = celldm[4]
    sb = np.sqrt(1.0 - cb * cb)
    return np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [c * cb, 0.0, c * sb]])


def _ibrav_13(celldm):
    """Base-centred monoclinic, unique axis c (celldm[3] = cos(ab))."""
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    cg = celldm[3]
    sg = np.sqrt(1.0 - cg * cg)
    return np.array([[a / 2.0, 0.0, -c / 2.0], [b * cg, b * sg, 0.0], [a / 2.0, 0.0, c / 2.0]])


def _ibrav_m13(celldm):
    """Base-centred monoclinic, unique axis b (celldm[4] = cos(ac)).

    Uses the current (QE >= 6.4) convention.
    """
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    cb = celldm[4]
    sb = np.sqrt(1.0 - cb * cb)
    return np.array([[a / 2.0, b / 2.0, 0.0], [-a / 2.0, b / 2.0, 0.0], [c * cb, 0.0, c * sb]])


def _ibrav_14(celldm):
    """Triclinic.

    ``celldm[3] = cos(bc) = cos alpha``,
    ``celldm[4] = cos(ac) = cos beta``,
    ``celldm[5] = cos(ab) = cos gamma``.
    """
    a = celldm[0]
    b = celldm[1] * a
    c = celldm[2] * a
    calpha = celldm[3]
    cbeta = celldm[4]
    cgamma = celldm[5]
    sgamma = np.sqrt(1.0 - cgamma * cgamma)

    v1 = [a, 0.0, 0.0]
    v2 = [b * cgamma, b * sgamma, 0.0]
    v3x = c * cbeta
    v3y = c * (calpha - cbeta * cgamma) / sgamma
    v3z = (
        c
        * np.sqrt(
            1.0 + 2.0 * calpha * cbeta * cgamma - calpha * calpha - cbeta * cbeta - cgamma * cgamma
        )
        / sgamma
    )
    return np.array([v1, v2, [v3x, v3y, v3z]])


# Registry: ibrav -> builder.  Keep this the single source of truth for the
# canonical lattice forms so an inverse (lattice -> ibrav) map can reuse it.
_IBRAV_BUILDERS = {
    1: _ibrav_1,
    2: _ibrav_2,
    3: _ibrav_3,
    -3: _ibrav_m3,
    4: _ibrav_4,
    5: _ibrav_5,
    -5: _ibrav_m5,
    6: _ibrav_6,
    7: _ibrav_7,
    8: _ibrav_8,
    9: _ibrav_9,
    -9: _ibrav_m9,
    91: _ibrav_91,
    10: _ibrav_10,
    11: _ibrav_11,
    12: _ibrav_12,
    -12: _ibrav_m12,
    13: _ibrav_13,
    -13: _ibrav_m13,
    14: _ibrav_14,
}


def lattice_format_QE(ibrav, celldm):
    """Return the primitive lattice vectors for a QE ``ibrav``.

    Parameters
    ----------
    ibrav : int
        Quantum ESPRESSO Bravais-lattice index.  All non-zero QE values
        are supported: ``1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10,
        11, 12, -12, 13, -13, 14``.
    celldm : array_like
        Length-6 array of QE ``celldm`` parameters.  ``celldm[0]`` is the
        lattice parameter ``a`` in **Bohr**; ``celldm[1] = b/a``,
        ``celldm[2] = c/a`` and ``celldm[3:6]`` are the cosines whose
        meaning depends on ``ibrav`` (see module docstring).

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` array whose rows are the primitive lattice vectors in
        **Bohr**.

    Raises
    ------
    ValueError
        If ``ibrav`` is ``0`` (explicit ``CELL_PARAMETERS`` required) or
        not a recognised QE Bravais-lattice index.
    """
    ibrav = int(ibrav)
    celldm = np.asarray(celldm, dtype=float)
    if celldm.size < 6:
        celldm = np.concatenate([celldm, np.zeros(6 - celldm.size)])

    if ibrav == 0:
        raise ValueError(
            'ibrav=0 has no implicit lattice; an explicit CELL_PARAMETERS card is required.'
        )

    try:
        builder = _IBRAV_BUILDERS[ibrav]
    except KeyError:
        raise ValueError('Unsupported QE ibrav value: {}'.format(ibrav))

    return builder(celldm)


def _to_float(value):
    """Parse a QE numeric token, tolerating Fortran ``d``/``D`` exponents."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip().strip('"').strip("'").replace('d', 'e').replace('D', 'e'))


def celldm_from_namelist(system_block, ibrav):
    """Assemble a length-6 ``celldm`` array from a parsed QE ``&system`` block.

    Accepts either of the two mutually exclusive QE conventions:

    - ``celldm(1)``..``celldm(6)`` directly; or
    - ``A``, ``B``, ``C`` (Ångström) with ``cosAB``, ``cosAC``, ``cosBC``.

    The cosine slots are filled following QE's ``cell_base`` mapping, which
    depends on ``ibrav``:

    - ``ibrav = 14``: ``celldm[3]=cosBC``, ``celldm[4]=cosAC``,
      ``celldm[5]=cosAB``.
    - ``ibrav in {-12, -13}`` (unique axis b): ``celldm[4]=cosAC``.
    - all other ``ibrav`` needing one cosine: ``celldm[3]=cosAB``.

    Parameters
    ----------
    system_block : dict
        Mapping of lower-case ``&system`` keywords to their raw string
        values, as produced by
        :func:`PAOFLOW.inputs.file_io.struct_from_inputfile_QE`.
    ibrav : int
        QE Bravais-lattice index (controls the cosine mapping above).

    Returns
    -------
    numpy.ndarray
        Length-6 ``celldm`` array with ``celldm[0]`` in **Bohr**.

    Raises
    ------
    ValueError
        If neither ``celldm(1)`` nor ``A`` is present in the block.
    """
    ibrav = int(ibrav)
    sys = {k.lower().replace(' ', ''): v for k, v in system_block.items()}
    celldm = np.zeros(6, dtype=float)

    has_celldm = any('celldm({})'.format(i) in sys for i in range(1, 7))
    if has_celldm:
        for i in range(1, 7):
            key = 'celldm({})'.format(i)
            if key in sys:
                celldm[i - 1] = _to_float(sys[key])
        if celldm[0] == 0.0:
            raise ValueError('celldm(1) is required but missing or zero.')
        return celldm

    if 'a' not in sys:
        raise ValueError('Cannot build celldm: neither celldm(1) nor A found in &system block.')

    a = _to_float(sys['a'])
    b = _to_float(sys['b']) if 'b' in sys else 0.0
    c = _to_float(sys['c']) if 'c' in sys else 0.0

    celldm[0] = a / BOHR_RADIUS_ANGS  # A is in Ångström -> Bohr
    celldm[1] = (b / a) if b else 0.0
    celldm[2] = (c / a) if c else 0.0

    cosab = _to_float(sys['cosab']) if 'cosab' in sys else 0.0
    cosac = _to_float(sys['cosac']) if 'cosac' in sys else 0.0
    cosbc = _to_float(sys['cosbc']) if 'cosbc' in sys else 0.0

    if ibrav == 14:
        celldm[3] = cosbc
        celldm[4] = cosac
        celldm[5] = cosab
    elif ibrav in (-12, -13):
        celldm[4] = cosac
    else:
        celldm[3] = cosab

    return celldm


# =========================================================================== #
# Inverse map: explicit lattice vectors -> QE ibrav + celldm.                  #
#                                                                              #
# Used to translate an explicit cell (e.g. an AFLOW ``ibrav=0`` CONTCAR) into  #
# the canonical Quantum ESPRESSO ``ibrav`` + ``celldm`` description.  The      #
# strategy is metadata-primary with geometric validation: an AFLOW Bravais     #
# symbol / space-group number narrows the candidate ``ibrav``; the geometry    #
# is then used to (a) read the conventional cell parameters into ``celldm``    #
# and (b) prove that the canonical QE cell built from those parameters         #
# describes the *same* lattice as the input (an integer, unimodular change of  #
# basis exists).  Anything that fails validation falls back to ``ibrav=0`` so  #
# a wrong cell is never emitted silently.                                      #
# =========================================================================== #

# Ratio of conventional-cell volume to primitive-cell volume for each ibrav
# (i.e. the centering multiplicity).
_IBRAV_INDEX = {
    1: 1,
    2: 4,
    3: 2,
    -3: 2,
    4: 1,
    5: 1,
    -5: 1,
    6: 1,
    7: 2,
    8: 1,
    9: 2,
    -9: 2,
    91: 2,
    10: 4,
    11: 2,
    12: 1,
    -12: 1,
    13: 2,
    -13: 2,
    14: 1,
}

# AFLOW Bravais-lattice symbols (and a few common aliases) -> QE ibrav.
# AFLOW reports the Bravais lattice of the relaxed structure in fields such as
# ``Bravais_lattice_relax`` / ``bravais_lattice_lattice_type`` using these
# short symbols.  Several map to a base ibrav whose sign/variant is then
# resolved geometrically (e.g. -3 vs 3, -9 vs 9 vs 91, -12 vs 12).
_BRAVAIS_TO_IBRAV = {
    'CUB': 1,
    'cP': 1,
    'FCC': 2,
    'cF': 2,
    'BCC': 3,
    'cI': 3,
    'HEX': 4,
    'hP': 4,
    'RHL': 5,
    'hR': 5,
    'TET': 6,
    'tP': 6,
    'BCT': 7,
    'tI': 7,
    'ORC': 8,
    'oP': 8,
    'ORCC': 9,
    'oS': 9,
    'oC': 9,
    'ORCA': 91,
    'oA': 91,
    'ORCF': 10,
    'oF': 10,
    'ORCI': 11,
    'oI': 11,
    'MCL': 12,
    'mP': 12,
    'MCLC': 13,
    'mS': 13,
    'mC': 13,
    'TRI': 14,
    'aP': 14,
}


def cell_lengths_angles(lattice):
    """Return ``(a, b, c, alpha, beta, gamma)`` for a set of cell vectors.

    Parameters
    ----------
    lattice : array_like
        ``(3, 3)`` array whose *rows* are the cell vectors (any length unit).

    Returns
    -------
    tuple
        Lengths ``a, b, c`` (same unit as ``lattice``) and angles
        ``alpha, beta, gamma`` in **degrees**.  ``alpha`` is the angle
        between vectors 2 and 3, ``beta`` between 1 and 3, ``gamma`` between
        1 and 2.
    """
    lat = np.asarray(lattice, dtype=float)
    a, b, c = (float(np.linalg.norm(v)) for v in lat)
    alpha = np.degrees(np.arccos(np.clip(lat[1] @ lat[2] / (b * c), -1.0, 1.0)))
    beta = np.degrees(np.arccos(np.clip(lat[0] @ lat[2] / (a * c), -1.0, 1.0)))
    gamma = np.degrees(np.arccos(np.clip(lat[0] @ lat[1] / (a * b), -1.0, 1.0)))
    return a, b, c, float(alpha), float(beta), float(gamma)


def bravais_to_ibrav(symbol, spacegroup=None):
    """Map an AFLOW Bravais-lattice symbol to a candidate QE ``ibrav``.

    Parameters
    ----------
    symbol : str
        AFLOW Bravais-lattice symbol (e.g. ``'FCC'``, ``'BCC'``, ``'HEX'``,
        ``'ORCC'``) or Pearson-style two-letter code (e.g. ``'cF'``).
    spacegroup : int, optional
        Space-group number.  Currently unused for the base mapping but
        accepted so callers can pass it through; variant/sign disambiguation
        is done geometrically.

    Returns
    -------
    int or None
        The candidate ``ibrav`` (the magnitude/base variant), or ``None`` if
        the symbol is not recognised.
    """
    if symbol is None:
        return None
    key = str(symbol).strip()
    if key in _BRAVAIS_TO_IBRAV:
        return _BRAVAIS_TO_IBRAV[key]
    # Case-insensitive retry on the upper-case three/four-letter symbols.
    upper = key.upper()
    for k, v in _BRAVAIS_TO_IBRAV.items():
        if k.upper() == upper:
            return v
    return None


def _integer_lattice_points(bound):
    """Integer coordinate triples in ``[-bound, bound]^3`` excluding origin."""
    pts = []
    for i in range(-bound, bound + 1):
        for j in range(-bound, bound + 1):
            for k in range(-bound, bound + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                pts.append((i, j, k))
    return pts


def _candidate_vectors(lattice, bound=2):
    """Return ``(coords, carts, lengths)`` for bounded integer lattice vectors.

    ``coords`` is an ``(N, 3)`` int array of integer combinations, ``carts``
    the corresponding Cartesian vectors (rows of ``lattice``), ``lengths``
    their norms, all sorted by increasing length.
    """
    lat = np.asarray(lattice, dtype=float)
    coords = np.array(_integer_lattice_points(bound), dtype=int)
    carts = coords @ lat
    lengths = np.linalg.norm(carts, axis=1)
    order = np.argsort(lengths, kind='stable')
    return coords[order], carts[order], lengths[order]


def _angles_ok(angles, expected, atol):
    """Check measured angles against expected fixed values (None = free)."""
    for meas, exp in zip(angles, expected):
        if exp is not None and abs(meas - exp) > atol:
            return False
    return True


# Per-ibrav conventional-cell symmetry constraints used to *select* the
# conventional cell from the lattice:
#   'angles'  : expected (alpha, beta, gamma) in degrees, None = free.
#   'eq'      : tuple of index groups whose lengths must be equal
#               (0=a, 1=b, 2=c).
#   'eq_ang'  : True if all three angles must be equal (rhombohedral).
_IBRAV_CONSTRAINTS = {
    1: {'angles': (90, 90, 90), 'eq': ((0, 1, 2),), 'eq_ang': False},
    2: {'angles': (90, 90, 90), 'eq': ((0, 1, 2),), 'eq_ang': False},
    3: {'angles': (90, 90, 90), 'eq': ((0, 1, 2),), 'eq_ang': False},
    -3: {'angles': (90, 90, 90), 'eq': ((0, 1, 2),), 'eq_ang': False},
    4: {'angles': (90, 90, 120), 'eq': ((0, 1),), 'eq_ang': False},
    5: {'angles': (None, None, None), 'eq': ((0, 1, 2),), 'eq_ang': True},
    -5: {'angles': (None, None, None), 'eq': ((0, 1, 2),), 'eq_ang': True},
    6: {'angles': (90, 90, 90), 'eq': ((0, 1),), 'eq_ang': False},
    7: {'angles': (90, 90, 90), 'eq': ((0, 1),), 'eq_ang': False},
    8: {'angles': (90, 90, 90), 'eq': (), 'eq_ang': False},
    9: {'angles': (90, 90, 90), 'eq': (), 'eq_ang': False},
    -9: {'angles': (90, 90, 90), 'eq': (), 'eq_ang': False},
    91: {'angles': (90, 90, 90), 'eq': (), 'eq_ang': False},
    10: {'angles': (90, 90, 90), 'eq': (), 'eq_ang': False},
    11: {'angles': (90, 90, 90), 'eq': (), 'eq_ang': False},
    12: {'angles': (90, 90, None), 'eq': (), 'eq_ang': False},
    13: {'angles': (90, 90, None), 'eq': (), 'eq_ang': False},
    -12: {'angles': (90, None, 90), 'eq': (), 'eq_ang': False},
    -13: {'angles': (90, None, 90), 'eq': (), 'eq_ang': False},
    14: {'angles': (None, None, None), 'eq': (), 'eq_ang': False},
}


def _conventional_constraint_ok(conv, ibrav, atol_len, atol_ang):
    """True if a candidate conventional cell matches the ibrav symmetry."""
    a, b, c, alpha, beta, gamma = cell_lengths_angles(conv)
    spec = _IBRAV_CONSTRAINTS[ibrav]
    if not _angles_ok((alpha, beta, gamma), spec['angles'], atol_ang):
        return False
    lengths = (a, b, c)
    for group in spec['eq']:
        ref = lengths[group[0]]
        for idx in group[1:]:
            if abs(lengths[idx] - ref) > atol_len * max(1.0, ref):
                return False
    if spec['eq_ang']:
        if abs(alpha - beta) > atol_ang or abs(beta - gamma) > atol_ang:
            return False
    return True


def _celldm_from_conventional(conv, ibrav):
    """Read a length-6 ``celldm`` (Bohr) from a conventional cell (Bohr)."""
    a, b, c, alpha, beta, gamma = cell_lengths_angles(conv)
    celldm = np.zeros(6, dtype=float)
    celldm[0] = a
    celldm[1] = b / a
    celldm[2] = c / a
    ca = np.cos(np.radians(alpha))
    cb = np.cos(np.radians(beta))
    cg = np.cos(np.radians(gamma))
    if ibrav in (5, -5):
        celldm[1] = 0.0
        celldm[2] = 0.0
        celldm[3] = cg  # angle between the equal-length rhombohedral axes
    elif ibrav == 14:
        celldm[3] = ca
        celldm[4] = cb
        celldm[5] = cg
    elif ibrav in (-12, -13):
        celldm[4] = cb  # unique axis b: beta is the oblique angle
    elif ibrav in (12, 13):
        celldm[3] = cg  # unique axis c: gamma is the oblique angle
    return celldm


def _find_cell_map(g_in, g_qe, tol, bound=3):
    """Find an integer unimodular ``M`` with ``M G_in M^T = G_qe``.

    Both ``g_in`` and ``g_qe`` are ``(3, 3)`` Gram (metric) matrices of two
    bases of the *same* lattice.  Returns the integer matrix ``M`` (rows are
    the QE primitive vectors expressed in the input basis) or ``None``.
    """
    coords = np.array(_integer_lattice_points(bound), dtype=int)
    # Squared length of each candidate in the input metric.
    len2 = np.einsum('ni,ij,nj->n', coords, g_in, coords)
    targets = np.diag(g_qe)
    cand = []
    for t in targets:
        mask = np.abs(len2 - t) <= tol
        cand.append(coords[mask])
    if any(len(c) == 0 for c in cand):
        return None
    for v1 in cand[0]:
        for v2 in cand[1]:
            if abs(v1 @ g_in @ v2 - g_qe[0, 1]) > tol:
                continue
            for v3 in cand[2]:
                if abs(v1 @ g_in @ v3 - g_qe[0, 2]) > tol:
                    continue
                if abs(v2 @ g_in @ v3 - g_qe[1, 2]) > tol:
                    continue
                M = np.array([v1, v2, v3], dtype=int)
                det = int(round(np.linalg.det(M)))
                if det not in (1, -1):
                    continue
                if np.allclose(M @ g_in @ M.T, g_qe, atol=tol):
                    return M
    return None


def _permute_conventional(conv):
    """Yield axis permutations and per-axis sign flips of a conventional cell.

    Sign flips are restricted to keep a right-handed cell.  Used to try every
    reasonable axis assignment so the validation step can confirm the correct
    one regardless of the order in which the conventional vectors were found.
    """
    import itertools

    conv = np.asarray(conv, dtype=float)
    for perm in itertools.permutations(range(3)):
        base = conv[list(perm)]
        for signs in itertools.product((1, -1), repeat=3):
            cand = base * np.array(signs)[:, None]
            if np.linalg.det(cand) > 0:
                yield cand


def qe_ibrav_from_lattice(
    lattice,
    bravais_hint=None,
    spacegroup=None,
    symprec=1e-4,
):
    """Classify an explicit lattice as a QE ``ibrav`` + ``celldm``.

    Parameters
    ----------
    lattice : array_like
        ``(3, 3)`` primitive cell, rows are the lattice vectors in **Bohr**.
    bravais_hint : str, optional
        AFLOW Bravais-lattice symbol (e.g. ``'FCC'``); when given it narrows
        the candidate ``ibrav`` and makes classification robust.  When
        omitted, every supported ``ibrav`` is tried in order of decreasing
        symmetry.
    spacegroup : int, optional
        Space-group number (passed through to :func:`bravais_to_ibrav`).
    symprec : float, optional
        Absolute tolerance (Bohr) for length comparisons; the angular
        tolerance is derived from it.  Defaults to ``1e-4``.

    Returns
    -------
    dict
        ``{'ibrav': int, 'celldm': ndarray, 'M': ndarray}`` on success, where
        ``M`` is the integer unimodular matrix mapping the input basis to the
        QE primitive basis (so fractional coordinates transform as
        ``f_qe = f_in @ inv(M)``).  On failure returns
        ``{'ibrav': 0, 'celldm': None, 'M': None}`` and the caller should keep
        the explicit ``CELL_PARAMETERS`` description.
    """
    lat = np.asarray(lattice, dtype=float)
    fail = {'ibrav': 0, 'celldm': None, 'M': None}
    if lat.shape != (3, 3) or abs(np.linalg.det(lat)) < 1e-12:
        return fail

    g_in = lat @ lat.T
    v_prim = abs(np.linalg.det(lat))
    scale = v_prim ** (1.0 / 3.0)
    atol_len = symprec
    atol_ang = np.degrees(symprec / max(scale, 1e-6))
    atol_ang = max(atol_ang, 0.05)
    tol_gram = max(symprec, 1e-6) * max(scale, 1.0) * 4.0

    # Candidate ibrav ordering.
    if bravais_hint is not None:
        base = bravais_to_ibrav(bravais_hint, spacegroup)
        if base is None:
            candidates = list(_IBRAV_CONSTRAINTS.keys())
        elif base == 3:
            candidates = [3, -3]
        elif base == 9:
            candidates = [9, -9, 91]
        elif base == 12:
            candidates = [12, -12]
        elif base == 13:
            candidates = [13, -13]
        else:
            candidates = [base]
    else:
        candidates = [1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14]

    coords, carts, _lengths = _candidate_vectors(lat, bound=3)
    # Limit the conventional-cell search to the shortest vectors for speed.
    n_short = min(len(coords), 60)
    short_carts = carts[:n_short]

    for ibrav in candidates:
        index = _IBRAV_INDEX[ibrav]
        v_conv_target = index * v_prim
        spec = _IBRAV_CONSTRAINTS[ibrav]

        # Find a conventional cell: a triple of lattice vectors with the
        # right volume and the ibrav's symmetry pattern.
        found = None
        for i in range(n_short):
            v1 = short_carts[i]
            for j in range(n_short):
                if j == i:
                    continue
                v2 = short_carts[j]
                for k in range(n_short):
                    if k in (i, j):
                        continue
                    v3 = short_carts[k]
                    conv = np.array([v1, v2, v3])
                    det = np.linalg.det(conv)
                    if det <= 0:
                        continue
                    if abs(det - v_conv_target) > tol_gram:
                        continue
                    if not _conventional_constraint_ok(conv, ibrav, atol_len, atol_ang):
                        continue
                    found = conv
                    break
                if found is not None:
                    break
            if found is not None:
                break

        if found is None:
            continue

        # Try axis assignments; validate each by requiring an integer
        # unimodular map between the input and the canonical QE cell.
        for conv in _permute_conventional(found):
            if not _conventional_constraint_ok(conv, ibrav, atol_len, atol_ang):
                continue
            celldm = _celldm_from_conventional(conv, ibrav)
            if celldm[0] <= 0:
                continue
            try:
                lat_qe = lattice_format_QE(ibrav, celldm)
            except ValueError:
                continue
            g_qe = lat_qe @ lat_qe.T
            M = _find_cell_map(g_in, g_qe, tol_gram, bound=3)
            if M is not None:
                return {'ibrav': ibrav, 'celldm': celldm, 'M': M}

    return fail
