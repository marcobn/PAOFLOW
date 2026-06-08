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

__all__ = ['lattice_format_QE', 'celldm_from_namelist']


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
