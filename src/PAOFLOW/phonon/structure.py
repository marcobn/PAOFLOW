"""Structure bridge between PAOFLOW and phonopy/phono3py.

PAOFLOW stores the crystal structure (parsed from the DFT output) in the
``DataController`` using the following conventions:

``alat``        Lattice parameter in Bohr (atomic units).
``a_vectors``   ``(3, 3)`` array; rows are the direct lattice vectors expressed
                in units of ``alat`` (so the Cartesian lattice in Bohr is
                ``a_vectors * alat``).
``tau``         ``(natoms, 3)`` array of Cartesian atomic positions in Bohr.
``atoms``       length-``natoms`` list of atom labels (e.g. ``'Si'``, ``'Fe1'``).
``species``     list of ``(label, pseudo_file)`` tuples.
``omega``       cell volume in Bohr^3.

When phonopy is driven through its Quantum ESPRESSO interface
(``Phonopy(..., calculator='qe')``) the lattice and positions are expressed in
**Bohr** (the QE-native length unit), forces in Ry/Bohr, and the frequency
conversion factor is set accordingly.  PAOFLOW already stores the structure in
Bohr, so the bridge keeps everything in Bohr: lattice row-wise in Bohr and
atomic positions as fractional (scaled) coordinates.  This module performs the
bidirectional conversion and provides a round-trip verification helper.
"""

import numpy as np


def _element_symbol(label):
    """Return the chemical element symbol embedded in a PAOFLOW atom label.

    QE/PAOFLOW labels may carry trailing digits or markers used to distinguish
    inequivalent sites of the same element (e.g. ``'Fe1'``, ``'Fe2'``).  phonopy
    needs valid chemical symbols to assign atomic masses, so we strip every
    non-alphabetic character and capitalise canonically (``'fe'`` -> ``'Fe'``).
    """
    sym = ''.join(ch for ch in str(label) if ch.isalpha())
    if len(sym) == 0:
        raise ValueError(f"Could not extract an element symbol from label '{label}'.")
    return sym[0].upper() + sym[1:].lower()


def paoflow_to_phonopy(data_controller, scale=1.0):
    """Build a :class:`phonopy.structure.atoms.PhonopyAtoms` from PAOFLOW data.

    Parameters
    ----------
    data_controller : DataController
        Provides the structural arrays/attributes described in the module
        docstring.
    scale : float, optional
        Isotropic linear scale factor applied to the lattice vectors (the cell
        volume scales as ``scale**3``).  The fractional atomic positions are
        preserved, so atoms move with the cell.  Used by the quasi-harmonic
        volume scan; defaults to ``1.0`` (no scaling).

    Returns
    -------
    phonopy.structure.atoms.PhonopyAtoms
        Primitive/unit cell in the phonopy QE convention (Bohr lattice, scaled
        positions, chemical symbols and default isotopic masses).
    """
    from phonopy.structure.atoms import PhonopyAtoms

    arry, attr = data_controller.data_dicts()

    alat = attr['alat']
    a_vectors = np.asarray(arry['a_vectors'], dtype=float)
    tau = np.asarray(arry['tau'], dtype=float)
    atoms = list(arry['atoms'])

    # Cartesian lattice in Bohr (rows are lattice vectors).
    cell_bohr = a_vectors * alat

    # Fractional positions are unitless: solve tau = scaled @ cell_bohr.  They
    # are scale-invariant, so the isotropic strain enters only through the cell.
    scaled_positions = tau @ np.linalg.inv(cell_bohr)

    symbols = [_element_symbol(a) for a in atoms]

    return PhonopyAtoms(
        symbols=symbols,
        cell=cell_bohr * float(scale),
        scaled_positions=scaled_positions,
    )


def phonopy_to_paoflow(cell, alat=None):
    """Convert a :class:`PhonopyAtoms` back to PAOFLOW structural quantities.

    Parameters
    ----------
    cell : phonopy.structure.atoms.PhonopyAtoms
        Cell to convert (Bohr lattice, scaled positions; QE convention).
    alat : float, optional
        Lattice parameter in Bohr to use as the ``alat`` reference.  When not
        supplied the norm of the first lattice vector (in Bohr) is used, which
        matches QE's default ``alat`` convention for many Bravais lattices.

    Returns
    -------
    dict
        Mapping with keys ``alat``, ``a_vectors``, ``tau``, ``atoms``,
        ``omega`` expressed in PAOFLOW conventions (Bohr / units of ``alat``).
    """
    cell_bohr = np.asarray(cell.cell, dtype=float)
    scaled_positions = np.asarray(cell.scaled_positions, dtype=float)
    symbols = [str(s) for s in cell.symbols]

    if alat is None:
        alat = np.linalg.norm(cell_bohr[0])

    a_vectors = cell_bohr / alat
    tau = scaled_positions @ cell_bohr
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))

    return {
        'alat': alat,
        'a_vectors': a_vectors,
        'tau': tau,
        'atoms': symbols,
        'omega': omega,
    }


def verify_round_trip(data_controller, rtol=1.0e-8, atol=1.0e-8):
    """Check that PAOFLOW -> PhonopyAtoms -> PAOFLOW preserves the structure.

    Compares the Cartesian lattice (Bohr), Cartesian atomic positions (Bohr)
    and chemical symbols/masses before and after the conversion.

    Parameters
    ----------
    data_controller : DataController
        Source of the reference structure.
    rtol, atol : float
        Relative/absolute tolerances forwarded to :func:`numpy.allclose`.

    Returns
    -------
    dict
        Diagnostics with the maximum lattice/position deviations (in Bohr),
        the symbol/mass match flags and an overall ``ok`` boolean.
    """
    arry, attr = data_controller.data_dicts()

    alat = attr['alat']
    cell_bohr_ref = np.asarray(arry['a_vectors'], dtype=float) * alat
    tau_ref = np.asarray(arry['tau'], dtype=float)
    symbols_ref = [_element_symbol(a) for a in arry['atoms']]

    cell = paoflow_to_phonopy(data_controller)
    back = phonopy_to_paoflow(cell, alat=alat)

    cell_bohr_rt = np.asarray(back['a_vectors'], dtype=float) * back['alat']
    tau_rt = np.asarray(back['tau'], dtype=float)
    symbols_rt = [_element_symbol(a) for a in back['atoms']]

    masses_ref = list(cell.masses)

    lattice_dev = float(np.max(np.abs(cell_bohr_rt - cell_bohr_ref)))
    position_dev = float(np.max(np.abs(tau_rt - tau_ref)))
    symbols_match = symbols_ref == symbols_rt

    lattice_ok = np.allclose(cell_bohr_rt, cell_bohr_ref, rtol=rtol, atol=atol)
    position_ok = np.allclose(tau_rt, tau_ref, rtol=rtol, atol=atol)

    return {
        'ok': bool(lattice_ok and position_ok and symbols_match),
        'lattice_dev_bohr': lattice_dev,
        'position_dev_bohr': position_dev,
        'symbols_match': symbols_match,
        'symbols': symbols_rt,
        'masses': masses_ref,
    }


def primitive_atom_info(data_controller):
    """Return the phonopy primitive-cell atom ordering for Born-charge I/O.

    Born effective charges are a property of the primitive cell, so a phonopy
    ``BORN`` file lists one tensor per primitive atom in the phonopy primitive
    order (which may differ from the PAOFLOW ``tau`` order).  This helper
    exposes that ordering and the per-atom masses/positions.

    Returns
    -------
    dict
        ``symbols`` (list), ``masses`` (ndarray), ``scaled_positions``
        ``(natom_prim, 3)``, ``cell`` (Bohr lattice rows) and ``natom``.
    """
    arry, _ = data_controller.data_dicts()
    phonon = arry['phonopy']
    primitive = phonon.primitive

    return {
        'symbols': [str(s) for s in primitive.symbols],
        'masses': np.asarray(primitive.masses, dtype=float),
        'scaled_positions': np.asarray(primitive.scaled_positions, dtype=float),
        'cell': np.asarray(primitive.cell, dtype=float),
        'natom': len(primitive),
    }
