"""Symmetry-reduced displacement bookkeeping for the finite-difference e-phonon.

``dV_{kappa,alpha} = dH/du_{kappa,alpha}`` is obtained by finite-differencing the
PAO Hamiltonian.  The displacement set is reduced by the **crystal symmetry**,
exactly as in the harmonic-phonon workflow: we delegate to phonopy's
``generate_displacements`` so only the symmetry-inequivalent displacements are
computed (e.g. a *single* displacement for fcc metals such as Al), and the full
derivative tensor is reconstructed from them by symmetry.
"""

import numpy as np

CARTESIAN = ('x', 'y', 'z')


def reference_supercell_atoms(phonon):
    """Supercell atom indices of the reference primitive-cell atoms (p2s_map)."""
    return np.asarray(phonon.primitive.p2s_map, dtype=int)


def generate_eph_displacements(phonon, distance, is_plusminus='auto'):
    """Symmetry-reduced finite displacements (reuses phonopy's reduction).

    Delegates to :meth:`phonopy.Phonopy.generate_displacements`, so the crystal
    symmetry reduces the set to the inequivalent displacements.  With
    ``is_plusminus='auto'`` (the phonopy/phonon default) the minus displacement
    is added only when it is not related to the plus by a symmetry operation
    (for fcc Al this yields a single displacement); ``is_plusminus=True`` always
    adds the minus so a central difference can be taken.

    Parameters
    ----------
    phonon : phonopy.Phonopy
        Initialised object (``phonon.supercell`` / ``phonon.primitive`` set).
    distance : float
        Displacement amplitude in Bohr.
    is_plusminus : {'auto', True, False}
        Passed through to phonopy.

    Returns
    -------
    cells : list[phonopy.structure.atoms.PhonopyAtoms]
        The symmetry-reduced displaced supercells.
    meta : list[dict]
        One entry per cell: ``{index, sc_atom, displacement:[dx, dy, dz], distance}``
        with the Cartesian displacement vector (Bohr).
    """
    phonon.generate_displacements(distance=distance, is_plusminus=is_plusminus)
    cells = list(phonon.supercells_with_displacements)

    rows = np.asarray(phonon.displacements, dtype=float)  # (ndisp, 4): [atom, dx, dy, dz]
    meta = []
    for i, row in enumerate(rows):
        meta.append(
            {
                'index': int(i),
                'sc_atom': int(round(row[0])),
                'displacement': [float(row[1]), float(row[2]), float(row[3])],
                'distance': float(distance),
            }
        )
    return cells, meta
