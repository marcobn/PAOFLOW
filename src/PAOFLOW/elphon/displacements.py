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


def generate_eph_displacements(phonon, distance, is_plusminus='auto', displacement_mode='symmetry'):
    """Finite displacements for the electron-phonon derivative.

    Two modes are available:

    * ``displacement_mode='symmetry'`` (default): delegate to
      :meth:`phonopy.Phonopy.generate_displacements`, so the crystal symmetry
      reduces the set to the inequivalent displacements (a single one for fcc
      Al).  The full Cartesian tensor is then reconstructed by the symmetry
      expansion.  ``is_plusminus='auto'`` adds the minus displacement only when
      it is not symmetry-related to the plus; ``True`` always adds it.
    * ``displacement_mode='cartesian'``: displace every reference primitive-cell
      atom explicitly along x, y and z (``is_plusminus=True`` -> central ``+/-``
      pairs, otherwise a single ``+`` per axis with a forward difference against
      the reference).  This yields the full Cartesian derivative directly, with
      no symmetry expansion, at the cost of more DFT runs.

    Parameters
    ----------
    phonon : phonopy.Phonopy
        Initialised object (``phonon.supercell`` / ``phonon.primitive`` set).
    distance : float
        Displacement amplitude in Bohr.
    is_plusminus : {'auto', True, False}
        Central-difference control (see above).
    displacement_mode : {'symmetry', 'cartesian'}
        Displacement generation strategy.

    Returns
    -------
    cells : list[phonopy.structure.atoms.PhonopyAtoms]
        The displaced supercells.
    meta : list[dict]
        One entry per cell: ``{index, sc_atom, displacement:[dx, dy, dz], distance}``
        with the Cartesian displacement vector (Bohr).
    """
    if displacement_mode == 'cartesian':
        return _cartesian_displacements(phonon, distance, is_plusminus)
    if displacement_mode != 'symmetry':
        raise ValueError("displacement_mode must be 'symmetry' or 'cartesian'.")

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


def _cartesian_displacements(phonon, distance, is_plusminus):
    """Explicit x/y/z displacements of every reference primitive-cell atom."""
    from phonopy.structure.atoms import PhonopyAtoms

    supercell = phonon.supercell
    p2s = reference_supercell_atoms(phonon)
    base_pos = np.asarray(supercell.positions, dtype=float)
    symbols = list(supercell.symbols)
    cell = np.asarray(supercell.cell, dtype=float)
    masses = np.asarray(supercell.masses, dtype=float)

    signs = (+1, -1) if is_plusminus is True else (+1,)

    cells, meta = [], []
    index = 0
    for sc_atom in p2s:
        for alpha in range(3):
            for sign in signs:
                pos = base_pos.copy()
                pos[sc_atom, alpha] += sign * distance
                cells.append(PhonopyAtoms(symbols=symbols, cell=cell, positions=pos, masses=masses))
                vec = [0.0, 0.0, 0.0]
                vec[alpha] = sign * distance
                meta.append(
                    {
                        'index': index,
                        'sc_atom': int(sc_atom),
                        'displacement': vec,
                        'distance': float(distance),
                    }
                )
                index += 1
    return cells, meta
