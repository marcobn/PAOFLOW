"""Finite slabs from the bulk PAO Hamiltonian for surface-sensitive photoemission."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .basis import PhotoemissionBasis
from .bloch import BlochHamiltonian
from .radial import BOHR_ANGSTROM


def build_slab(
    bloch: BlochHamiltonian,
    basis: PhotoemissionBasis,
    nlayers: int,
    axis: int = 2,
) -> tuple[BlochHamiltonian, PhotoemissionBasis]:
    """Stack ``nlayers`` cells along a lattice vector with open boundaries.

    Parameters
    ----------
    bloch : BlochHamiltonian
        Bulk Hamiltonian (Wigner-Seitz images of the FFT cells).
    basis : PhotoemissionBasis
        Bulk orbitals.
    nlayers : int
        Number of cells in the slab.
    axis : int, optional
        Lattice vector along which the slab is stacked (``0, 1, 2``); the surface is the
        plane of the other two vectors, so the stacking vector should be chosen (with a
        supercell if needed) to be the surface normal or close to it.

    Returns
    -------
    slab : BlochHamiltonian
        Two-dimensional Hamiltonian of ``nlayers * nawf`` orbitals (cells with a zero
        component along ``axis``), same lattice vectors.
    slab_basis : PhotoemissionBasis
        The orbitals of every layer, positions shifted by the layer offset and ``depth`` set
        from the topmost layer along the normal of the surface plane (Å).

    Notes
    -----
    Orbital ``(L, i)`` of layer ``L`` couples to ``(L', j)`` through every bulk image cell
    ``n`` with ``n_axis = L' - L``: :math:`H^{\\rm slab}_{(L i),(L' j)}(\\mathbf k_\\parallel) =
    \\sum_{n:\\,n_{\\rm axis} = L'-L} W_n H_{ij}(R_n)\\,e^{-2\\pi i\\,\\mathbf k_\\parallel\\cdot n_\\parallel}`.
    The hoppings beyond the range of the bulk images are zero (the FFT grid of ``HRs`` sets
    the range); layer ``0`` is the top surface.
    """
    nawf = bloch.nawf
    nl = int(nlayers)
    if nl < 1:
        raise ValueError('nlayers must be positive')
    cells = bloch.cells
    inplane = np.delete(cells, axis, axis=1)
    unique_inplane, inverse = np.unique(inplane, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    hn = np.zeros((unique_inplane.shape[0], nl * nawf, nl * nawf), dtype=complex)
    offsets = cells[:, axis]
    for L in range(nl):
        for Lp in range(nl):
            select = np.nonzero(offsets == Lp - L)[0]
            for n in select:
                block = hn[inverse[n], L * nawf : (L + 1) * nawf, Lp * nawf : (Lp + 1) * nawf]
                block += bloch.hn[n]
    slab_cells = np.insert(unique_inplane, axis, 0, axis=1)
    slab = BlochHamiltonian(hn, slab_cells, bloch.at, bloch.alat_bohr, bloch.fermi_level)

    lattice = bloch.at * bloch.alat_bohr * BOHR_ANGSTROM  # rows, Å
    stacking = lattice[axis]
    others = [lattice[i] for i in range(3) if i != axis]
    normal = np.cross(others[0], others[1])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, stacking) < 0.0:
        normal = -normal
    orbitals = []
    for L in range(nl):
        shift = -L * stacking  # layer 0 on top, deeper layers below
        for o in basis.orbitals:
            orbitals.append(
                replace(o, index=L * nawf + o.index, atom=L * (basis.positions.shape[0]) + o.atom,
                        position=o.position + shift)
            )  # fmt: skip
    slab_basis = PhotoemissionBasis(orbitals).with_depths(normal)
    return slab, slab_basis
