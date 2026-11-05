"""Cell doubling as pure index arithmetic on the bond list.

Replicates ``hamiltonian.do_doubling.doubling_HRs`` without forming the dense
``(2 nawf, 2 nawf, nk1, nk2, nk3, nspin)`` array: ``nnz`` doubles per step (plus the split
Nyquist plane), where the dense kernel quadruples.  The dense kernel indexes its ``cell_index``
by the *negated* folded coordinate (``ix = -round(Rx*nk1)``), so in the folded coordinates of
:func:`~PAOFLOW.sparse.hamiltonian.folded_R_triples` it builds

.. math::

    H^{2\\times}(M) = \\begin{pmatrix}
        H(2M) & H(2M-1) \\\\ H(2M+1) & H(2M)
    \\end{pmatrix}

Inverse map per bond with folded axis index ``m`` (floor division, exact for negative ``m``):

- ``m`` even: same replica, blocks (0,0) and (1,1) at ``M = m/2``, ``dnm`` kept;
- ``m`` odd: cross replica, block (0,1) at ``M = (m+1)/2`` and (1,0) at ``M = (m-1)/2``,
  ``dnm`` shifted by ``-/+ alat * a_axis``.

The cross-replica ``dnm`` and the Nyquist split depart from the dense kernel on purpose; see
:func:`double_axis`.  ``eig_bound`` carries over (rows are only rearranged and duplicated), and
thresholding commutes with doubling.  The driver updates ``tau``/``a_vectors`` in the
DataController and the scalar attributes (dense ``doubling_attr_arry``); ``double_axis`` only
updates the container's own ``a_vectors``.
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import SparseHamiltonian, split_nyquist_axis, unique_R


def double_axis(sph: SparseHamiltonian, axis: int, split_nyquist: bool = True) -> SparseHamiltonian:
    """Return a new bond list for the cell doubled along one axis.

    Parameters
    ----------
    sph : SparseHamiltonian
        Bond list of the current cell; left unmodified, must not be compacted.
    axis : int
        Lattice axis to double, 0, 1 or 2.
    split_nyquist : bool, optional
        Split the ``-nk/2`` plane of ``axis`` before mapping (see Notes).  ``False`` gives the
        dense kernel's bonds exactly, for the parity tests only.

    Returns
    -------
    SparseHamiltonian
        ``2 * nawf`` orbitals, ``2 * nnz`` bonds (plus two per bond on the split Nyquist plane),
        the same R grid, and ``a_vectors`` with the doubled axis scaled by two.

    Raises
    ------
    AssertionError
        If a mapped index leaves the folded window, i.e. no longer names a supercell lattice vector.
    RuntimeError
        If ``sph`` has been compacted.

    Notes
    -----
    ``dnm`` stays the supercell's ``x_i - x_j``, with replica 1 at ``x + alat * a_axis``: block
    (0,1) gets ``dnm - alat * a_axis``, (1,0) ``dnm + alat * a_axis``.  The ``dnm`` term adds
    ``i [x, H(k)]`` to ``dH/dk``, harmless on band diagonals only in that commutator form.  The
    dense ``doubling_attr_arry`` zeroes cross-replica ``Dnm``, which on Si (example01) puts
    velocities off by ~35% and sigma by ~50% on every doubled cell.

    Doubling maps ``-nk/2`` to ``-nk/4``, no longer a Nyquist plane, so the plan cannot split it
    later and :meth:`~SparseHamiltonian.hermitize` only adds the ``R -> -R`` mirror.  That suffices
    on one axis, but a bond on two Nyquist planes (a quarter on each of four corners) gets two of
    four copies and the doubled H(K) stops folding onto the primitive one (Pt, exact H(R): 1.6 meV
    with two axes doubled, 4.4 meV with three).  Splitting here, while still Nyquist, is exact.
    """
    sph._require_bonds('double_axis')
    triples = sph.R_int[sph.ridx].astype(np.int64)
    rows, cols, vals, dnm = sph.rows, sph.cols, sph.vals, sph.dnm
    nk = sph.nk_grid[axis]
    # nk = 2: +1 maps outside the window, and the plan still splits the doubled -1 plane
    if split_nyquist and nk >= 4:
        rows, cols, triples, vals, dnm = split_nyquist_axis(
            rows, cols, triples, vals, dnm, nk, axis
        )
    m = triples[:, axis]
    even = (m % 2) == 0
    odd = ~even
    nawf = sph.nawf

    t_diag = triples[even].copy()
    t_diag[:, axis] = m[even] // 2
    t_ur = triples[odd].copy()
    t_ur[:, axis] = (m[odd] + 1) // 2
    t_ll = triples[odd].copy()
    t_ll[:, axis] = (m[odd] - 1) // 2

    lo = -(nk // 2)
    hi = (nk - 1) // 2
    for t in (t_diag, t_ur, t_ll):
        if len(t) and (t[:, axis].min() < lo or t[:, axis].max() > hi):
            raise AssertionError(
                'double_axis: mapped lattice index outside the folded window '
                f'[{lo}, {hi}] of the {nk}-point grid'
            )

    rows = np.concatenate(
        (
            rows[even],
            rows[even] + nawf,
            rows[odd],
            rows[odd] + nawf,
        )
    )
    cols = np.concatenate(
        (
            cols[even],
            cols[even] + nawf,
            cols[odd] + nawf,
            cols[odd],
        )
    )
    new_triples = np.concatenate((t_diag, t_diag, t_ur, t_ll))
    vals = np.concatenate((vals[even], vals[even], vals[odd], vals[odd]))
    shift = sph.alat * sph.a_vectors[axis]
    dnm = np.concatenate((dnm[even], dnm[even], dnm[odd] - shift, dnm[odd] + shift))

    a_vectors = sph.a_vectors.copy()
    a_vectors[axis, :] *= 2.0

    R_uniq, ridx = unique_R(new_triples, sph.nk_grid)
    out = SparseHamiltonian(
        nawf=2 * nawf,
        nspin=sph.nspin,
        alat=sph.alat,
        a_vectors=a_vectors,
        nk_grid=sph.nk_grid,
        R_int=R_uniq.astype(np.int32),
        rows=rows,
        cols=cols,
        ridx=ridx,
        vals=vals,
        dnm=dnm,
        threshold=sph.threshold,
        drop_report=dict(sph.drop_report),
    )
    out._doubled = True
    return out
