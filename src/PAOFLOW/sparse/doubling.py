"""Cell doubling as pure index arithmetic on the bond list.

Replicates ``hamiltonian.do_doubling.doubling_HRs`` exactly, without ever
materializing the dense ``(2 nawf, 2 nawf, nk1, nk2, nk3, nspin)`` array.
The dense kernel indexes its ``cell_index`` map by the *negated* folded
lattice coordinate (``ix = -round(Rx*nk1)``, "the minus sign is due to
the Fourier transformation"), so in the true folded coordinates produced
by :func:`~PAOFLOW.sparse.hamiltonian.folded_R_triples` it builds

.. math::

    H^{2\\times}(M) = \\begin{pmatrix}
        H(2M) & H(2M-1) \\\\ H(2M+1) & H(2M)
    \\end{pmatrix}

and the inverse map per bond with folded axis index ``m`` is:

- ``m`` even: contributes the (0,0) and (1,1) blocks at ``M = m/2``
  (``dnm`` kept — same replica);
- ``m`` odd: contributes the (0,1) block at ``M = (m+1)/2`` and the (1,0)
  block at ``M = (m-1)/2`` (``dnm`` shifted by ``-/+ alat * a_axis``, the
  offset between the two replicas).

The cross-replica ``dnm`` is where this kernel deliberately departs from
the dense ``doubling_attr_arry``, which doubles ``Dnm`` block-diagonally
and so leaves those pairs at zero; see :func:`double_axis`.

Each doubling therefore exactly doubles ``nnz``: memory grows linearly in
the number of orbitals, not quadratically as in the dense kernel.  The
truncation ``eig_bound`` carries over unchanged (doubling only rearranges
and duplicates row content).  Thresholding commutes with doubling.

The driver remains responsible for the accompanying metadata updates
(``tau``/``a_vectors`` in the DataController plus the dense
``doubling_attr_arry`` for scalar attributes); ``double_axis`` updates the
container's own ``a_vectors`` copy.
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import SparseHamiltonian, unique_R


def double_axis(sph: SparseHamiltonian, axis: int) -> SparseHamiltonian:
    """Return a new bond list for the cell doubled along one axis.

    Parameters
    ----------
    sph : SparseHamiltonian
        Bond list of the current cell.  Left unmodified; its raw bond
        arrays must still be present, i.e. it must not have been
        compacted.
    axis : int
        Lattice axis to double, 0, 1 or 2.

    Returns
    -------
    SparseHamiltonian
        Bond list of the doubled cell: ``2 * nawf`` orbitals, ``2 * nnz``
        bonds, the same R grid, and ``a_vectors`` with the doubled axis
        scaled by two.

    Raises
    ------
    AssertionError
        If a mapped lattice index leaves the folded window of the R grid,
        which would mean the halved index no longer names a vector of the
        supercell's own lattice.
    RuntimeError
        If ``sph`` has been compacted and no longer holds the bond arrays.

    Notes
    -----
    Doubling a cell along an axis means describing the same crystal with a
    unit cell twice as long, holding two copies of the original basis.  The
    orbital index therefore doubles: orbital ``i`` of the first replica
    keeps index ``i``, orbital ``i`` of the second gets ``i + nawf``.  A
    hopping that used to reach from cell ``m`` to cell ``0`` now either
    stays inside one supercell (connecting the two replicas) or crosses to
    a neighbouring supercell, depending on whether ``m`` is even or odd:

    - even ``m`` connects like-numbered replicas and lands on supercell
      ``m/2``, contributing to both diagonal blocks;
    - odd ``m`` connects the two different replicas and lands on
      supercells ``(m ± 1) / 2``, contributing to the two off-diagonal
      blocks.

    Python's floor division implements the even case exactly, negative
    indices included, and the ``±1`` offsets of the odd case pick out the
    two supercells a cross-replica hopping reaches.  So the whole operation
    is integer arithmetic on the axis component of each bond plus an offset
    on its row or column index — no matrix is ever built, and the bond
    count merely doubles.  The dense kernel, by
    contrast, allocates the full ``(2 nawf, 2 nawf, ...)`` array and pays
    four times the memory per doubling.

    The intra-cell orbital position differences ``dnm``, which enter the
    velocity operator, stay those of the doubled cell: ``x_i - x_j`` for the
    orbital positions ``x`` of the supercell, whose second replica sits at
    ``x + alat * a_axis``.  Like-numbered replicas keep the base value, and
    a bond from replica 0 to replica 1 (the (0,1) block) gets
    ``dnm - alat * a_axis``, its mirror ``dnm + alat * a_axis``.

    This matters for band velocities.  The ``dnm`` term adds
    ``i [x, H(k)]`` to ``dH/dk``, whose band-diagonal elements vanish, so it
    is harmless only while ``dnm`` keeps that commutator form.  The dense
    ``doubling_attr_arry`` doubles ``Dnm`` block-diagonally and leaves the
    cross-replica pairs at zero, which breaks the form: on Si (example01)
    the velocities are then off by ~35% and sigma by ~50% on every doubled
    cell.  So this kernel differs from the dense doubling in the
    cross-replica ``dnm`` and in nothing else; the bond values are identical.
    """
    sph._require_bonds('double_axis')
    triples = sph.R_int[sph.ridx].astype(np.int64)
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

    nk = sph.nk_grid[axis]
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
            sph.rows[even],
            sph.rows[even] + nawf,
            sph.rows[odd],
            sph.rows[odd] + nawf,
        )
    )
    cols = np.concatenate(
        (
            sph.cols[even],
            sph.cols[even] + nawf,
            sph.cols[odd] + nawf,
            sph.cols[odd],
        )
    )
    new_triples = np.concatenate((t_diag, t_diag, t_ur, t_ll))
    vals = np.concatenate((sph.vals[even], sph.vals[even], sph.vals[odd], sph.vals[odd]))
    shift = sph.alat * sph.a_vectors[axis]
    dnm = np.concatenate((sph.dnm[even], sph.dnm[even], sph.dnm[odd] - shift, sph.dnm[odd] + shift))

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
