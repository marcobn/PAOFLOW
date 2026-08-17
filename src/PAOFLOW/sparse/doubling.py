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
  block at ``M = (m-1)/2`` (``dnm = 0`` — the dense ``doubling_attr_arry``
  doubles ``Dnm`` block-diagonally, zeroing cross-replica pairs).

Each doubling therefore exactly doubles ``nnz``: memory grows linearly in
the number of orbitals, not quadratically as in the dense kernel.  The
truncation ``eig_bound`` carries over unchanged (doubling only rearranges
and duplicates row content).  Thresholding commutes with doubling.

The driver remains responsible for the accompanying metadata updates
(``tau``/``a_vectors`` in the DataController plus the dense
``doubling_attr_arry`` for scalar attributes); ``double_axis`` updates the
container's own ``a_vectors`` copy.
"""

import numpy as np

from .hamiltonian import SparseHamiltonian, unique_R


def double_axis(sph, axis):
    """Return a new :class:`SparseHamiltonian` doubled along ``axis`` (0..2)."""
    sph._require_bonds('double_axis')
    triples = sph.R_int[sph.ridx].astype(np.int64)
    m = triples[:, axis]
    even = (m % 2) == 0
    odd = ~even
    nawf = sph.nawf

    t_diag = triples[even].copy()
    t_diag[:, axis] = m[even] // 2  # exact for negatives (even m)
    t_ur = triples[odd].copy()
    t_ur[:, axis] = (m[odd] + 1) // 2  # (0,1) block: m = 2M - 1
    t_ll = triples[odd].copy()
    t_ll[:, axis] = (m[odd] - 1) // 2  # (1,0) block: m = 2M + 1

    nk = sph.nk_grid[axis]
    lo = -(nk // 2)
    hi = (nk - 1) // 2
    for t in (t_diag, t_ur, t_ll):
        if len(t) and (t[:, axis].min() < lo or t[:, axis].max() > hi):
            raise AssertionError(
                'double_axis: mapped lattice index outside the folded window '
                '[%d, %d] of the %d-point grid' % (lo, hi, nk)
            )

    rows = np.concatenate(
        (
            sph.rows[even],
            sph.rows[even] + nawf,  # (0,0), (1,1)
            sph.rows[odd],
            sph.rows[odd] + nawf,  # (0,1), (1,0)
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
    zeros = np.zeros((int(odd.sum()), 3))
    dnm = np.concatenate((sph.dnm[even], sph.dnm[even], zeros, zeros))

    a_vectors = sph.a_vectors.copy()
    a_vectors[axis, :] *= 2.0

    # R triples take at most nk1*nk2*nk3 distinct values, so the unique is a
    # presence mask plus a lookup table -- O(nnz), no sort.
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
