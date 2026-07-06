"""Typed sparse containers for the sparse PAOFLOW backend.

These containers make the sparse data contract explicit: instead of overloading
the dense ``data_arrays`` dictionary with ambiguous dense tensors, the sparse
path passes a :class:`SparseHamiltonian` (the thresholded real-space hopping
list) and :class:`SparseEigenpairs` (selected-window eigenpairs) between stages.

Design notes
------------
The Hamiltonian is stored as a *flat* COO-style bond list
``(rows, cols, vals, ridx)`` shared across all real-space vectors ``R`` (with
``ridx`` indexing into ``R``), exactly analogous to
:mod:`PAOFLOW.spectrum.sparse_bands`.  This makes ``H(k)`` assembly a single
vectorised CSR construction rather than a Python loop that sums one sparse
block per ``R`` — the flat form is both faster and never materialises a dense
intermediate.  The Bloch phase convention matches
:func:`PAOFLOW.spectrum.do_bands.band_loop_H`:

.. math::

    H(\\mathbf{k}) = \\sum_{\\mathbf{R}} H(\\mathbf{R})\\,
        e^{2\\pi i\\, \\mathbf{k}\\cdot\\mathbf{R}}

with ``k`` in Cartesian units of :math:`2\\pi/a` and ``R`` the Cartesian
real-space grid produced by :func:`PAOFLOW.utils.get_R_grid_fft.get_R_grid_fft`.
"""

import numpy as np
from scipy.sparse import csr_matrix


class SparseHamiltonian:
    """Thresholded real-space PAO Hamiltonian with matrix-free ``H(k)`` assembly.

    Parameters
    ----------
    nawf : int
        Number of PAO basis functions (matrix dimension).
    nspin : int
        Number of spin channels.
    R : np.ndarray, shape ``(nR, 3)``
        Cartesian real-space grid vectors (units of the lattice constant), as
        returned by :func:`get_R_grid_fft`.
    rows, cols : list of np.ndarray
        Per-spin integer arrays of orbital row/column indices of the retained
        (above-threshold) hopping entries.
    vals : list of np.ndarray
        Per-spin complex arrays of hopping amplitudes ``H(R)[row, col]``.
    ridx : list of np.ndarray
        Per-spin integer arrays indexing into ``R`` for each retained entry.
    alat : float
        Lattice constant (Bohr), used to scale the velocity operator to the
        PAOFLOW convention (see :meth:`build_dHk`).
    threshold : float
        The magnitude below which real-space entries were discarded.

    Notes
    -----
    The container is deliberately backend-agnostic about *how* the bond list was
    produced (dense coarse build + threshold, or a future fully-sparse builder).
    """

    format = 'csr'

    def __init__(self, nawf, nspin, R, rows, cols, vals, ridx, alat, threshold):
        self.nawf = int(nawf)
        self.nspin = int(nspin)
        self.R = np.ascontiguousarray(R, dtype=float)
        self.nR = self.R.shape[0]
        self.rows = [np.ascontiguousarray(r, dtype=np.int32) for r in rows]
        self.cols = [np.ascontiguousarray(c, dtype=np.int32) for c in cols]
        self.vals = [np.ascontiguousarray(v, dtype=complex) for v in vals]
        self.ridx = [np.ascontiguousarray(i, dtype=np.int64) for i in ridx]
        self.alat = float(alat)
        self.threshold = float(threshold)
        # Per-spin, per-entry position-operator values Dnm[row, col, l]; filled
        # lazily by set_position_operator.  Kept on the H-sparsity pattern so it
        # never becomes a dense (nawf, nawf) object in the sparse hot path.
        self._dnm = None

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dense_HRs(cls, HRs, R, alat, threshold):
        """Build from a dense coarse-grid real-space Hamiltonian.

        This is the *bounded input-stage* conversion: the dense ``HRs`` on the
        coarse QE (or doubled) grid is thresholded into the sparse hopping list
        and can then be discarded.  No dense fine-grid tensor is ever formed.

        Parameters
        ----------
        HRs : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``, complex
            Dense real-space Hamiltonian (``ifftn`` of ``Hks``).
        R : np.ndarray, shape ``(nk1*nk2*nk3, 3)``
            Cartesian real-space grid vectors (from :func:`get_R_grid_fft`),
            flattened consistently with ``HRs`` reshaped in C order.
        alat : float
            Lattice constant (Bohr).
        threshold : float
            Entries with ``abs(H) < threshold`` are dropped.

        Returns
        -------
        SparseHamiltonian
        """
        nawf = HRs.shape[0]
        nspin = HRs.shape[-1]
        nk1, nk2, nk3 = HRs.shape[2], HRs.shape[3], HRs.shape[4]
        nR = nk1 * nk2 * nk3
        Hflat = np.reshape(HRs, (nawf, nawf, nR, nspin), order='C')

        rows_s, cols_s, vals_s, ridx_s = [], [], [], []
        for ispin in range(nspin):
            block = Hflat[:, :, :, ispin]  # (nawf, nawf, nR)
            mask = np.abs(block) >= threshold
            ii, jj, rr = np.nonzero(mask)
            rows_s.append(ii.astype(np.int32))
            cols_s.append(jj.astype(np.int32))
            vals_s.append(block[ii, jj, rr].astype(complex))
            ridx_s.append(rr.astype(np.int64))

        return cls(nawf, nspin, R, rows_s, cols_s, vals_s, ridx_s, alat, threshold)

    def set_position_operator(self, Dnm):
        """Attach the tight-binding position operator for velocity assembly.

        The velocity operator carries the intra-cell orbital position
        correction ``i * Dnm[n, m, l] * H(k)[n, m]`` (see
        :func:`PAOFLOW.hamiltonian.do_gradient.do_gradient`).  Because this term
        is only ever multiplied by ``H(k)`` — which is sparse — we store
        ``Dnm`` values *only on the Hamiltonian's sparsity pattern*, avoiding a
        dense ``(nawf, nawf)`` object.

        Parameters
        ----------
        Dnm : np.ndarray, shape ``(nawf, nawf, 3)``
            Orbital position differences ``tau_n - tau_m`` (lattice-constant
            units), as built by the projection stage.  Read only at the sparse
            entries; the full array is not retained.
        """
        self._dnm = []
        for ispin in range(self.nspin):
            r = self.rows[ispin]
            c = self.cols[ispin]
            # (nnz_spin, 3): the position difference at each retained entry.
            self._dnm.append(np.ascontiguousarray(Dnm[r, c, :], dtype=float))

    # ------------------------------------------------------------------
    #  Matrix-free k-space assembly
    # ------------------------------------------------------------------
    def _phase(self, k_cart, ispin):
        """Per-entry Bloch phase ``exp(2πi k·R)`` for the current spin."""
        # k·R for every retained entry via its ridx into the R grid.
        kR = self.R[self.ridx[ispin]] @ np.asarray(k_cart, dtype=float)
        return np.exp(2.0j * np.pi * kR)

    def build_hk(self, k_cart, ispin=0):
        """Assemble the sparse Bloch Hamiltonian ``H(k)`` at one k-point.

        Parameters
        ----------
        k_cart : array_like, shape ``(3,)``
            k-point in Cartesian coordinates (units of :math:`2\\pi/a`).
        ispin : int
            Spin channel.

        Returns
        -------
        scipy.sparse.csr_matrix, shape ``(nawf, nawf)``, complex
        """
        data = self.vals[ispin] * self._phase(k_cart, ispin)
        Hk = csr_matrix(
            (data, (self.rows[ispin], self.cols[ispin])),
            shape=(self.nawf, self.nawf),
        )
        return Hk

    def build_dHk(self, k_cart, ispin=0):
        """Assemble the three Cartesian velocity operators ``dH/dk_l`` at one k.

        Matches the dense convention of
        :func:`PAOFLOW.hamiltonian.do_gradient.do_gradient`:

        .. math::

            \\frac{\\partial H(\\mathbf{k})}{\\partial k_l}
              = i\\, a_{\\rm lat} \\sum_{\\mathbf{R}} R_l\\, H(\\mathbf{R})\\,
                e^{2\\pi i \\mathbf{k}\\cdot\\mathbf{R}}
              + i\\, D^{nm}_l \\odot H(\\mathbf{k})

        Both terms share the Hamiltonian sparsity pattern, so each ``dH/dk_l``
        is assembled directly as a CSR matrix with no dense intermediate.

        Parameters
        ----------
        k_cart : array_like, shape ``(3,)``
            k-point in Cartesian coordinates (units of :math:`2\\pi/a`).
        ispin : int
            Spin channel.

        Returns
        -------
        list of scipy.sparse.csr_matrix
            ``[dH/dk_x, dH/dk_y, dH/dk_z]``, each ``(nawf, nawf)`` complex.
        """
        phase = self._phase(k_cart, ispin)
        hk_data = self.vals[ispin] * phase  # H(k) entries on the pattern
        R_entry = self.R[self.ridx[ispin]]  # (nnz, 3) Cartesian R per entry
        rows, cols = self.rows[ispin], self.cols[ispin]

        dHk = []
        for l in range(3):
            weight = self.alat * R_entry[:, l]
            if self._dnm is not None:
                weight = weight + self._dnm[ispin][:, l]
            data = 1.0j * hk_data * weight
            dHk.append(csr_matrix((data, (rows, cols)), shape=(self.nawf, self.nawf)))
        return dHk

    # ------------------------------------------------------------------
    #  Reporting
    # ------------------------------------------------------------------
    @property
    def nnz(self):
        """Total number of retained nonzero entries across all spins."""
        return int(sum(v.size for v in self.vals))

    def density(self):
        """Fraction of dense ``(nawf, nawf, nR, nspin)`` elements retained."""
        dense = self.nawf * self.nawf * self.nR * self.nspin
        return self.nnz / dense if dense else 0.0


class SparseEigenpairs:
    """Selected-window eigenpairs produced by the sparse eigensolver.

    Only the retained (energy-window) bands are stored, and eigenvectors keep
    just their selected columns, so no dense ``(nkpnts, nawf, nawf, nspin)``
    eigenvector tensor is ever allocated.

    Parameters
    ----------
    E_k : np.ndarray, shape ``(nkpnts, n_sel, nspin)``
        Selected eigenvalues (eV), ascending per k-point.
    v_k : np.ndarray or None, shape ``(nkpnts, nawf, n_sel, nspin)``
        Selected eigenvectors (columns), or ``None`` when only eigenvalues were
        requested (e.g. band-structure paths).
    window : tuple of float or None
        The ``(emin, emax)`` energy window, or ``None`` for a fixed band count.
    n_sel : int
        Number of selected bands per k-point.
    converged : int
        Number of k-points whose solve converged.
    solver : str
        Solver identifier used (for reporting).
    """

    def __init__(self, E_k, v_k, window, n_sel, converged, solver):
        self.E_k = E_k
        self.v_k = v_k
        self.window = window
        self.n_sel = int(n_sel)
        self.converged = int(converged)
        self.solver = solver
