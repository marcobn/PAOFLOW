"""Sparse Hamiltonian interpolation (matrix-free).

Dense PAOFLOW interpolation zero-pads ``H(R)`` from the ``(nk1, nk2, nk3)`` grid
to a finer ``(nfft1, nfft2, nfft3)`` grid in real space and FFTs back to a dense
``H(k)`` on the fine mesh — the single largest allocation in the dense pipeline
(:func:`PAOFLOW.hamiltonian.do_double_grid.do_double_grid`).

Zero-padding in real space adds only zero hoppings: it does not change the
retained hopping list at all.  It is *mathematically identical* to keeping the
same sparse ``H(R)`` and evaluating ``H(k)`` on the finer k-mesh.  So sparse
"interpolation" is a metadata operation — it records the finer target grid and
finalizes the sparse Hamiltonian — and the finer ``H(k)`` is assembled
matrix-free at diagonalisation time.  Sparsity is exactly preserved.
"""

from .hamiltonian_builder import finalize_sparse_hamiltonian


def set_interpolation_grid(data_controller, nfft1=0, nfft2=0, nfft3=0, threshold=1.0e-6):
    """Record the finer interpolation k-grid and finalize the sparse Hamiltonian.

    Parameters
    ----------
    data_controller : DataController
        Must provide the coarse-grid attributes (``nk1``, ``nk2``, ``nk3``) and
        either a dense ``HRs`` (to finalize) or an already-built ``sparse_H``.
    nfft1, nfft2, nfft3 : int
        Target grid dimensions.  A value of ``0`` defaults to twice the
        corresponding coarse dimension (matching the dense default).
    threshold : float
        Sparsification threshold used when finalizing from dense ``HRs``.

    Returns
    -------
    SparseHamiltonian
        The finalized sparse Hamiltonian (also stored under
        ``data_arrays['sparse_H']``).  Its real-space grid is unchanged; only
        the evaluation k-mesh is refined.
    """
    arry, attr = data_controller.data_dicts()

    nfft1 = 2 * attr['nk1'] if nfft1 == 0 else nfft1
    nfft2 = 2 * attr['nk2'] if nfft2 == 0 else nfft2
    nfft3 = 2 * attr['nk3'] if nfft3 == 0 else nfft3

    attr['nfft1'], attr['nfft2'], attr['nfft3'] = nfft1, nfft2, nfft3
    # The eigensolver reads this flag to build H(k) on the finer mesh.
    attr['sparse_interpolated'] = True

    return finalize_sparse_hamiltonian(data_controller, threshold)
