"""Sparse supercell doubling.

Cell doubling is performed on the *bounded coarse* real-space Hamiltonian by
reusing the dense block-doubling kernel
(:func:`PAOFLOW.hamiltonian.do_doubling.doubling_HRs`).  That kernel also
correctly updates every derived quantity the sparse pipeline depends on — the
position operator ``Dnm``, ``tau``, ``a_vectors``, ``omega``, ``nawf``,
``natoms`` — via ``doubling_attr_arry``.  Reimplementing all of that on the
hopping list would duplicate subtle index bookkeeping for no memory win at the
coarse-grid scale, so the responsible first implementation delegates and then
lets :func:`finalize_sparse_hamiltonian` threshold the doubled ``HRs`` into the
sparse container.

The doubled dense coarse ``HRs`` is a transient, size-gated intermediate; it is
freed as soon as it is converted to sparse.
"""

# The doubled dense coarse Hamiltonian must fit under this transient gate.
_DOUBLED_DENSE_GATE_BYTES = 8 * 1024**3  # 8 GB


def sparse_doubling(data_controller, nx, ny, nz):
    """Double the cell ``nx``/``ny``/``nz`` times along each axis.

    Parameters
    ----------
    data_controller : DataController
        Must provide ``data_arrays['HRs']`` (dense coarse Hamiltonian) and the
        associated metadata (``tau``, ``a_vectors``, ``Dnm``, ...).
    nx, ny, nz : int
        Number of doublings along each lattice direction.

    Returns
    -------
    None
        Updates ``HRs`` and all derived metadata in place (delegated to the
        dense doubling kernel).  Any previously finalized ``sparse_H`` is
        invalidated so it will be rebuilt from the doubled ``HRs``.

    """
    from ..hamiltonian.do_doubling import doubling_HRs

    arry, attr = data_controller.data_dicts()

    if 'HRs' not in arry:
        raise KeyError(
            'sparse_doubling: doubling must run before the sparse Hamiltonian '
            'is finalized (needs dense coarse HRs). Call pao_hamiltonian() '
            'then doubling_Hamiltonian() before pao_eigh().'
        )

    attr['nx'], attr['ny'], attr['nz'] = nx, ny, nz
    doubling_HRs(data_controller)

    # A doubled Hamiltonian supersedes any earlier sparse finalization.
    if 'sparse_H' in arry:
        arry['sparse_H'] = None
