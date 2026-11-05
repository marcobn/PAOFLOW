"""The only converter between the dense ``HRs`` and the base-cell bond list.

Both describe the same truncated model.  The bond-list archive (:mod:`PAOFLOW.sparse.io`, which
owns the file format) is the interchange: either pipeline writes and reads it, and the reading
``PAOFLOW``'s ``sparse=`` flag picks the engine; a live sparse run switches with
``PAOFLOW.to_dense()``.  :func:`sparsify` delegates to
:meth:`SparseHamiltonian.from_data_controller`, the single truncation implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hamiltonian import SparseHamiltonian

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController


def sparsify(
    data_controller: DataController,
    threshold: float | None = None,
    rcut: float | None = None,
    bond_order: int | None = None,
) -> SparseHamiltonian:
    """Truncate the dense ``HRs`` of a controller into a base-cell bond list.

    Parameters
    ----------
    data_controller : DataController
        Run state after ``pao_hamiltonian``; ``HRs`` is read, not modified.
    threshold, rcut, bond_order
        See :meth:`SparseHamiltonian.from_data_controller`, which this delegates to unchanged.

    Returns
    -------
    SparseHamiltonian
    """
    return SparseHamiltonian.from_data_controller(
        data_controller, threshold, rcut=rcut, bond_order=bond_order
    )


def densify(
    data_controller: DataController, H: SparseHamiltonian, Dnm: np.ndarray | None = None
) -> None:
    """Hand a base-cell bond list to the dense pipeline.

    Parameters
    ----------
    data_controller : DataController
        Already holds geometry and run attributes (from
        :func:`~PAOFLOW.sparse.io.restore_data_controller` or a live sparse run).  Receives
        ``HRs``, ``Hks`` (rank 0) and ``Dnm``.
    H : SparseHamiltonian
        A base-cell (never doubled, not compacted) bond list.
    Dnm : np.ndarray or None, optional
        Orbital-centre offsets ``(nawf, nawf, 3)``, stored verbatim (unit depends on the source);
        needed by dense ``gradient_and_momenta``.  ``None`` leaves any existing value.

    Raises
    ------
    RuntimeError
        If ``H`` has been doubled, or (from ``to_dense_HRs``) compacted.

    Notes
    -----
    The controller then matches a dense run right after ``pao_hamiltonian()`` with the truncated
    model: ``HRs`` on every rank, ``Hks = fftn(HRs)`` on rank 0 for ``pao_eigh`` when no
    ``interpolated_hamiltonian`` runs first.

    The R grid (``R``, ``Rfft``, ``idx``, ``R_wght``) is deliberately not built, as in dense:
    ``do_berry_curvature`` reuses an existing ``Rfft`` without checking its size, so a base-grid
    copy would be picked up after interpolation to a finer mesh.

    The archive lacks ``basis``, ``kpnts``, ``Sks``, ``my_eigsmat``, so the projection,
    ``write_PAO_bin``, ``density`` and the DFT-level dielectric tensor are unavailable after a
    bond-list restart.
    """
    if H._doubled:
        raise RuntimeError(
            'densify: the bond list has been doubled. Densify the base cell and use the '
            'dense doubling_Hamiltonian() instead; a doubled dense HRs is what the sparse '
            'backend exists to avoid.'
        )
    HRs = H.to_dense_HRs()

    arrays, _ = data_controller.data_dicts()
    arrays['HRs'] = HRs
    if Dnm is not None:
        arrays['Dnm'] = np.array(Dnm, dtype=float)
    if getattr(data_controller, 'rank', 0) == 0:
        arrays['Hks'] = np.fft.fftn(HRs, axes=(2, 3, 4))
    else:
        arrays.pop('Hks', None)


def archive_Dnm(bundle: dict) -> np.ndarray:
    """``Dnm`` of an archive, rebuilt from its orbital map when not stored.

    Parameters
    ----------
    bundle : dict
        Second return value of :func:`~PAOFLOW.sparse.io.read_sparse_hamiltonian`.

    Returns
    -------
    np.ndarray, shape (nawf, nawf, 3)
        The stored ``Dnm``, else ``tau_i - tau_j`` of the atoms carrying each orbital (Bohr).

    Notes
    -----
    Older sparse archives lack ``Dnm``.  For a QE projection, the only source an archive accepts,
    ``Dnm`` is exactly this geometric offset (checked to 0.0 on example01): the rebuild is lossless.
    """
    if 'Dnm' in bundle:
        return np.asarray(bundle['Dnm'], dtype=float)
    centres = np.asarray(bundle['tau'], dtype=float)[np.asarray(bundle['orbital_atom'])]
    return centres[:, None, :] - centres[None, :, :]


def init_restart_session(data_controller, comm, workpath, outputdir, npool, smearing, verbose):
    """Give a ``restart=True`` controller the session state a fresh run has.

    On restart the dense ``DataController`` leaves both data dictionaries ``None`` for a JSON dump
    to fill; a bond-list restart fills them from an archive instead, but the log and the output
    directory are needed before that.
    """
    from os import makedirs
    from os.path import join

    dc = data_controller
    if dc.data_attributes is None:
        dc.data_arrays = {}
        dc.data_attributes = {}
        dc.add_default_arrays()
    attr = dc.data_attributes
    attr.update(
        workpath=workpath,
        outputdir=outputdir,
        opath=join(workpath, outputdir),
        npool=npool,
        smearing=smearing,
        verbose=verbose,
        mpisize=comm.Get_size(),
    )
    attr.setdefault('abort_on_exception', True)
    attr.setdefault('scipyfft', True)
    if comm.Get_rank() == 0:
        makedirs(attr['opath'], exist_ok=True)
    comm.Barrier()
