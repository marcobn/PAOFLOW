"""The one boundary between the dense and the sparse representation of H(R).

A base-cell :class:`~PAOFLOW.sparse.hamiltonian.SparseHamiltonian` and the
dense ``HRs`` of :class:`PAOFLOW.PAOFLOW` describe the same truncated
model.  The bond-list archive (:mod:`PAOFLOW.sparse.io`) is the
interchange format between the two pipelines: either can write it, either
can read it, and the engine that runs afterwards is chosen by the
``sparse=`` flag of the ``PAOFLOW`` that reads it.  A live sparse run
switches to the dense pipeline with ``PAOFLOW.to_dense()``.

This module is the only place that converts between the two forms:

- :func:`sparsify` (dense -> bond list) delegates to
  :meth:`SparseHamiltonian.from_data_controller`, the single truncation
  implementation;
- :func:`densify` (bond list -> dense) puts a ``DataController`` in the
  state dense ``pao_hamiltonian()`` leaves it in, so the dense FFT +
  LAPACK pipeline continues with every dense property available.

The file format stays in :mod:`PAOFLOW.sparse.io`.
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
    threshold : float or None, optional
        Magnitude in eV below which a hopping is dropped; ``None`` picks the
        default for the truncation mode.  Exclusive with a real-space cutoff.
    rcut : float or None, optional
        Bond-length cutoff in Bohr.
    bond_order : int or None, optional
        Neighbour-shell form of the same cutoff.  Mutually exclusive with
        ``rcut``.

    Returns
    -------
    SparseHamiltonian
        See :meth:`SparseHamiltonian.from_data_controller`, which this
        delegates to unchanged.
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
        Target container, already holding the geometry and run attributes
        (from :func:`~PAOFLOW.sparse.io.restore_data_controller` or a live
        sparse run).  Receives ``HRs``, ``Hks`` (rank 0) and ``Dnm``.
    H : SparseHamiltonian
        A base-cell (never doubled, not compacted) bond list.
    Dnm : np.ndarray or None, optional
        Orbital-centre offsets, shape ``(nawf, nawf, 3)``, stored verbatim
        (its length unit depends on the source).  Required by the dense
        ``gradient_and_momenta``; ``None`` leaves any existing value alone.

    Raises
    ------
    RuntimeError
        If ``H`` has been doubled, or (from ``to_dense_HRs``) compacted.

    Notes
    -----
    Afterwards the controller matches a dense run right after
    ``pao_hamiltonian()``, with the truncated model in place of the full
    one: ``HRs`` replicated on every rank and ``Hks = fftn(HRs)`` on rank
    0, which ``pao_eigh`` consumes when no ``interpolated_hamiltonian``
    runs first.

    The real-space grid (``R``, ``Rfft``, ``idx``, ``R_wght``) is
    deliberately **not** built.  The dense ``pao_hamiltonian`` does not
    build it either: ``do_gradient`` builds its own at the current mesh,
    and ``do_berry_curvature`` reuses an existing ``Rfft`` without
    checking its size, so a base-grid copy left here would be picked up
    after interpolation to a finer mesh.

    Arrays the archive does not carry (``basis``, ``kpnts``, ``Sks``,
    ``my_eigsmat``) are only read by stages that need the DFT input
    itself (the projection, ``write_PAO_bin``, ``density``, the DFT-level
    dielectric tensor); those are unavailable after any bond-list restart.
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
        The stored ``Dnm`` if present, else ``tau_i - tau_j`` of the atoms
        carrying each orbital pair (Bohr).

    Notes
    -----
    Archives written by a sparse run before ``Dnm`` was passed to the
    writer explicitly lack it.  For a QE projection, the only source an
    archive accepts, the projection's ``Dnm`` is exactly this geometric
    offset (checked to 0.0 on example01), so the rebuild is lossless.
    """
    if 'Dnm' in bundle:
        return np.asarray(bundle['Dnm'], dtype=float)
    centres = np.asarray(bundle['tau'], dtype=float)[np.asarray(bundle['orbital_atom'])]
    return centres[:, None, :] - centres[None, :, :]


def init_restart_session(data_controller, comm, workpath, outputdir, npool, smearing, verbose):
    """Give a ``restart=True`` controller the session state a fresh run has.

    The dense ``DataController`` leaves both data dictionaries ``None``
    on restart, expecting a JSON dump to fill them.  A bond-list restart
    fills them from an archive instead (``load_sparse_hamiltonian`` of
    either pipeline), but the log and the output directory are needed
    before that, so the session part is set here.
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
