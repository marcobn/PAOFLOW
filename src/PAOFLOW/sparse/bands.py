"""Band structure along a high-symmetry path, sparse iterative version.

Mirrors ``spectrum.do_bands`` scaffolding exactly — same k-path generation
(``kpnts_interpolation_mesh``, including the Angstrom/Bohr ``alat`` dance
and the in-place rotation of ``kq`` to Cartesian by ``b_vectors``), same
MPI k-scatter — but replaces the dense Fourier sum + ``eigh`` with the
fixed-pattern sparse assembly (``sign=+1``, the band-path phase
convention) and the iterative ``solve_lowest``, computing only the lowest
``nsel`` bands.  Eigenvectors are used solely to warm-start the next
k-point and are never stored.

Convention note: the dense path diagonalizes the upper triangle of a
slightly non-Hermitian interpolated H(k) (``eigh(lower=False)``); the
sparse path solves the exactly Hermitian interpolant (Nyquist-split, the
``zero_pad`` convention).  The two differ by O(|H(R)| at the Nyquist
shell), far below plotting resolution for converged R grids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from .hamiltonian import SparseHamiltonian

comm = MPI.COMM_WORLD


def do_bands_sparse(
    data_controller: DataController,
    sparse_h: SparseHamiltonian,
    nsel: int,
    verbose: bool = False,
    hk_solver: str = 'auto',
    ehi: float | None = None,
    interior: tuple[float, float] | None = None,
) -> None:
    """Compute the band structure along the interpolation path.

    Parameters
    ----------
    data_controller : DataController
        Run state.  Supplies the k-path (or the parameters to build one)
        and receives ``arrays['E_k']``, this rank's slice of the bands,
        shaped ``(nkpi_local, nsel, nspin)`` as in the dense pipeline.
    sparse_h : SparseHamiltonian
        Bond list the Bloch Hamiltonian is assembled from at each k-point.
    nsel : int
        Number of lowest bands to compute.  Ignored when ``interior`` is
        given.
    verbose : bool, optional
        Log progress every few percent of this rank's k-points.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        Per-k-point kernel, chosen once for the whole path (see
        :func:`~PAOFLOW.sparse.solver.select_hk_solver`).
    ehi : float or None, optional
        Top of the energy window ``nsel`` was sized for (eV).  Enables the
        same window-coverage guard the mesh pass uses: a path k-point whose
        highest computed band falls below ``ehi`` is counted, and a non-zero
        global count raises after the loop.
    interior : (float, float) or None, optional
        Energy window ``(elo, ehi)`` in eV.  Solves *inside* the window
        instead of from the bottom of the spectrum.

    Returns
    -------
    None
        Results are written into ``data_controller`` arrays, matching the
        dense layout.

    Notes
    -----
    A band structure is the eigenvalue spectrum of :math:`H(\\mathbf{k})`
    sampled along a path through the Brillouin zone.  Each k-point is an
    independent Hermitian eigenproblem, so the path is scattered across MPI
    ranks and solved point by point; nothing couples neighbouring k-points
    except the warm start.

    Two pieces of unit and frame bookkeeping are inherited verbatim from
    the dense ``do_bands``, because the path generator and the assembler
    expect them.  ``alat`` is converted from Bohr to Angstrom around the
    call that builds the path and converted back afterwards, and the
    resulting fractional k-points are rotated into Cartesian coordinates in
    place with the reciprocal lattice vectors — the sparse assembler
    consumes Cartesian k on the band path (``cart=True``), in units of
    :math:`2\\pi/a`.

    Neighbouring points on a path are close in k, so their lowest
    eigenvectors are nearly the same vector.  The previous point's ground
    state is therefore handed to the iterative solver as its starting
    vector, which shortens the Krylov iteration measurably; it is a pure
    convergence aid and never affects the converged result.  Apart from
    that, the eigenvectors are discarded — a band plot needs only energies.

    Under ``interior`` the number of states inside the window is a property
    of the band structure at that k-point, so it varies along the path (a
    band crossing an edge is in at one point and out at the next).  The
    output is padded to the global maximum with **NaN**, unlike the mesh
    pass, which pads with a far-away energy because its consumers evaluate
    smearing kernels there.  Here the numbers are written straight to
    ``bands_*.dat`` and plotted, so a missing state must read as missing
    rather than as a band at some plausible-looking energy.  The columns
    are then no longer band indices.
    """
    from .log import get_sparse_log
    from .mesh import check_window_coverage
    from .solver import describe_hk_solver, solve_interior, solve_lowest

    arrays, attr = data_controller.data_dicts()

    kq_aux, nkpi = prepare_path(data_controller)
    nk_local = kq_aux.shape[0]
    nspin = sparse_h.nspin

    log = get_sparse_log(data_controller)
    log.section('Bands (path solve)')
    log.field('k-points on path', nkpi)
    if interior is None:
        log.field('bands requested', nsel)
    else:
        log.field('interior window (eV)', f'[{interior[0]:.3f}, {interior[1]:.3f}]')
        log.field('bands requested', 'k-dependent (interior solve); NaN-padded')
    log.field('window top ehi (eV)', 'none' if ehi is None else f'{ehi:.3f}')
    log.write(describe_hk_solver(sparse_h.nawf, nsel, hk_solver=hk_solver))

    if interior is None:
        E_k = np.zeros((nk_local, nsel, nspin), dtype=float)
    else:
        acc = {}
    deficit = 0
    step = max(1, min(100, nk_local // 10))
    for ispin in range(nspin):
        v0 = None
        k0 = None
        for ik in range(nk_local):
            hk = sparse_h.assemble_hk(kq_aux[ik], ispin=ispin, sign=+1, cart=True)
            if interior is None:
                E, V = solve_lowest(hk, nsel, v0=v0, hk_solver=hk_solver)
                E_k[ik, :, ispin] = E
                v0 = np.ascontiguousarray(V[:, 0])
                if ehi is not None and E[-1] < ehi:
                    deficit += 1
            else:
                E, _ = solve_interior(hk, interior[0], interior[1], k0=k0, hk_solver=hk_solver)
                acc[(ispin, ik)] = E
                k0 = max(8, 2 * len(E))
            if verbose and (ik + 1) % step == 0:
                log.write(f'  progress: {ik + 1}/{nk_local} local k-points')

    if interior is not None:
        local_max = max((len(E) for E in acc.values()), default=0)
        m = int(comm.allreduce(int(local_max), op=MPI.MAX))
        log.write(
            f'  interior window: up to {local_max} states per k locally (padded to {m} with NaN)'
        )
        E_k = np.full((nk_local, max(m, 1), nspin), np.nan, dtype=float)
        for (ispin, ik), E in acc.items():
            E_k[ik, : len(E), ispin] = E

    arrays['E_k'] = E_k
    if interior is None:
        check_window_coverage(deficit, nsel, ehi, 'bands')


def prepare_path(data_controller: DataController) -> tuple[np.ndarray, int]:
    """This rank's share of the band path, Cartesian in units of ``2 pi / alat``.

    Parameters
    ----------
    data_controller : DataController
        Supplies ``ibrav`` (the path is rebuilt from it) or an external
        ``kq`` (crystal coordinates, as given), and ``b_vectors``.

    Returns
    -------
    (kq_aux, nkpi) : tuple
        Local path points ``(nkpi_local, 3)`` and the total path length.

    Notes
    -----
    Inherited from the dense ``do_bands``: ``alat`` is converted to
    Angstrom around the path generator, and the points are rotated into
    Cartesian coordinates in place in ``arrays['kq']``.  The rotation is
    recorded (``attr['sparse_kq_cartesian']``), so an external path is
    rotated once however many path passes follow; a path built from
    ``ibrav`` is rebuilt and rotated afresh each time, as in the dense code.
    """
    from ..spectrum.kpnts_interpolation_mesh import kpnts_interpolation_mesh
    from ..utils.communication import scatter_full
    from ..utils.constants import ANGSTROM_AU

    arrays, attr = data_controller.data_dicts()

    attr['alat'] /= ANGSTROM_AU
    if 'ibrav' in attr:
        kpnts_interpolation_mesh(data_controller)
        attr['sparse_kq_cartesian'] = False
    attr['alat'] *= ANGSTROM_AU
    if 'kq' not in arrays:
        raise RuntimeError('sparse bands: need external kq for bands')

    nkpi = arrays['kq'].shape[1]
    if not attr.get('sparse_kq_cartesian', False):
        for n in range(nkpi):
            arrays['kq'][:, n] = np.dot(arrays['kq'][:, n], arrays['b_vectors'])
        attr['sparse_kq_cartesian'] = True

    return scatter_full(arrays['kq'].T.copy(), attr['npool']), nkpi


def run_path(
    data_controller: DataController,
    sparse_h: SparseHamiltonian,
    consumers: list,
    nsolve: int,
    with_dnm: bool = True,
    hk_solver: str = 'auto',
    verbose: bool = False,
) -> None:
    """Stream the band path to path properties, one :class:`KPoint` at a time.

    Parameters
    ----------
    data_controller : DataController
        Run state; supplies the path (see :func:`prepare_path`).
    sparse_h : SparseHamiltonian
    consumers : list of MeshProperty
        Path properties; each receives every local path point in ``on_k``.
        They are finalized by the engine.
    nsolve : int
        States solved per point: ``attr['bnd']``, or ``nawf`` when a
        consumer needs the full spectrum.
    with_dnm : bool, optional
        Whether ``dH/dk`` carries the intra-cell ``Dnm`` terms; the dense
        path kernels differ in this (see
        ``SparseHamiltonian.assemble_derivatives``).
    hk_solver : {'auto', 'sparse', 'dense'}, optional
    verbose : bool, optional

    Notes
    -----
    The path convention of :func:`do_bands_sparse`: ``sign=+1`` and
    Cartesian k (``cart=True``).  There is no adaptive smearing on a path,
    so ``KPoint.vel`` and ``KPoint.delta`` are ``None``; ``KPoint.kglobal``
    is the index along the path.
    """
    from ..utils.communication import scatter_full
    from .kpoint import KPoint
    from .log import get_sparse_log
    from .solver import describe_hk_solver, solve_lowest

    attr = data_controller.data_attributes
    kq_aux, nkpi = prepare_path(data_controller)
    kglobal = scatter_full(np.arange(nkpi), attr['npool'])
    nawf = sparse_h.nawf
    solver = 'dense' if nsolve >= nawf else hk_solver

    log = get_sparse_log(data_controller)
    log.section('Path pass (%s)' % ', '.join(sorted({c.method for c in consumers})))
    log.field('k-points on path', nkpi)
    log.field('states per k', '%d of nawf = %d' % (nsolve, nawf))
    log.write(describe_hk_solver(nawf, nsolve, hk_solver=solver))

    bnd = min(int(attr['bnd']), nsolve)
    step = max(1, min(100, kq_aux.shape[0] // 10))
    for ispin in range(sparse_h.nspin):
        v0 = None
        for ik in range(kq_aux.shape[0]):
            hk, dhk = sparse_h.assemble_derivatives(
                kq_aux[ik], ispin=ispin, sign=+1, cart=True, order=1, with_dnm=with_dnm
            )
            E, V = solve_lowest(hk, nsolve, v0=v0, hk_solver=solver)
            v0 = np.ascontiguousarray(V[:, 0])
            kp = KPoint(
                sparse_h,
                ik,
                ispin,
                kq_aux[ik],
                E,
                V,
                hk,
                dhk,
                None,
                None,
                bnd,
                None,
                None,
                sign=+1,
                cart=True,
                kglobal=int(kglobal[ik]),
                with_dnm=with_dnm,
            )
            for c in consumers:
                c.activate()
                c.on_k(kp)
            kp = None
            if verbose and (ik + 1) % step == 0:
                log.write(f'  progress: {ik + 1}/{kq_aux.shape[0]} local path points')
