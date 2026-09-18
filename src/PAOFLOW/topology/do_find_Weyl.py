# import matplotlib.pyplot as plt
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as OP
from mpi4py import MPI
from numpy.typing import NDArray

from ..utils.communication import gather_full, scatter_full
from ..utils.constants import BOHR_RADIUS_ANGS

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

# initialize parallel execution
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# np.set_printoptions(precision=8, threshold=100, edgeitems=50, linewidth=350, suppress=True)


def pack_HR(HRaux: NDArray[np.complexfloating], ispin: int = 0) -> NDArray[np.complexfloating]:
    """Pack one spin channel of ``H(R)`` into a contiguous matrix for BLAS.

    Parameters
    ----------
    HRaux : ndarray, shape ``(nawf, nawf, nR, nspin)``, complex
        Real-space Hamiltonian on the R-grid in crystal coordinates.
    ispin : int, optional
        Spin channel to extract (default ``0``).

    Returns
    -------
    ndarray, shape ``(nawf*nawf, nR)``, complex
        C-contiguous packing of the requested spin channel.  For
        ``nspin == 1`` this is a view and no data is copied.

    Notes
    -----
    In this layout the Fourier sum over ``R`` is a single matrix-vector
    (or matrix-matrix) product.  Slicing the spin axis of the original
    ``(nawf, nawf, nR, nspin)`` array yields a strided view, which forces
    NumPy to copy the whole Hamiltonian on *every* contraction; packing
    once here removes that copy from the optimisation loop.
    """
    nawf = HRaux.shape[0]
    return np.ascontiguousarray(HRaux[:, :, :, ispin]).reshape(nawf * nawf, -1)


def build_Hk(
    HRpack: NDArray[np.complexfloating],
    nawf: int,
    kq: NDArray[np.floating],
    R: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Fourier-transform the packed ``H(R)`` to ``H(k)`` at a single k-point.

    Parameters
    ----------
    HRpack : ndarray, shape ``(nawf*nawf, nR)``, complex
        Packed Hamiltonian from :func:`pack_HR`.
    nawf : int
        Dimension of the Hamiltonian matrix.
    kq : ndarray, shape ``(3,)``, float
        k-point in crystal (fractional) coordinates.
    R : ndarray, shape ``(nR, 3)``, float
        Real-space lattice vectors produced by :func:`get_R_grid_fft`.

    Returns
    -------
    ndarray, shape ``(nawf, nawf)``, complex
        The Hamiltonian at ``kq``.

    Notes
    -----
    .. math::

        H(\\mathbf{k}) = \\sum_{\\mathbf{R}}
                         H(\\mathbf{R})\\,e^{2\\pi i\\mathbf{k}\\cdot\\mathbf{R}}

    The dot product :math:`\\mathbf{k}\\cdot\\mathbf{R}` is accumulated in
    real arithmetic before exponentiation, and the R-sum is a single
    ``ZGEMV`` call with no temporaries.
    """
    phase = np.exp((2.0j * np.pi) * (R @ kq))
    return HRpack.dot(phase).reshape(nawf, nawf)


def build_Hk_batch(
    HRpack: NDArray[np.complexfloating],
    nawf: int,
    kq: NDArray[np.floating],
    R: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Fourier-transform the packed ``H(R)`` to ``H(k)`` for a block of k-points.

    Parameters
    ----------
    HRpack : ndarray, shape ``(nawf*nawf, nR)``, complex
        Packed Hamiltonian from :func:`pack_HR`.
    nawf : int
        Dimension of the Hamiltonian matrix.
    kq : ndarray, shape ``(nk, 3)``, float
        k-points in crystal (fractional) coordinates.
    R : ndarray, shape ``(nR, 3)``, float
        Real-space lattice vectors.

    Returns
    -------
    ndarray, shape ``(nk, nawf, nawf)``, complex
        The Hamiltonian at each k-point.

    Notes
    -----
    Identical to :func:`build_Hk` but evaluates the R-sum as a single
    ``ZGEMM``.  The one-k-point kernel is memory-bandwidth bound (it
    streams the whole Hamiltonian for :math:`\\mathcal{O}(n_{awf}^2 n_R)`
    flops), so amortising that traffic over a block of k-points is much
    faster per point.
    """
    phases = np.exp((2.0j * np.pi) * (kq @ R.T))
    return phases.dot(HRpack.T).reshape(-1, nawf, nawf)


def band_energies(
    HRpack: NDArray[np.complexfloating],
    nawf: int,
    kq: NDArray[np.floating],
    R: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Compute band eigenvalues for one or many k-points.

    Parameters
    ----------
    HRpack : ndarray, shape ``(nawf*nawf, nR)``, complex
        Packed Hamiltonian from :func:`pack_HR`.
    nawf : int
        Dimension of the Hamiltonian matrix.
    kq : ndarray, shape ``(3,)`` or ``(nk, 3)``, float
        k-point(s) in crystal (fractional) coordinates.
    R : ndarray, shape ``(nR, 3)``, float
        Real-space lattice vectors.

    Returns
    -------
    ndarray, shape ``(nk, nawf)``, float
        Eigenvalues in ascending order, Hermitian upper triangle.

    Notes
    -----
    k-points are processed in blocks sized to keep the batched
    Hamiltonian below roughly 16 MB.
    """
    kq = np.atleast_2d(kq)
    nk = kq.shape[0]
    E = np.empty((nk, nawf), dtype=np.float64)

    chunk = int(min(1024, max(1, 1000000 // (nawf * nawf))))
    for ini_ik in range(0, nk, chunk):
        end_ik = min(ini_ik + chunk, nk)
        Hks = build_Hk_batch(HRpack, nawf, kq[ini_ik:end_ik], R)
        E[ini_ik:end_ik] = np.linalg.eigvalsh(Hks, UPLO='U')

    return E


def get_gap(
    HRpack: NDArray[np.complexfloating],
    nawf: int,
    kq: NDArray[np.floating],
    R: NDArray[np.floating],
    nelec: int,
) -> float:
    """Return the direct band gap at k-point ``kq`` between bands ``nelec-1`` and ``nelec``.

    This is the scalar objective function minimised by :func:`find_min` to
    locate Weyl (band-crossing) points.  A gap of zero indicates a
    band-touching or crossing.

    Parameters
    ----------
    HRpack : ndarray, shape ``(nawf*nawf, nR)``, complex
        Packed Hamiltonian from :func:`pack_HR`.
    nawf : int
        Dimension of the Hamiltonian matrix.
    kq : ndarray, shape ``(3,)``, float
        k-point in crystal (fractional) coordinates.
    R : ndarray, shape ``(nR, 3)``, float
        Real-space lattice vectors.
    nelec : int
        Number of occupied bands; the gap is evaluated between band
        index ``nelec - 1`` (HOMO) and ``nelec`` (LUMO).

    Returns
    -------
    float
        Energy difference ``E[nelec] - E[nelec - 1]`` at ``kq``.

    Notes
    -----
    Only the first spin channel is built, since the gap is defined on
    ``ispin = 0``.
    """
    eigs = np.linalg.eigvalsh(build_Hk(HRpack, nawf, kq, R), UPLO='U')
    return float(eigs[nelec] - eigs[nelec - 1])


# corners of the unit cube, used to sample each search box
_BOX_CORNERS = np.array([[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)])


def screen_boxes(
    HRpack: NDArray[np.complexfloating],
    nawf: int,
    R: NDArray[np.floating],
    nelec: int,
    lo: NDArray[np.floating],
    hi: NDArray[np.floating],
    factor: float = 2.0,
) -> NDArray[np.bool_]:
    """Flag the search boxes that can plausibly contain a band crossing.

    Parameters
    ----------
    HRpack : ndarray, shape ``(nawf*nawf, nR)``, complex
        Packed Hamiltonian from :func:`pack_HR`.
    nawf : int
        Dimension of the Hamiltonian matrix.
    R : ndarray, shape ``(nR, 3)``, float
        Real-space lattice vectors.
    nelec : int
        Number of occupied bands.
    lo, hi : ndarray, shape ``(nbox, 3)``, float
        Lower and upper corners of each search box, in crystal
        coordinates.
    factor : float, optional
        Safety margin on the gap-variation bound (default ``2.0``).
        Larger values keep more boxes.

    Returns
    -------
    ndarray, shape ``(nbox,)``, bool
        ``True`` for boxes that must be handed to the optimiser.

    Notes
    -----
    The gap is sampled on a 9-point stencil (centre plus the eight
    corners) of every box in a single batched evaluation; adjacent boxes
    share corners, so the stencil is de-duplicated first.  A box is
    discarded when its smallest sampled gap exceeds ``factor`` times the
    largest gap variation observed across the box, i.e. when a linear
    (Lipschitz) extrapolation of the sampled slope cannot reach zero
    inside the box.  This is a heuristic, not a proof: it assumes the gap
    does not develop structure finer than the box on which it is
    sampled.  Pass ``factor <= 0`` in :func:`find_min` to disable.
    """
    nbox = lo.shape[0]
    bbox = hi - lo

    stencil = np.empty((nbox, 1 + _BOX_CORNERS.shape[0], 3))
    stencil[:, 0] = lo + 0.5 * bbox
    stencil[:, 1:] = lo[:, None, :] + _BOX_CORNERS[None] * bbox[:, None, :]

    # corners shared between neighbouring boxes agree only to round-off
    kpts, inv = np.unique(np.around(stencil.reshape(-1, 3), 12), axis=0, return_inverse=True)
    E = band_energies(HRpack, nawf, kpts, R)
    gaps = (E[:, nelec] - E[:, nelec - 1])[inv.reshape(-1)].reshape(nbox, -1)

    spread = np.max(np.abs(gaps[:, 1:] - gaps[:, :1]), axis=1)
    return gaps.min(axis=1) <= factor * spread


def get_R_grid_fft(nr1: int, nr2: int, nr3: int) -> NDArray[np.float64]:
    """Build the real-space R-grid corresponding to an FFT supercell.

    Generates all ``nr1 * nr2 * nr3`` lattice vectors in fractional
    coordinates and folds them into the centred interval ``[-0.5, 0.5)``
    so that the Fourier sums converge correctly with the PAOFLOW
    convention.

    Parameters
    ----------
    nr1, nr2, nr3 : int
        Number of FFT grid points along each reciprocal-space direction.

    Returns
    -------
    R : ndarray, shape (nr1 * nr2 * nr3, 3)
        Lattice vectors in units of the corresponding primitive vectors,
        ordered as ``k + j*nr3 + i*nr2*nr3``.
    """
    nr = np.array([nr1, nr2, nr3], dtype=np.float64)[:, None]

    R = np.indices((nr1, nr2, nr3), dtype=np.float64).reshape(3, -1) / nr
    R[R >= 0.5] -= 1.0

    return np.ascontiguousarray((R * nr).T)


def get_search_grid(
    nk1: int,
    nk2: int,
    nk3: int,
    snk1_range=[-0.5, 0.5],
    snk2_range=[-0.5, 0.5],
    snk3_range=[-0.5, 0.5],
    endpoint: bool = False,
) -> NDArray[np.float64]:
    """Generate a uniform 3-D search grid in fractional BZ coordinates.

    Creates a full-factorial mesh with ``nk1 * nk2 * nk3`` points by
    taking the outer product of three ``np.linspace`` arrays and reshaping
    to a flat list of k-vectors.  These grid points are used as the lower
    bounds of the optimisation boxes in :func:`find_min`.

    Parameters
    ----------
    nk1, nk2, nk3 : int
        Number of grid points along each reciprocal direction.
    snk1_range, snk2_range, snk3_range : list of float, optional
        ``[start, stop]`` range for each direction.  Default is the full
        first Brillouin zone ``[-0.5, 0.5]``.
    endpoint : bool, optional
        If ``False`` (default), the stop value is excluded, matching the
        Monkhorst–Pack convention and avoiding zone-boundary duplication.

    Returns
    -------
    grid : ndarray, shape (nk1 * nk2 * nk3, 3)
        Cartesian product of the three 1-D grids, ordered with the first
        index varying slowest (``indexing='ij'``).
    """
    nk1_arr = np.linspace(snk1_range[0], snk1_range[1], num=nk1, endpoint=endpoint)
    nk2_arr = np.linspace(snk2_range[0], snk2_range[1], num=nk2, endpoint=endpoint)
    nk3_arr = np.linspace(snk3_range[0], snk3_range[1], num=nk3, endpoint=endpoint)

    # nk_str = np.zeros((nk1*nk2*nk3,3), order='C')
    return np.array(np.meshgrid(nk1_arr, nk2_arr, nk3_arr, indexing='ij')).T.reshape(-1, 3)


def find_weyl(data_controller: DataController, test_rad: float, search_grid: list[int]) -> None:
    """Locate and classify Weyl points in the Brillouin zone.

    Orchestrates the full Weyl-point search workflow:

    1. Calls :func:`find_min` to minimise the band gap over a coarse
       search grid and collect band-touching candidates.
    2. If symmetry is enabled (``attr['symmetrize']``), expands candidates
       to all symmetry-equivalent k-points via :func:`get_equiv_k`.
    3. Optionally uses ``z2pack`` to compute the Chern number on a small
       sphere around each candidate; points with non-zero Chern number are
       confirmed as Weyl points.  If ``z2pack`` is unavailable, chirality
       is recorded as ``'?'``.  The candidates are distributed over the
       MPI ranks, each of which builds its own ``tbmodels`` model from
       ``hamiltonian.dat``.
    4. Writes a summary table to ``weyl_points.dat`` in ``attr['opath']``.

    Parameters
    ----------
    data_controller : DataController
        Provides ``HRs``, ``sym_rot``, ``b_vectors``, ``sym_TR``,
        ``nelec``, ``symmetrize``, ``verbose``, ``dftMAG``, ``dftSO``,
        ``alat``, ``opath``.  The optional attribute
        ``weyl_screen_factor`` (default ``2.0``) tunes the box screening
        of :func:`screen_boxes`; set it to ``0`` to disable.
    test_rad : float
        Radius (in fractional BZ units) of the ``z2pack`` Chern-number
        sphere.  Currently unused: the radius is fixed at ``0.005``.
    search_grid : list of int, length 3
        ``[nk1, nk2, nk3]`` number of cells to divide the BZ into for the
        initial coarse gap-minimisation search.

    Returns
    -------
    None
        Writes ``weyl_points.dat`` to ``attr['opath']`` on rank ``0``.

    Output files (written to ``opath``)
    ------------------------------------
    ``weyl_points.dat``
        Formatted table with columns: index, Cartesian k-coordinates in
        units of ``2\u03c0/alat``, Chern number, and energy.
    """
    import os

    arry, attr = data_controller.data_dicts()

    symf = attr['symmetrize']
    nelec, verbose = attr['nelec'], attr['verbose']
    HRs, symops, b_vectors, TR_flag = (
        arry['HRs'],
        arry['sym_rot'],
        arry['b_vectors'],
        arry['sym_TR'],
    )

    nawf, _, nk1, nk2, nk3, nspin = HRs.shape
    R = get_R_grid_fft(nk1, nk2, nk3)

    HRpack = pack_HR(np.reshape(HRs, (nawf, nawf, nk1 * nk2 * nk3, nspin)))

    mag_soc = np.logical_and(attr['dftMAG'], attr['dftSO'])

    CAND, _ = find_min(
        HRpack,
        nawf,
        nelec,
        R,
        symf,
        verbose,
        search_grid,
        attr.get('weyl_screen_factor', 2.0),
    )

    if rank == 0 and symf:
        # get all equiv k
        CAND = get_equiv_k(CAND, symops, TR_flag, mag_soc)
    CAND = comm.bcast(CAND if rank == 0 else None, root=0)

    eigs = band_energies(HRpack, nawf, CAND, R)
    ene = eigs[:, nelec]
    gaps = eigs[:, nelec] - eigs[:, nelec - 1]

    if rank == 0 and verbose:
        print()
        for i in range(CAND.shape[0]):
            tup = (i + 1,) + tuple(CAND[i][j] for j in range(3)) + (ene[i], gaps[i])
            print(
                'Weyl point candidate #%d crystal coord: [ %6.4f %6.4f %6.4f ] ene=%.4f gap=%.6e'
                % tup
            )

            in_cart = b_vectors.T.dot(CAND[i])
            tup = (i + 1,) + tuple(in_cart[j] for j in range(3)) + (ene[i], gaps[i])
            print(
                'Weyl point candidate #%d 2pi/alat     : [ %6.4f %6.4f %6.4f ] ene=%.4f gap=%.6e'
                % tup
            )
            print()

    ncand = CAND.shape[0]
    chirality = np.zeros(ncand)
    have_chern = True

    try:
        import tbmodels
        import z2pack
    except ModuleNotFoundError:
        have_chern = False
        if rank == 0:
            print('Could not load z2pack to verify chirality of weyl points')

    if have_chern:
        if not verbose:
            import logging

            logging.getLogger('z2pack').setLevel(logging.WARNING)

        model = tbmodels.Model.from_wannier_files(
            hr_file=os.path.join(attr['opath'], 'hamiltonian.dat')
        )
        system = z2pack.tb.System(model, bands=nelec)

        # NOTE: the test_rad / nearest-candidate clamp is inactive, radius is fixed
        k_rad = 0.005

        # candidates are spread round-robin over the ranks
        local = []
        for i in range(rank, ncand, size):
            surface = z2pack.shape.Sphere(center=tuple(CAND[i]), radius=k_rad)
            result_1 = z2pack.surface.run(system=system, surface=surface)
            local.append((i, z2pack.invariant.chern(result_1)))

        for part in comm.allgather(local):
            for i, invariant in part:
                chirality[i] = invariant

        confirmed = np.flatnonzero(chirality != 0.0)
    else:
        confirmed = np.arange(ncand)

    if rank != 0:
        return

    if verbose:
        print()

    wcs = '{0:>3} {1:>10} {2:>10} {3:>10} {4:>2} {5:>7}\n'.format(
        '#', '2pi/alat', '2pi/alat', '2pi/alat', 'C', 'ene'
    )
    wcs += '  #   alat = {0} angstrom\n'.format(attr['alat'] * BOHR_RADIUS_ANGS)
    wcs += '-' * 72 + '\n'

    for j, i in enumerate(confirmed):
        fm = b_vectors.T.dot(CAND[i])
        chern = int(chirality[i]) if have_chern else '?'
        if verbose and have_chern:
            fstring = 'Found Candidate No. {0} at [{1:>7.4f} {2:>7.4f} {3:>7.4f}] with Chirality:{4:>2} ene={5:>7.4f}'
            print(fstring.format(j + 1, fm[0], fm[1], fm[2], chern, ene[i]))
        wcs += '{0:>3d} {1:>10.4f} {2:>10.4f} {3:>10.4f} {4:>2} {5:>7.4f}\n'.format(
            j + 1, fm[0], fm[1], fm[2], chern, ene[i]
        )

    with open(os.path.join(attr['opath'], 'weyl_points.dat'), 'w') as ofo:
        ofo.write(wcs)


def find_min(
    HRpack: NDArray[np.complexfloating],
    nawf: int,
    nelec: int,
    R: NDArray[np.floating],
    symf: bool,
    verbose: bool,
    search_grid: list[int] = [8, 8, 8],
    screen_factor: float = 2.0,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Find band-gap minima across the BZ using local optimisation.

    Divides the first BZ into ``nk1 * nk2 * nk3`` rectangular boxes and
    runs an ``L-BFGS-B`` minimisation of :func:`get_gap` inside each box,
    with the box centre as the initial guess.  The BZ boxes are distributed
    across MPI ranks for parallelism; partial results are collected with
    :func:`~.communication.gather_full`.

    Boxes are first screened with :func:`screen_boxes`, which samples the
    gap on a batched 9-point stencil and skips boxes that cannot reach
    zero; this removes the great majority of the optimiser calls.

    On rank 0, candidates with a gap smaller than ``1e-5`` are retained.
    If symmetry is enabled (``symf``), they are further de-duplicated by
    sorting on energy and removing entries with identical energies (to
    ``4`` decimal places), leaving only symmetry-inequivalent Weyl points.

    Parameters
    ----------
    HRpack : ndarray, shape ``(nawf*nawf, nR)``, complex
        Packed Hamiltonian from :func:`pack_HR`.
    nawf : int
        Dimension of the Hamiltonian matrix.
    nelec : int
        Number of occupied bands; gap is ``E[nelec] - E[nelec-1]``.
    R : ndarray, shape (nR, 3)
        Real-space lattice vectors from :func:`get_R_grid_fft`.
    symf : bool
        If ``True``, de-duplicate candidates using energy degeneracy.
    verbose : bool
        Print intermediate candidate information to stdout.
    search_grid : list of int, optional
        ``[nk1, nk2, nk3]`` subdivision of the BZ.  Default ``[8, 8, 8]``.
    screen_factor : float, optional
        Safety margin passed to :func:`screen_boxes`.  Values ``<= 0``
        disable screening and optimise inside every box.

    Returns
    -------
    candidates : ndarray, shape (nc, 3) or ``None`` on non-root ranks
        k-coordinates (fractional) of the band-touching candidates.
    ene : ndarray, shape (nc,) or ``None`` on non-root ranks
        Energy at the LUMO band for each candidate.
    """
    snk = (search_grid[0], search_grid[1], search_grid[2])
    grid_K = get_search_grid(snk[0], snk[1], snk[2])

    # get the search grid off possible HSP
    end = np.array([0.5] * 3)
    start = np.array([-0.5] * 3)
    bbox = np.array([(end[i] - start[i]) / snk[i] for i in range(3)])

    # only this rank's share of the boxes is materialised
    box_lo = scatter_full(grid_K, 1)
    box_hi = box_lo + bbox
    print('finding Weyl points... rank={0} npoints={1}'.format(rank, box_lo.shape[0]))

    # bounds for each search subsection of FBZ, initial guess in the middle
    bounds_K = np.stack((box_lo, box_hi), axis=-1)
    guess_K = box_lo + 0.5 * bbox

    if screen_factor > 0.0:
        alive = screen_boxes(HRpack, nawf, R, nelec, box_lo, box_hi, screen_factor)
    else:
        alive = np.ones(box_lo.shape[0], dtype=bool)

    # column 3 flags a converged box; gather_full requires the full partition
    candidates = np.zeros((box_lo.shape[0], 4))

    lam_XiP = lambda K: get_gap(HRpack, nawf, K, R, nelec)
    for i in np.flatnonzero(alive):
        solx = OP.minimize(
            lam_XiP,
            guess_K[i],
            bounds=bounds_K[i],
            method='L-BFGS-B',
            options={'ftol': 1.0e-14, 'gtol': 1.0e-12},
        )

        if np.abs(solx.fun) < 1.0e-5:
            candidates[i, :3] = solx.x
            candidates[i, 3] = 1.0

    candidates = gather_full(candidates, 1)

    ene = None
    if rank == 0:
        # filter out non hits
        candidates = candidates[candidates[:, 3] > 0.5, :3]
        # calculate energy at each candidate to reduce equiv ones
        eigs = band_energies(HRpack, nawf, candidates, R)
        ene = eigs[:, nelec]
        gaps = eigs[:, nelec] - eigs[:, nelec - 1]

        if symf:
            # sort by gap size (should be nearly zero)
            idx = np.argsort(gaps)
            ene = ene[idx]
            candidates = candidates[idx]
            # filter out duplicates by degeneracy
            _, idx = np.unique(np.around(ene, decimals=4), return_index=True)
            ene = ene[idx]
            candidates = candidates[idx]
            if verbose:
                print('\nfound %s non-equivilent candidates' % candidates.shape[0])
                for i in range(ene.shape[0]):
                    tup = tuple(candidates[i][j] for j in range(3)) + (ene[i],)
                    print('[ % 7.4f % 7.4f % 7.4f ] % 4.4f' % tup)
                print()

    comm.Barrier()
    return (candidates, ene)


def get_equiv_k(kp, symop, sym_TR, mag_soc):
    """Expand a set of k-points to all symmetry-equivalent images.

    Applies each space-group rotation ``symop[isym]`` (and optionally its
    time-reversal partner ``-k``) to every k-point in ``kp``, then
    de-duplicates the result to return only unique images within the
    first BZ (folded to ``[-0.5, 0.5)``).

    Time-reversal symmetry adds the set ``{-k}`` unless the system is a
    magnetic SOC calculation (``mag_soc = dftMAG and dftSO``).

    Parameters
    ----------
    kp : ndarray, shape (nc, 3)
        Input k-points in crystal (fractional) coordinates.
    symop : ndarray, shape (nsym, 3, 3)
        Rotation matrices of the crystal point group.
    sym_TR : ndarray of bool, shape (nsym,)
        ``True`` for operations that include time reversal (i.e. act as
        ``k' = -R k``).
    mag_soc : bool
        ``True`` when both ``dftMAG`` and ``dftSO`` are active; disables
        the automatic ``{-k}`` extension.

    Returns
    -------
    newk_tot : ndarray, shape (nu, 3)
        All unique symmetry-equivalent k-points, rounded to ``4`` decimal
        places.
    """
    from ..hamiltonian.pao_sym import correct_roundoff

    # if we have time inversion sym
    if not mag_soc:
        kp = np.vstack([kp, -kp])

    kp = correct_roundoff(kp)
    newk_tot = np.copy(kp)

    for isym in range(symop.shape[0]):
        # transform k -> k' with the sym op
        if sym_TR[isym]:
            newk = ((((-symop[isym] @ (kp.T % 1.0)) % 1.0) + 0.5) % 1.0) - 0.5
        else:
            newk = ((((symop[isym] @ (kp.T % 1.0)) % 1.0) + 0.5) % 1.0) - 0.5

        newk = correct_roundoff(newk)
        newk[np.where(np.isclose(newk, 0.5))] = -0.5
        newk[np.where(np.isclose(newk, -1.0))] = 0.0
        newk[np.where(np.isclose(newk, 1.0))] = 0.0

        newk_tot = np.vstack([newk_tot, newk.T])

    # filter duplicates
    newk_r = np.around(newk_tot, decimals=4)
    newk_r[np.where(np.abs(newk_r) < 1.0e-4)] = 0

    b_pos = np.ascontiguousarray(newk_r).view(
        np.dtype((np.void, newk_r.dtype.itemsize * newk_r.shape[1]))
    )
    _, idx = np.unique(b_pos, return_index=True)
    newk_tot = newk_tot[idx]

    return newk_tot
