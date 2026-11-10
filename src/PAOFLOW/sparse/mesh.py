"""Fused BZ-mesh pass: eigenpairs, velocities, adaptive widths, consumers.

One loop over this rank's k-slice, eigenvectors never stored, fills in the dense layout:

- ``E_k`` (nk_local, nev, nspin): eigenvalues (eV),
- ``velkp`` (nk_local, 3, nev, nspin): Hellmann-Feynman velocities, degeneracies resolved by
  :func:`_band_diagonal_velocities`,
- ``deltakp`` (nk_local, nev, nspin): Yates widths ``afac * dk * |v_n|``, optionally bounded
  by ``min_width``/``max_width`` (as ``do_adaptive_smearing``),
- ``d2Ed2k`` (6, nk_local, nev, nspin): band curvature, only if requested,

so ``do_dos_adaptive``, Boltzmann transport and ``do_effective_mass`` consume them unchanged.
Consumers (:class:`~PAOFLOW.sparse.properties.MeshProperty`) get one
:class:`~PAOFLOW.sparse.kpoint.KPoint` per k and must not retain it: its eigenvector block is the
only dense workspace. Phases use ``sign=-1`` and k is ordered ``n = k + j*nk3 + i*nk2*nk3``, as
in the dense FFT grid, so per-k data are index-comparable with dense arrays.

``COMMUTE_TOL``: on example01 folded copies commute to <= 5e-5, base-cell symmetry multiplets
measure 0.4-1.5.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI
from scipy.sparse import issparse

from .solver import DENSE_N_MAX, DENSE_RATIO

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from .hamiltonian import SparseHamiltonian
    from .log import SparseLog
    from .properties import MeshProperty

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PAD_ENERGY_OFFSET = 1.0e3
PAD_DELTA = 1.0
OPERATOR_DENSE_FILL = 0.1
COMMUTE_TOL = 1.0e-3
# relative singular value below which a matrix counts as commuting with all
# three velocity blocks (the exact zeros between folded copies are ~1e-15)
SECTOR_TOL = 1.0e-8
# rationally independent, so c.v = c.v' only if v = v', even for the
# permuted and sign-flipped velocities that symmetry makes degenerate
JOINT_WEIGHTS = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])


def dense_operators(n: int, nnz: int, dense_n_max: int = DENSE_N_MAX) -> bool:
    """Whether ``dH/dk`` (``nnz`` nonzeros, ``n = nawf``) should be multiplied as dense arrays.

    True when fill >= ``OPERATOR_DENSE_FILL`` and ``n <= dense_n_max``: scipy's sparse-times-block
    is single-threaded, and BLAS was 15x faster on the full ``H(k)`` of Si at ``bond_order=36``.
    """
    return n <= dense_n_max and nnz >= OPERATOR_DENSE_FILL * n * n


def _blas_threads() -> int:
    """Threads this rank was given for BLAS, read from the usual
    environment variables; 1 when none is set."""
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        try:
            return max(1, int(os.environ[name]))
        except (KeyError, ValueError):
            continue
    return 1


def _times_block(A: Any, V: np.ndarray, dense: bool) -> np.ndarray:
    """``A @ V`` for sparse ``A`` (n, n) and ``V`` (n, m) on all this rank's threads.

    ``dense``: one ``zgemm`` (88 s -> 7 s per k on doubled Si, n = 4608, where the sparse product
    was 80% of the pass). Otherwise column slices on a thread pool (scipy releases the GIL; 2-3x
    on four threads).
    """
    if dense:
        return A.toarray() @ V
    m = V.shape[1]
    threads = min(_blas_threads(), m)
    if threads <= 1:
        return A @ V
    out = np.empty((A.shape[0], m), dtype=np.result_type(A.dtype, V.dtype))
    bounds = np.linspace(0, m, threads + 1).astype(int)

    def work(i):
        out[:, bounds[i] : bounds[i + 1]] = A @ V[:, bounds[i] : bounds[i + 1]]

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(work, range(threads)))
    return out


def _band_diagonal_velocities(
    E: np.ndarray, V: np.ndarray, dhk: Sequence[Any], dense_n_max: int = DENSE_N_MAX
) -> np.ndarray:
    """Band velocities ``v_ln = <n| dH/dk_l |n>``, degeneracies resolved.

    Parameters
    ----------
    E : np.ndarray, shape (m,)
        Ascending eigenvalues (eV).
    V : np.ndarray, shape (nawf, m)
        Orthonormal eigenvectors.
    dhk : sequence of 3 sparse matrices or :class:`~PAOFLOW.sparse.kpoint.DenseOnDemand`
        ``dH/dk_l``.
    dense_n_max : int, optional
        Above it ``(dH/dk_l) V`` stays sparse whatever the fill.

    Returns
    -------
    vel : np.ndarray, shape (3, m)
    resolved : list of (D, R, parts)
        Per degenerate group ``D`` not left to the per-direction rule: ``V[:, D] @ R`` are the
        states ``vel[:, D]`` belongs to (the caller must rotate ``V``), and ``parts`` the index
        ranges in ``D`` that still take the per-direction rule.

    Notes
    -----
    In a degenerate group the diagonal is gauge-dependent, so the Hermitized blocks
    :math:`B_l = V_D^\\dagger (\\partial H/\\partial k_l) V_D` decide:

    - commuting (``COMMUTE_TOL``; folded copies from different primitive k): joint eigenbasis of
      ``JOINT_WEIGHTS . B``, each state's three components read in it, ordered by ``v_x``;
    - not commuting (base-cell symmetry multiplets): per-direction eigenvalues of each
      :math:`B_l`, sorted, as ``utils.perturb_split``. If the group also holds copies (``k``/``-k``,
      exact copies of a near-degenerate pair), it is first split by :func:`_invariant_subspaces`
      and each sector takes the rule that fits it.

    The per-direction rule on a commuting group mispairs ``(v_x, v_y, v_z)`` across states:
    :math:`\\sum v_l^2` survives but per-state :math:`|v|` (smearing width) and :math:`v_x v_y` do
    not (error 0.09 on doubled Si). On the base cell both rules agree on :math:`\\sum v_l^2` and
    :math:`\\sum |v|` to 1e-12; :math:`\\sigma_{xy}` moves by ~2e-8 of :math:`\\sigma_{xx}`.
    ``(dH/dk_l) V`` is formed once (:func:`_times_block`), the dominant cost, and column-sliced per
    group.
    """
    from .kpoint import degenerate_groups

    m = len(E)
    vel = np.empty((3, m), dtype=float)
    resolved = []
    if m == 0:
        return vel, resolved

    if issparse(dhk[0]):
        dense = dense_operators(V.shape[0], dhk[0].nnz, dense_n_max)
        W = [_times_block(dhk[l], V, dense) for l in range(3)]
    else:
        W = [dhk[l] @ V for l in range(3)]
    for l in range(3):
        vel[l] = np.einsum('an,an->n', np.conj(V), W[l]).real

    for D in degenerate_groups(E, m):
        VD = V[:, D]
        B = []
        for l in range(3):
            block = VD.conj().T @ W[l][:, D]
            B.append(0.5 * (block + block.conj().T))
        B = np.array(B)
        if _commute(B):
            U, vel[:, D] = _joint_basis(B)
            resolved.append((D, U, []))
            continue
        Q, sectors = _invariant_subspaces(B)
        if len(sectors) == 1:
            for l in range(3):
                vel[l][D] = np.linalg.eigvalsh(B[l])
            continue
        R = np.empty_like(Q)
        parts = []
        start = 0
        for sector in sectors:
            cols = slice(start, start + len(sector))
            Qs = Q[:, sector]
            Bs = np.einsum('ia,lij,jb->lab', Qs.conj(), B, Qs)
            if _commute(Bs):
                U, vel[:, D[cols]] = _joint_basis(Bs)
                R[:, cols] = Qs @ U
            else:
                R[:, cols] = Qs
                for l in range(3):
                    vel[l][D[cols]] = np.linalg.eigvalsh(Bs[l])
                parts.append(D[cols])
            start += len(sector)
        resolved.append((D, R, parts))
    return vel, resolved


def _commute(B: np.ndarray) -> bool:
    """Whether the three Hermitian blocks ``B`` (3, g, g) commute to ``COMMUTE_TOL``."""
    scale = np.abs(B).max() ** 2
    commutator = max(np.abs(B[i] @ B[j] - B[j] @ B[i]).max() for i, j in ((0, 1), (0, 2), (1, 2)))
    return commutator <= COMMUTE_TOL * scale


def _joint_basis(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Common eigenbasis ``U`` of commuting blocks ``B`` (3, g, g), ordered by
    ``v_x`` (then ``v_y``, ``v_z``), and the velocities ``v`` (3, g) in it."""
    _, U = np.linalg.eigh(np.tensordot(JOINT_WEIGHTS, B, axes=1))
    v = np.einsum('in,lij,jn->ln', U.conj(), B, U).real
    order = np.lexsort(v[::-1])
    return U[:, order], v[:, order]


def _invariant_subspaces(B: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Finest subspaces all three Hermitian blocks ``B`` (3, g, g) leave invariant.

    Returns unitary ``Q`` (g, g) and ``sectors``, the column indices of ``Q`` per subspace.
    The commutant of the ``B_l`` (null space of ``X -> [B_l, X]``, SVD of a ``(3 g^2, g^2)``
    matrix) is a *-algebra; a generic Hermitian element is a distinct multiple of the identity on
    each irreducible subspace, so its eigenspaces are the sectors. Only the identity: one sector.
    """
    g = B.shape[1]
    eye = np.eye(g)
    # column-stacked vec: vec(B X - X B) = (I (x) B - B^T (x) I) vec(X)
    L = np.concatenate([np.kron(eye, b) - np.kron(b.T, eye) for b in B])
    try:
        _, s, vh = np.linalg.svd(L)
    except np.linalg.LinAlgError:
        if not np.isfinite(L).all():
            raise ValueError(
                'sparse: the velocity blocks of a degenerate group of %d states hold NaN or '
                'inf, so the SVD of their commutator cannot converge; the eigenvectors or '
                'dH/dk of this k-point are not finite' % g
            ) from None
        from scipy.linalg import svd

        # gesdd (numpy) sometimes fails on this structured L; gesvd is slower but converges
        _, s, vh = svd(L, lapack_driver='gesvd')
    null = vh[s <= SECTOR_TOL * s[0]].conj()
    if len(null) <= 1:
        return eye.astype(complex), [np.arange(g)]
    weights = np.random.default_rng(0).standard_normal(len(null))
    X = np.tensordot(weights, null.reshape(-1, g, g).transpose(0, 2, 1), axes=1)
    w, Q = np.linalg.eigh(X + X.conj().T)
    cuts = np.flatnonzero(np.diff(w) > SECTOR_TOL**0.5 * np.abs(w).max()) + 1
    return Q, np.split(np.arange(g), cuts)


def run_mesh(
    data_controller: DataController,
    sparse_h: SparseHamiltonian,
    nev: int,
    consumers: Sequence[MeshProperty] = (),
    products: Sequence[str] = (),
    afac: float | None = None,
    smearing: str = 'gauss',
    min_width: float | None = None,
    max_width: float | None = None,
    verbose: bool = False,
    hk_solver: str = 'auto',
    ehi: float | None = None,
    interior: tuple[float, float] | None = None,
    ptop: float | None = None,
    dense_ratio: float = DENSE_RATIO,
    dense_n_max: int = DENSE_N_MAX,
    label: str = 'Mesh pass',
) -> None:
    """Walk the local share of the BZ mesh, producing all band-diagonal data.

    Parameters
    ----------
    data_controller : DataController
        Supplies mesh and ``omega``; receives this rank's k-slice of ``E_k``, ``velkp``,
        ``deltakp`` and products, plus ``attr['sparse_dmax']`` (largest real width, for the
        engine's smearing-tail margin) and, interior only, ``attr['bnd']``.
    sparse_h : SparseHamiltonian
        Bond list ``H(k)`` and ``dH/dk`` are assembled from.
    nev : int
        Bands per k kept in the stored arrays; ignored with ``interior``.
    consumers : sequence of MeshProperty, optional
        Fed one ``KPoint`` per k; finalized by the engine, in call order, not here.
    products : sequence of str, optional
        Only ``'d2Ed2k'`` (band curvature).
    afac : float or None, optional
        Adaptive-width prefactor; default is the dense pipeline's value for ``smearing``.
    smearing : str, optional
        Smearing kernel name; sets the default ``afac``.
    min_width : float or None, optional
        Lower bound of the widths (eV), as in ``do_adaptive_smearing``: a
        state narrower than the energy grid of a consumer drops out of its
        sums.  ``None`` (default) keeps the raw widths.
    max_width : float or None, optional
        Upper bound of the widths (eV); ``None`` leaves them uncapped.
    verbose : bool, optional
        Log progress every ``min(100, nk_local // 10)`` k (a fixed 100 is silent on coarse meshes).
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        Chosen once from ``(nawf, nev)``; a full-spectrum pass is always ``'dense'``.
    ehi : float or None, optional
        Top of the window ``nev`` was sized for (eV). k-points whose solve stops below it are
        counted and raise collectively after the loop with the ``nev`` to re-run with.
    interior : (float, float) or None, optional
        ``(elo, ehi)`` in eV: solve every state inside, none below (no ``ehi`` check needed).
    ptop : float or None, optional
        Top of the well-projected range (``SparseEngine._projectable_top``). Only states below it
        count in ``KPoint.bnd`` and the stored arrays (padded above); the rest stay in
        ``KPoint.E``/``V`` as intermediate states of full-spectrum sums.
    dense_ratio, dense_n_max : optional
        Kernel dispatch limits from ``sparse_config``; ``dense_n_max`` also caps the full spectrum.
    label : str, optional
        Section title in ``sparse.log``.

    Raises
    ------
    NotImplementedError
        Full spectrum needed (``'full_spectrum'`` consumer or ``'d2Ed2k'``) and
        ``nawf > dense_n_max``. No truncated-sum fallback; Sternheimer is the intended route.
    ValueError
        Full spectrum with ``interior``, or an unknown product.

    Notes
    -----
    Fused so eigenvectors never outlive their k. Full-spectrum passes solve all ``nawf`` states
    per k with the dense kernel but store only the window. Warm starts (previous ground state as
    ``v0``, previous count as ``k0``) only aid convergence. Interior counts vary with k, so results
    are padded to the global maximum (:func:`_pad_interior`): energy ``PAD_ENERGY_OFFSET`` above
    the top so every smearing kernel vanishes, zero velocity so no transport weight, and width
    ``PAD_DELTA`` rather than 0 because the kernels divide by it.
    """
    from ..spectrum.do_adaptive_smearing import adaptive_widths
    from ..utils.communication import scatter_full
    from ..utils.get_K_grid_fft import get_K_grid_fft_crystal
    from .curvature import band_curvature_k
    from .kpoint import DenseOnDemand, KPoint, degenerate_groups
    from .log import get_sparse_log
    from .solver import creep_warning, hk_solver_fields, solve_k

    arrays, attr = data_controller.data_dicts()
    nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
    attr['nkpnts'] = nkpnts = nk1 * nk2 * nk3

    products = set(products)
    unknown = products - {'d2Ed2k'}
    if unknown:
        raise ValueError(f'run_mesh: unknown products {sorted(unknown)}')
    curvature = 'd2Ed2k' in products
    needs = set().union(*(c.needs for c in consumers))
    nawf = sparse_h.nawf
    full = curvature or 'full_spectrum' in needs
    if full:
        wanting = sorted(
            {c.method for c in consumers if 'full_spectrum' in c.needs}
            | ({'band curvature (d2Ed2k)'} if curvature else set())
        )
        if interior is not None:
            raise ValueError(
                'run_mesh: %s need the full spectrum, which an interior window never '
                'computes.' % ', '.join(wanting)
            )
        if nawf > dense_n_max:
            raise NotImplementedError(
                f'sparse mesh: {", ".join(wanting)} sum over interband pairs that include every '
                f'state, which needs the full spectrum per k-point (nawf = {nawf}). That is a '
                f'dense (nawf, nawf) solve, admitted only up to dense_n_max = {dense_n_max}, '
                f'where it would take {16.0 * nawf * nawf / 1024**3:.2f} GB per k-point. '
                'The exact route past it is a Sternheimer (linear-response) solve for the '
                'interband sums, which is not implemented yet; there is no truncated-sum '
                'fallback. Run the property on a smaller cell, or on the dense pipeline.'
            )
        nsolve, solver = nawf, 'dense'
    else:
        nsolve, solver = nev, hk_solver

    kfrac_all = get_K_grid_fft_crystal(nk1, nk2, nk3)
    kloc = scatter_full(kfrac_all, attr['npool'])
    kglobal = scatter_full(np.arange(nkpnts), attr['npool'])
    nk_local = kloc.shape[0]
    nspin = sparse_h.nspin

    dk = (8.0 * np.pi**3 / attr['omega'] / nkpnts) ** (1.0 / 3.0)
    if afac is None:
        afac = 1.0 if smearing == 'm-p' else 0.7
    floor = 0.0 if min_width is None else min_width
    if floor < 0.0 or (max_width is not None and max_width < floor):
        raise ValueError('need 0 <= min_width <= max_width (got %r, %r)' % (min_width, max_width))
    bounds = '' if min_width is None else f', widths floored at {min_width:.3g} eV'
    bounds += '' if max_width is None else f', capped at {max_width:.3g} eV'

    extras = sorted({c.method for c in consumers}) + (['band curvature'] if curvature else [])
    log = get_sparse_log(data_controller)
    log.section(label)
    log.field('k-mesh', f'{nk1} x {nk2} x {nk3}  ({nkpnts} k-points)')
    log.field('computed per k-point', ', '.join(['eigenvalues', 'velocities'] + extras))
    if interior is None:
        log.field('bands per k-point', nev)
    else:
        log.field('bands per k-point', 'varies with k (interior solve)')
    if full:
        log.field('full spectrum', f'all {nawf} states per k-point for {", ".join(wanting)}')
    log.field('smearing', f'{smearing}, afac {afac:.3f}{bounds}')
    limits = {'dense_ratio': dense_ratio, 'dense_n_max': dense_n_max}
    log.rows(hk_solver_fields(nawf, nsolve, hk_solver=solver, **limits))
    warning = None if (full or interior is not None) else creep_warning(nawf, nsolve)
    if warning:
        log.write('  WARNING: ' + warning)

    if interior is None:
        # the states dropped above ptop take the interior padding values
        pad = (ptop if ptop is not None else 0.0) + PAD_ENERGY_OFFSET
        E_k = np.full((nk_local, nev, nspin), pad, dtype=float)
        velkp = np.zeros((nk_local, 3, nev, nspin), dtype=float)
        deltakp = np.full((nk_local, nev, nspin), PAD_DELTA, dtype=float)
        if curvature:
            d2Ed2k = np.zeros((6, nk_local, nev, nspin), dtype=float)
            # a unit diagonal for the dropped states: finite in the Hall sums,
            # where their occupation weight vanishes, and invertible for the
            # effective-mass writer, whose rows at the padding energy are not bands
            d2Ed2k[:3] = 1.0
    else:
        acc = {}
    deficit = 0
    dropped = 0
    solved_top = np.inf
    step = max(1, min(100, nk_local // 10))
    # consumers and the curvature reuse dH/dk many times per k-point, so they
    # get it dense for the whole k-point; the velocities alone densify one
    # direction at a time (_times_block)
    dense_ops = bool(consumers or curvature) and dense_operators(
        nawf, len(sparse_h.plan['indices']), dense_n_max
    )
    if dense_ops:
        log.field('dH/dk scratch per k-point (GB)', '%.2f' % (4 * 16.0 * nawf * nawf / 1024**3))
    dmax = 0.0

    for ispin in range(nspin):
        warm = None
        for ik in range(nk_local):
            assembled = sparse_h.assemble_derivatives(
                kloc[ik], ispin=ispin, sign=-1, order=2 if curvature else 1
            )
            hk, dhk = assembled[0], assembled[1]
            if dense_ops:
                dhk = DenseOnDemand(dhk)
            # an interior window never takes the full spectrum, so there solver == hk_solver
            E, V, warm = solve_k(hk, nsolve, interior, warm, solver, **limits)
            nsolved = nev if interior is None else len(E)
            bnd = nsolved
            if ptop is not None:
                # the placeholders at the PAO shift lie above every real state
                bnd = int(np.searchsorted(E[:nsolved], ptop))
                dropped += nsolved - bnd

            vel, resolved = _band_diagonal_velocities(
                E[:bnd], V[:, :bnd], dhk, dense_n_max=dense_n_max
            )
            # the interband kernels (perturb_split) would split these groups
            # again one direction at a time and pair matrix elements and
            # widths of different states; KPoint.degen hands them only the
            # parts that still take that rule
            for D, R, _ in resolved:
                V[:, D] = V[:, D] @ R
            delta = adaptive_widths(vel, afac, dk, min_width=min_width, max_width=max_width)
            if len(delta):
                dmax = max(dmax, float(np.max(delta)))
            kp = KPoint(
                sparse_h,
                ik,
                ispin,
                kloc[ik],
                E,
                V,
                hk,
                dhk,
                vel,
                delta,
                bnd,
                afac,
                dk,
                sign=-1,
                kglobal=int(kglobal[ik]),
                dense_operators=dense_ops,
                resolved=[(D, parts) for D, _, parts in resolved],
                min_width=min_width,
                max_width=max_width,
            )
            if curvature:
                d2hk = assembled[2]
                if dense_ops:
                    d2hk = DenseOnDemand(d2hk, cache=False)
                kp.__dict__['d2hk'] = d2hk
                d2Ed2k[:, ik, :bnd, ispin] = band_curvature_k(
                    E, dhk, d2hk, V, kp.degen, bnd, degenerate_groups(E, bnd)
                )

            if interior is None:
                solved_top = min(solved_top, float(E[nsolved - 1]))
                if ehi is not None and E[nsolved - 1] < ehi:
                    deficit += 1
                E_k[ik, :bnd, ispin] = E[:bnd]
                velkp[ik, :, :bnd, ispin] = vel
                deltakp[ik, :bnd, ispin] = delta
            else:
                acc[(ispin, ik)] = (E[:bnd], vel, delta)

            for c in consumers:
                c.activate()
                c.on_k(kp)
            kp = None

            if verbose and (ik + 1) % step == 0:
                log.field('progress', f'{ik + 1}/{nk_local} local k-points')

    dropped = comm.allreduce(dropped, op=MPI.SUM)
    if ptop is not None:
        log.field(
            'states dropped at the PAO shift',
            '%.2f per k-point' % (dropped / float(comm.allreduce(nk_local * nspin, op=MPI.SUM))),
        )
    if interior is not None:
        E_k, velkp, deltakp = _pad_interior(acc, nk_local, nspin, interior[1], log)
        attr['bnd'] = E_k.shape[1]
    attr['sparse_dmax'] = float(comm.allreduce(dmax, op=MPI.MAX))
    log.field('max adaptive width (eV)', '%.3f' % attr['sparse_dmax'])

    arrays['E_k'] = E_k
    arrays['velkp'] = velkp
    arrays['deltakp'] = deltakp
    if curvature:
        arrays['d2Ed2k'] = d2Ed2k
    if interior is None:
        # how high every k-point's solve reached, placeholders included: the
        # last column of E_k is padding wherever states were dropped
        attr['sparse_solved_top'] = float(comm.allreduce(solved_top, op=MPI.MIN))

    if interior is None:
        check_window_coverage(deficit, nev, ehi, 'mesh')


def _pad_interior(
    acc: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    nk_local: int,
    nspin: int,
    ehi: float,
    log: SparseLog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad per-k interior results ``acc[(ispin, ik)] = (E, vel, delta)`` to rectangular
    ``E_k``, ``velkp``, ``deltakp`` in the dense layout, padding values as in :func:`run_mesh`.

    The width ``m`` is the max over *all* ranks, so a later ``gather_full`` never straddles two
    band counts. Raises ``RuntimeError`` if no rank found any state (window likely outside the
    spectrum rather than in a gap).
    """
    local_max = max((len(E) for (E, _, _) in acc.values()), default=0)
    m = int(comm.allreduce(int(local_max), op=MPI.MAX))
    local_min = min((len(E) for (E, _, _) in acc.values()), default=0)
    log.field(
        'interior states per k-point', f'{local_min} to {local_max} on this rank; padded to {m}'
    )
    if m == 0:
        raise RuntimeError(
            'sparse mesh: the interior window contains no states at any k-point on any '
            'rank. Widen the window, or check that it straddles the energy range you '
            'meant (the PAO zero of energy is E_F).'
        )

    E_k = np.full((nk_local, m, nspin), ehi + PAD_ENERGY_OFFSET, dtype=float)
    velkp = np.zeros((nk_local, 3, m, nspin), dtype=float)
    deltakp = np.full((nk_local, m, nspin), PAD_DELTA, dtype=float)
    for (ispin, ik), (E, vel, delta) in acc.items():
        j = len(E)
        E_k[ik, :j, ispin] = E
        velkp[ik, :, :j, ispin] = vel
        deltakp[ik, :j, ispin] = delta
    return E_k, velkp, deltakp


def check_window_coverage(deficit: int, nev: int, ehi: float | None, tag: str) -> None:
    """Raise ``RuntimeError`` if any k-point's ``nev``-band solve stopped below ``ehi`` (eV).

    ``deficit`` is the local count of such k-points; ``ehi=None`` disables the check; ``tag``
    names the stage. Collective (allreduce), so all ranks raise together instead of some hanging
    in a later reduction. The message names the ``nev`` to re-run with.
    """
    if ehi is None:
        return
    total = comm.allreduce(int(deficit), op=MPI.SUM)
    if total:
        suggested = nev + max(8, int(0.1 * nev) + 1)
        raise RuntimeError(
            f'sparse {tag}: {total} k-point(s) had their highest computed band below the '
            f'window top {ehi:.3f} eV, so the {nev}-band solve does not cover it. Re-run with '
            f"'nev': {suggested} in sparse_config. No results were truncated silently."
        )


def probe_k(
    sparse_h: SparseHamiltonian,
    kfrac: np.ndarray,
    ispin: int,
    ehi: float | None,
    interior: tuple[float, float] | None = None,
    hk_solver: str = 'auto',
    dense_n_max: int = DENSE_N_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and band-velocity norms at one probe k-point.

    Parameters
    ----------
    sparse_h : SparseHamiltonian
    kfrac : np.ndarray, shape (3,)
        Crystal coordinates (mesh convention, ``sign=-1``).
    ispin : int
    ehi : float or None
        Solve every state below this energy (eV); ``None`` solves all. Ignored with ``interior``.
    interior : (float, float) or None, optional
        Solve only the states inside this window instead.
    hk_solver, dense_n_max : optional
        As :func:`run_mesh`.

    Returns
    -------
    (E, vnorm) : (np.ndarray, np.ndarray)
        Ascending eigenvalues (eV) and ``|v_n|``.

    Raises
    ------
    NotImplementedError
        From the bottom with ``n > dense_n_max`` (the probe densifies H(k)).

    Notes
    -----
    The engine sizes a pass from it: the count below the window top gives ``nev``, the largest
    ``|v_n|`` the largest width ``afac * dk * |v_n|`` and hence the smearing margin.
    """
    import scipy.linalg

    from .solver import solve_interior

    hk, dhk = sparse_h.assemble_derivatives(kfrac, ispin=ispin, sign=-1, order=1)[:2]
    if interior is not None:
        E, V = solve_interior(
            hk, interior[0], interior[1], hk_solver=hk_solver, dense_n_max=dense_n_max
        )
    else:
        n = hk.shape[0]
        if n > dense_n_max:
            raise NotImplementedError(
                f'sparse window probe: counting the states below the window top densifies '
                f'H(k), which at n = {n} would need {16.0 * n * n / 1024**3:.2f} GB '
                f"(dense_n_max = {dense_n_max}). Give 'nev' in sparse_config to skip the probe."
            )
        A = hk.toarray()
        if ehi is None:
            E, V = scipy.linalg.eigh(A, driver='evr', check_finite=False, overwrite_a=True)
        else:
            E, V = scipy.linalg.eigh(
                A,
                subset_by_value=(-np.inf, float(ehi)),
                driver='evr',
                check_finite=False,
                overwrite_a=True,
            )
    if len(E) == 0:
        return np.asarray(E, dtype=float), np.zeros(0)
    vel, _ = _band_diagonal_velocities(E, V, dhk, dense_n_max=dense_n_max)
    return np.asarray(E, dtype=float), np.linalg.norm(np.real(vel), axis=0)
