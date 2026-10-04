"""Fused BZ-mesh pass: eigenpairs, velocities, adaptive widths, consumers.

One loop over the local share of the full k-mesh produces everything the
band-diagonal property kernels need, without ever storing eigenvectors:

- ``E_k``       (nk_local, nev, nspin)  — lowest ``nev`` eigenvalues,
- ``velkp``     (nk_local, 3, nev, nspin) — Hellmann-Feynman band-diagonal
  velocities  v_ln = <n| dH/dk_l |n>, with degenerate groups resolved as
  the ascending eigenvalues of the group block (replicating
  ``utils.perturb_split`` as used by ``do_momentum``),
- ``deltakp``   (nk_local, nev, nspin)  — Yates adaptive smearing widths
  ``afac * dk * |v_n|`` (``do_adaptive_smearing.adaptive_widths``),
- ``d2Ed2k``    (6, nk_local, nev, nspin) — band curvature, only when the
  ``'d2Ed2k'`` product is requested (``do_band_curvature.band_curvature_k_ij``),

stored k-scattered in the dense conventions so ``do_dos_adaptive``, the
Boltzmann transport stack and ``do_effective_mass`` consume them unchanged.
Consumers (:class:`~PAOFLOW.sparse.properties.MeshProperty`) receive one
:class:`~PAOFLOW.sparse.kpoint.KPoint` per k-point and must not retain it:
the eigenvector block is the only dense workspace and is discarded before
the next k-point.

The mesh phase convention is ``sign=-1`` (the dense ``fftn`` convention),
and the k ordering matches the dense pipeline's FFT-grid linearization
``n = k + j*nk3 + i*nk2*nk3``, so per-k quantities are index-comparable
with the dense arrays.

``PAD_ENERGY_OFFSET`` and ``PAD_DELTA`` are the sentinel energy offset and
smearing width used to pad an interior-window solve to a rectangular block;
:func:`run_mesh` explains why they take these values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from mpi4py import MPI

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

    from .hamiltonian import SparseHamiltonian
    from .log import SparseLog, _NullLog
    from .properties import MeshProperty

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PAD_ENERGY_OFFSET = 1.0e3
PAD_DELTA = 1.0


def _band_diagonal_velocities(E: np.ndarray, V: np.ndarray, dhk: Sequence[Any]) -> np.ndarray:
    """Band velocities ``v_ln = <n| dH/dk_l |n>``, degeneracies resolved.

    Parameters
    ----------
    E : np.ndarray, shape (m,)
        Eigenvalues at this k-point (eV), ascending.
    V : np.ndarray, shape (nawf, m)
        Matching eigenvector block, orthonormal.
    dhk : sequence of scipy.sparse matrix
        The three Cartesian derivatives ``dH/dk_l`` at this k-point.

    Returns
    -------
    np.ndarray, shape (3, m)
        Band-diagonal velocity of each state along each Cartesian
        direction.

    Notes
    -----
    By the Hellmann-Feynman theorem the group velocity of band :math:`n` is
    the expectation value of the Hamiltonian's k-derivative in that band's
    own state, :math:`v_{ln} = \\langle n | \\partial H / \\partial k_l | n
    \\rangle` — no derivative of the eigenvector is needed.

    That formula is ambiguous when states are degenerate, because any
    rotation within a degenerate subspace is an equally valid eigenbasis and
    the diagonal elements change under it.  The physically meaningful
    velocities are the eigenvalues of the velocity operator *restricted to
    the degenerate subspace*, so for each degenerate group the small block
    :math:`V_D^\\dagger (\\partial H/\\partial k_l) V_D` is formed,
    Hermitized against round-off, and diagonalized; its ascending
    eigenvalues replace the diagonal entries.  This is the same convention
    the dense pipeline applies through ``utils.perturb_split``, and it
    matters at exactly the folded supercells this backend targets, where
    multiplets are the rule.

    The products ``(dH/dk_l) V`` are computed once for the whole block and
    reused inside the degeneracy loop, since restricting to a group is just
    a column selection of the same product.  The routine is shared by the
    from-the-bottom and interior solves; they differ only in how many states
    arrive, which is read from ``E``.
    """
    from ..spectrum.do_eigh import get_degeneracies

    m = len(E)
    vel = np.empty((3, m), dtype=float)
    if m == 0:
        return vel

    W = [dhk[l] @ V for l in range(3)]
    for l in range(3):
        vel[l] = np.einsum('an,an->n', np.conj(V), W[l]).real

    for D in get_degeneracies(E[None, :, None], m)[0][0]:
        VD = V[:, D]
        for l in range(3):
            block = VD.conj().T @ W[l][:, D]
            block = 0.5 * (block + block.conj().T)
            vel[l][D] = np.linalg.eigvalsh(block)
    return vel


def run_mesh(
    data_controller: DataController,
    sparse_h: SparseHamiltonian,
    nev: int,
    consumers: Sequence[MeshProperty] = (),
    products: Sequence[str] = (),
    afac: float | None = None,
    smearing: str = 'gauss',
    verbose: bool = False,
    hk_solver: str = 'auto',
    ehi: float | None = None,
    interior: tuple[float, float] | None = None,
) -> None:
    """Walk the local share of the BZ mesh, producing all band-diagonal data.

    Parameters
    ----------
    data_controller : DataController
        Run state.  Supplies the mesh dimensions and cell volume, and
        receives ``arrays['E_k']``, ``arrays['velkp']``, ``arrays['deltakp']``
        and each requested product — this rank's k-slice, in the dense
        layout.
    sparse_h : SparseHamiltonian
        Bond list the Bloch Hamiltonian and its derivatives are assembled
        from at each k-point.
    nev : int
        Number of lowest bands per k-point kept in the stored arrays (the
        window).  Ignored when ``interior`` is given.
    consumers : sequence of MeshProperty, optional
        Streaming properties fed one :class:`~PAOFLOW.sparse.kpoint.KPoint`
        at a time.  They are *not* finalized here; the engine does that, in
        call order, after the pass.
    products : sequence of str, optional
        Extra band-diagonal arrays to store.  ``'d2Ed2k'`` is the only one:
        the band curvature ``(6, nk_local, nev, nspin)``.
    afac : float or None, optional
        Prefactor of the adaptive smearing width.  Defaults to the value
        the dense pipeline uses for the chosen ``smearing``.
    smearing : str, optional
        Smearing kernel name; sets the default ``afac``.
    verbose : bool, optional
        Log progress every few percent of this rank's k-points.
    hk_solver : {'auto', 'sparse', 'dense'}, optional
        Per-k-point kernel, selected once for the whole loop (see
        :func:`~PAOFLOW.sparse.solver.select_hk_solver`).  It depends only
        on ``(nawf, nev)``, so it cannot change from k-point to k-point.
        A full-spectrum pass always uses ``'dense'``.
    ehi : float or None, optional
        Top of the energy window ``nev`` was sized for (eV).  Every k-point
        whose highest window band falls below it is counted, and a
        non-zero global count raises after the loop with the ``nev`` needed
        to re-run — no silent truncation, and no whole-mesh redo triggered
        from inside the loop.
    interior : (float, float) or None, optional
        Energy window ``(elo, ehi)`` in eV.  Switches to the interior
        solver: every state inside the window, none below it.

    Returns
    -------
    None
        Results are written into ``data_controller`` arrays and pushed to
        the consumers.

    Raises
    ------
    NotImplementedError
        If the full spectrum is needed (a ``'full_spectrum'`` consumer, or
        the ``'d2Ed2k'`` product) and ``nawf`` exceeds ``DENSE_N_MAX``.
    ValueError
        If the full spectrum is needed together with ``interior``, or an
        unknown product is requested.

    Notes
    -----
    Every band-diagonal property in the pipeline — density of states,
    projected DOS, Boltzmann transport — is a Brillouin-zone sum over
    quantities that depend on one band at one k-point: its energy, its
    velocity, and the width of the smearing that stands in for the delta
    function.  The dense pipeline computes those in separate passes, each
    reading a stored eigenvector tensor.  Here they are produced in a
    single fused loop, so the eigenvectors never outlive the k-point that
    produced them.

    Three things come out of each k-point.  The eigenvalues are the band
    energies.  The band-diagonal velocities follow from the Hellmann-Feynman
    theorem (see :func:`_band_diagonal_velocities`) using the same sparse
    assembly of :math:`\\partial H/\\partial k` that gives :math:`H(k)`.
    The adaptive smearing width is the Yates estimate ``afac * dk * |v_n|``:
    the energy of a band changes by roughly its velocity times the k-point
    spacing between one mesh point and the next, so a band that disperses
    steeply must be smeared more widely than a flat one to make a discrete
    mesh sum approximate a continuous integral.

    Interband quantities — the band curvature, Berry curvatures, optical
    matrix elements — sum over pairs of states that include every band, not
    only the window.  A consumer that needs them declares
    ``'full_spectrum'``, and the pass then solves all ``nawf`` states per
    k-point with the dense kernel, exactly as the dense pipeline's ``eigh``
    does; the stored arrays still keep only the window ``E[:nev]``.  This is
    the per-k ``(nawf, nawf)`` scratch the memory contract admits up to
    ``DENSE_N_MAX``; above it there is no exact alternative here (a
    Sternheimer solve for the interband sums is the intended follow-up), so
    the pass refuses rather than truncating the sum.

    Consecutive mesh points are close in k, so each solve is seeded from
    its predecessor: the from-the-bottom branch reuses the previous ground
    state as the starting vector, and the interior branch reuses the
    previous state count as its Krylov size estimate.  Both are convergence
    aids only and cannot change the converged result.

    Which states are computed depends on the mode.  From the bottom, each
    k-point returns exactly ``nev`` states and the output arrays are
    rectangular by construction.  With an ``interior`` window, the number of
    states inside the window is k-dependent (a band crossing an edge is in
    at one k-point and out at the next), so the per-k results are collected
    and padded to the global maximum afterwards; ``attr['bnd']`` is set to
    that maximum.  The padding is deliberately not neutral-looking: the
    energy sits far outside any plotted range so every smearing kernel
    evaluates to zero there, the velocity is zero so the state carries no
    transport weight, and the width is ``O(1)`` rather than zero only
    because the smearing kernels divide by it.  ``solve_interior``
    guarantees completeness inside the window or raises, so the ``ehi``
    deficit counter does not apply there and is skipped.

    An interior solve also records ``attr['sparse_interior_dmax']``, the
    largest *real* adaptive width over the whole mesh with padding excluded.
    Consumers need it to know how far a smearing tail reaches past the
    window edge: a state just below ``elo`` still contributes inside the
    window, so a DOS plotted too close to the edge is contaminated by states
    that were never computed.  It is recorded here because only the mesh
    pass knows the widths.

    Progress logging uses an interval of one tenth of this rank's k-points,
    capped at 100.  A fixed interval of 100 would print nothing at all on a
    coarse supercell mesh (216 k-points over four ranks is 54 each), which
    is exactly the run slow enough to want progress from.
    """
    from ..hamiltonian.do_d2Hd2k import IJ_PAIRS
    from ..spectrum.do_adaptive_smearing import adaptive_widths
    from ..spectrum.do_band_curvature import band_curvature_k_ij
    from ..utils.communication import scatter_full
    from ..utils.get_K_grid_fft import get_K_grid_fft_crystal
    from .kpoint import KPoint
    from .log import get_sparse_log
    from .solver import DENSE_N_MAX, describe_hk_solver, solve_interior, solve_lowest

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
        if nawf > DENSE_N_MAX:
            raise NotImplementedError(
                f'sparse mesh: {", ".join(wanting)} sum over interband pairs that include every '
                f'state, which needs the full spectrum per k-point (nawf = {nawf}). That is a '
                f'dense (nawf, nawf) solve, admitted only up to DENSE_N_MAX = {DENSE_N_MAX}, '
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

    extras = sorted({c.method for c in consumers}) + (['band curvature'] if curvature else [])
    log = get_sparse_log(data_controller)
    log.section('Mesh pass (eigenvalues + velocities%s)' % ''.join(' + ' + name for name in extras))
    log.field('mesh', f'{nk1} x {nk2} x {nk3}  ({nkpnts} k-points)')
    if interior is None:
        log.field('bands per k (nev)', nev)
    else:
        log.field('interior window (eV)', f'[{interior[0]:.3f}, {interior[1]:.3f}]')
        log.field('bands per k', 'k-dependent (interior solve); padded after the loop')
    log.field('smearing', f'{smearing}, afac = {afac:.3f}')
    log.field('window top ehi (eV)', 'none' if ehi is None else f'{ehi:.3f}')
    if full:
        log.field(
            'full spectrum',
            f'all {nawf} states per k-point for {", ".join(wanting)}; window keeps {nev}',
        )
    log.write(describe_hk_solver(nawf, nsolve, hk_solver=solver))

    if interior is None:
        E_k = np.zeros((nk_local, nev, nspin), dtype=float)
        velkp = np.zeros((nk_local, 3, nev, nspin), dtype=float)
        deltakp = np.zeros((nk_local, nev, nspin), dtype=float)
        if curvature:
            d2Ed2k = np.zeros((6, nk_local, nev, nspin), dtype=float)
    else:
        acc = {}
    deficit = 0
    step = max(1, min(100, nk_local // 10))

    for ispin in range(nspin):
        v0 = None
        k0 = None
        for ik in range(nk_local):
            assembled = sparse_h.assemble_derivatives(
                kloc[ik], ispin=ispin, sign=-1, order=2 if curvature else 1
            )
            hk, dhk = assembled[0], assembled[1]
            if interior is None:
                E, V = solve_lowest(hk, nsolve, v0=v0, hk_solver=solver)
                v0 = np.ascontiguousarray(V[:, 0])
                bnd = nev
            else:
                E, V = solve_interior(hk, interior[0], interior[1], k0=k0, hk_solver=hk_solver)
                k0 = max(8, 2 * len(E))
                bnd = len(E)

            vel = _band_diagonal_velocities(E[:bnd], V[:, :bnd], dhk)
            delta = adaptive_widths(vel, afac, dk)
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
            )
            if curvature:
                d2hk = kp.__dict__['d2hk'] = assembled[2]
                for ij, (i, j) in enumerate(IJ_PAIRS):
                    d2Ed2k[ij, ik, :, ispin] = band_curvature_k_ij(
                        E, dhk[i], dhk[j], d2hk[ij], V, kp.degen, bnd
                    )

            if interior is None:
                if ehi is not None and E[bnd - 1] < ehi:
                    deficit += 1
                E_k[ik, :, ispin] = E[:bnd]
                velkp[ik, :, :, ispin] = vel
                deltakp[ik, :, ispin] = delta
            else:
                acc[(ispin, ik)] = (E, vel, delta)

            for c in consumers:
                c.activate()
                c.on_k(kp)
            kp = None

            if verbose and (ik + 1) % step == 0:
                log.write(f'  progress: {ik + 1}/{nk_local} local k-points')

    if interior is not None:
        E_k, velkp, deltakp = _pad_interior(acc, nk_local, nspin, interior[1], log)
        attr['bnd'] = E_k.shape[1]
        local = max((float(d.max()) for (_, _, d) in acc.values() if len(d)), default=0.0)
        attr['sparse_interior_dmax'] = float(comm.allreduce(local, op=MPI.MAX))

    arrays['E_k'] = E_k
    arrays['velkp'] = velkp
    arrays['deltakp'] = deltakp
    if curvature:
        arrays['d2Ed2k'] = d2Ed2k

    if interior is None:
        check_window_coverage(deficit, nev, ehi, 'mesh')


def _pad_interior(
    acc: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    nk_local: int,
    nspin: int,
    ehi: float,
    log: SparseLog | _NullLog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad k-dependent interior results into one rectangular block.

    Parameters
    ----------
    acc : dict
        Per-k results, keyed by ``(ispin, ik)`` and holding
        ``(E, vel, delta)`` with a k-dependent number of states.
    nk_local : int
        Number of k-points on this rank.
    nspin : int
        Number of spin channels.
    ehi : float
        Top of the interior window (eV); the padding energy is placed far
        above it.
    log : SparseLog or _NullLog
        Logger for the achieved and padded state counts.

    Returns
    -------
    (E_k, velkp, deltakp) : tuple of np.ndarray
        Rectangular arrays in the dense layout, ``(nk_local, m, nspin)``,
        ``(nk_local, 3, m, nspin)`` and ``(nk_local, m, nspin)``.

    Raises
    ------
    RuntimeError
        If no k-point on any rank found a state in the window, which almost
        always means the window was placed outside the spectrum rather than
        inside a real gap.

    Notes
    -----
    The width ``m`` is the maximum over *all* ranks, not the local one, so
    the scattered arrays stay index-comparable with the dense conventions
    and a later ``gather_full`` cannot straddle two different band counts.
    See :func:`run_mesh` for what the padding values mean and why they are
    chosen to be inert rather than zero.
    """
    local_max = max((len(E) for (E, _, _) in acc.values()), default=0)
    m = int(comm.allreduce(int(local_max), op=MPI.MAX))
    local_min = min((len(E) for (E, _, _) in acc.values()), default=0)
    log.write(
        f'  interior window: {local_min}-{local_max} states per k locally, '
        f'padded to {m} (global max)'
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
    """Raise if any k-point's computed spectrum stopped below the window.

    Parameters
    ----------
    deficit : int
        Number of local k-points whose highest computed band fell below
        ``ehi``.
    nev : int
        Band count the solve was run with, quoted in the message.
    ehi : float or None
        Top of the requested energy window (eV).  ``None`` disables the
        check entirely.
    tag : str
        Name of the calling stage, for the message.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If any rank reported a deficit.

    Notes
    -----
    A from-the-bottom solve returns a fixed number of bands, but how far up
    in energy those bands reach is k-dependent.  If the requested window
    extends above the highest computed band at some k-point, every property
    integrated over that window is quietly missing states there — the kind
    of error that produces a plausible plot with the wrong answer.

    The check is collective: the local counts are summed across ranks, so
    every rank raises together rather than one rank raising while the others
    hang in a later reduction.  The message names the exact ``nev`` to
    re-run with.
    """
    if ehi is None:
        return
    total = comm.allreduce(int(deficit), op=MPI.SUM)
    if total:
        suggested = nev + max(8, int(0.1 * nev) + 1)
        raise RuntimeError(
            f'sparse {tag}: {total} k-point(s) had their highest computed band below the '
            f'requested window top ehi = {ehi:.3f} eV, so the {nev}-band solve does not cover '
            f'it. Re-run with a larger nev (energy_window(..., nev={suggested}) or a wider '
            'margin). No results were truncated silently.'
        )
