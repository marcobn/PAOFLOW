"""Fused BZ-mesh pass: eigenpairs, velocities, adaptive widths, consumers.

One loop over the local share of the full k-mesh produces everything the
band-diagonal property kernels need, without ever storing eigenvectors:

- ``E_k``       (nk_local, nev, nspin)  — lowest ``nev`` eigenvalues,
- ``velkp``     (nk_local, 3, nev, nspin) — Hellmann-Feynman band-diagonal
  velocities  v_ln = <n| dH/dk_l |n>, with degenerate groups resolved as
  the ascending eigenvalues of the group block (replicating
  ``utils.perturb_split`` as used by ``do_momentum``),
- ``deltakp``   (nk_local, nev, nspin)  — Yates adaptive smearing widths
  ``afac * dk * |v_n|`` (replicating ``do_adaptive_smearing``),

stored k-scattered in the dense conventions so ``do_dos_adaptive`` and the
Boltzmann transport stack consume them unchanged.  Registered consumers
receive ``(ik, ispin, E, V, vel, delta)`` per k-point and must not retain
``V`` — the ``(nawf, nev)`` eigenvector block is the only dense workspace
and is discarded before the next k-point.

The mesh phase convention is ``sign=-1`` (the dense ``fftn`` convention),
and the k ordering matches the dense pipeline's FFT-grid linearization
``n = k + j*nk3 + i*nk2*nk3``, so per-k quantities are index-comparable
with the dense arrays.
"""

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Sentinels for the padding an interior window needs (its band count varies
# from k-point to k-point).  The energy sits far outside any plotted range so
# every smearing kernel evaluates to exactly zero there, the velocity is zero
# so the state carries no transport weight, and the width is O(1) rather than
# 0 because the smearing kernels divide by it.
PAD_ENERGY_OFFSET = 1.0e3
PAD_DELTA = 1.0


def _band_diagonal_velocities(E, V, dhk):
    """``v_ln = <n| dH/dk_l |n>`` with degenerate groups resolved.

    Shared by the from-the-bottom and interior paths; the only difference
    between them is how many states arrive here, so ``m = len(E)`` is read
    from the input rather than passed in.  Returns ``(3, m)``.
    """
    from ..spectrum.do_eigh import get_degeneracies

    m = len(E)
    vel = np.empty((3, m), dtype=float)
    if m == 0:
        return vel

    W = [dhk[l] @ V for l in range(3)]  # (nawf, m) each
    for l in range(3):
        vel[l] = np.einsum('an,an->n', np.conj(V), W[l]).real

    # degenerate groups: replace diagonal entries by the ascending
    # eigenvalues of the group block (perturb_split convention).
    # (dhk_l @ V)[:, D] == dhk_l @ V[:, D] exactly, so the group
    # block reuses W instead of re-running the sparse matmul.
    for D in get_degeneracies(E[None, :, None], m)[0][0]:
        VD = V[:, D]
        for l in range(3):
            block = VD.conj().T @ W[l][:, D]
            block = 0.5 * (block + block.conj().T)
            vel[l][D] = np.linalg.eigvalsh(block)
    return vel


def run_mesh(
    data_controller,
    sparse_h,
    nev,
    consumers=(),
    afac=None,
    smearing='gauss',
    verbose=False,
    hk_solver='auto',
    ehi=None,
    interior=None,
):
    """Fused mesh pass.

    ``hk_solver`` selects the per-k kernel once for the whole loop (see
    :func:`~PAOFLOW.sparse.solver.select_hk_solver`); it depends only on
    ``(nawf, nev)``, so it cannot change from k-point to k-point.

    ``ehi`` (eV), when given, is the top of the energy window the caller
    sized ``nev`` for.  Every k-point whose highest computed band falls
    below it is counted, and a non-zero global count raises after the loop
    with the ``nev`` needed to re-run — no silent truncation, and no
    whole-mesh redo triggered from inside the loop.

    ``interior`` (``(elo, ehi)``, eV) switches to the interior solver: every
    state inside the window, none below it.  The count is then k-dependent by
    construction, so the per-k results are accumulated and padded to the
    global maximum afterwards (see ``PAD_ENERGY_OFFSET``); ``nev`` is ignored
    and ``attr['bnd']`` is set to that maximum.  ``solve_interior`` guarantees
    completeness inside the window or raises, so the ``ehi`` deficit counter
    does not apply and is skipped.
    """
    from ..utils.communication import scatter_full
    from ..utils.get_K_grid_fft import get_K_grid_fft_crystal
    from .log import get_sparse_log
    from .solver import describe_hk_solver, solve_interior, solve_lowest

    arrays, attr = data_controller.data_dicts()
    nk1, nk2, nk3 = attr['nk1'], attr['nk2'], attr['nk3']
    attr['nkpnts'] = nkpnts = nk1 * nk2 * nk3

    kfrac_all = get_K_grid_fft_crystal(nk1, nk2, nk3)  # (nktot, 3), dense ordering
    kloc = scatter_full(kfrac_all, attr['npool'])
    nk_local = kloc.shape[0]
    nspin = sparse_h.nspin

    # Yates widths, as in do_adaptive_smearing
    dk = (8.0 * np.pi**3 / attr['omega'] / nkpnts) ** (1.0 / 3.0)
    if afac is None:
        afac = 1.0 if smearing == 'm-p' else 0.7

    log = get_sparse_log(data_controller)
    log.section('Mesh pass (eigenvalues + velocities%s)' % (' + PDOS' if consumers else ''))
    log.field('mesh', '%d x %d x %d  (%d k-points)' % (nk1, nk2, nk3, nkpnts))
    if interior is None:
        log.field('bands per k (nev)', nev)
    else:
        log.field('interior window (eV)', '[%.3f, %.3f]' % (interior[0], interior[1]))
        log.field('bands per k', 'k-dependent (interior solve); padded after the loop')
    log.field('smearing', '%s, afac = %.3f' % (smearing, afac))
    log.field('window top ehi (eV)', 'none' if ehi is None else '%.3f' % ehi)
    log.write(describe_hk_solver(sparse_h.nawf, nev, hk_solver=hk_solver))

    # from-the-bottom: rectangular by construction, so preallocate as before.
    # interior: the count is k-dependent, so collect and pad afterwards.
    if interior is None:
        E_k = np.zeros((nk_local, nev, nspin), dtype=float)
        velkp = np.zeros((nk_local, 3, nev, nspin), dtype=float)
        deltakp = np.zeros((nk_local, nev, nspin), dtype=float)
    else:
        acc = {}
    deficit = 0
    # a fixed interval of 100 prints nothing at all on a coarse supercell mesh
    # (216 k-points over 4 ranks is 54 each), which is exactly the run that
    # takes long enough to want progress
    step = max(1, min(100, nk_local // 10))

    for ispin in range(nspin):
        v0 = None
        k0 = None
        for ik in range(nk_local):
            hk, dhk = sparse_h.assemble_hk_dhk(kloc[ik], ispin=ispin, sign=-1)
            if interior is None:
                E, V = solve_lowest(hk, nev, v0=v0, hk_solver=hk_solver)
                v0 = np.ascontiguousarray(V[:, 0])
            else:
                # the neighbouring k-point's count is a good Krylov estimate,
                # the same idea as the v0 warm start on the other branch
                E, V = solve_interior(hk, interior[0], interior[1], k0=k0, hk_solver=hk_solver)
                k0 = max(8, 2 * len(E))

            vel = _band_diagonal_velocities(E, V, dhk)
            delta = afac * dk * np.linalg.norm(vel, axis=0)  # (m,)

            if interior is None:
                if ehi is not None and E[-1] < ehi:
                    deficit += 1
                E_k[ik, :, ispin] = E
                velkp[ik, :, :, ispin] = vel
                deltakp[ik, :, ispin] = delta
            else:
                acc[(ispin, ik)] = (E, vel, delta)

            for c in consumers:
                c.on_k(ik, ispin, E, V, vel, delta)
            # V goes out of scope here — never stored

            if verbose and (ik + 1) % step == 0:
                log.write('  progress: %d/%d local k-points' % (ik + 1, nk_local))

    if interior is not None:
        E_k, velkp, deltakp = _pad_interior(acc, nk_local, nspin, interior[1], log)
        attr['bnd'] = E_k.shape[1]
        # Largest *real* adaptive width, padding excluded.  Consumers need it
        # to know how far a smearing tail reaches past the window edge: a state
        # just below elo still contributes inside the window, so a DoS plotted
        # too close to the edge is contaminated by states that were never
        # computed.  Recorded here because only the mesh knows the widths.
        local = max((float(d.max()) for (_, _, d) in acc.values() if len(d)), default=0.0)
        attr['sparse_interior_dmax'] = float(comm.allreduce(local, op=MPI.MAX))

    arrays['E_k'] = E_k
    arrays['velkp'] = velkp
    arrays['deltakp'] = deltakp

    if interior is None:
        check_window_coverage(deficit, nev, ehi, 'mesh')

    for c in consumers:
        c.finalize(data_controller)


def _pad_interior(acc, nk_local, nspin, ehi, log):
    """Pad k-dependent interior results to one rectangular block.

    The width is the maximum over *all* ranks, not the local one, so the
    scattered arrays stay index-comparable with the dense conventions and a
    later ``gather_full`` cannot straddle two different band counts.
    """
    local_max = max((len(E) for (E, _, _) in acc.values()), default=0)
    m = int(comm.allreduce(int(local_max), op=MPI.MAX))
    log.write(
        '  interior window: %d-%d states per k locally, padded to %d (global max)'
        % (min((len(E) for (E, _, _) in acc.values()), default=0), local_max, m)
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


def check_window_coverage(deficit, nev, ehi, tag):
    """Raise if any k-point's computed spectrum stopped below the window.

    Collective: the local deficit counts are summed across ranks, so
    every rank raises together rather than one rank hanging in a later
    reduction.  The message names the exact ``nev`` to re-run with.
    """
    if ehi is None:
        return
    total = comm.allreduce(int(deficit), op=MPI.SUM)
    if total:
        raise RuntimeError(
            'sparse %s: %d k-point(s) had their highest computed band below the requested '
            'window top ehi = %.3f eV, so the %d-band solve does not cover it. Re-run with a '
            'larger nev (energy_window(..., nev=%d) or a wider margin). No results were '
            'truncated silently.' % (tag, total, ehi, nev, nev + max(8, int(0.1 * nev) + 1))
        )
