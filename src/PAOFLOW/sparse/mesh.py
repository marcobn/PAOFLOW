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


def run_mesh(
    data_controller,
    sparse_h,
    nev,
    consumers=(),
    afac=None,
    smearing='gauss',
    verbose=False,
    method='auto',
    ehi=None,
):
    """Fused mesh pass.

    ``method`` selects the eigensolver branch once for the whole loop (see
    :func:`~PAOFLOW.sparse.solver.select_method`); it depends only on
    ``(nawf, nev)``, so it cannot change from k-point to k-point.

    ``ehi`` (eV), when given, is the top of the energy window the caller
    sized ``nev`` for.  Every k-point whose highest computed band falls
    below it is counted, and a non-zero global count raises after the loop
    with the ``nev`` needed to re-run — no silent truncation, and no
    whole-mesh redo triggered from inside the loop.
    """
    from ..spectrum.do_eigh import get_degeneracies
    from ..utils.communication import scatter_full
    from ..utils.get_K_grid_fft import get_K_grid_fft_crystal
    from .solver import describe_method, solve_lowest

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

    if rank == 0:
        print(describe_method(sparse_h.nawf, nev, method=method), flush=True)

    E_k = np.zeros((nk_local, nev, nspin), dtype=float)
    velkp = np.zeros((nk_local, 3, nev, nspin), dtype=float)
    deltakp = np.zeros((nk_local, nev, nspin), dtype=float)
    deficit = 0
    # a fixed interval of 100 prints nothing at all on a coarse supercell mesh
    # (216 k-points over 4 ranks is 54 each), which is exactly the run that
    # takes long enough to want progress
    step = max(1, min(100, nk_local // 10))

    for ispin in range(nspin):
        v0 = None
        for ik in range(nk_local):
            hk, dhk = sparse_h.assemble_hk_dhk(kloc[ik], ispin=ispin, sign=-1)
            E, V = solve_lowest(hk, nev, v0=v0, method=method)
            v0 = np.ascontiguousarray(V[:, 0])

            W = [dhk[l] @ V for l in range(3)]  # (nawf, nev) each
            vel = np.empty((3, nev), dtype=float)
            for l in range(3):
                vel[l] = np.einsum('an,an->n', np.conj(V), W[l]).real

            # degenerate groups: replace diagonal entries by the ascending
            # eigenvalues of the group block (perturb_split convention).
            # (dhk_l @ V)[:, D] == dhk_l @ V[:, D] exactly, so the group
            # block reuses W instead of re-running the sparse matmul.
            degen = get_degeneracies(E[None, :, None], nev)[0][0]
            for D in degen:
                VD = V[:, D]
                for l in range(3):
                    block = VD.conj().T @ W[l][:, D]
                    block = 0.5 * (block + block.conj().T)
                    vel[l][D] = np.linalg.eigvalsh(block)

            if ehi is not None and E[-1] < ehi:
                deficit += 1

            delta = afac * dk * np.linalg.norm(vel, axis=0)  # (nev,)

            E_k[ik, :, ispin] = E
            velkp[ik, :, :, ispin] = vel
            deltakp[ik, :, ispin] = delta
            for c in consumers:
                c.on_k(ik, ispin, E, V, vel, delta)
            # V goes out of scope here — never stored

            if verbose and rank == 0 and (ik + 1) % step == 0:
                print('Sparse mesh progress: %d/%d local k-points' % (ik + 1, nk_local), flush=True)

    arrays['E_k'] = E_k
    arrays['velkp'] = velkp
    arrays['deltakp'] = deltakp

    check_window_coverage(deficit, nev, ehi, 'mesh')

    for c in consumers:
        c.finalize(data_controller)


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
