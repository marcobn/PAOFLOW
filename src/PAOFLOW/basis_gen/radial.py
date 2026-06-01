"""Radial Schroedinger solver for norm-conserving pseudopotentials.

The pseudo-atom Hamiltonian acting on the radial part u(r) = r R(r) of
psi_{lm}(r) = R(r) Y_{lm} is, for a given (l, j) channel,

    H u(r) = -1/2 u''(r) + [V_loc(r) + l(l+1)/(2 r^2)] u(r)
             + r * sum_{i,j in (l,j)} beta_i(r) D_ij <beta_j | R>

where <beta_j | R> = int beta_j(r) R(r) r^2 dr.  Using the UPF-stored
quantity a_i(r) = r * beta_i(r) the nonlocal kernel in u-space becomes
the symmetric outer product

    M(r, r') = sum_{i,j} a_i(r) D_ij a_j(r') .

We discretise on a uniform mesh r_k = k * dr, k = 1, ..., N-1 with
u(0) = u(R_box) = 0, dr = R_box / N.  V_loc and a_i are interpolated
(cubic spline) from the UPF log mesh onto this uniform grid; outside
each projector's cutoff_radius a_i is zero.

For SO UPFs the projectors are filtered by matching (l, j) -- callers
that request (n, l) with j = None on an SO UPF get the j = l + 1/2
channel by default (override via the j argument).

All quantities are in Hartree atomic units.  Output radial functions
are returned in the QE convention used by build_aewfc_basis: the
returned wfc[k] equals r_k * R(r_k) (i.e. u(r_k)), normalised so that
sum_k wfc[k]**2 * dr = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _interp_to_uniform(r_log, f_log, r_uni, cutoff_index=None):
    """Cubic-spline interpolate f(r) defined on r_log onto r_uni.

    Values of r_uni > r_log[-1] (or beyond cutoff_index, when given) are
    set to zero.  Values at r_uni < r_log[0] use the spline extrapolation;
    for NC UPFs r_log[0] is ~1e-5 Bohr so this is usually irrelevant.
    """
    if cutoff_index is not None and cutoff_index < len(r_log):
        r_src = r_log[: cutoff_index + 1]
        f_src = f_log[: cutoff_index + 1]
        r_max_src = r_src[-1]
    else:
        r_src = r_log
        f_src = f_log
        r_max_src = r_log[-1]

    cs = CubicSpline(r_src, f_src, bc_type='natural', extrapolate=False)
    out = np.zeros_like(r_uni)
    mask = (r_uni >= r_src[0]) & (r_uni <= r_max_src)
    out[mask] = cs(r_uni[mask])
    return out


def _select_projectors(upf, l, j):
    """Return indices into upf.beta whose channel matches (l, j).

    For non-SO UPFs j is ignored and all beta with matching l are returned.
    For SO UPFs j must match exactly (within 1e-6).
    """
    idx = []
    for i, b in enumerate(upf.beta):
        if int(b['l']) != int(l):
            continue
        if getattr(upf, 'has_spinorbit', False) and j is not None:
            bj = b.get('j')
            if bj is None or abs(float(bj) - float(j)) > 1e-6:
                continue
        idx.append(i)
    return idx


def _default_j(upf, l, j):
    """If SO UPF and j not provided, pick j = l + 1/2."""
    if not getattr(upf, 'has_spinorbit', False):
        return None
    if j is not None:
        return float(j)
    return float(l) + 0.5 if l > 0 else 0.5


def solve_radial_channel(
    upf,
    l: int,
    j: float | None = None,
    r_box: float | None = None,
    n_points: int = 2000,
    n_states: int = 6,
):
    """Solve H u = eps u for a single (l, j) channel in a box.

    Parameters
    ----------
    upf : PAOFLOW.inputs.read_upf.UPF
        Parsed pseudopotential (norm-conserving).
    l : int
        Orbital angular momentum.
    j : float, optional
        Total angular momentum.  Required (or auto-defaulted) only for
        spin-orbit UPFs.
    r_box : float, optional
        Confining box radius (Bohr).  Default min(upf.r[-1], 10.0).
    n_points : int
        Number of intervals on the uniform mesh; the eigenproblem has
        n_points - 1 interior unknowns.
    n_states : int
        Number of lowest eigenstates to return.

    Returns
    -------
    eps : ndarray, shape (n_states,)
        Eigenvalues in Hartree.
    u : ndarray, shape (n_states, n_points - 1)
        Eigenfunctions u_n(r) = r * R_n(r) on the interior mesh,
        normalised so that sum u_n[k]**2 * dr = 1.
    r : ndarray, shape (n_points - 1,)
        Interior mesh r_k = k * dr.
    """
    from .atomic_potential import frozen_effective_potential

    if r_box is None:
        r_box = float(min(upf.r[-1], 10.0))

    j_used = _default_j(upf, l, j)

    dr = r_box / n_points
    r = np.arange(1, n_points) * dr  # interior points r_1 .. r_{N-1}
    N = r.size

    # Effective local potential: V_loc + V_H[rho_val] + V_xc[rho_val].
    # The Hartree and XC pieces are required because the UPF PSWFC are
    # eigenstates of the *self-consistent* pseudo-atom, not of bare V_loc.
    v_eff_log = frozen_effective_potential(upf)
    vloc_uni = _interp_to_uniform(upf.r, v_eff_log, r)
    # Outside the UPF mesh the neutral pseudo-atom potential decays to 0:
    # V_loc(r) -> -z_val/r is cancelled by V_H(r) -> +z_val/r, and V_xc -> 0.
    r_max_src = upf.r[-1]
    far = r > r_max_src
    if np.any(far):
        vloc_uni[far] = 0.0

    centrifugal = l * (l + 1) / (2.0 * r**2)

    # Kinetic operator -1/2 d^2/dr^2 with Dirichlet BCs (u_0 = u_N = 0).
    inv_dr2 = 1.0 / (dr * dr)
    diag = np.full(N, inv_dr2) + vloc_uni + centrifugal
    offdiag = np.full(N - 1, -0.5 * inv_dr2)

    H = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)

    # Nonlocal KB block (only projectors matching this (l, j) channel).
    pidx = _select_projectors(upf, l, j_used)
    if pidx:
        n_p = len(pidx)
        A = np.zeros((N, n_p))
        for col, ip in enumerate(pidx):
            b = upf.beta[ip]
            A[:, col] = _interp_to_uniform(upf.r, b['wfc'], r, cutoff_index=b.get('cutoff_index'))
        D = upf.dion[np.ix_(pidx, pidx)]
        # Nonlocal matrix in u-space: M = A D A^T * dr  (symmetric).
        H += dr * (A @ D @ A.T)

    # Diagonalise; eigh assumes symmetric.
    eps, V = np.linalg.eigh(H)
    eps = eps[:n_states]
    U = V[:, :n_states].T  # shape (n_states, N)

    # Normalise so that sum U[n, k]**2 * dr = 1.
    norms = np.sqrt(np.sum(U * U, axis=1) * dr)
    U = U / norms[:, None]

    # Sign convention: make the radial function R(r) ~ U(r)/r positive
    # near the origin (matches QE PSWFC convention).
    for n in range(n_states):
        # Use the value at the smallest r (k=0) divided by r_1 > 0.
        if U[n, 0] < 0.0:
            U[n] *= -1.0

    return eps, U, r


def pseudize_shell(
    upf,
    n: int,
    l: int,
    j: float | None = None,
    r_box: float | None = None,
    n_points: int = 2000,
):
    """Return the (n, l[, j]) radial function on a uniform mesh.

    Pseudopotentials do not bind the core states, so the principal
    quantum number cannot be identified with ``node_count + l + 1``
    (the all-electron rule).  Instead the requested state is selected
    as the ``(n - n_lowest)``-th eigenstate of the radial pseudo-atom
    Hamiltonian for this ``(l, j)`` channel, where ``n_lowest`` is

    * the smallest principal quantum number among the UPF PSWFC entries
      with matching ``(l, j)``, if any are present, or
    * the atomic-physics minimum ``n`` for that ``l``
      (``S->1, P->2, D->3, F->4``) otherwise (augmentation channels
      that have no PSWFC counterpart).

    For Pt ONCV-fr ``(l=0)`` PSWFC = ``5S, 6S`` => ``n_lowest = 5``,
    so ``pseudize_shell(upf, 5, 0)`` picks rank 0, ``(6, 0)`` rank 1,
    and ``(7, 0)`` rank 2.  For Si ONCV ``(l=2)`` (no PSWFC D entry)
    ``n_lowest = 3``, so ``(3, 2)`` picks rank 0.
    """
    n_lowest = _lowest_n_for_channel(upf, l, j)
    target = n - n_lowest
    if target < 0:
        raise ValueError(
            f'Requested (n, l) = ({n}, {l}) below lowest pseudo-atom level '
            f'(n_lowest = {n_lowest}).'
        )
    eps_all, U, r = solve_radial_channel(
        upf, l, j=j, r_box=r_box, n_points=n_points, n_states=max(6, target + 3)
    )
    return r, U[target], float(eps_all[target])


def _lowest_n_for_channel(upf, l, j):
    """Smallest principal quantum number for the given (l, j) channel."""
    j_used = _default_j(upf, l, j)
    candidates = []
    for i, c in enumerate(upf.pswfc):
        label = c['label']
        l_char = label[1].upper()
        if l_char not in 'SPDF' or 'SPDF'.index(l_char) != l:
            continue
        if j_used is not None and i < len(getattr(upf, 'jchia', [])):
            if abs(float(upf.jchia[i]) - j_used) > 1e-6:
                continue
        candidates.append(int(label[0]))
    if candidates:
        return min(candidates)
    return {0: 1, 1: 2, 2: 3, 3: 4}.get(l, l + 1)
