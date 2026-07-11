r"""Radial Schroedinger solver for norm-conserving, USPP and PAW pseudopotentials.

The pseudo-atom Hamiltonian acting on the radial part :math:`u(r) = r R(r)` of
:math:`\psi_{lm}(r) = R(r) Y_{lm}` is, for a given :math:`(l, j)` channel,

.. math::

   H u(r) = -\frac{1}{2} u''(r) + \left[V_{\mathrm{loc}}(r) + \frac{l(l+1)}{2r^2}\right] u(r)
             + r \sum_{i,j \in (l,j)} \beta_i(r) D_{ij} \langle\beta_j | R\rangle

where :math:`\langle\beta_j | R\rangle = \int \beta_j(r) R(r) r^2 \, dr`.  Using the UPF-stored
quantity :math:`a_i(r) = r \beta_i(r)` the nonlocal kernel in :math:`u`-space becomes
the symmetric outer product

.. math::

   M(r, r') = \sum_{i,j} a_i(r) D_{ij} a_j(r').

For ultrasoft / PAW pseudopotentials the augmentation overlap operator

.. math::

   S = 1 + \sum_{ij} q_{ij} |\beta_i\rangle\langle\beta_j|
   \quad\rightarrow\quad
   S_{uu'} = I + \mathrm{d}r\, a Q a^T

is built from ``upf.qqq`` and the eigenproblem becomes the generalized
:math:`H u = \varepsilon S u` (solved with ``scipy.linalg.eigh(H, S)``); the
returned :math:`u(r)` is then normalised to :math:`\langle u|S|u\rangle \, \mathrm{d}r = 1`.  For NC pseudos
(no augmentation) the path collapses to ``np.linalg.eigh(H)`` with the
ordinary :math:`L^2` normalisation.

We discretise on a uniform mesh :math:`r_k = k \, \mathrm{d}r`, :math:`k = 1, \ldots, N-1`, with
:math:`u(0) = u(R_{\mathrm{box}}) = 0`, :math:`\mathrm{d}r = R_{\mathrm{box}} / N`.  :math:`V_{\mathrm{loc}}`
and :math:`a_i` are interpolated (cubic spline) from the UPF log mesh onto this uniform grid;
outside each projector's cutoff_radius :math:`a_i` is zero.

For SO UPFs the projectors are filtered by matching :math:`(l, j)` -- callers
that request :math:`(n, l)` with ``j = None`` on an SO UPF get the :math:`j = l + 1/2`
channel by default (override via the ``j`` argument).

All quantities are in Hartree atomic units.  Output radial functions
are returned in the QE convention used by ``build_aewfc_basis``: the
returned ``wfc[k]`` equals :math:`r_k R(r_k)` (i.e. :math:`u(r_k)`), normalised so that
:math:`\sum_k \mathrm{wfc}[k]^2 \, \mathrm{d}r = 1`.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
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
    # For ultrasoft/PAW pseudos the same projector outer-product also builds
    # the augmentation overlap operator S = I + dr * A Q A^T, leading to a
    # generalized eigenproblem H u = eps S u.  For NC (qqq is None) S = I and
    # the path collapses to the original np.linalg.eigh call.
    pidx = _select_projectors(upf, l, j_used)
    S = None
    if pidx:
        n_p = len(pidx)
        A = np.zeros((N, n_p))
        for col, ip in enumerate(pidx):
            b = upf.beta[ip]
            A[:, col] = _interp_to_uniform(upf.r, b['wfc'], r, cutoff_index=b.get('cutoff_index'))
        D = upf.dion[np.ix_(pidx, pidx)]
        # Nonlocal matrix in u-space: M = A D A^T * dr  (symmetric).
        H += dr * (A @ D @ A.T)

        qqq = getattr(upf, 'qqq', None)
        if qqq is not None:
            Q = qqq[np.ix_(pidx, pidx)]
            if np.any(Q):
                S = np.eye(N) + dr * (A @ Q @ A.T)

    # Diagonalise.  Switch to a generalized solve if S != I (USPP/PAW).
    if S is None:
        eps, V = np.linalg.eigh(H)
        weight = None
    else:
        eps, V = scipy.linalg.eigh(H, S)
        weight = S
    eps = eps[:n_states]
    U = V[:, :n_states].T  # shape (n_states, N)

    # Normalise so that <u_n | S | u_n> * dr = 1 (collapses to L^2 norm for NC).
    if weight is None:
        norms = np.sqrt(np.sum(U * U, axis=1) * dr)
    else:
        norms = np.sqrt(np.einsum('nk,kl,nl->n', U, weight, U) * dr)
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

    .. note::

       For PAW pseudopotentials with deep projector subspaces (e.g. an
       extra semi-core projector at a given ``l``) the lowest-rank
       eigenstate of the generalized augmented problem can be a
       *spurious* deep "ghost" state at one or two Hartree below the
       physical valence level.  The rank heuristic above does not
       distinguish ghosts from physical states; affected channels
       should be inspected by the caller (overlap with the stored
       PSWFC is a reliable filter).
    """
    n_lowest = _lowest_n_for_channel(upf, l, j)
    target = n - n_lowest
    if target < 0:
        raise ValueError(
            f'Requested (n, l) = ({n}, {l}) below lowest pseudo-atom level (n_lowest = {n_lowest}).'
        )
    eps_all, U, r = solve_radial_channel(
        upf, l, j=j, r_box=r_box, n_points=n_points, n_states=max(6, target + 3)
    )
    eps = float(eps_all[target])

    # If the frozen-density solver returned a continuum (unbound) state but
    # the UPF carries a PSWFC for this (n, l[, j]) channel, fall back to the
    # PSWFC.  This happens for the valence d shell of late transition-metal
    # USPP/PAW pseudos (e.g. Pt 5d), where the pseudo-atom potential alone
    # does not bind d and the rank-0 box state is a confinement mode rather
    # than the physical valence orbital.  The PSWFC stored in the UPF was
    # constructed by the pseudopotential generator with full atomic SCF and
    # is the correct radial for those shells.
    if eps >= 0.0:
        u_psw = _pswfc_on_uniform(upf, n, l, j, r)
        if u_psw is not None:
            return r, u_psw, eps
    return r, U[target], eps


def _pswfc_on_uniform(upf, n, l, j, r_uni):
    """Return the matching UPF PSWFC ``u(r) = r R(r)`` on ``r_uni``, or None.

    Looks for a ``PP_CHI`` entry with label starting ``f"{n}{L}"`` and
    matching ``j`` (under SO).  The returned array is normalised so that
    ``sum(u**2) * dr = 1``, the same convention as the solver output.
    """
    j_used = _default_j(upf, l, j)
    l_char = 'SPDF'[l]
    target_label = f'{n}{l_char}'
    for i, c in enumerate(upf.pswfc):
        if c['label'].upper() != target_label:
            continue
        if j_used is not None and i < len(getattr(upf, 'jchia', [])):
            if abs(float(upf.jchia[i]) - j_used) > 1e-6:
                continue
        u_log = c['wfc']  # u(r) = r R(r) on the UPF log mesh
        u_uni = _interp_to_uniform(upf.r, u_log, r_uni)
        dr = r_uni[1] - r_uni[0]
        norm = np.sqrt(np.sum(u_uni * u_uni) * dr)
        if norm > 0.0:
            u_uni = u_uni / norm
        if u_uni.size > 0 and u_uni[0] < 0.0:
            u_uni = -u_uni
        return u_uni
    return None


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


def _pswfc_max_n(upf):
    """Largest principal quantum number among UPF PSWFC labels (0 if none)."""
    nmax = 0
    for c in upf.pswfc:
        label = c['label']
        if len(label) < 2 or label[1].upper() not in 'SPDF':
            continue
        try:
            nmax = max(nmax, int(label[0]))
        except ValueError:
            continue
    return nmax


def _has_pswfc_label(upf, n, l, j=None):
    """True if the UPF carries a PSWFC whose label matches ``(n, l[, j])``."""
    j_used = _default_j(upf, l, j)
    l_char = 'SPDF'[l]
    target_label = f'{n}{l_char}'
    for i, c in enumerate(upf.pswfc):
        if c['label'].upper() != target_label:
            continue
        if j_used is not None and i < len(getattr(upf, 'jchia', [])):
            if abs(float(upf.jchia[i]) - j_used) > 1e-6:
                continue
        return True
    return False


def is_frozen_core_shell(upf, n, l, eps, j=None):
    """Heuristic: is ``(n, l[, j])`` a spurious frozen-core request?

    A pseudopotential that freezes a shell into the core (e.g. As 3d,
    whose 3d electrons are not in the As pseudo valence) carries no
    PSWFC for that shell and cannot bind it.  Requesting such a shell
    makes :func:`pseudize_shell` return an unbound, diffuse box mode
    that is linearly dependent with the genuine polarization shells and
    corrupts the projection basis.

    The request is flagged as frozen-core when *all* of the following
    hold:

    * the UPF has no PSWFC with the exact ``(n, l[, j])`` label
      (a genuine semicore shell, e.g. Ga 3D, *is* present and so is
      never flagged);
    * ``n`` lies below the largest PSWFC principal quantum number
      ``nmax`` (above-valence polarization such as Ga 4D has ``n >= nmax``
      and is never flagged); and
    * the solver eigenvalue is non-negative (the returned state is
      unbound, the signature of a confinement mode rather than a bound
      orbital).
    """
    if _has_pswfc_label(upf, n, l, j):
        return False
    if n >= _pswfc_max_n(upf):
        return False
    return eps >= 0.0
