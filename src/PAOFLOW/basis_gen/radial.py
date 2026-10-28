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


def _pswfc_channel_states(upf, l, j, r_uni):
    """The UPF PSWFC of the ``(l, j)`` channel on ``r_uni`` (plain-normalised), with their ``n``."""
    j_used = _default_j(upf, l, j)
    out = []
    for i, c in enumerate(upf.pswfc):
        label = c['label'].upper()
        if len(label) < 2 or label[1] not in 'SPDF' or 'SPDF'.index(label[1]) != l:
            continue
        if j_used is not None and i < len(getattr(upf, 'jchia', [])):
            if abs(float(upf.jchia[i]) - j_used) > 1e-6:
                continue
        u = _interp_to_uniform(upf.r, c['wfc'], r_uni)
        dr = r_uni[1] - r_uni[0]
        norm = np.sqrt(np.sum(u * u) * dr)
        if norm > 0.0:
            u = u / norm
        out.append((int(label[0]), u))
    return out


def _augmented_overlap_operator(upf, l, j, r_uni):
    """``S = 1 + dr A Q A^T`` of the channel on the uniform mesh, or ``None`` without augmentation."""
    pidx = _select_projectors(upf, l, _default_j(upf, l, j))
    qqq = getattr(upf, 'qqq', None)
    if not pidx or qqq is None:
        return None
    Q = qqq[np.ix_(pidx, pidx)]
    if not np.any(Q):
        return None
    dr = r_uni[1] - r_uni[0]
    A = np.zeros((r_uni.size, len(pidx)))
    for col, ip in enumerate(pidx):
        b = upf.beta[ip]
        A[:, col] = _interp_to_uniform(upf.r, b['wfc'], r_uni, cutoff_index=b.get('cutoff_index'))
    return np.eye(r_uni.size) + dr * (A @ Q @ A.T)


def select_physical_states(
    upf, l, j, eps_all, U, r_uni, ghost_norm_ratio=3.0, ghost_sphere_weight=0.85, pswfc_overlap=0.5
):
    """Indices of the solver states that are neither ghosts nor the stored pseudo-orbitals.

    Parameters
    ----------
    upf : UPF
    l, j : channel
    eps_all, U, r_uni : output of :func:`solve_radial_channel` (``U`` rows ``S``-normalised).
    ghost_norm_ratio : float, optional
        A state whose plain norm exceeds this multiple of its augmented norm lives where ``S``
        is nearly singular: a ghost of the generalized problem (Ga PAW, d channel: ratio 10.8
        for the spurious state at -3.4 Ha, 1.0 for physical ones).
    ghost_sphere_weight : float, optional
        A state with more than this fraction of its plain norm inside the largest projector
        radius of the channel is confined to the augmentation sphere (Ga PAW, s channel: the
        spurious state at -4.8 Ha); the polarisation shells the solver has to provide are
        diffuse. Not applied to channels without projectors.
    pswfc_overlap : float, optional
        States whose augmented overlap with a stored pseudo-orbital of the channel exceeds it
        are that orbital (or a ghost mixing with it) and are excluded: the pseudo-orbital itself
        is used for the shell it belongs to.

    Returns
    -------
    list of int
        Solver ranks of the remaining states, in order of increasing energy.
    """
    dr = r_uni[1] - r_uni[0]
    S = _augmented_overlap_operator(upf, l, j, r_uni)
    refs = [u for _, u in _pswfc_channel_states(upf, l, j, r_uni)]
    pidx = _select_projectors(upf, l, _default_j(upf, l, j))
    r_cut = 0.0
    for ip in pidx:
        b = upf.beta[ip]
        if b.get('cutoff_radius') is not None:
            r_cut = max(r_cut, float(b['cutoff_radius']))
        elif b.get('cutoff_index'):
            r_cut = max(r_cut, float(upf.r[min(int(b['cutoff_index']), upf.r.size) - 1]))
    inside = r_uni <= r_cut
    keep = []
    for rank in range(U.shape[0]):
        u = U[rank]
        plain = float(np.sum(u * u)) * dr
        if S is not None:
            s_norm = float(u @ S @ u) * dr
            if plain > ghost_norm_ratio * s_norm:
                continue
        if r_cut > 0.0 and float(np.sum(u[inside] ** 2)) * dr > ghost_sphere_weight * plain:
            continue
        bad = False
        for ref in refs:
            su = (S @ ref) * dr if S is not None else ref * dr
            ref_norm = float(ref @ su) if S is not None else float(np.sum(ref * ref)) * dr
            ov = abs(float(u @ su)) / np.sqrt(max(ref_norm, 1e-30))
            if ov > pswfc_overlap:
                bad = True
                break
        if not bad:
            keep.append(rank)
    return keep


def pseudize_shell(
    upf,
    n: int,
    l: int,
    j: float | None = None,
    r_box: float | None = None,
    n_points: int = 2000,
    prefer_pswfc: bool = True,
):
    """Return the (n, l[, j]) radial function on a uniform mesh.

    Shells that the pseudopotential carries as pseudo-orbitals (``PP_PSWFC``) are returned as
    stored (interpolated, plain-normalised): they are the pseudo-atom's own solutions, generated
    with the pseudopotential and the full atomic SCF, and the only functions consistent with the
    augmented metric of ultrasoft / PAW pseudopotentials. ``prefer_pswfc=False`` restores the
    radial solver for them (the behaviour before October 2026, useful to test the solver).

    The other shells (polarisation and radial-doubling shells of the presets) are eigenstates
    of the frozen-density pseudo-atom Hamiltonian in a box (:func:`solve_radial_channel`),
    selected by physics rather than by rank: ghosts of the generalized augmented problem and
    the states that reproduce a stored pseudo-orbital are discarded
    (:func:`select_physical_states`), then the requested shell is the ``k``-th remaining state
    with ``k = n - n_first`` and ``n_first`` the principal quantum number of the first shell the
    solver has to provide, i.e. one above the highest pseudo-orbital of the channel, or, for a
    channel without pseudo-orbitals, the lowest valence ``n`` of that ``l`` given the core of
    the pseudopotential (:func:`lowest_valence_n`; Si ``3D``, As ``4D``).

    Returns
    -------
    r : ndarray
        Uniform mesh (bohr).
    u : ndarray
        ``u(r) = r R(r)``.
    eps : float
        Solver eigenvalue (Hartree), ``nan`` for a stored pseudo-orbital.
    """
    r_box_used = float(min(upf.r[-1], 10.0)) if r_box is None else float(r_box)
    if prefer_pswfc and _has_pswfc_label(upf, n, l, j):
        r = np.linspace(0.0, r_box_used, n_points + 1)[1:-1]
        return r, _pswfc_on_uniform(upf, n, l, j, r), float('nan')
    if prefer_pswfc:
        pswfc_n = [m for m, _ in _pswfc_channel_states(upf, l, j, np.array([0.0, 1.0]))]
        n_first = max(pswfc_n) + 1 if pswfc_n else lowest_valence_n(upf, l)
    else:
        n_first = _lowest_n_for_channel(upf, l, j)
    target = n - n_first
    if target < 0:
        raise ValueError(
            f'Requested (n, l) = ({n}, {l}) below lowest pseudo-atom level (n_first = {n_first}).'
        )
    eps_all, U, r = solve_radial_channel(
        upf, l, j=j, r_box=r_box, n_points=n_points, n_states=max(8, target + 5)
    )
    if prefer_pswfc:
        ranks = select_physical_states(upf, l, j, eps_all, U, r)
        if target >= len(ranks):
            raise ValueError(
                f'Requested (n, l) = ({n}, {l}): only {len(ranks)} physical states found'
            )
        rank = ranks[target]
        return r, U[rank], float(eps_all[rank])
    eps = float(eps_all[target])
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


_ELEMENTS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se '
    'Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb '
    'Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm '
    'Bk Cf Es Fm Md No Lr'
).split()
_AUFBAU = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1),
           (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]  # fmt: skip


def lowest_valence_n(upf, l):
    """Lowest principal quantum number of the valence shells of angular momentum ``l``.

    The core of the pseudopotential holds ``Z - z_valence`` electrons; filling the shells in
    the aufbau order with that many electrons gives the frozen core, and the first shell of
    each ``l`` above it is the lowest one the pseudo-atom can describe (As: core [Ar]3d10, so
    the lowest d shell is 4d; Si: core [Ne], 3d). Falls back to the atomic minimum
    (``S->1, P->2, D->3, F->4``) when the element or the valence charge is unknown.
    """
    minimum = {0: 1, 1: 2, 2: 3, 3: 4}.get(l, l + 1)
    elem = str(getattr(upf, 'element', '')).strip()
    zval = getattr(upf, 'val', None)
    if elem not in _ELEMENTS or zval is None:
        return minimum
    core = int(round(_ELEMENTS.index(elem) + 1 - float(zval)))
    if core <= 0:
        return minimum
    highest = {}
    # pseudopotential cores: a noble-gas core plus, possibly, the filled d (and 4f) shells of
    # the same period ([Ar]3d10 for As, [Kr]4d10 for Sn, [Xe]4f14 5d10 for Au)
    noble = {2: 1, 10: 2, 18: 3, 36: 4, 54: 5, 86: 6}
    period = max((p for ne, p in noble.items() if ne <= core), default=0)
    ne_noble = max((ne for ne in noble if ne <= core), default=0)
    remainder = core - ne_noble
    if period >= 5 and remainder >= 14:
        highest[3] = 4
        remainder -= 14
    if period >= 3 and remainder >= 10:
        highest[2] = period
        remainder -= 10
    if remainder != 0:  # not a closed-shell core: fall back to the aufbau filling
        highest = {}
        filled = 0
        for n, ll in _AUFBAU:
            if filled >= core:
                break
            filled += 2 * (2 * ll + 1)
            highest[ll] = max(highest.get(ll, 0), n)
    else:
        for n, ll in _AUFBAU:  # the noble-gas part: every shell up to the period
            if ll == 0 and n <= period:
                highest[0] = max(highest.get(0, 0), n)
            if ll == 1 and n <= period:
                highest[1] = max(highest.get(1, 0), n)
            if ll == 2 and n <= period - 1 and n >= 3:
                highest[2] = max(highest.get(2, 0), n)
            if ll == 3 and n <= period - 2 and n >= 4:
                highest[3] = max(highest.get(3, 0), n)
    return max(minimum, highest.get(l, 0) + 1)


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
