"""Special displacement method of Zacharias and Giustino (the ZG configuration).

One supercell configuration whose atoms are displaced along every phonon of the
supercell-commensurate q-grid with the thermal amplitude ``sigma_{q nu}`` and a sign
pattern chosen so that the second-order expansion of any clamped-ion observable
reproduces its Williams-Lax thermal average in the large-supercell limit (Zacharias
and Giustino, Phys. Rev. B 94, 075125 (2016); Phys. Rev. Research 2, 013357 (2020)).
The recipe follows ``ZG.x`` (QE 7.5, ``EPW/ZG``): the q-points are split into the set
A (``2q = G``) and the set B (one of each ``+-q`` pair), the sign sequence cycles
through the rows of an antithetic binary table and is permuted until the relative
error of the mean-square displacements falls below a threshold, and the displacement
of atom ``kappa`` in cell ``R`` is

.. math::

    \\Delta\\tau_{R\\kappa\\alpha} = \\sum_{q\\in B,\\nu}\\sqrt{\\frac{2}{N_pM_\\kappa}}\\,
      \\mathrm{Re}\\!\\left[e^{2\\pi i q\\cdot R}(1+i)\\,e_{\\kappa\\alpha,\\nu}(q)\\right]
      S_{q\\nu}\\sigma_{q\\nu}
      + \\sum_{q\\in A,\\nu}\\sqrt{\\frac{1}{N_pM_\\kappa}}\\cos(2\\pi q\\cdot R)\\,
      \\mathrm{Re}\\!\\left[e_{\\kappa\\alpha,\\nu}(q)\\right]S_{q\\nu}\\sigma_{q\\nu},

with ``sigma_{q nu} = sqrt(hbar/(2 omega_{q nu} m_u)) sqrt(2 n_{q nu} + 1)`` (reference
mass one atomic mass unit, the atomic mass entering through ``1/sqrt(M_kappa)``) and
``e`` the orthonormal mass-weighted eigenvectors.  Lengths are in bohr (PAOFLOW's
phonopy cells are built in bohr), masses in amu, frequencies in THz.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

HBAR_SI = 1.054571817e-34
KB_SI = 1.380649e-23
AMU_SI = 1.66053906660e-27
BOHR_M = 5.29177210903e-11
THZ = 1.0e12


# ----------------------------------------------------------------------------- q-points
def supercell_qpoints(dims: tuple[int, int, int]) -> NDArray[np.float64]:
    """The q-grid commensurate with a diagonal supercell, in the order ``ZG.x`` uses.

    Parameters
    ----------
    dims : tuple of int
        Supercell dimensions ``(dim1, dim2, dim3)``.

    Returns
    -------
    NDArray[np.float64]
        ``(dim1*dim2*dim3, 3)`` crystal coordinates in ``[0, 1)``, the third index
        running fastest (``q = (i/dim1, j/dim2, k/dim3)``), Gamma first.
    """
    d1, d2, d3 = (int(n) for n in dims)
    q = [(i / d1, j / d2, k / d3) for i in range(d1) for j in range(d2) for k in range(d3)]
    return np.asarray(q, dtype=float)


def _is_zero_mod_1(v: NDArray[np.float64], tol: float) -> bool:
    return bool(np.all(np.abs(v - np.round(v)) < tol))


def split_qpoints(
    q_all: NDArray[np.float64], tol: float = 1.0e-6
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Keep the ``ZG.x`` set ``A + B`` of a q-grid: ``2q = G`` and one of each ``+-q`` pair.

    Parameters
    ----------
    q_all : NDArray[np.float64]
        ``(nq, 3)`` crystal coordinates (see :func:`supercell_qpoints`).
    tol : float
        Tolerance on the lattice-vector test.

    Returns
    -------
    q_ab : NDArray[np.float64]
        ``(nab, 3)`` the kept points, Gamma first, then in grid order.
    is_a : NDArray[np.bool_]
        ``(nab,)`` ``True`` for the set A (time-reversal-invariant points).
    """
    q_all = np.asarray(q_all, dtype=float)
    nq = q_all.shape[0]
    keep, is_a = [], []
    for i in range(nq):
        if _is_zero_mod_1(2.0 * q_all[i], tol):
            keep.append(i)
            is_a.append(True)
            continue
        for j in range(i + 1, nq):
            if _is_zero_mod_1(q_all[i] + q_all[j], tol):
                keep.append(i)
                is_a.append(False)
                break
    gamma = [i for i in keep if _is_zero_mod_1(q_all[i], tol)]
    if len(gamma) != 1:
        raise ValueError('the q-grid must contain Gamma exactly once')
    order = gamma + [i for i in keep if i != gamma[0]]
    flags = dict(zip(keep, is_a))
    return q_all[order], np.asarray([flags[i] for i in order], dtype=bool)


# ---------------------------------------------------------------------------- amplitudes
def bose_occupation(freq_thz: NDArray[np.float64], temperature_k: float) -> NDArray[np.float64]:
    """Bose-Einstein occupation of phonons of frequency ``freq_thz`` at ``temperature_k``.

    Parameters
    ----------
    freq_thz : NDArray[np.float64]
        Frequencies in THz (non-positive entries give zero occupation).
    temperature_k : float
        Temperature in kelvin (``0`` gives zero occupation).

    Returns
    -------
    NDArray[np.float64]
        ``n = 1 / (exp(hbar omega / k_B T) - 1)``, same shape.
    """
    f = np.asarray(freq_thz, dtype=float)
    n = np.zeros_like(f)
    if temperature_k <= 0.0:
        return n
    x = HBAR_SI * 2.0 * np.pi * f * THZ / (KB_SI * float(temperature_k))
    ok = f > 0.0
    n[ok] = 1.0 / np.expm1(np.minimum(x[ok], 700.0))
    return n


def zg_amplitudes(
    freq_thz: NDArray[np.float64],
    temperature_k: float,
    is_gamma: NDArray[np.bool_] | None = None,
    min_freq_thz: float = 1.0e-3,
) -> NDArray[np.float64]:
    """Thermal amplitudes ``sigma_{q nu}`` of the ZG displacement.

    Parameters
    ----------
    freq_thz : NDArray[np.float64]
        ``(nq, nmode)`` phonon frequencies in THz.
    temperature_k : float
        Temperature in kelvin.
    is_gamma : NDArray[np.bool_], optional
        ``(nq,)`` marks the Gamma point, whose three lowest modes (acoustic) are frozen.
    min_freq_thz : float
        Modes below this frequency (soft or imaginary) are frozen.

    Returns
    -------
    NDArray[np.float64]
        ``(nq, nmode)`` ``sigma = sqrt(hbar / (2 omega m_u)) sqrt(2n + 1)`` in
        bohr times sqrt(amu).

    Notes
    -----
    .. math:: \\sigma_{q\\nu}^2 = \\frac{\\hbar}{2 m_u \\omega_{q\\nu}}\\,(2 n_{q\\nu} + 1)
    """
    f = np.asarray(freq_thz, dtype=float)
    omega = 2.0 * np.pi * f * THZ
    n = bose_occupation(f, temperature_k)
    sigma = np.zeros_like(f)
    ok = f > float(min_freq_thz)
    sigma[ok] = np.sqrt(HBAR_SI / (2.0 * omega[ok] * AMU_SI) * (2.0 * n[ok] + 1.0)) / BOHR_M
    if is_gamma is not None:
        for iq in np.nonzero(np.asarray(is_gamma, dtype=bool))[0]:
            order = np.argsort(f[iq])
            sigma[iq, order[:3]] = 0.0
    return sigma


# --------------------------------------------------------------------------------- signs
def sign_table(nmode: int) -> NDArray[np.int64]:
    """The antithetic binary sign table of ``ZG.x`` (``M_mat``).

    Parameters
    ----------
    nmode : int
        Number of phonon branches ``3 nat``.

    Returns
    -------
    NDArray[np.int64]
        ``(2 pn, nmode)`` entries ``+-1`` with ``pn = 2**(nmode-1)`` capped at ``2**11``:
        a binary counting pattern (mode ``i`` alternates in blocks of
        ``2**(nmode-1-i)`` rows), with every second row negated.
    """
    pn = 2 ** (min(int(nmode), 12) - 1)
    rows = np.arange(2 * pn)
    table = np.ones((2 * pn, nmode), dtype=np.int64)
    for i in range(nmode):
        block = 2 ** (nmode - 1 - i)
        table[:, i] = np.where((rows // block) % 2 == 0, 1, -1)
    table[1::2] *= -1
    return table


def _initial_rows(table: NDArray[np.int64]) -> NDArray[np.int64]:
    """The ``pn`` rows ``ZG.x`` draws from the table (``Mx_mat_or``), 1-based bookkeeping."""
    pn = table.shape[0] // 2
    out = np.ones((pn, table.shape[1]), dtype=np.int64)
    half = pn // 2
    for i in range(1, half + 1, 2):  # odd i <= pn/2
        out[i - 1] = table[i - 1]
    for i in range(2, half + 1, 2):  # even i <= pn/2
        out[i - 1] = table[2 * pn - i]  # M(2 pn + 1 - i)
    for i in range(half + 1, pn + 1, 2):  # odd i in (pn/2, pn]
        out[i - 1] = table[i]  # M(i + 1)
    for i in range(half + 2, pn + 1, 2):  # even i in (pn/2, pn]
        out[i - 1] = table[2 * pn + 1 - i]  # M(2 pn + 2 - i)
    return out


def zg_sign_sequence(
    nmode: int, nq: int, permutation: NDArray[np.int64] | None = None
) -> NDArray[np.int64]:
    """Sign sequence ``S_{q nu}`` of ``ZG.x`` for ``nq`` q-points.

    Parameters
    ----------
    nmode : int
        Number of branches.
    nq : int
        Number of q-points of the set ``A + B`` (Gamma first).
    permutation : NDArray[np.int64], optional
        Permutation of the ``pn`` rows of the initial table (``None``: identity).

    Returns
    -------
    NDArray[np.int64]
        ``(nq, nmode)`` signs: the rows of the (permuted) table are cycled through
        the q-points; every ``pn`` q-points the table is rotated by one row and
        negated (the antithetic block of ``ZG.x``).
    """
    rows = _initial_rows(sign_table(nmode))
    pn = rows.shape[0]
    if permutation is not None:
        rows = rows[np.asarray(permutation, dtype=int)]
    mx = rows.copy()
    signs = np.empty((nq, nmode), dtype=np.int64)
    for ii in range(1, nq + 1):
        if ii % pn == 1:  # a new antithetic block every pn q-points (never for pn == 1)
            mx = np.vstack([mx[1:], mx[:1]])  # second row to the top, first to the bottom
            mx = -mx
        r = ii % pn
        signs[ii - 1] = mx[(r if r > 0 else pn) - 1]
    return signs


def zg_error(
    signs: NDArray[np.int64],
    sigma: NDArray[np.float64],
    eigenvectors: NDArray[np.complex128],
    is_a: NDArray[np.bool_] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Relative error of the mean-square displacements of a sign choice.

    Parameters
    ----------
    signs : NDArray[np.int64]
        ``(nq, nmode)``.
    sigma : NDArray[np.float64]
        ``(nq, nmode)`` amplitudes.
    eigenvectors : NDArray[np.complex128]
        ``(nq, nat, 3, nmode)`` orthonormal mass-weighted eigenvectors.
    is_a : NDArray[np.bool_], optional
        ``(nq,)`` set-A flags: those points weigh 1 against 2 for the set B and enter
        with the real-gauge eigenvectors, as in :func:`zg_displacements`.  ``None``
        weighs every point equally (the ``ZG.x`` criterion).

    Returns
    -------
    total : float
        Mean over the Cartesian atomic components of ``|ratio|``.
    ratio : NDArray[np.float64]
        ``(3 nat,)`` ``(<u^2>_config - <u^2>_thermal) / <u^2>_thermal`` per component,
        at the level of the second-order expansion of the configuration.

    Notes
    -----
    With ``C_{q nu} = S_{q nu} sigma_{q nu}`` and ``v_{q,p} = sum_nu e_{p nu}(q) C_{q nu}``,

    .. math::

        \\mathrm{ratio}_p = \\frac{\\sum_q w_q\\,(|v_{q,p}|^2 - \\sum_\\nu |e_{p\\nu}(q)|^2 C_{q\\nu}^2)}
        {\\sum_{q\\nu} w_q\\,|e_{p\\nu}(q)|^2 C_{q\\nu}^2},

    the cross-branch terms of the same q relative to the diagonal ones; the cross terms
    between different q-points vanish exactly on the cell average.  The configuration of
    :func:`zg_displacements` has exactly ``<u^2>_thermal (1 + ratio)``.
    """
    nq, nmode = np.asarray(sigma).shape
    e = np.asarray(eigenvectors).reshape(nq, -1, nmode)
    c = np.asarray(signs, dtype=float) * np.asarray(sigma, dtype=float)
    if is_a is None:
        w = np.ones(nq)
        a = np.zeros(nq, dtype=bool)
    else:
        a = np.asarray(is_a, dtype=bool)
        w = np.where(a, 1.0, 2.0)
    v = np.einsum('qpn,qn->qp', e, c)
    if a.any():
        er = np.stack([real_gauge(e[iq]) for iq in np.nonzero(a)[0]])  # (nqa, p, nmode)
        v[a] = np.real(np.einsum('qpn,qn->qp', er, c[a]))
    config = np.einsum('q,qp->p', w, np.abs(v) ** 2)
    diag = np.einsum('q,qpn,qn->p', w, np.abs(e) ** 2, c**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(diag > 0.0, (config - diag) / diag, 0.0)
    return float(np.mean(np.abs(ratio))), ratio


def optimise_signs(
    sigma: NDArray[np.float64],
    eigenvectors: NDArray[np.complex128],
    niters: int = 15000,
    error_thresh: float = 0.05,
    seed: int | None = 0,
    is_a: NDArray[np.bool_] | None = None,
    shuffle_qpoints: bool = True,
) -> tuple[NDArray[np.int64], float, int]:
    """Search the row permutations of the sign table until the error is below the threshold.

    Parameters
    ----------
    sigma, eigenvectors, is_a
        As in :func:`zg_error`.
    niters : int
        Maximum number of attempts (the first is the unpermuted table).
    error_thresh : float
        Stop when :func:`zg_error` falls below it.  ``ZG.x`` defaults to 0.3 and its
        tutorials use 0.02; on Pb 4x4x4 a threshold of 0.05 is reached within about a hundred
        permutations.
    seed : int, optional
        Seed of the permutations.
    shuffle_qpoints : bool
        From the second attempt on, also permute the assignment of the table rows to the
        q-points (``ZG.x`` permutes only the rows, which for a monoatomic cell gives
        ``pn! = 24`` distinct sequences); different seeds then give distinct configurations.

    Returns
    -------
    signs : NDArray[np.int64]
        The best sequence found.
    error : float
        Its error.
    attempts : int
        Number of sequences tried.
    """
    nq, nmode = np.asarray(sigma).shape
    rng = np.random.default_rng(seed)
    pn = 2 ** (min(nmode, 12) - 1)
    best = None
    for attempt in range(1, max(int(niters), 1) + 1):
        perm = None if attempt == 1 else rng.permutation(pn)
        signs = zg_sign_sequence(nmode, nq, perm)
        if attempt > 1 and shuffle_qpoints:
            # the rows of the table are also redistributed over the q-points (Gamma kept first):
            # for a few branches the pn! row permutations alone give a handful of sequences
            order = np.concatenate([[0], 1 + rng.permutation(nq - 1)])
            signs = signs[order]
        err, _ = zg_error(signs, sigma, eigenvectors, is_a=is_a)
        if best is None or err < best[1]:
            best = (signs, err, attempt)
        if err < error_thresh:
            break
    return best


# -------------------------------------------------------------------------- displacements
def real_gauge(eigvec: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Multiply each mode by the phase that makes it as real as possible.

    Parameters
    ----------
    eigvec : NDArray[np.complex128]
        ``(..., nmode)`` eigenvectors (components on the leading axes).

    Returns
    -------
    NDArray[np.complex128]
        ``e exp(-i phi)`` with ``phi = arg(sum_p e_p^2) / 2``, exactly real when the
        mode is real up to a phase (the time-reversal-invariant q-points).
    """
    e = np.asarray(eigvec)
    comps = e.reshape(-1, e.shape[-1])
    phi = 0.5 * np.angle(np.sum(comps**2, axis=0))
    return e * np.exp(-1j * phi)


def zg_displacements(
    q_ab: NDArray[np.float64],
    is_a: NDArray[np.bool_],
    eigenvectors: NDArray[np.complex128],
    sigma: NDArray[np.float64],
    signs: NDArray[np.int64],
    masses_amu: NDArray[np.float64],
    translations: NDArray[np.int64],
    s2p: NDArray[np.int64],
    ncells: int,
    include_set_a: bool = True,
) -> NDArray[np.float64]:
    """Displacement (bohr) of every supercell atom from the ZG formula of the module docstring.

    Parameters
    ----------
    q_ab, is_a
        From :func:`split_qpoints`.
    eigenvectors : NDArray[np.complex128]
        ``(nab, nat, 3, nmode)``.
    sigma, signs : NDArray
        ``(nab, nmode)``.
    masses_amu : NDArray[np.float64]
        ``(nat,)`` primitive-cell masses.
    translations : NDArray[np.int64]
        ``(natom_sc, 3)`` primitive lattice translation of every supercell atom.
    s2p : NDArray[np.int64]
        ``(natom_sc,)`` primitive atom of every supercell atom.
    ncells : int
        ``N_p``, number of primitive cells in the supercell.
    include_set_a : bool
        Include the modes of the set A (``ZG.x`` ``incl_qA``).

    Returns
    -------
    NDArray[np.float64]
        ``(natom_sc, 3)`` displacements in bohr.
    """
    e = np.asarray(eigenvectors)
    c = np.asarray(signs, dtype=float) * np.asarray(sigma, dtype=float)
    phase = np.exp(
        2j * np.pi * np.asarray(translations, dtype=float) @ np.asarray(q_ab).T
    )  # (nsc, nab)
    amp = np.einsum('qkan,qn->qka', e, c)  # (nab, nat, 3): sum_nu e C
    u = np.zeros((len(s2p), 3))
    b = ~np.asarray(is_a, dtype=bool)
    if b.any():
        term = (1.0 + 1j) * np.einsum('sq,qka->ska', phase[:, b], amp[b])
        u += np.sqrt(2.0 / ncells) * np.real(term)[np.arange(len(s2p)), s2p]
    if include_set_a and np.asarray(is_a).any():
        a = np.asarray(is_a, dtype=bool)
        amp_a = np.einsum('qkan,qn->qka', real_gauge(e[a]), c[a])
        term = np.einsum('sq,qka->ska', np.real(phase[:, a]), np.real(amp_a))
        u += np.sqrt(1.0 / ncells) * term[np.arange(len(s2p)), s2p]
    return u / np.sqrt(np.asarray(masses_amu, dtype=float)[s2p])[:, None]


def thermal_mean_square_displacement(
    is_a: NDArray[np.bool_],
    eigenvectors: NDArray[np.complex128],
    sigma: NDArray[np.float64],
    masses_amu: NDArray[np.float64],
    ncells: int,
) -> NDArray[np.float64]:
    """Harmonic thermal mean-square displacement ``<u^2_{kappa alpha}>`` (bohr^2).

    Parameters
    ----------
    is_a, eigenvectors, sigma, masses_amu, ncells
        As in :func:`zg_displacements`.

    Returns
    -------
    NDArray[np.float64]
        ``(nat, 3)``.

    Notes
    -----
    .. math:: \\langle u_{\\kappa\\alpha}^2\\rangle = \\frac{1}{N_p M_\\kappa}\\sum_{q\\nu} w_q\\,\\sigma_{q\\nu}^2\\,|e_{\\kappa\\alpha,\\nu}(q)|^2

    with ``w_q = 1`` for the set A and ``2`` for the set B (both members of the pair).
    """
    w = np.where(np.asarray(is_a, dtype=bool), 1.0, 2.0)
    msd = np.einsum(
        'q,qkan,qn->ka', w, np.abs(np.asarray(eigenvectors)) ** 2, np.asarray(sigma) ** 2
    )
    return msd / (float(ncells) * np.asarray(masses_amu, dtype=float)[:, None])


def configuration_mean_square_displacement(
    displacements: NDArray[np.float64], s2p: NDArray[np.int64], nat: int
) -> NDArray[np.float64]:
    """Cell-averaged ``u^2_{kappa alpha}`` of a configuration (bohr^2), ``(nat, 3)``."""
    u2 = np.zeros((nat, 3))
    counts = np.zeros(nat)
    for a, k in enumerate(np.asarray(s2p, dtype=int)):
        u2[k] += np.asarray(displacements)[a] ** 2
        counts[k] += 1
    return u2 / counts[:, None]


# -------------------------------------------------------------------------------- driver
def phonopy_from_q2r(unitcell, force_constants_file: str):
    """A phonopy object with the force constants of a ``q2r.x`` file.

    Parameters
    ----------
    unitcell : phonopy.structure.atoms.PhonopyAtoms
        The cell the force constants refer to (bohr, PAOFLOW's phonopy convention).
    force_constants_file : str
        ``flfrc`` output of ``q2r.x`` (the file ``ZG.x`` and ``matdyn.x`` read).

    Returns
    -------
    phonopy.Phonopy
        With ``force_constants`` set on the supercell of the ``q2r`` grid.
    """
    from phonopy import Phonopy
    from phonopy.interface.qe import PH_Q2R

    q2r = PH_Q2R(force_constants_file)
    q2r.run(unitcell, is_full_fc=True)
    phonon = Phonopy(
        unitcell, supercell_matrix=np.diag(q2r.dimension), primitive_matrix='P', calculator='qe'
    )
    phonon.force_constants = q2r.fc
    return phonon


def zg_configuration(
    phonon,
    supercell_matrix,
    temperature_k: float,
    niters: int = 15000,
    error_thresh: float = 0.05,
    seed: int | None = 0,
    include_set_a: bool = True,
    min_freq_thz: float = 1.0e-3,
    shuffle_qpoints: bool = True,
) -> dict:
    """The ZG configuration of a phonopy object with force constants.

    Parameters
    ----------
    phonon : phonopy.Phonopy
        With force constants produced (on any supercell: the phonons are interpolated
        to the q-grid of the ZG supercell); positions in bohr, as PAOFLOW builds them.
    supercell_matrix : int or array_like
        Diagonal supercell matrix of the ZG cell (a scalar or three integers).
    temperature_k : float
        Temperature in kelvin.
    niters, error_thresh, seed
        Sign search (:func:`optimise_signs`).
    include_set_a : bool
        Include the time-reversal-invariant q-points.
    min_freq_thz : float
        Soft-mode threshold of :func:`zg_amplitudes`.
    shuffle_qpoints : bool
        See :func:`optimise_signs`.

    Returns
    -------
    dict
        ``cell`` (``phonopy.structure.atoms.PhonopyAtoms`` of the displaced supercell),
        ``reference`` (the undisplaced one), ``displacements`` ``(natom_sc, 3)`` bohr,
        ``signs``, ``error``, ``attempts``, ``q_ab``, ``is_a``, ``frequencies_thz``,
        ``sigma``, ``msd_thermal`` and ``msd_configuration`` ``(nat, 3)`` bohr^2,
        ``dims`` and ``temperature_k``.
    """
    from types import SimpleNamespace

    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.cells import get_supercell

    from ..elphon.fold import supercell_atom_translations
    from ..elphon.gkq import phonon_modes

    S = np.asarray(supercell_matrix)
    dims = tuple(int(n) for n in (np.diag(S) if S.ndim == 2 else np.broadcast_to(S, (3,))))
    q_ab, is_a = split_qpoints(supercell_qpoints(dims))
    freq, eig, masses = phonon_modes(phonon, q_ab, with_eigenvectors=True)
    is_gamma = np.array([_is_zero_mod_1(q, 1.0e-8) for q in q_ab])
    sigma = zg_amplitudes(freq, temperature_k, is_gamma=is_gamma, min_freq_thz=min_freq_thz)
    signs, error, attempts = optimise_signs(
        sigma,
        eig,
        niters=niters,
        error_thresh=error_thresh,
        seed=seed,
        is_a=is_a,
        shuffle_qpoints=shuffle_qpoints,
    )
    sc = get_supercell(phonon.primitive, np.diag(dims))
    s2p, translations = supercell_atom_translations(
        SimpleNamespace(primitive=phonon.primitive, supercell=sc)
    )
    ncells = int(np.prod(dims))
    u = zg_displacements(
        q_ab,
        is_a,
        eig,
        sigma,
        signs,
        masses,
        translations,
        s2p,
        ncells,
        include_set_a=include_set_a,
    )
    reference = PhonopyAtoms(
        symbols=list(sc.symbols), cell=np.asarray(sc.cell), positions=np.asarray(sc.positions),
        masses=list(sc.masses),
    )  # fmt: skip
    cell = PhonopyAtoms(
        symbols=list(sc.symbols), cell=np.asarray(sc.cell), positions=np.asarray(sc.positions) + u,
        masses=list(sc.masses),
    )  # fmt: skip
    return {
        'cell': cell,
        'reference': reference,
        'displacements': u,
        'signs': signs,
        'error': error,
        'attempts': attempts,
        'q_ab': q_ab,
        'is_a': is_a,
        'frequencies_thz': freq,
        'sigma': sigma,
        'msd_thermal': thermal_mean_square_displacement(is_a, eig, sigma, masses, ncells),
        'msd_configuration': configuration_mean_square_displacement(u, s2p, len(masses)),
        'dims': dims,
        'temperature_k': float(temperature_k),
    }


MANIFEST = 'zg.json'


def zg_prefix(base_prefix: str, temperature_k: float | None) -> str:
    """QE prefix of the ZG cell at a temperature (``<base>_zg_T<T>K``; ``<base>_zg_ref`` undisplaced)."""
    if temperature_k is None:
        return '%s_zg_ref' % base_prefix
    return '%s_zg_T%gK' % (base_prefix, float(temperature_k))


def reference_configuration(configuration: dict) -> dict:
    """The undisplaced supercell of a ZG configuration (``temperature_k`` ``None``).

    Parameters
    ----------
    configuration : dict
        Output of :func:`zg_configuration`.

    Returns
    -------
    dict
        The same dictionary with the ideal ``cell``, zero displacements and mean-square
        displacements, ``error`` 0 and no temperature.  Its scf gives the supercell bands without
        disorder: unfolded, they measure the offset of the supercell PAO Hamiltonian (basis,
        k-sampling) from the primitive one, to which the thermal shifts are referred.
    """
    out = dict(configuration)
    out['cell'] = configuration['reference']
    for key in ('displacements', 'msd_configuration', 'msd_thermal', 'sigma'):
        out[key] = np.zeros_like(np.asarray(configuration[key]))
    out['temperature_k'] = None
    out['error'] = 0.0
    out['attempts'] = 0
    return out


def write_zg_inputs(
    data_controller,
    configurations: list[dict],
    zg_dir: str = 'zg',
    pp_dir: str | None = None,
    prefix: str | None = None,
    kgrid=None,
    nbnd: int | None = None,
    hubbard_card: str | None = None,
) -> list[str]:
    """Write one ``pw.x`` scf input per ZG configuration and the manifest ``zg.json``.

    Parameters
    ----------
    data_controller : DataController
        Of the unit cell (cut-offs, k-grid, species and pseudopotentials).
    configurations : list of dict
        Outputs of :func:`zg_configuration`, one per temperature (same ``dims``), optionally
        with the undisplaced cell of :func:`reference_configuration`.
    zg_dir : str
        Sub-directory of the output path.
    pp_dir, prefix, kgrid, nbnd, hubbard_card
        As in :func:`~PAOFLOW.elphon.io.write_eph_displaced_supercells`: pseudopotential
        directory (default the save path), base prefix, supercell k-grid (default the
        unit-cell grid divided by the dimensions), band count, Hubbard card.

    Returns
    -------
    list of str
        The written input files.  Next to each input, ``<prefix>.npz`` keeps the
        displacements (bohr), signs, q-points, amplitudes and mean-square displacements.
    """
    import json

    from .io import _pp_filenames, _qe_input_text, _supercell_kgrid, resolve_phonon_dir

    arry, attr = data_controller.data_dicts()
    out_dir = resolve_phonon_dir(data_controller, zg_dir)
    if prefix is None:
        savedir = attr.get('savedir', None)
        prefix = os.path.basename(str(savedir)).replace('.save', '') if savedir else 'zg'
    if pp_dir is None:
        pp_dir = attr.get('fpath', '.')
    dims = configurations[0]['dims']
    S = np.diag(dims)
    if kgrid is None:
        kgrid = _supercell_kgrid(data_controller, S)
    pp = _pp_filenames(data_controller)
    written, entries = [], []
    if getattr(data_controller, 'rank', 0) != 0:
        return written
    for conf in configurations:
        pfx = zg_prefix(prefix, conf['temperature_k'])
        path = os.path.join(out_dir, pfx + '.in')
        with open(path, 'w') as f:
            f.write(
                _qe_input_text(
                    data_controller,
                    conf['cell'],
                    S,
                    pfx,
                    pp_dir,
                    pp,
                    kgrid,
                    hubbard_card=hubbard_card,
                    nbnd=nbnd,
                )  # fmt: skip
            )
        np.savez(
            os.path.join(out_dir, pfx + '.npz'), displacements=conf['displacements'],
            signs=conf['signs'], q_ab=conf['q_ab'], is_a=conf['is_a'], sigma=conf['sigma'],
            frequencies_thz=conf['frequencies_thz'], msd_thermal=conf['msd_thermal'],
            msd_configuration=conf['msd_configuration'], positions=conf['cell'].positions,
            reference_positions=conf['reference'].positions, cell=conf['cell'].cell,
        )  # fmt: skip
        written.append(path)
        entries.append(
            {
                'temperature_k': None
                if conf['temperature_k'] is None
                else float(conf['temperature_k']),
                'prefix': pfx,
                'input': os.path.basename(path),
                'error': float(conf['error']),
                'attempts': int(conf['attempts']),
                'rms_displacement_bohr': float(
                    np.sqrt(conf['msd_configuration'].sum(axis=1).mean())
                ),
                'msd_thermal_bohr2': np.asarray(conf['msd_thermal']).tolist(),
            }
        )
    with open(os.path.join(out_dir, MANIFEST), 'w') as f:
        json.dump(
            {
                'dims': [int(n) for n in dims],
                'kgrid': [int(n) for n in kgrid],
                'nbnd': nbnd,
                'natoms': len(configurations[0]['cell']),
                'configurations': entries,
            },
            f,
            indent=2,
        )
    return written
