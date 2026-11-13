r"""Vectorised two-centre overlap and dipole blocks between two atom-centred orbital sets.

The element-wise primitives of :mod:`PAOFLOW.hamiltonian._two_center` evaluate one
:math:`(l_A m_A, l_B m_B)` matrix element at a time, with a Simpson quadrature per
:math:`(\text{channel}_A, \text{channel}_B, L)` and a Python loop over :math:`(L, M)` and the
Gaunt coefficients.  For a basis with many shells and hundreds of lattice pairs this costs tens
of seconds (Si, 44 orbitals: :math:`1.4\times10^5` quadratures, :math:`2.5\times10^6` angular
terms).  This module evaluates the same expressions for a whole pair of sites at once:

.. math::

    S_{(a m_a)(b m_b)}(\mathbf d) = 8 \sum_L i^{\,l_a - l_b - L}\, I^{ab}_L(d)
        \sum_M G(l_a m_a; l_b m_b; L M)\, Y_{LM}(\hat{\mathbf d}),
    \qquad
    I^{ab}_L(d) = \int_0^\infty q^2 J^a_{l_a}(q) J^b_{l_b}(q)\, j_L(q d)\, dq,

with the radial integrals of every channel pair and every :math:`L` from one Simpson call on a
``(channels, L, q)`` array, and the angular sums as contractions of precomputed Gaunt tensors
with the vector :math:`Y_{LM}(\hat{\mathbf d})`.  The dipole block uses the modified bra
:math:`g_a(r) = r R_a(r)` at :math:`L' = l_a \pm 1` and the product expansion
:math:`Y_{1 m_\alpha} Y_{l_a m_a} = \sum G(1 m_\alpha; l_a m_a; L' M') Y_{L' M'}` of
:func:`~PAOFLOW.hamiltonian._two_center.two_center_dipole_overlap`.  Rows and columns follow
PAOFLOW's orbital order (channel-major, Quantum ESPRESSO magnetic order), and the numbers agree
with the element-wise primitives to rounding.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import scipy.integrate
import scipy.special
from numpy.typing import NDArray

from ._two_center import _CART_TO_YLM_M, real_gaunt_coefficient, real_spherical_harmonic
from .nonlocal_velocity import qe_m_index_to_std


@cache
def gaunt_tensor(l_a: int, l_b: int, big_l: int) -> NDArray[np.float64]:
    r"""Real Gaunt coefficients :math:`G(l_a m_a; l_b m_b; L M)` as an array ``(2l_a+1, 2l_b+1, 2L+1)``.

    Magnetic indices run from :math:`-l` to :math:`+l` along each axis.
    """
    out = np.zeros((2 * l_a + 1, 2 * l_b + 1, 2 * big_l + 1))
    for i, m_a in enumerate(range(-l_a, l_a + 1)):
        for j, m_b in enumerate(range(-l_b, l_b + 1)):
            for k, big_m in enumerate(range(-big_l, big_l + 1)):
                out[i, j, k] = real_gaunt_coefficient(l_a, m_a, l_b, m_b, big_l, big_m)
    return out


@cache
def qe_order(l: int) -> NDArray[np.int_]:
    """Positions of the standard ``m = -l..l`` harmonics in PAOFLOW's (Quantum ESPRESSO) order."""
    return np.array([qe_m_index_to_std(qe_m, l) + l for qe_m in range(1, 2 * l + 2)])


def _allowed_l(l_a: int, l_b: int) -> list[int]:
    return [big_l for big_l in range(abs(l_a - l_b), l_a + l_b + 1) if (big_l - l_a - l_b) % 2 == 0]


def _dipole_bra_l(l: int) -> list[int]:
    return sorted({abs(l - 1), l + 1})


def two_center_pair_blocks(
    bra_channels: list[tuple[int, NDArray[np.float64], dict | None]],
    ket_channels: list[tuple[int, NDArray[np.float64]]],
    displacement: NDArray[np.float64],
    q_grid: NDArray[np.float64],
    *,
    include_dipole: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Overlap and intrinsic-dipole blocks of all orbitals of two sites at one displacement.

    Parameters
    ----------
    bra_channels : list of ``(l, J_l(q), J_g_by_Lp)``
        Radial channels of the bra site: angular momentum, radial Bessel transform
        :math:`J_l(q)` of :math:`R(r)` on ``q_grid``, and (for the dipole) the transforms of the
        modified bra :math:`g(r) = r R(r)` keyed by :math:`L' \in \{|l-1|, l+1\}` (``None`` if
        ``include_dipole`` is false).
    ket_channels : list of ``(l, J_l(q))``
        Radial channels of the ket site.
    displacement : ndarray, shape ``(3,)``
        :math:`\mathbf d = \boldsymbol\tau_{\rm ket} + \Delta\mathbf R - \boldsymbol\tau_{\rm bra}` (bohr).
    q_grid : ndarray, shape ``(n_q,)``
        Reciprocal-space Simpson grid shared by all transforms.
    include_dipole : bool, optional
        Also build the dipole block (origin on the bra site).

    Returns
    -------
    overlap : ndarray, shape ``(n_bra, n_ket)``
        :math:`\langle \varphi_a | \varphi_b(\cdot - \mathbf d)\rangle` for every
        ``(channel, m)`` of the bra rows and ket columns, channel-major, QE magnetic order.
    dipole : ndarray, shape ``(3, n_bra, n_ket)``
        :math:`\langle \varphi_a | r_\alpha | \varphi_b(\cdot - \mathbf d)\rangle` with
        :math:`\mathbf r` measured from the bra centre (bohr); zeros when ``include_dipole``
        is false.
    """
    displacement = np.asarray(displacement, dtype=float)
    distance = float(np.linalg.norm(displacement))
    direction = displacement[np.newaxis, :]
    qq = q_grid * q_grid
    n_bra = sum(2 * l + 1 for l, _, _ in bra_channels)
    n_ket = sum(2 * l + 1 for l, _ in ket_channels)
    overlap = np.zeros((n_bra, n_ket))
    dipole = np.zeros((3, n_bra, n_ket))

    l_max = max(
        [l_a + l_b for l_a, _, _ in bra_channels for l_b, _ in ket_channels]
        + (
            [l_a + 1 + l_b for l_a, _, _ in bra_channels for l_b, _ in ket_channels]
            if include_dipole
            else [0]
        )
    )
    bessel = np.array(
        [scipy.special.spherical_jn(big_l, q_grid * distance) for big_l in range(l_max + 1)]
    )  # (l_max+1, n_q)
    ylm = {
        big_l: np.array(
            [
                float(real_spherical_harmonic(big_l, big_m, direction)[0])
                for big_m in range(-big_l, big_l + 1)
            ]
        )
        for big_l in range(l_max + 1)
    }

    # Radial integrals of every (bra radial function, ket channel, L) in one quadrature:
    # the bra functions are the channels themselves and, for the dipole, the modified bras at L'.
    bra_functions: list[tuple[int, NDArray[np.float64]]] = []  # (angular momentum, J(q))
    bra_slot: dict = {}
    for a, (l_a, j_a, j_g) in enumerate(bra_channels):
        bra_slot[('phi', a)] = len(bra_functions)
        bra_functions.append((l_a, j_a))
        if include_dipole:
            for lp in _dipole_bra_l(l_a):
                bra_slot[('g', a, lp)] = len(bra_functions)
                bra_functions.append((lp, j_g[lp]))
    products = np.array(
        [[qq * j_f * j_b for l_b, j_b in ket_channels] for _, j_f in bra_functions]
    )  # (n_f, n_ket_ch, n_q)
    integrals = scipy.integrate.simpson(
        products[:, :, None, :] * bessel[None, None, :, :], x=q_grid, axis=-1
    )  # (n_f, n_ket_ch, l_max+1)

    def angular(l_f: int, l_b: int, radial: NDArray[np.float64]) -> NDArray[np.float64]:
        """8 sum_L i^(l_f - l_b - L) I_L sum_M G(l_f m; l_b m'; L M) Y_LM, shape (2l_f+1, 2l_b+1), std m order."""
        block = np.zeros((2 * l_f + 1, 2 * l_b + 1))
        for big_l in _allowed_l(l_f, l_b):
            phase = 1j ** (l_f - l_b - big_l)
            block += phase.real * radial[big_l] * (gaunt_tensor(l_f, l_b, big_l) @ ylm[big_l])
        return 8.0 * block

    prefactor = np.sqrt(4.0 * np.pi / 3.0)
    row = 0
    for a, (l_a, _j_a, _j_g) in enumerate(bra_channels):
        rows = qe_order(l_a)
        col = 0
        for b, (l_b, _j_b) in enumerate(ket_channels):
            cols = qe_order(l_b)
            block = angular(l_a, l_b, integrals[bra_slot[('phi', a)], b])
            overlap[row : row + 2 * l_a + 1, col : col + 2 * l_b + 1] = block[np.ix_(rows, cols)]
            if include_dipole:
                for lp in _dipole_bra_l(l_a):
                    inner = angular(
                        lp, l_b, integrals[bra_slot[('g', a, lp)], b]
                    )  # (2lp+1, 2l_b+1)
                    for alpha in range(3):
                        m_alpha = _CART_TO_YLM_M[alpha] + 1
                        coupling = gaunt_tensor(1, l_a, lp)[m_alpha]  # (2l_a+1, 2lp+1)
                        contribution = prefactor * (coupling @ inner)  # (2l_a+1, 2l_b+1)
                        dipole[alpha, row : row + 2 * l_a + 1, col : col + 2 * l_b + 1] += (
                            contribution[np.ix_(rows, cols)]
                        )
            col += 2 * l_b + 1
        row += 2 * l_a + 1
    return overlap, dipole
