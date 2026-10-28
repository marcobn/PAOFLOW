r"""Position dipoles of the PAO basis and the velocity term :math:`i[H, \mathbf d](\mathbf k)`.

PAOFLOW's velocity operator in the orthonormal PAO Bloch basis,

.. math::

    \mathbf v(\mathbf k) = \partial_{\mathbf k} H(\mathbf k) + i[H(\mathbf k), \boldsymbol\tau],

takes the position operator diagonal on the orbital centres :math:`\boldsymbol\tau_i`
(:func:`~PAOFLOW.elphon.elph_bloch.band_velocities`, the ``Dnm`` term of
:func:`~PAOFLOW.hamiltonian.do_gradient.do_gradient`).  For orthonormal localised orbitals
:math:`\omega_i` the exact tight-binding velocity has a third term,

.. math::

    \mathbf v(\mathbf k) = \partial_{\mathbf k} H + i[H, \boldsymbol\tau] + i[H, \mathbf d(\mathbf k)],
    \qquad
    \mathbf d_{ij}(\mathbf R) = \langle \omega_i(\mathbf r - \boldsymbol\tau_i)\,|\,
        \mathbf r - \mathbf R - \boldsymbol\tau_j\,|\,
        \omega_j(\mathbf r - \mathbf R - \boldsymbol\tau_j)\rangle,

whose band-basis form :math:`i(\varepsilon_n - \varepsilon_m)\,d_{nm}` is the analogue of EPW's
Berry-connection term in ``vme = 'wannier'``.  Its largest part is the on-site
:math:`\Delta l = \pm 1` dipole (:math:`\langle s|\mathbf r|p\rangle`,
:math:`\langle p|\mathbf r|d\rangle`) between orbitals of the same atom.  It vanishes on the
diagonal and inside degenerate blocks, so group velocities (transport, adaptive smearing) are
unchanged.

PAOFLOW's basis functions are the atomic orbitals :math:`\varphi` symmetrically (Löwdin)
orthogonalised, :math:`\omega = \varphi\,S^{-1/2}`.  With the two-centre tables of the atomic
orbitals, the overlap :math:`S_{ij}(\mathbf R) = \langle\varphi_i|\varphi_j(\mathbf R)\rangle` and
the intrinsic dipole :math:`X_{ij}(\mathbf R) = \langle\varphi_i(\mathbf r - \boldsymbol\tau_i)|
\mathbf r - \boldsymbol\tau_i|\varphi_j(\mathbf r - \mathbf R - \boldsymbol\tau_j)\rangle`
(origin on the bra site), the Bloch-space dipole of the orthonormal basis is

.. math::

    \mathbf d(\mathbf k) = S^{-1/2} X S^{-1/2} + S^{-1/2}\boldsymbol\tau S^{1/2} - \boldsymbol\tau
        + i\,S^{-1/2}\,\partial_{\mathbf k} S^{1/2},

all matrices at :math:`\mathbf k` (:math:`S(\mathbf k) = \sum_{\mathbf R} e^{i\mathbf k\cdot\mathbf R}
S(\mathbf R)`, likewise :math:`X`), with :math:`\boldsymbol\tau` the diagonal matrix of orbital
centres.  The last three terms come from the cell-position operator, which does not commute with
:math:`S^{\pm 1/2}` (:math:`[\hat{\mathbf R}, M](\mathbf k) = i\,\partial_{\mathbf k} M(\mathbf k)`),
and from the bra-site origin of :math:`X`; :math:`\mathbf d(\mathbf k)` is Hermitian.
``level='intra-atomic'`` keeps the textbook approximation :math:`S \equiv 1` with the on-site
blocks of :math:`X` only.

The real-space tables are built with the primitives of :mod:`PAOFLOW.hamiltonian._two_center`
(the PAO radial function as bra, :math:`g(r) = r\,R(r)` at :math:`l \pm 1` for the dipole),
vectorised over the orbitals of a site pair in :mod:`PAOFLOW.hamiltonian.two_center_blocks`.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray

from ..inputs.read_upf import UPF
from ._two_center import _SPHERE_DIRS, _SPHERE_WEIGHTS, _SPHERE_YLM, radial_bessel_transform
from .nonlocal_velocity import (
    PAOCatalog,
    _lattice_search_bounds,
    load_pao_orbitals,
    pao_orbital_signs,
    site_pao_cutoff,
)
from .two_center_blocks import qe_order, two_center_pair_blocks

DIPOLE_LEVELS = ('full', 'intra-atomic')


@dataclass(frozen=True)
class PAOPair:
    r"""One (bra site, ket site, lattice translation) triple of the position tables.

    Attributes
    ----------
    bra_site : int
        Site of the bra orbitals (home cell).
    ket_site : int
        Site of the ket orbitals, in the periodic image ``deltaR_lattice``.
    deltaR_lattice : tuple of int
        Integer lattice translation ``(n1, n2, n3)`` of the ket site.
    deltaR_cart : ndarray, shape ``(3,)``
        The same translation in Cartesian bohr; it carries the Bloch phase
        :math:`e^{i\mathbf k\cdot\Delta\mathbf R}`.
    displacement : ndarray, shape ``(3,)``
        :math:`\boldsymbol\tau_{\rm ket} + \Delta\mathbf R - \boldsymbol\tau_{\rm bra}` (bohr).
    distance : float
        Length of ``displacement`` (bohr).
    """

    bra_site: int
    ket_site: int
    deltaR_lattice: tuple
    deltaR_cart: NDArray[np.float64]
    displacement: NDArray[np.float64]
    distance: float


@dataclass
class PositionTables:
    r"""Real-space overlap and intrinsic-dipole blocks of the atomic orbitals.

    Attributes
    ----------
    pairs : list of PAOPair
        Pairs of :func:`enumerate_pao_pairs`.
    overlap : list of ndarray
        ``overlap[p]`` has shape ``(n_bra, n_ket)``: :math:`S_{ij}(\Delta\mathbf R)`.
    dipole : list of ndarray
        ``dipole[p]`` has shape ``(3, n_bra, n_ket)``: the intrinsic dipole
        :math:`X_{ij}(\Delta\mathbf R)` with the origin on the bra site (bohr).
    basis_offsets : list of int
        First global orbital index of every site.
    orbitals_per_site : list of int
        Number of orbitals of every site.
    tau_orbitals : ndarray, shape ``(nawf, 3)``
        Centre of every orbital (Cartesian bohr).
    nawf : int
        Number of PAO orbitals.
    """

    pairs: list
    overlap: list
    dipole: list
    basis_offsets: list
    orbitals_per_site: list
    tau_orbitals: NDArray[np.float64]
    nawf: int


def enumerate_pao_pairs(
    pao_catalog: PAOCatalog,
    a_vectors_cart: NDArray[np.float64],
    *,
    extra_pad: float = 0.0,
    pao_tol: float = 1.0e-4,
) -> list[PAOPair]:
    r"""Enumerate the (bra site, ket site, lattice translation) triples within the orbital cutoffs.

    Parameters
    ----------
    pao_catalog : PAOCatalog
        Loaded by :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.load_pao_orbitals`.
    a_vectors_cart : ndarray, shape ``(3, 3)``
        Cartesian lattice vectors (rows, bohr).
    extra_pad : float, optional
        Padding added to every pair cutoff :math:`r_{\rm bra} + r_{\rm ket}` (bohr).
    pao_tol : float, optional
        Relative tail threshold of
        :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.pao_cutoff_radius`.

    Returns
    -------
    list of PAOPair
        Every ordered pair of sites and every image whose centres are closer than the pair
        cutoff, including the on-site pair ``(I, I, 0)``.  The list is symmetric: with
        ``(I, J, dR)`` it also holds ``(J, I, -dR)``.

    Raises
    ------
    ValueError
        If ``a_vectors_cart`` is not ``(3, 3)``.
    """
    a_cart = np.asarray(a_vectors_cart, dtype=float)
    if a_cart.shape != (3, 3):
        raise ValueError(f'a_vectors_cart must have shape (3, 3); got {a_cart.shape}.')
    cutoffs = [site_pao_cutoff(site, tol=pao_tol) for site in pao_catalog.sites]
    max_pair_cut = 2.0 * (max(cutoffs) if cutoffs else 0.0) + extra_pad
    N1, N2, N3 = _lattice_search_bounds(a_cart, max_pair_cut)

    pairs: list[PAOPair] = []
    for bra, bra_site in enumerate(pao_catalog.sites):
        for ket, ket_site in enumerate(pao_catalog.sites):
            cut = cutoffs[bra] + cutoffs[ket] + extra_pad
            if cut <= 0.0:
                continue
            cut2 = cut * cut
            for n1 in range(-N1, N1 + 1):
                for n2 in range(-N2, N2 + 1):
                    for n3 in range(-N3, N3 + 1):
                        deltaR = n1 * a_cart[0] + n2 * a_cart[1] + n3 * a_cart[2]
                        displacement = ket_site.tau + deltaR - bra_site.tau
                        d2 = float(displacement @ displacement)
                        if d2 > cut2:
                            continue
                        pairs.append(
                            PAOPair(
                                bra_site=bra,
                                ket_site=ket,
                                deltaR_lattice=(n1, n2, n3),
                                deltaR_cart=deltaR.copy(),
                                displacement=displacement.copy(),
                                distance=float(np.sqrt(d2)),
                            )
                        )
    return pairs


def _dipole_bra_channels(l: int) -> tuple[int, ...]:
    """Parity-allowed angular momenta ``l +- 1`` of the dipole-modified bra."""
    return tuple(sorted({abs(l - 1), l + 1}))


def build_position_tables(
    pao_catalog: PAOCatalog,
    pairs: list[PAOPair],
    *,
    q_max: float = 20.0,
    n_q: int = 600,
    include_dipole: bool = True,
) -> PositionTables:
    r"""Two-centre overlap and intrinsic-dipole blocks for every pair.

    Parameters
    ----------
    pao_catalog : PAOCatalog
        Loaded by :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.load_pao_orbitals`.
    pairs : list of PAOPair
        Output of :func:`enumerate_pao_pairs`.
    q_max, n_q : float, int, optional
        Reciprocal-space Simpson grid of the radial Bessel transforms
        (:func:`~PAOFLOW.hamiltonian._two_center.radial_bessel_transform`).
    include_dipole : bool, optional
        ``False`` computes the overlap blocks only (the dipole blocks are zero): the input of
        the Löwdin transformation of the non-local velocity correction, at a third of the cost.

    Returns
    -------
    PositionTables

    Notes
    -----
    For a pair with displacement :math:`\mathbf d = \boldsymbol\tau_j + \Delta\mathbf R -
    \boldsymbol\tau_i` the blocks are

    .. math::

        S_{ij} = \int \varphi_i(\mathbf r)\,\varphi_j(\mathbf r - \mathbf d)\,d^3r,
        \qquad
        X^\alpha_{ij} = \int \varphi_i(\mathbf r)\, r_\alpha\, \varphi_j(\mathbf r - \mathbf d)\,d^3r,

    with the origin on the bra orbital.  The radial Bessel transforms of every channel (and of
    the modified bra :math:`g(r) = r R(r)` at :math:`l \pm 1`) are computed once; every pair is
    then one vectorised evaluation of
    :func:`~PAOFLOW.hamiltonian.two_center_blocks.two_center_pair_blocks` (all channels, all
    :math:`L` and all magnetic indices at once), which is what makes the tables of a 44-orbital
    basis a matter of seconds.
    """
    q_grid = np.linspace(0.0, q_max, n_q)
    bessel_phi: dict = {}
    bessel_bra: dict = {}
    for label, species in pao_catalog.species.items():
        for channel_index, channel in enumerate(species.channels):
            bessel_phi[(label, channel_index)] = radial_bessel_transform(
                species.r, channel.R_radial, channel.l, q_grid
            )
            modified_bra = species.r * channel.R_radial
            for lp in _dipole_bra_channels(channel.l) if include_dipole else ():
                bessel_bra[(label, channel_index, lp)] = radial_bessel_transform(
                    species.r, modified_bra, lp, q_grid
                )

    sites = pao_catalog.sites
    overlap_blocks: list[np.ndarray] = []
    dipole_blocks: list[np.ndarray] = []
    channels_of_site = {}
    for index, site in enumerate(sites):
        bra = [
            (
                channel.l,
                bessel_phi[(site.label, channel_index)],
                {
                    lp: bessel_bra[(site.label, channel_index, lp)]
                    for lp in _dipole_bra_channels(channel.l)
                }
                if include_dipole
                else None,
            )
            for channel_index, channel in enumerate(site.species.channels)
        ]
        channels_of_site[index] = (bra, [(l, j) for l, j, _ in bra])
    for pair in pairs:
        overlap_block, dipole_block = two_center_pair_blocks(
            channels_of_site[pair.bra_site][0],
            channels_of_site[pair.ket_site][1],
            pair.displacement,
            q_grid,
            include_dipole=include_dipole,
        )
        overlap_blocks.append(overlap_block)
        dipole_blocks.append(dipole_block)

    tau_orbitals = np.array([sites[orbital.site_index].tau for orbital in pao_catalog.basis])
    return PositionTables(
        pairs=pairs,
        overlap=overlap_blocks,
        dipole=dipole_blocks,
        basis_offsets=[site.basis_offset for site in sites],
        orbitals_per_site=[len(site.orbitals) for site in sites],
        tau_orbitals=tau_orbitals.reshape(-1, 3),
        nawf=pao_catalog.total_nlm,
    )


def _stacked_blocks(tables: PositionTables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair blocks padded to full ``(nawf, nawf)`` matrices, stacked along the pair axis."""
    npairs, nawf = len(tables.pairs), tables.nawf
    overlap = np.zeros((npairs, nawf, nawf))
    dipole = np.zeros((npairs, 3, nawf, nawf))
    translations = np.zeros((npairs, 3))
    for p, pair in enumerate(tables.pairs):
        rows = slice(
            tables.basis_offsets[pair.bra_site],
            tables.basis_offsets[pair.bra_site] + tables.orbitals_per_site[pair.bra_site],
        )
        cols = slice(
            tables.basis_offsets[pair.ket_site],
            tables.basis_offsets[pair.ket_site] + tables.orbitals_per_site[pair.ket_site],
        )
        overlap[p, rows, cols] = tables.overlap[p]
        dipole[p, :, rows, cols] = tables.dipole[p]
        translations[p] = pair.deltaR_cart
    return overlap, dipole, translations


def assemble_position_kspace(
    tables: PositionTables,
    k_points_cart: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    r"""Bloch sums of the overlap, its k-derivative and the intrinsic dipole.

    Parameters
    ----------
    tables : PositionTables
        Output of :func:`build_position_tables`.
    k_points_cart : ndarray, shape ``(nk, 3)``
        k-points in Cartesian inverse bohr (no :math:`2\pi` factor).

    Returns
    -------
    overlap_k : ndarray, shape ``(nk, nawf, nawf)``, complex
        :math:`S(\mathbf k) = \sum_{\Delta\mathbf R} e^{i\mathbf k\cdot\Delta\mathbf R}
        S(\Delta\mathbf R)`.
    overlap_gradient_k : ndarray, shape ``(nk, 3, nawf, nawf)``, complex
        :math:`\partial_{\mathbf k} S(\mathbf k) = \sum_{\Delta\mathbf R} i\,\Delta\mathbf R\,
        e^{i\mathbf k\cdot\Delta\mathbf R} S(\Delta\mathbf R)` (bohr).
    dipole_k : ndarray, shape ``(nk, 3, nawf, nawf)``, complex
        :math:`X(\mathbf k)` (bohr).

    Notes
    -----
    The phase carries the lattice translation only, as the Hamiltonian of
    :func:`~PAOFLOW.elphon.elph_bloch.band_velocities`; the orbital centres enter through
    :math:`\boldsymbol\tau` in :func:`lowdin_dipole_kspace`.
    """
    k_points_cart = np.asarray(k_points_cart, dtype=float)
    if k_points_cart.ndim != 2 or k_points_cart.shape[1] != 3:
        raise ValueError('k_points_cart must have shape (nk, 3).')
    overlap, dipole, translations = _stacked_blocks(tables)
    phases = np.exp(1j * (k_points_cart @ translations.T))  # (nk, npairs)
    overlap_k = np.tensordot(phases, overlap, axes=(1, 0))
    dipole_k = np.tensordot(phases, dipole, axes=(1, 0))
    gradient_phases = phases[:, :, None] * (1j * translations)[None, :, :]  # (nk, npairs, 3)
    overlap_gradient_k = np.tensordot(gradient_phases, overlap, axes=(1, 0))  # (nk, 3, nawf, nawf)
    return overlap_k, overlap_gradient_k, dipole_k


def assemble_overlap_kspace(
    tables: PositionTables, k_points_cart: NDArray[np.float64]
) -> NDArray[np.complex128]:
    r"""Bloch overlap :math:`S(\mathbf k) = \sum_{\Delta\mathbf R} e^{i\mathbf k\cdot\Delta\mathbf R} S(\Delta\mathbf R)` alone.

    Parameters
    ----------
    tables : PositionTables
        Output of :func:`build_position_tables` (``include_dipole=False`` is enough).
    k_points_cart : ndarray, shape ``(nk, 3)``
        k-points in Cartesian inverse bohr.

    Returns
    -------
    ndarray, shape ``(nk, nawf, nawf)``, complex
        The input of :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.lowdin_transform`.
    """
    k_points_cart = np.asarray(k_points_cart, dtype=float)
    if k_points_cart.ndim != 2 or k_points_cart.shape[1] != 3:
        raise ValueError('k_points_cart must have shape (nk, 3).')
    npairs, nawf = len(tables.pairs), tables.nawf
    overlap = np.zeros((npairs, nawf, nawf))
    translations = np.zeros((npairs, 3))
    for p, pair in enumerate(tables.pairs):
        rows = slice(
            tables.basis_offsets[pair.bra_site],
            tables.basis_offsets[pair.bra_site] + tables.orbitals_per_site[pair.bra_site],
        )
        cols = slice(
            tables.basis_offsets[pair.ket_site],
            tables.basis_offsets[pair.ket_site] + tables.orbitals_per_site[pair.ket_site],
        )
        overlap[p, rows, cols] = tables.overlap[p]
        translations[p] = pair.deltaR_cart
    phases = np.exp(1j * (k_points_cart @ translations.T))
    return np.tensordot(phases, overlap, axes=(1, 0))


def on_site_dipoles(tables: PositionTables) -> NDArray[np.float64]:
    r"""Intra-atomic dipole matrix :math:`\langle\varphi_i|\mathbf r - \boldsymbol\tau|\varphi_j\rangle`.

    Parameters
    ----------
    tables : PositionTables

    Returns
    -------
    ndarray, shape ``(3, nawf, nawf)``
        The ``(I, I, 0)`` blocks of :attr:`PositionTables.dipole` (bohr); zero between sites.
    """
    dipole = np.zeros((3, tables.nawf, tables.nawf))
    for p, pair in enumerate(tables.pairs):
        if pair.bra_site != pair.ket_site or any(pair.deltaR_lattice):
            continue
        block = slice(
            tables.basis_offsets[pair.bra_site],
            tables.basis_offsets[pair.bra_site] + tables.orbitals_per_site[pair.bra_site],
        )
        dipole[:, block, block] = tables.dipole[p]
    return dipole


def overlap_sqrt_and_derivative(
    overlap_k: NDArray[np.complex128],
    overlap_gradient_k: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    r"""``S^{1/2}``, ``S^{-1/2}`` and :math:`\partial_{\mathbf k} S^{1/2}` from the eigendecomposition.

    Parameters
    ----------
    overlap_k : ndarray, shape ``(nk, nawf, nawf)``, complex
        Hermitian positive-definite overlap :math:`S(\mathbf k)`.
    overlap_gradient_k : ndarray, shape ``(nk, 3, nawf, nawf)``, complex
        :math:`\partial_{\mathbf k} S(\mathbf k)`.

    Returns
    -------
    sqrt_k, inverse_sqrt_k : ndarray, shape ``(nk, nawf, nawf)``, complex
    sqrt_gradient_k : ndarray, shape ``(nk, 3, nawf, nawf)``, complex

    Raises
    ------
    ValueError
        If :math:`S(\mathbf k)` is not positive definite (linearly dependent orbitals).

    Notes
    -----
    With :math:`S = U\Lambda U^\dagger`, the derivative of the matrix square root solves
    :math:`S^{1/2}\,\partial S^{1/2} + \partial S^{1/2}\,S^{1/2} = \partial S`, i.e.

    .. math::

        (U^\dagger\,\partial S^{1/2}\,U)_{ab}
            = \frac{(U^\dagger\,\partial S\,U)_{ab}}{\sqrt{\lambda_a} + \sqrt{\lambda_b}}.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(overlap_k)
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            f'the PAO overlap S(k) is not positive definite (smallest eigenvalue '
            f'{eigenvalues.min():.3e}): linearly dependent orbitals'
        )
    if eigenvalues.min() < 1.0e-4 * eigenvalues.max():
        warnings.warn(
            'the PAO overlap S(k) is ill-conditioned (eigenvalues %.1e .. %.1e): the Löwdin '
            'position dipoles amplify the nearly dependent combinations; use a basis without '
            'redundant shells (for ultrasoft / PAW the UPF orbitals plus augmentation shells)'
            % (eigenvalues.min(), eigenvalues.max()),
            RuntimeWarning,
            stacklevel=2,
        )
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    adjoint = np.conj(np.transpose(eigenvectors, (0, 2, 1)))
    sqrt_k = (eigenvectors * sqrt_eigenvalues[:, None, :]) @ adjoint
    inverse_sqrt_k = (eigenvectors / sqrt_eigenvalues[:, None, :]) @ adjoint
    denominator = sqrt_eigenvalues[:, None, :, None] + sqrt_eigenvalues[:, None, None, :]
    rotated_gradient = adjoint[:, None] @ overlap_gradient_k @ eigenvectors[:, None]
    sqrt_gradient_k = eigenvectors[:, None] @ (rotated_gradient / denominator) @ adjoint[:, None]
    return sqrt_k, inverse_sqrt_k, sqrt_gradient_k


def lowdin_dipole_kspace(
    overlap_k: NDArray[np.complex128],
    overlap_gradient_k: NDArray[np.complex128],
    dipole_k: NDArray[np.complex128],
    tau_orbitals: NDArray[np.float64],
    *,
    level: str = 'full',
    on_site: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    r"""Position dipoles :math:`\mathbf d(\mathbf k)` of the Löwdin-orthonormal PAO basis.

    Parameters
    ----------
    overlap_k, overlap_gradient_k, dipole_k : ndarray
        Output of :func:`assemble_position_kspace`.
    tau_orbitals : ndarray, shape ``(nawf, 3)``
        Orbital centres (Cartesian bohr).
    level : {'full', 'intra-atomic'}, optional
        ``'full'``: the Löwdin formula of the module docstring.  ``'intra-atomic'``: the
        on-site blocks ``on_site`` alone (:math:`S \equiv 1`).
    on_site : ndarray, shape ``(3, nawf, nawf)``, optional
        :func:`on_site_dipoles`; required for ``level='intra-atomic'``.

    Returns
    -------
    ndarray, shape ``(nk, 3, nawf, nawf)``, complex
        :math:`\mathbf d(\mathbf k)` (bohr), Hermitian at every k up to the quadrature
        accuracy of the tables.

    Raises
    ------
    ValueError
        For an unknown ``level`` or a missing ``on_site`` with ``'intra-atomic'``.
    """
    if level not in DIPOLE_LEVELS:
        raise ValueError(f'level must be one of {DIPOLE_LEVELS}, got {level!r}')
    nk = overlap_k.shape[0]
    if level == 'intra-atomic':
        if on_site is None:
            raise ValueError("level='intra-atomic' needs the on_site dipoles")
        return np.broadcast_to(on_site.astype(complex), (nk,) + on_site.shape).copy()

    sqrt_k, inverse_sqrt_k, sqrt_gradient_k = overlap_sqrt_and_derivative(
        overlap_k, overlap_gradient_k
    )
    tau = np.asarray(tau_orbitals, dtype=float)
    dipole = np.empty_like(dipole_k)
    for alpha in range(3):
        # S^{-1/2} X S^{-1/2}
        term_dipole = inverse_sqrt_k @ dipole_k[:, alpha] @ inverse_sqrt_k
        # S^{-1/2} tau S^{1/2} - tau  (tau diagonal: scale the columns of S^{-1/2})
        term_centres = (inverse_sqrt_k * tau[None, None, :, alpha]) @ sqrt_k
        term_centres[:, np.arange(tau.shape[0]), np.arange(tau.shape[0])] -= tau[None, :, alpha]
        # i S^{-1/2} d_k S^{1/2}
        term_gradient = 1j * (inverse_sqrt_k @ sqrt_gradient_k[:, alpha])
        dipole[:, alpha] = term_dipole + term_centres + term_gradient
    return dipole


def hermiticity_residual(dipole_k: NDArray[np.complex128]) -> float:
    """Largest ``|d - d^dagger|`` over k and Cartesian components (bohr)."""
    return float(np.abs(dipole_k - np.conj(np.transpose(dipole_k, (0, 1, 3, 2)))).max())


def dipole_velocity_term(
    hamiltonian_k: NDArray[np.complex128],
    dipole_k: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""The velocity term :math:`i[H(\mathbf k), \mathbf d(\mathbf k)]` in the orbital basis.

    Parameters
    ----------
    hamiltonian_k : ndarray, shape ``(nk, nawf, nawf)``, complex
        :math:`H(\mathbf k)` (any energy unit).
    dipole_k : ndarray, shape ``(nk, 3, nawf, nawf)``, complex
        :math:`\mathbf d(\mathbf k)` (bohr).

    Returns
    -------
    ndarray, shape ``(nk, 3, nawf, nawf)``, complex
        :math:`i(H\mathbf d - \mathbf d H)` in (energy unit) bohr; in the band basis
        :math:`i(\varepsilon_n - \varepsilon_m)\,d_{nm}`, zero on the diagonal.
    """
    hamiltonian = hamiltonian_k[:, None]
    return 1j * (hamiltonian @ dipole_k - dipole_k @ hamiltonian)


def tesseral_signs(pao_catalog: PAOCatalog) -> NDArray[np.float64]:
    """Sign of every PAOFLOW orbital with respect to the orbitals of the tables.

    Alias of :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.pao_orbital_signs`, where the
    convention :math:`(-1)^l (-1)^{|m| \\bmod 2}` is documented.
    """
    return pao_orbital_signs(pao_catalog)


# --------------------------------------------------------------------------- PAW augmentation
@dataclass
class ProjectorAugmentation:
    r"""PAW augmentation of the overlap and position matrices of the PAO Bloch basis.

    With the projectors :math:`\tilde p^a_i` and the partial waves :math:`\phi^a_i`,
    :math:`\tilde\phi^a_i` of the PAW datasets, a local operator between the all-electron
    counterparts of two pseudo functions is
    :math:`\langle\tilde f|A|\tilde g\rangle + \sum_{a,ij}\langle\tilde f|\tilde p^a_i\rangle
    [\langle\phi^a_i|A|\phi^a_j\rangle - \langle\tilde\phi^a_i|A|\tilde\phi^a_j\rangle]
    \langle\tilde p^a_j|\tilde g\rangle`. For :math:`A = 1` the bracket is :math:`q^a_{ij}`
    (``PP_Q``); for :math:`A = \mathbf r - \boldsymbol\tau_a` it is the dipole of the partial
    waves, a radial integral of :math:`(\phi_i\phi_j - \tilde\phi_i\tilde\phi_j)\,r` times the
    angular factor :math:`\langle Y_{l_i m_i}|\hat r_\alpha|Y_{l_j m_j}\rangle`.

    Attributes
    ----------
    atoms : list of int
        Atom index of every projector block.
    tau_atoms : ndarray ``(nat, 3)``
        Atom positions (Cartesian bohr).
    q_blocks, dipole_blocks : list of ndarray
        ``q^a`` ``(np_a, np_a)`` and ``Delta r^a`` ``(3, np_a, np_a)`` with the magnetic index
        resolved (QE order inside every channel), per atom.
    overlaps : list of tuple
        ``(atom, site, deltaR_cart, block)``: :math:`\langle\tilde p^a_i|\tilde\varphi_\mu
        (\boldsymbol\tau_\mu + \Delta\mathbf R)\rangle`, ``block`` of shape ``(np_a, n_site)``.
    basis_offsets, orbitals_per_site, nawf : as in :class:`PositionTables`.
    """

    atoms: list
    tau_atoms: NDArray[np.float64]
    q_blocks: list
    dipole_blocks: list
    overlaps: list
    basis_offsets: list
    orbitals_per_site: list
    nawf: int


def partial_wave_dipole_radial(upf) -> NDArray[np.float64]:
    r"""Radial dipole integrals of the PAW partial waves, :math:`\int (u_i u_j - \tilde u_i \tilde u_j)\, r\, dr`.

    Parameters
    ----------
    upf : UPF
        A PAW dataset with ``PP_FULL_WFC`` (:attr:`aewfc`, :attr:`pswfc_partial`).

    Returns
    -------
    ndarray ``(nbeta, nbeta)``
        Bohr; zero beyond the projector cutoff by construction (the partial waves coincide).
    """
    from ..elphon.pseudopotential_terms import qe_simpson

    nb = len(upf.beta)
    out = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            n = max(int(upf.beta[i].get('cutoff_index') or upf.r.size), int(upf.beta[j].get('cutoff_index') or upf.r.size))
            n = min(n, upf.r.size)
            diff = (upf.aewfc[i][:n] * upf.aewfc[j][:n] - upf.pswfc_partial[i][:n] * upf.pswfc_partial[j][:n]) * upf.r[:n]
            out[i, j] = qe_simpson(diff, upf.rab[:n])
    return out


@cache
def _angular_dipole(l_i: int, l_j: int) -> NDArray[np.float64]:
    r"""``<Y_{l_i m}|r_hat_alpha|Y_{l_j m'}>`` as ``(3, 2l_i+1, 2l_j+1)`` in QE magnetic order."""
    dirs, weights, ylm = _SPHERE_DIRS, _SPHERE_WEIGHTS, _SPHERE_YLM
    out = np.zeros((3, 2 * l_i + 1, 2 * l_j + 1))
    for a, m_i in enumerate(range(-l_i, l_i + 1)):
        for b, m_j in enumerate(range(-l_j, l_j + 1)):
            prod = weights * ylm[(l_i, m_i)] * ylm[(l_j, m_j)]
            out[:, a, b] = prod @ dirs
    rows, cols = qe_order(l_i), qe_order(l_j)
    return out[:, rows][:, :, cols]


def build_projector_augmentation(
    data_controller,
    pao_catalog: PAOCatalog,
    *,
    q_max: float = 20.0,
    n_q: int = 600,
    pao_tol: float = 1.0e-4,
) -> ProjectorAugmentation | None:
    r"""The PAW augmentation of the PAO overlap and position matrices, or ``None`` without PAW.

    Parameters
    ----------
    data_controller : DataController
        ``arrays['species']``, ``arrays['atoms']``, ``arrays['tau']`` (bohr), ``attr['fpath']``.
    pao_catalog : PAOCatalog
        The PAO orbitals (:func:`~PAOFLOW.hamiltonian.nonlocal_velocity.load_pao_orbitals`).
    q_max, n_q, pao_tol : optional
        As in :func:`build_position_tables` / :func:`enumerate_pao_pairs`.

    Returns
    -------
    ProjectorAugmentation or None
        ``None`` when no species carries augmentation charges. Ultrasoft datasets without
        partial waves (``PP_FULL_WFC``) are skipped with a warning: their overlap augmentation
        ``q_ij`` is applied, their dipole augmentation is not available.
    """
    arrays, attributes = data_controller.data_dicts()
    species_upf = {name: UPF(os.path.join(attributes['fpath'], pseudo)) for name, pseudo in arrays['species']}
    augmented = {
        name
        for name, upf in species_upf.items()
        if getattr(upf, 'ptype', 'NC') != 'NC' or getattr(upf, 'has_augmentation', False)
    }
    if not augmented:
        return None
    q_grid = np.linspace(0.0, q_max, n_q)
    per_species = {}
    for name in augmented:
        upf = species_upf[name]
        if upf.qqq is None:
            continue
        betas = [(int(b['l']), np.asarray(b['wfc'], dtype=float)) for b in upf.beta]
        if upf.aewfc is None:
            warnings.warn(
                '%s: no PAW partial waves (PP_FULL_WFC) in the UPF, the position dipoles are '
                'augmented for the overlap only' % name,
                RuntimeWarning,
                stacklevel=2,
            )
            radial_dipole = np.zeros((len(betas), len(betas)))
        else:
            radial_dipole = partial_wave_dipole_radial(upf)
        # m-resolved q and Delta r, channel-major with QE magnetic order
        sizes = [2 * l + 1 for l, _ in betas]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        nproj = int(offsets[-1])
        q_block = np.zeros((nproj, nproj))
        d_block = np.zeros((3, nproj, nproj))
        for i, (l_i, _) in enumerate(betas):
            for j, (l_j, _) in enumerate(betas):
                si, sj = slice(offsets[i], offsets[i + 1]), slice(offsets[j], offsets[j + 1])
                if l_i == l_j:
                    q_block[si, sj] = upf.qqq[i, j] * np.eye(2 * l_i + 1)
                if abs(l_i - l_j) == 1 and radial_dipole[i, j] != 0.0:
                    d_block[:, si, sj] = radial_dipole[i, j] * _angular_dipole(l_i, l_j)
        cutoff = max(
            float(b['cutoff_radius']) if b.get('cutoff_radius') else float(upf.r[min(int(b['cutoff_index'] or upf.r.size), upf.r.size) - 1])
            for b in upf.beta
        )
        channels = []
        for l, u in betas:
            radial = np.zeros_like(u)
            radial[1:] = u[1:] / upf.r[1:]
            radial[0] = radial[1]
            channels.append((l, radial_bessel_transform(upf.r, radial, l, q_grid)))
        per_species[name] = (q_block, d_block, channels, cutoff)
    if not per_species:
        return None
    # Bessel transforms of the PAO channels of every species
    bessel_phi = {}
    for label, species in pao_catalog.species.items():
        for channel_index, channel in enumerate(species.channels):
            bessel_phi[(label, channel_index)] = radial_bessel_transform(
                species.r, channel.R_radial, channel.l, q_grid
            )
    sites = pao_catalog.sites
    site_cutoffs = [site_pao_cutoff(site, tol=pao_tol) for site in sites]
    a_cart = np.asarray(arrays['a_vectors'], dtype=float) * float(attributes['alat'])
    tau_atoms = np.asarray(arrays['tau'], dtype=float)
    atoms, q_blocks, d_blocks, overlaps = [], [], [], []
    for atom_index, name in enumerate(arrays['atoms']):
        if name not in per_species:
            continue
        q_block, d_block, proj_channels, cutoff = per_species[name]
        atoms.append(atom_index)
        q_blocks.append(q_block)
        d_blocks.append(d_block)
        bra = [(l, j, None) for l, j in proj_channels]
        for site_index, site in enumerate(sites):
            ket = [
                (channel.l, bessel_phi[(site.label, channel_index)])
                for channel_index, channel in enumerate(site.species.channels)
            ]
            reach = cutoff + site_cutoffs[site_index]
            N1, N2, N3 = _lattice_search_bounds(a_cart, reach)
            for n1 in range(-N1, N1 + 1):
                for n2 in range(-N2, N2 + 1):
                    for n3 in range(-N3, N3 + 1):
                        deltaR = n1 * a_cart[0] + n2 * a_cart[1] + n3 * a_cart[2]
                        displacement = site.tau + deltaR - tau_atoms[atom_index]
                        if float(displacement @ displacement) > reach * reach:
                            continue
                        block, _ = two_center_pair_blocks(bra, ket, displacement, q_grid, include_dipole=False)
                        overlaps.append((len(atoms) - 1, site_index, deltaR.copy(), block))
    return ProjectorAugmentation(
        atoms=atoms,
        tau_atoms=tau_atoms,
        q_blocks=q_blocks,
        dipole_blocks=d_blocks,
        overlaps=overlaps,
        basis_offsets=[site.basis_offset for site in sites],
        orbitals_per_site=[len(site.orbitals) for site in sites],
        nawf=pao_catalog.total_nlm,
    )


def augmentation_kspace(
    aug: ProjectorAugmentation,
    k_points_cart: NDArray[np.float64],
    tau_orbitals: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    r"""PAW corrections to ``S(k)``, ``d_k S(k)`` and ``X(k)`` of :func:`assemble_position_kspace`.

    Parameters
    ----------
    aug : ProjectorAugmentation
    k_points_cart : ndarray ``(nk, 3)``
        Cartesian inverse bohr (no :math:`2\pi`).
    tau_orbitals : ndarray ``(nawf, 3)``
        Orbital centres (bohr), the bra origin of :math:`X`.

    Returns
    -------
    overlap, overlap_gradient, dipole : ndarray
        ``(nk, nawf, nawf)``, ``(nk, 3, nawf, nawf)``, ``(nk, 3, nawf, nawf)``: with
        :math:`B^a(\mathbf k) = \sum_{\Delta\mathbf R} e^{i\mathbf k\cdot\Delta\mathbf R}
        \langle\tilde p^a|\tilde\varphi(\Delta\mathbf R)\rangle`,
        :math:`\Delta S = \sum_a B^{a\dagger} q^a B^a`, its k-gradient, and
        :math:`\Delta X_\alpha = \sum_a B^{a\dagger}(\Delta r^a_\alpha + \tau_{a\alpha} q^a)B^a
        - \tau_{\mu\alpha}\Delta S_{\mu\nu}` (origin on the bra orbital).
    """
    k = np.asarray(k_points_cart, dtype=float)
    nk, nawf = k.shape[0], aug.nawf
    tau = np.asarray(tau_orbitals, dtype=float)
    dS = np.zeros((nk, nawf, nawf), dtype=complex)
    dS_grad = np.zeros((nk, 3, nawf, nawf), dtype=complex)
    dX = np.zeros((nk, 3, nawf, nawf), dtype=complex)
    for ia, atom_index in enumerate(aug.atoms):
        nproj = aug.q_blocks[ia].shape[0]
        B = np.zeros((nk, nproj, nawf), dtype=complex)
        dB = np.zeros((nk, 3, nproj, nawf), dtype=complex)
        for ja, site_index, deltaR, block in aug.overlaps:
            if ja != ia:
                continue
            phase = np.exp(1j * (k @ deltaR))  # (nk,)
            start = aug.basis_offsets[site_index]
            stop = start + aug.orbitals_per_site[site_index]
            B[:, :, start:stop] += phase[:, None, None] * block[None]
            dB[:, :, :, start:stop] += (phase[:, None] * (1j * deltaR)[None, :])[:, :, None, None] * block[None, None]
        Bh = np.conj(np.transpose(B, (0, 2, 1)))
        q = aug.q_blocks[ia]
        dS += Bh @ q @ B
        for alpha in range(3):
            dBh = np.conj(np.transpose(dB[:, alpha], (0, 2, 1)))
            dS_grad[:, alpha] += dBh @ q @ B + Bh @ q @ dB[:, alpha]
            op = aug.dipole_blocks[ia][alpha] + aug.tau_atoms[atom_index, alpha] * q
            dX[:, alpha] += Bh @ op @ B
    dX -= tau.T[None, :, :, None] * dS[:, None, :, :]
    return dS, dS_grad, dX


def position_dipole_operator(
    data_controller,
    bg: NDArray[np.float64],
    alat: float,
    *,
    level: str = 'full',
    q_max: float = 20.0,
    n_q: int = 600,
    pao_tol: float = 1.0e-4,
    extra_pad: float = 0.0,
    augmentation: bool | str = 'auto',
) -> Callable[[NDArray[np.float64]], NDArray[np.complex128]]:
    r"""Position dipoles of the PAO basis at arbitrary k, in PAOFLOW's orbital conventions.

    Parameters
    ----------
    data_controller : DataController
        PAOFLOW data after ``projections`` (PAO basis, species, UPF files, ``a_vectors``).
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, :math:`2\pi/a` units).
    alat : float
        Lattice parameter (bohr).
    level : {'full', 'intra-atomic'}, optional
        See :func:`lowdin_dipole_kspace`.
    q_max, n_q, pao_tol, extra_pad : optional
        Table parameters of :func:`build_position_tables` and :func:`enumerate_pao_pairs`.
    augmentation : {'auto', True, False}, optional
        PAW augmentation of the overlap and of the position matrix
        (:class:`ProjectorAugmentation`): the PAO orbitals are pseudo functions whose
        all-electron counterparts carry the partial-wave corrections inside the augmentation
        spheres. ``'auto'`` applies it when a species is ultrasoft / PAW (the ``level='full'``
        Löwdin formula only); ``False`` gives the plain tables.

    Returns
    -------
    callable
        ``K_cryst (nk, 3) -> (nk, 3, nawf, nawf)``: :math:`\mathbf d(\mathbf k)` in bohr, the
        ``position_dipoles`` input of :func:`~PAOFLOW.elphon.elph_bloch.band_velocities`.

    Raises
    ------
    NotImplementedError
        For fully-relativistic runs (scalar-relativistic bases only).

    Notes
    -----
    The tables are built once; every call Fourier-sums them at the requested k (folded to
    ``[-0.5, 0.5)``), applies the Löwdin formula, symmetrises the result
    (:math:`(\mathbf d + \mathbf d^\dagger)/2` removes the quadrature noise) and restores
    PAOFLOW's tesseral signs on both orbital indices (:func:`tesseral_signs`).
    """
    arrays, attributes = data_controller.data_dicts()
    if bool(attributes.get('dftSO', False)):
        raise NotImplementedError(
            'position_dipole_operator supports scalar-relativistic runs only.'
        )
    if level not in DIPOLE_LEVELS:
        raise ValueError(f'level must be one of {DIPOLE_LEVELS}, got {level!r}')
    pao_catalog = load_pao_orbitals(data_controller)
    a_cart = np.asarray(arrays['a_vectors'], dtype=float) * float(alat)
    pairs = enumerate_pao_pairs(pao_catalog, a_cart, extra_pad=extra_pad, pao_tol=pao_tol)
    tables = build_position_tables(pao_catalog, pairs, q_max=q_max, n_q=n_q)
    on_site = on_site_dipoles(tables)
    signs = tesseral_signs(pao_catalog)
    sign_matrix = signs[:, None] * signs[None, :]
    bg = np.asarray(bg, dtype=float)
    aug = None
    if augmentation and level == 'full':
        aug = build_projector_augmentation(
            data_controller, pao_catalog, q_max=q_max, n_q=n_q, pao_tol=pao_tol
        )

    def dipoles(k_cryst: NDArray[np.float64]) -> NDArray[np.complex128]:
        folded = k_cryst - np.floor(k_cryst + 0.5)
        k_cart = (folded @ bg) * (2.0 * np.pi / float(alat))
        overlap_k, overlap_gradient_k, dipole_k = assemble_position_kspace(tables, k_cart)
        if aug is not None:
            d_overlap, d_gradient, d_dipole = augmentation_kspace(aug, k_cart, tables.tau_orbitals)
            overlap_k = overlap_k + d_overlap
            overlap_gradient_k = overlap_gradient_k + d_gradient
            dipole_k = dipole_k + d_dipole
        dipole = lowdin_dipole_kspace(
            overlap_k,
            overlap_gradient_k,
            dipole_k,
            tables.tau_orbitals,
            level=level,
            on_site=on_site,
        )
        dipole = 0.5 * (dipole + np.conj(np.transpose(dipole, (0, 1, 3, 2))))
        return dipole * sign_matrix

    return dipoles


def position_dipoles_on_grid(
    data_controller,
    kgrid_2pi_alat: NDArray[np.float64],
    *,
    level: str = 'full',
    q_max: float = 20.0,
    n_q: int = 600,
    pao_tol: float = 1.0e-3,
) -> NDArray[np.complex128]:
    r"""Position dipoles :math:`\mathbf d(\mathbf k)` on PAOFLOW's k-grid, in the basis of ``dHksp``.

    Parameters
    ----------
    data_controller : DataController
        PAOFLOW data after ``projections`` (PAO basis, species, UPF files, ``a_vectors``,
        ``alat``, ``dftSO``, ``adhoc_SO``).  The tables are cached under
        ``'_PD_tables'`` / ``'_PD_on_site'`` so repeated calls (MPI slices) are cheap.
    kgrid_2pi_alat : ndarray, shape ``(3, nk)`` or ``(nk, 3)``
        Cartesian k-points in units of :math:`2\pi/a` (PAOFLOW's ``arrays['kgrid']``).
    level : {'full', 'intra-atomic'}, optional
        See :func:`lowdin_dipole_kspace`.
    q_max, n_q, pao_tol : optional
        Table parameters (:func:`build_position_tables`, :func:`enumerate_pao_pairs`).

    Returns
    -------
    ndarray, shape ``(nk, 3, n, n)``, complex
        :math:`\mathbf d(\mathbf k)` in bohr with PAOFLOW's orbital signs, Hermitian; ``n`` is
        ``nawf`` for scalar runs, ``2 nawf`` for fully relativistic ones (the scalar matrix
        rotated to the ``(j, m_j)`` basis of ``dHksp`` with
        :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.scalar_to_jm_matrix`) and for the
        ad-hoc spin-orbit path (block-diagonal in spin).

    Notes
    -----
    The input of :func:`inject_dipole_velocity_into_dHksp`; the k-grid layout is the caller's
    (global, or the rank's slice under MPI, as in
    :meth:`PAOFLOW.PAOFLOW.position_dipole_correction`).
    """
    from .nonlocal_velocity import _kgrid_to_cart, _pao_jm_channel_info, scalar_to_jm_matrix

    arrays, attributes = data_controller.data_dicts()
    if level not in DIPOLE_LEVELS:
        raise ValueError(f'level must be one of {DIPOLE_LEVELS}, got {level!r}')
    alat = float(attributes['alat'])
    if '_PD_tables' not in arrays:
        pao_catalog = arrays.get('_NL_pao_catalog') or load_pao_orbitals(data_controller)
        arrays['_NL_pao_catalog'] = pao_catalog
        a_cart = np.asarray(arrays['a_vectors'], dtype=float) * alat
        pairs = enumerate_pao_pairs(pao_catalog, a_cart, pao_tol=pao_tol)
        arrays['_PD_tables'] = build_position_tables(pao_catalog, pairs, q_max=q_max, n_q=n_q)
        arrays['_PD_on_site'] = on_site_dipoles(arrays['_PD_tables'])
    pao_catalog, tables, on_site = (
        arrays['_NL_pao_catalog'],
        arrays['_PD_tables'],
        arrays['_PD_on_site'],
    )
    k_cart = _kgrid_to_cart(kgrid_2pi_alat, alat)
    # blocks of k keep the Bloch sums (nk, 3, nawf, nawf) within a few hundred MB
    kblock = max(1, int(2.0e8 // (3 * 16 * tables.nawf**2)))
    dipole = np.empty((k_cart.shape[0], 3, tables.nawf, tables.nawf), dtype=complex)
    for start in range(0, k_cart.shape[0], kblock):
        overlap_k, overlap_gradient_k, dipole_k = assemble_position_kspace(
            tables, k_cart[start : start + kblock]
        )
        dipole[start : start + kblock] = lowdin_dipole_kspace(
            overlap_k,
            overlap_gradient_k,
            dipole_k,
            tables.tau_orbitals,
            level=level,
            on_site=on_site,
        )
    dipole = 0.5 * (dipole + np.conj(np.transpose(dipole, (0, 1, 3, 2))))
    fully_rel = bool(attributes.get('dftSO', False)) and not bool(attributes.get('adhoc_SO', False))
    if fully_rel:
        # odd-|m| sign inside the shell transform; (-1)^l per channel on both indices
        n_rel = 2 * pao_catalog.total_nlm
        parity = np.ones(n_rel)
        for site in pao_catalog.sites:
            for l_p, _col0, n_m, rel_col0 in _pao_jm_channel_info(site):
                parity[rel_col0 : rel_col0 + 2 * n_m] = (-1.0) ** l_p
        rotated = np.stack(
            [scalar_to_jm_matrix(dipole[:, alpha], pao_catalog) for alpha in range(3)], axis=1
        )
        return rotated * parity[None, None, :, None] * parity[None, None, None, :]
    signs = pao_orbital_signs(pao_catalog)
    dipole = dipole * signs[None, None, :, None] * signs[None, None, None, :]
    if bool(attributes.get('dftSO', False)):  # ad-hoc spin-orbit: [up; down] blocks
        n = pao_catalog.total_nlm
        doubled = np.zeros(dipole.shape[:2] + (2 * n, 2 * n), dtype=complex)
        doubled[:, :, :n, :n] = dipole
        doubled[:, :, n:, n:] = dipole
        return doubled
    return dipole


def inject_dipole_velocity_into_dHksp(
    dHksp: NDArray[np.complex128],
    energies_k: NDArray[np.float64],
    eigenvectors_k: NDArray[np.complex128],
    dipole_k: NDArray[np.complex128],
) -> None:
    r"""Add the position-dipole velocity term to ``dHksp`` in place.

    Parameters
    ----------
    dHksp : ndarray, shape ``(nk, 3, n, n, nspin)``, complex
        PAOFLOW's gradient (eV bohr), which is minus the velocity operator
        (``pksp = -<n|p|m>``); modified in place.
    energies_k, eigenvectors_k : ndarray, shapes ``(nk, n, nspin)`` (eV) and ``(nk, n, n, nspin)``
        ``arrays['E_k']`` and ``arrays['v_k']`` of ``pao_eigh`` at the same k-points, from which
        :math:`H(\mathbf k) = V E V^\dagger` is rebuilt (``do_gradient`` has overwritten ``Hksp``).
    dipole_k : ndarray, shape ``(nk, 3, n, n)``, complex
        :func:`position_dipoles_on_grid` (bohr).

    Returns
    -------
    None
        ``dHksp`` gains :math:`-\,i[H(\mathbf k), \mathbf d(\mathbf k)]` (eV bohr), i.e. the
        velocity gains :math:`+i[H, \mathbf d]` as in
        :func:`~PAOFLOW.elphon.elph_bloch.band_velocities`.

    Raises
    ------
    ValueError
        On shape mismatches.
    """
    nk, _three, n, _, nspin = dHksp.shape
    if dipole_k.shape != (nk, 3, n, n):
        raise ValueError(f'dipole_k must have shape {(nk, 3, n, n)}; got {dipole_k.shape}')
    if energies_k.shape[:2] != (nk, n) or eigenvectors_k.shape[:3] != (nk, n, n):
        raise ValueError('E_k / v_k do not match dHksp')
    for ispin in range(nspin):
        vectors = eigenvectors_k[:, :, :, ispin]
        hamiltonian = (vectors * energies_k[:, None, :, ispin]) @ np.conj(
            np.transpose(vectors, (0, 2, 1))
        )
        dHksp[:, :, :, :, ispin] -= dipole_velocity_term(hamiltonian, dipole_k)
