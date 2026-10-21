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
(the PAO radial function as bra, :math:`g(r) = r\,R(r)` at :math:`l \pm 1` for the dipole, as the
projector bra of :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.build_nl_real_space_tables`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._two_center import (
    radial_bessel_transform,
    two_center_dipole_overlap_precomputed,
    two_center_overlap_precomputed,
)
from .nonlocal_velocity import (
    PAOCatalog,
    _lattice_search_bounds,
    load_pao_orbitals,
    pao_orbital_signs,
    site_pao_cutoff,
)

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
    the modified bra :math:`g(r) = r R(r)` at :math:`l \pm 1`) are computed once; the angular
    and radial pieces of every pair are memoised across magnetic indices as in
    :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.build_nl_real_space_tables`.
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
    for pair in pairs:
        bra_site = sites[pair.bra_site]
        ket_site = sites[pair.ket_site]
        n_bra, n_ket = len(bra_site.orbitals), len(ket_site.orbitals)
        overlap_block = np.zeros((n_bra, n_ket))
        dipole_block = np.zeros((3, n_bra, n_ket))
        pair_cache: dict = {}
        for a, bra in enumerate(bra_site.orbitals):
            bessel_a = bessel_phi[(bra_site.label, bra.channel_index)]
            bessel_g = {
                lp: bessel_bra[(bra_site.label, bra.channel_index, lp)]
                for lp in (_dipole_bra_channels(bra.l) if include_dipole else ())
            }
            key_a = ('phi', bra_site.label, bra.channel_index)
            key_g = ('g', bra_site.label, bra.channel_index)
            for b, ket in enumerate(ket_site.orbitals):
                bessel_b = bessel_phi[(ket_site.label, ket.channel_index)]
                key_b = ('phi', ket_site.label, ket.channel_index)
                overlap_block[a, b] = two_center_overlap_precomputed(
                    bessel_a,
                    bessel_b,
                    bra.l,
                    bra.m,
                    ket.l,
                    ket.m,
                    pair.displacement,
                    q_grid,
                    cache=pair_cache,
                    keyA=key_a,
                    keyB=key_b,
                )
                for alpha in range(3) if include_dipole else ():
                    dipole_block[alpha, a, b] = two_center_dipole_overlap_precomputed(
                        bessel_g,
                        bessel_b,
                        bra.l,
                        bra.m,
                        ket.l,
                        ket.m,
                        pair.displacement,
                        alpha,
                        q_grid,
                        cache=pair_cache,
                        keyA=key_g,
                        keyB=key_b,
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
    :func:`~PAOFLOW.elphon.elph_bloch.band_velocities` and the projector sums of
    :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.assemble_beta_projections_k`; the orbital
    centres enter through :math:`\boldsymbol\tau` in :func:`lowdin_dipole_kspace`.
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

    def dipoles(k_cryst: NDArray[np.float64]) -> NDArray[np.complex128]:
        folded = k_cryst - np.floor(k_cryst + 0.5)
        k_cart = (folded @ bg) * (2.0 * np.pi / float(alat))
        overlap_k, overlap_gradient_k, dipole_k = assemble_position_kspace(tables, k_cart)
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
    :meth:`PAOFLOW.PAOFLOW.nonlocal_velocity_correction`).
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
    overlap_k, overlap_gradient_k, dipole_k = assemble_position_kspace(tables, k_cart)
    dipole = lowdin_dipole_kspace(
        overlap_k, overlap_gradient_k, dipole_k, tables.tau_orbitals, level=level, on_site=on_site
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
    nk, three, n, _, nspin = dHksp.shape
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
