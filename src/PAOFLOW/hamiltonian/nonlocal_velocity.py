"""PAO orbital catalog and basis-convention helpers of the velocity operator.

Historically this module held the non-local pseudopotential velocity correction
:math:`\\Delta p = m/(i\\hbar)[V_{\\rm NL}, \\mathbf r]`, removed in October 2026: PAOFLOW's
velocity is the commutator :math:`i[H, \\mathbf r]` of a Hamiltonian that already contains
:math:`V_{\\rm NL}`, so the term double counted it (Si: 0.87 of the exact plane-wave
:math:`\\varepsilon_2` with the position dipoles, Fe anomalous Hall peak :math:`\\times1.7`); the
missing part of the orbital-centre velocity is the position-dipole term of
:mod:`PAOFLOW.hamiltonian.position_dipoles`.  The file keeps its name and the pieces that the
dipoles use: the PAO radial channels and per-site orbital mapping (:class:`PAOCatalog`,
:func:`load_pao_orbitals`), the orbital cutoffs and lattice search, PAOFLOW's orbital signs
(:func:`pao_orbital_signs`), the Löwdin transformation (:func:`lowdin_transform`) and the
scalar-to-``(j, m_j)`` rotation (:func:`scalar_to_jm_matrix`).  The catalog needs only the
atomic orbitals (the basis tables of the projection, or the UPF ``PP_PSWFC`` block), so any
pseudopotential type is accepted: the norm-conserving restriction of the removed term (its
Kleinman-Bylander projectors) went with it.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

import numpy as np

from ..inputs.read_upf import UPF


def qe_m_index_to_std(qe_m: int, l: int) -> int:
    r"""Convert Quantum-ESPRESSO 1-indexed real-Y_lm ``m`` to the standard
    :math:`m \in \{-l, \dots, +l\}` convention used by
    :mod:`PAOFLOW.hamiltonian._two_center`.

    QE orders the :math:`2l+1` real spherical harmonics as
    ``(Y_{l,0}, Y_{l,+1}, Y_{l,-1}, Y_{l,+2}, Y_{l,-2}, \\dots)``
    indexed by ``qe_m = 1, 2, 3, ...``.  The mapping is:

    * ``qe_m = 1`` :math:`\\to` ``m = 0``
    * ``qe_m = 2k``  :math:`\\to` ``m = +k``  (``k = 1, \\dots, l``)
    * ``qe_m = 2k+1`` :math:`\\to` ``m = -k``  (``k = 1, \\dots, l``)
    """
    if not (1 <= qe_m <= 2 * l + 1):
        raise ValueError(f'qe_m={qe_m} out of range [1, {2 * l + 1}] for l={l}.')
    if qe_m == 1:
        return 0
    k = qe_m // 2
    return +k if (qe_m % 2 == 0) else -k


def _upf_wfc_to_radial(r: np.ndarray, wfc: np.ndarray, l: int) -> np.ndarray:
    r"""Convert a UPF-stored radial array ``wfc`` (= :math:`r\,R(r)`) to
    :math:`R(r)`.

    For ``l = 0`` :math:`R(r)` tends to a finite constant at the origin;
    when the QE radial mesh starts at ``r[0] = 0``, we extrapolate
    :math:`R(0)` linearly from ``R(r[1])`` and ``R(r[2])``.  For
    ``l \\ge 1`` the orbital satisfies :math:`R(r) \\sim r^l`, so
    :math:`R(0) = 0`.
    """
    r = np.asarray(r, dtype=float)
    wfc = np.asarray(wfc, dtype=float)
    R = np.zeros_like(wfc)
    nz = r > 0.0
    R[nz] = wfc[nz] / r[nz]
    if not np.all(nz):
        # r[0] == 0 case.  Standard QE meshes use r[0] > 0; this guard is
        # purely defensive.
        if l == 0 and np.sum(nz) >= 2:
            r1, r2 = r[nz][0], r[nz][1]
            R1, R2 = R[nz][0], R[nz][1]
            # Linear extrapolation: R(0) = R1 - r1 * (R2 - R1) / (r2 - r1).
            R[~nz] = R1 - r1 * (R2 - R1) / (r2 - r1)
        # l>=1: R[~nz] remains 0.
    return R


@dataclass
class PAOChannelData:
    """One radial PAO channel from a UPF (per species, per shell).

    Attributes
    ----------
    label : str
        UPF shell label (e.g. ``'2S'``, ``'3P'``, ``'3D'``).
    l : int
        Angular momentum.
    R_radial : np.ndarray, shape ``(npoints,)``
        Pseudo-wavefunction radial part :math:`R(r) = (r\\,R(r))/r`
        (UPF's ``r\\,R`` divided by ``r`` with the ``r=0`` end-point
        extrapolated for ``l=0`` -- see :func:`_upf_wfc_to_radial`).
    wfc : np.ndarray, shape ``(npoints,)``
        Original UPF array :math:`r\\,R(r)` (kept verbatim for callers
        that prefer the UPF convention, e.g. matching
        :func:`PAOFLOW.projection.do_atwfc_proj.radialfft_simpson`).
    occupation : float
        UPF ``occupation`` attribute (non-negative; zero for unoccupied
        but available shells).
    """

    label: str
    l: int
    R_radial: np.ndarray
    wfc: np.ndarray
    occupation: float


@dataclass
class PAOSpeciesData:
    """Per-species PAO radial channels.

    Attributes
    ----------
    label : str
        Species label as it appears in ``data_arrays['species']``.
    pseudo_file : str
        UPF filename.
    upf : UPF
        Parsed UPF object (shared with :class:`BetaSpeciesData` when both
        catalogs are loaded for the same species).
    r : np.ndarray, shape ``(npoints,)``
        Radial mesh (Bohr).
    rab : np.ndarray, shape ``(npoints,)``
        Radial integration weights.
    channels : list[PAOChannelData]
        Radial channels in UPF order (one per occupied PSWFC shell).
    """

    label: str
    pseudo_file: str
    upf: UPF
    r: np.ndarray
    rab: np.ndarray
    channels: list[PAOChannelData]


@dataclass
class PAOOrbitalEntry:
    """One real-spherical-harmonic PAO basis function on a given site.

    Attributes
    ----------
    basis_index : int
        Position in the global :attr:`PAOCatalog.basis` list -- matches
        the PAOFLOW orbital index ``0..nawf-1`` produced by
        :func:`PAOFLOW.projection.do_atwfc_proj.build_pswfc_basis_all`.
    site_index : int
        Site index ``J`` (matches ``data_arrays['tau']``).
    channel_index : int
        Index into :attr:`PAOSpeciesData.channels`.
    l : int
        Angular momentum.
    m : int
        Standard real-Y_lm magnetic quantum number :math:`m \\in
        [-l, +l]` (converted from QE's 1-indexed ``qe_m`` via
        :func:`qe_m_index_to_std`).
    qe_m : int
        Original QE 1-indexed magnetic index (preserved for cross-checks
        against ``build_pswfc_basis_all`` and the QE projection matrices).
    label : str
        UPF shell label.
    """

    basis_index: int
    site_index: int
    channel_index: int
    l: int
    m: int
    qe_m: int
    label: str


@dataclass
class PAOSiteData:
    """Per-atom mapping into the PAO basis.

    Attributes
    ----------
    index : int
        Site index ``J``.
    label : str
        Species label.
    tau : np.ndarray, shape ``(3,)``
        Cartesian position :math:`\\boldsymbol{\\tau}_J` (Bohr).
    species : PAOSpeciesData
        Reference to the per-species PAO data.
    orbitals : list[PAOOrbitalEntry]
        Real-Y_lm basis functions belonging to this site, in PAOFLOW
        basis order (channel-major, m sweeps inside each channel).
    basis_offset : int
        Index in the global PAO basis where this site's block starts.
    """

    index: int
    label: str
    tau: np.ndarray
    species: PAOSpeciesData
    orbitals: list[PAOOrbitalEntry]
    basis_offset: int


@dataclass
class PAOCatalog:
    """Top-level container for the loaded PAO atomic-orbital data.

    Attributes
    ----------
    species : dict[str, PAOSpeciesData]
        Per-species data, keyed by species label.
    sites : list[PAOSiteData]
        Per-atom entries (length ``natoms``).
    basis : list[PAOOrbitalEntry]
        Flat PAO basis in PAOFLOW order; ``len(basis) == nawf``.
    total_nlm : int
        :math:`\\sum_J \\sum_\\text{channels} (2l+1)` -- the dimension of
        the PAO basis.
    """

    species: dict[str, PAOSpeciesData] = field(default_factory=dict)
    sites: list[PAOSiteData] = field(default_factory=list)
    basis: list[PAOOrbitalEntry] = field(default_factory=list)
    total_nlm: int = 0


def load_pao_orbitals(data_controller) -> PAOCatalog:
    """Load PAO atomic radial functions and the per-site basis mapping.

    The orbital catalog of the position dipoles
    (:mod:`PAOFLOW.hamiltonian.position_dipoles`); any pseudopotential type
    (norm-conserving, ultrasoft, PAW), since only the atomic orbitals are read.

    When ``data_arrays['atomic_basis']`` is present (PAOFLOW stashes it
    inside :meth:`PAOFLOW.projections` after the projector basis has
    been built), the catalog is reconstructed from those records so
    that the dipoles are those of **exactly the orbitals that produced**
    ``dHksp``.  This is required when the production run uses an
    AE / extended basis (``BASIS/<elem>/*.dat``) whose orbital count
    and radial mesh do not match the UPF pswfc set.

    Falls back to parsing ``upf.pswfc`` directly when
    ``atomic_basis`` is not present (legacy / unit-test path).

    Parameters
    ----------
    data_controller : DataController
        Required:

        - ``arrays['species']``: list of ``(label, pseudo_file)``.
        - ``arrays['atoms']``: per-site list of species labels.
        - ``arrays['tau']``: ``(natoms, 3)`` Cartesian positions (Bohr).
        - ``attributes['fpath']``: directory containing the UPF files.

        Optional:

        - ``arrays['atomic_basis']``: per-orbital dicts (see
          :func:`PAOFLOW.projection.do_atwfc_proj.build_pswfc_basis_all`)
          with keys ``atom, tau, l, m, label, r, wfc``.

    Returns
    -------
    PAOCatalog
        Loaded species data, per-site mapping, and flat PAO basis.

    Raises
    ------
    RuntimeError
        On missing species, a tau/atoms shape mismatch, or when
        ``atomic_basis`` records for the same species disagree on the
        radial mesh.
    """
    arry, attr = data_controller.data_dicts()
    fpath = attr['fpath']

    atomic_basis = arry.get('atomic_basis')

    species_data: dict[str, PAOSpeciesData] = {}
    for label, pseudo_file in arry['species']:
        if label in species_data:
            continue
        upf_path = os.path.join(fpath, pseudo_file)
        upf = UPF(upf_path)

        if atomic_basis:
            channels, r_basis = _channels_from_atomic_basis(label, atomic_basis)
            if channels:
                r_species = r_basis
                rab_species = _trapezoid_weights(r_species)
            else:
                # No basis records for this species (e.g. ghost atom);
                # fall back to UPF pswfc.
                channels = _channels_from_upf(upf, label)
                r_species = upf.r
                rab_species = upf.rab
        else:
            channels = _channels_from_upf(upf, label)
            r_species = upf.r
            rab_species = upf.rab

        species_data[label] = PAOSpeciesData(
            label=label,
            pseudo_file=pseudo_file,
            upf=upf,
            r=r_species,
            rab=rab_species,
            channels=channels,
        )

    atoms = arry['atoms']
    tau = np.asarray(arry['tau'], dtype=float)
    if tau.shape[0] != len(atoms):
        raise RuntimeError(
            f'tau ({tau.shape[0]} rows) does not match atoms list (length {len(atoms)}).'
        )

    sites: list[PAOSiteData] = []
    basis: list[PAOOrbitalEntry] = []
    basis_idx = 0
    for site_i, label in enumerate(atoms):
        if label not in species_data:
            raise RuntimeError(
                f"Atom site {site_i} has species '{label}' but no matching entry in arrays['species']."
            )
        sp = species_data[label]
        site_orbitals: list[PAOOrbitalEntry] = []
        offset = basis_idx
        for ch_idx, ch in enumerate(sp.channels):
            for qe_m in range(1, 2 * ch.l + 2):
                m_std = qe_m_index_to_std(qe_m, ch.l)
                entry = PAOOrbitalEntry(
                    basis_index=basis_idx,
                    site_index=site_i,
                    channel_index=ch_idx,
                    l=ch.l,
                    m=m_std,
                    qe_m=qe_m,
                    label=ch.label,
                )
                site_orbitals.append(entry)
                basis.append(entry)
                basis_idx += 1
        sites.append(
            PAOSiteData(
                index=site_i,
                label=label,
                tau=tau[site_i].copy(),
                species=sp,
                orbitals=site_orbitals,
                basis_offset=offset,
            )
        )

    return PAOCatalog(
        species=species_data,
        sites=sites,
        basis=basis,
        total_nlm=basis_idx,
    )


def _channels_from_upf(upf, species_label: str) -> list['PAOChannelData']:
    """Build the per-species channel list from ``upf.pswfc`` (legacy path)."""
    channels: list[PAOChannelData] = []
    for pao in upf.pswfc:
        shell_label = pao['label']
        l = 'SPDF'.find(shell_label[1].upper())
        if l == -1:
            raise RuntimeError(
                f'Cannot parse angular momentum from PSWFC shell label '
                f"'{shell_label}' for species '{species_label}'."
            )
        wfc = np.asarray(pao['wfc'], dtype=float)
        R_radial = _upf_wfc_to_radial(upf.r, wfc, l)
        channels.append(
            PAOChannelData(
                label=shell_label,
                l=l,
                R_radial=R_radial,
                wfc=wfc,
                occupation=float(pao['occ']),
            )
        )
    return channels


def _channels_from_atomic_basis(
    species_label: str, atomic_basis: list
) -> tuple[list['PAOChannelData'], np.ndarray]:
    """Reconstruct unique radial channels for ``species_label`` from
    PAOFLOW's ``arry['atomic_basis']`` records.

    Channels are deduplicated by ``(label, l)`` and returned in the
    order they first appear in ``atomic_basis``.  All channels for a
    given species must share an identical radial mesh (true for both
    :func:`build_pswfc_basis_all` and :func:`build_aewfc_basis`); the
    shared mesh is returned alongside.

    Returns
    -------
    channels : list[PAOChannelData]
        Per-channel radial data (empty if ``species_label`` has no
        entries in ``atomic_basis``).
    r_mesh : np.ndarray
        Shared radial mesh (empty array when ``channels`` is empty).
    """
    seen: dict[tuple[str, int], PAOChannelData] = {}
    order: list[tuple[str, int]] = []
    r_ref: np.ndarray | None = None
    for record in atomic_basis:
        if record.get('atom') != species_label:
            continue
        label_full = record['label']
        l = int(record['l'])
        key = (label_full, l)
        r = np.asarray(record['r'], dtype=float)
        wfc = np.asarray(record['wfc'], dtype=float)
        if r.shape != wfc.shape:
            raise RuntimeError(
                f"atomic_basis entry for species '{species_label}' shell "
                f"'{label_full}' has mismatched r ({r.shape}) and wfc "
                f'({wfc.shape}) shapes.'
            )
        if r_ref is None:
            r_ref = r
        elif r.shape != r_ref.shape or not np.allclose(r, r_ref, rtol=0, atol=1e-10):
            raise RuntimeError(
                f"atomic_basis records for species '{species_label}' "
                f"shell '{label_full}' use a radial mesh that differs "
                'from earlier shells of the same species; mixed meshes '
                'are not supported.'
            )
        if key in seen:
            continue
        R_radial = _upf_wfc_to_radial(r, wfc, l)
        seen[key] = PAOChannelData(
            label=label_full,
            l=l,
            R_radial=R_radial,
            wfc=wfc,
            occupation=0.0,
        )
        order.append(key)
    if r_ref is None:
        return [], np.empty(0, dtype=float)
    return [seen[k] for k in order], r_ref


def _trapezoid_weights(r: np.ndarray) -> np.ndarray:
    """Trapezoidal-rule integration weights ``rab`` on the mesh ``r``.

    Used in place of ``upf.rab`` when channels come from a
    PAOFLOW-built basis (no UPF-style rab table).  Internal radial
    integrals in the NL-velocity path use the spherical Bessel
    transform (:func:`radial_bessel_transform`), which builds its own
    quadrature; this stub is only stored on ``PAOSpeciesData.rab`` for
    bookkeeping.
    """
    r = np.asarray(r, dtype=float)
    w = np.empty_like(r)
    if r.size == 1:
        w[:] = 0.0
        return w
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    if r.size > 2:
        w[1:-1] = 0.5 * (r[2:] - r[:-2])
    return w


def pao_cutoff_radius(channel: PAOChannelData, r: np.ndarray, tol: float = 1.0e-4) -> float:
    r"""Return the numerical real-space cutoff of a PAO channel.

    The pseudo-wavefunction :math:`r\,R(r)` (stored as
    :attr:`PAOChannelData.wfc`) decays smoothly to zero outside the core.
    This helper returns the smallest ``r_c`` such that
    :math:`|r R(r)| < \mathrm{tol} \times \max_r |r R(r)|`
    for every grid point :math:`r > r_c`.

    Parameters
    ----------
    channel : PAOChannelData
        PAO radial channel.
    r : np.ndarray, shape ``(npoints,)``
        Radial mesh from :attr:`PAOSpeciesData.r`.
    tol : float, optional
        Relative threshold (default ``1e-4``).

    Returns
    -------
    float
        Cutoff radius in Bohr.  ``r[-1]`` is returned if the tail never
        drops below ``tol`` × peak.
    """
    wfc = np.abs(np.asarray(channel.wfc, dtype=float))
    peak = float(wfc.max())
    if peak <= 0.0:
        return 0.0
    threshold = tol * peak
    # Walk from the right: find the largest index where wfc still exceeds threshold.
    above = np.where(wfc > threshold)[0]
    if above.size == 0:
        return 0.0
    last = int(above[-1])
    if last + 1 < r.size:
        return float(r[last + 1])
    return float(r[-1])


def site_pao_cutoff(site: PAOSiteData, tol: float = 1.0e-4) -> float:
    r"""Largest PAO radial cutoff on a given site (via :func:`pao_cutoff_radius`)."""
    r = site.species.r
    cuts = [pao_cutoff_radius(ch, r, tol=tol) for ch in site.species.channels]
    return max(cuts) if cuts else 0.0


def _lattice_search_bounds(a_cart: np.ndarray, max_radius: float) -> tuple[int, int, int]:
    r"""Return half-widths ``(N1, N2, N3)`` such that every Cartesian
    point within radius ``max_radius`` of the origin can be reached by
    integer translations ``n_i ∈ [-N_i, +N_i]``.

    Uses the interplanar spacings of the direct lattice, computed from
    the reciprocal lattice rows (without the ``2π`` factor):
    ``h_i = 1 / |b_i|`` with ``b_i = (a_j × a_k) / Ω``.
    """
    a1, a2, a3 = a_cart[0], a_cart[1], a_cart[2]
    omega = float(np.dot(a1, np.cross(a2, a3)))
    if omega == 0.0:
        raise RuntimeError('Degenerate lattice vectors (zero cell volume).')
    b1 = np.cross(a2, a3) / omega
    b2 = np.cross(a3, a1) / omega
    b3 = np.cross(a1, a2) / omega
    spacings = np.array(
        [1.0 / np.linalg.norm(b1), 1.0 / np.linalg.norm(b2), 1.0 / np.linalg.norm(b3)]
    )
    return tuple(int(np.ceil(max_radius / h)) + 1 for h in spacings)  # type: ignore[return-value]


def pao_orbital_signs(pao_catalog: PAOCatalog) -> np.ndarray:
    r"""Sign of every PAOFLOW orbital with respect to the real-space orbitals of the tables.

    PAOFLOW's atomic Bloch orbitals (``calc_atwfc_k``) are built in reciprocal space with the
    phase :math:`i^l` and the ``calc_ylmg`` harmonics, which carry an extra ``-1`` on odd
    ``|m|``; transformed back to real space they are :math:`(-1)^l (-1)^{|m| \bmod 2}` times
    :math:`R(r) Y_{lm}(\hat r)` with the tesseral harmonics of
    :mod:`PAOFLOW.hamiltonian._two_center`.  A matrix with two PAO indices built from the
    tables therefore transforms with the product of the two signs.  Measured on the Si basis
    (3S 3P 3D 4S 4P) against the plane-wave overlaps and Berry connection of the PAOFLOW
    orbitals (agreement to :math:`10^{-3}` and 2 %); the odd-``|m|`` factor alone flips every
    :math:`s`--:math:`p` and :math:`p`--:math:`d` element.

    Parameters
    ----------
    pao_catalog : PAOCatalog

    Returns
    -------
    np.ndarray, shape ``(nawf,)``
        ``+1`` or ``-1`` per orbital in PAOFLOW basis order.
    """
    return np.array(
        [
            (-1.0) ** orbital.l * (-1.0 if abs(orbital.m) % 2 == 1 else 1.0)
            for orbital in pao_catalog.basis
        ],
        dtype=float,
    )


def lowdin_transform(matrix_k: np.ndarray, overlap_k: np.ndarray) -> np.ndarray:
    r"""Transform an operator matrix from the atomic to the Löwdin-orthonormal PAO basis.

    PAOFLOW's Hamiltonian, ``dHksp`` and band velocities are expressed in the symmetrically
    orthogonalised orbitals :math:`\omega = \varphi S^{-1/2}`; a matrix computed between the
    raw atomic orbitals :math:`\varphi` (such as the two-centre :math:`\Delta p`) must be
    transformed as

    .. math::

        M^{\omega}(\mathbf k) = S^{-1/2}(\mathbf k)\, M^{\varphi}(\mathbf k)\, S^{-1/2}(\mathbf k)

    before it is added to them.  For a diffuse basis this is not a small correction: on the
    standard Si set (Bloch-overlap eigenvalues :math:`5\cdot10^{-4}` to 20) the untransformed
    :math:`\Delta p` has its largest elements 4--5 times too small.

    Parameters
    ----------
    matrix_k : np.ndarray, shape ``(nk, 3, n, n)`` or ``(nk, n, n)``, complex
        Operator in the atomic-orbital Bloch basis.
    overlap_k : np.ndarray, shape ``(nk, n, n)``, complex
        Hermitian positive-definite Bloch overlap :math:`S(\mathbf k)` of the same orbitals,
        in the same phase convention.

    Returns
    -------
    np.ndarray
        The transformed operator, same shape as ``matrix_k``.

    Raises
    ------
    ValueError
        If :math:`S(\mathbf k)` is not positive definite (linearly dependent orbitals).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(overlap_k)
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            f'the PAO overlap S(k) is not positive definite (smallest eigenvalue '
            f'{eigenvalues.min():.3e}): linearly dependent orbitals'
        )
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)[:, None, :]) @ np.conj(
        np.transpose(eigenvectors, (0, 2, 1))
    )
    if matrix_k.ndim == 4:
        return inverse_sqrt[:, None] @ matrix_k @ inverse_sqrt[:, None]
    return inverse_sqrt @ matrix_k @ inverse_sqrt


@functools.lru_cache(maxsize=None)
def _jm_shell_transform(l: int) -> np.ndarray:
    r"""Per-shell unitary mapping RSH-tesseral :math:`\otimes` spin to
    :math:`(j, m_j)`.

    Returns ``T`` of shape ``(2(2l+1), 2(2l+1))`` such that

    .. math::
       \chi_{j,m_j} = \sum_{m_l,\sigma} T[(j,m_j),\,(m_l,\sigma)]\;
                      \tilde T^{m_l}_l\,|\sigma\rangle,

    where :math:`\tilde T^{m_l}_l` are the real cubic harmonics in the
    ``S_bp`` local order (``qe_m = m_l + 1``, i.e.
    :func:`qe_m_index_to_std`).  Rows run in :func:`assign_jm` local order
    (j=l-1/2 block first, then j=l+1/2); columns are flattened as
    ``m_l + (2l+1) * sigma`` with ``sigma in {0=up, 1=down}``.
    """
    from ..projection.do_atwfc_proj import clebsch_jm_matrix, tesseral_to_ylm_matrix

    n_m = 2 * l + 1
    U = tesseral_to_ylm_matrix(l)  # (n_m, n_m): PAOFLOW-tesseral -> Y_lm
    sgn = np.array(
        [(-1.0) ** abs(qe_m_index_to_std(k + 1, l)) for k in range(n_m)],
        dtype=float,
    )
    W = U * sgn[None, :]  # U @ diag(sgn): RSH-tesseral -> Y_lm
    C = clebsch_jm_matrix(l)  # (2 n_m, 2 n_m)
    n_jm = 2 * n_m
    W_spin = np.zeros((n_jm, n_jm), dtype=complex)
    W_spin[:n_m, :n_m] = W
    W_spin[n_m:, n_m:] = W
    # The relativistic Hks (built via topology.clebsch_gordan) uses the
    # complex-conjugate spinor/Y_lm phase convention relative to
    # clebsch_jm_matrix + tesseral_to_ylm_matrix here.  Take the conjugate
    # so that Delta_p transforms under the SAME C4z representation
    # D = diag(exp(-i m_j theta)) that diagonalises Hks covariantly; without
    # this conjugation Delta_p is covariant under D* instead, which leaves
    # the velocity matrix elements <psi|Delta_p|psi> mis-rotated and breaks
    # cubic isotropy of epsilon (eps_xx = eps_zz != eps_yy on Pt_REL).
    return np.conj(C @ W_spin)


def _pao_jm_channel_info(site: PAOSiteData):
    r"""Per-channel PAO layout for the :math:`(j, m_j)` assembly.

    Returns a list of tuples ``(l, col0_local, n_m, rel_col0)`` where
    ``col0_local`` is the channel's column offset in the scalar ``S_bp``
    PAO block, ``n_m = 2l+1``, and ``rel_col0`` is the channel's column
    offset in the global relativistic basis (``2 ×`` the scalar global
    index of the channel's first orbital -- ``assign_jm`` order, matching
    :func:`build_jm_transformation_matrix`).
    """
    seen: dict[int, tuple[int, int, int]] = {}
    for orb in site.orbitals:
        ci = orb.channel_index
        if ci not in seen:
            seen[ci] = (orb.l, orb.basis_index - site.basis_offset, orb.basis_index)
    info = []
    for ci in sorted(seen):
        l_p, col0_local, gidx_first = seen[ci]
        info.append((l_p, col0_local, 2 * l_p + 1, 2 * gidx_first))
    return info


def scalar_to_jm_matrix(matrix_k: np.ndarray, pao_catalog: PAOCatalog) -> np.ndarray:
    r"""Spin-double a scalar PAO matrix and rotate it to the relativistic ``(j, m_j)`` basis.

    Uses the same per-shell unitaries (:func:`_jm_shell_transform`) and channel layout as the
    PAO index of :func:`assemble_beta_projections_jm`, so that e.g. the Bloch overlap
    :math:`S(\mathbf k)` of the scalar orbitals becomes the overlap of the ``(j, m_j)`` basis
    in which the relativistic ``dHksp`` is expressed.

    Parameters
    ----------
    matrix_k : np.ndarray, shape ``(nk, nawf, nawf)``, complex
        Spin-independent matrix in the scalar (tesseral) PAO basis.
    pao_catalog : PAOCatalog

    Returns
    -------
    np.ndarray, shape ``(nk, 2 nawf, 2 nawf)``, complex
    """
    matrix_k = np.asarray(matrix_k)
    nk = matrix_k.shape[0]
    n_rel = 2 * pao_catalog.total_nlm
    out = np.zeros((nk, n_rel, n_rel), dtype=complex)
    channels = []
    for site in pao_catalog.sites:
        for l_p, col0_local, n_m, rel_col0 in _pao_jm_channel_info(site):
            channels.append((l_p, site.basis_offset + col0_local, n_m, rel_col0))
    for l_a, col_a, n_a, rel_a in channels:
        T_a = _jm_shell_transform(l_a)
        for l_b, col_b, n_b, rel_b in channels:
            T_b_h = _jm_shell_transform(l_b).conj().T
            block = matrix_k[:, col_a : col_a + n_a, col_b : col_b + n_b]
            doubled = np.zeros((nk, 2 * n_a, 2 * n_b), dtype=complex)
            doubled[:, :n_a, :n_b] = block
            doubled[:, n_a:, n_b:] = block
            out[:, rel_a : rel_a + 2 * n_a, rel_b : rel_b + 2 * n_b] = T_a @ doubled @ T_b_h
    return out


def _kgrid_to_cart(kgrid_2pi_alat: np.ndarray, alat: float) -> np.ndarray:
    """PAOFLOW's ``(3, nk)`` or ``(nk, 3)`` k-grid in units of 2 pi/alat -> ``(nk, 3)`` 1/bohr."""
    k_arr = np.asarray(kgrid_2pi_alat, dtype=float)
    if k_arr.ndim != 2:
        raise ValueError('kgrid_2pi_alat must be 2-D')
    if k_arr.shape[0] == 3 and k_arr.shape[1] != 3:
        k_arr = k_arr.T  # (3, nk) → (nk, 3)
    elif k_arr.shape[1] != 3:
        raise ValueError('kgrid_2pi_alat must have a length-3 axis')
    return k_arr * (2.0 * np.pi / float(alat))
