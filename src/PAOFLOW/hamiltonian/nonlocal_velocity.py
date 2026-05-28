"""Non-local pseudopotential velocity correction (Kleinman--Bylander form).

This module assembles the data needed to evaluate

.. math::
   \\Delta p^{\\mathrm{PAO}}_\\alpha(\\mathbf{k})
     = \\frac{m}{i\\hbar}\\,\\sum_I\\!\\left[
         P_I^{\\dagger}\\,D^I\\,P^{\\alpha}_I
       - (P^{\\alpha}_I)^{\\dagger}\\,D^I\\,P_I
       \\right](\\mathbf{k}),

the non-local contribution to the velocity (momentum) matrix elements that
is otherwise missing from PAOFLOW's ``dH^{PAO}/dk`` operator (see
``TODOs/nonlocal_velocity_correction.md`` §2 for the derivation).

This file currently provides Phase 3a/3b loader infrastructure only.  The
real-space overlap tables and k-space assembly are added in subsequent
phases.

Phase 3 covers **norm-conserving** pseudopotentials only.  Ultrasoft and
PAW augmentation contributions (the :math:`Q_{ij}` charges) are deferred
to a later phase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..inputs.read_upf import UPF
from ._two_center import (
    radial_bessel_transform,
    two_center_dipole_overlap_precomputed,
    two_center_overlap_precomputed,
)


@dataclass
class BetaSpeciesData:
    """Per-species Kleinman--Bylander projector data extracted from a UPF.

    Attributes
    ----------
    label : str
        Species label as it appears in ``data_arrays['species']``.
    pseudo_file : str
        UPF filename (basename), as recorded by Quantum ESPRESSO.
    upf : UPF
        Parsed UPF object.  Beta/D data live in :attr:`UPF.beta` and
        :attr:`UPF.dion` (the latter already in Hartree).
    r : np.ndarray, shape ``(npoints,)``
        Radial mesh (Bohr).
    rab : np.ndarray, shape ``(npoints,)``
        Radial integration weights.
    nproj : int
        Number of KB projectors :math:`\\beta_i(r)` (no :math:`m` index).
    lchannels : list of int
        Angular momentum :math:`l_i` of each projector (length ``nproj``).
    dion : np.ndarray, shape ``(nproj, nproj)``
        Coupling matrix :math:`D_{ij}` in **Hartree**.
    """

    label: str
    pseudo_file: str
    upf: UPF
    r: np.ndarray
    rab: np.ndarray
    nproj: int
    lchannels: list[int]
    dion: np.ndarray


@dataclass
class BetaSiteData:
    """One entry per atomic site in the unit cell.

    Attributes
    ----------
    index : int
        Site index ``I`` (0-based), matching ``data_arrays['tau']``.
    label : str
        Species label.
    tau : np.ndarray, shape ``(3,)``
        Cartesian position :math:`\\boldsymbol{\\tau}_I` (Bohr).
    species : BetaSpeciesData
        Reference to the per-species KB data (shared across sites of the
        same species).
    """

    index: int
    label: str
    tau: np.ndarray
    species: BetaSpeciesData


@dataclass
class BetaCatalog:
    """Top-level container for the loaded KB projector data.

    Attributes
    ----------
    species : dict[str, BetaSpeciesData]
        Per-species data, keyed by species label.
    sites : list of BetaSiteData
        Per-atom entries (length ``natoms``).
    total_nproj_radial : int
        :math:`\\sum_I n_\\beta^I`, ignoring the :math:`m` (orientation)
        index.  The full number of (β,m) basis functions is
        :math:`\\sum_I \\sum_i (2 l_{I,i} + 1)`; see
        :attr:`total_nproj_lm`.
    total_nproj_lm : int
        :math:`\\sum_I \\sum_i (2 l_{I,i} + 1)` -- the dimension of the
        atom-resolved projector block used to build :math:`P_I(\\mathbf{k})`.
    """

    species: dict[str, BetaSpeciesData] = field(default_factory=dict)
    sites: list[BetaSiteData] = field(default_factory=list)
    total_nproj_radial: int = 0
    total_nproj_lm: int = 0


def load_beta_projectors(data_controller) -> BetaCatalog:
    """Load Kleinman--Bylander projectors and per-site mapping.

    Parses each species' UPF (via :class:`PAOFLOW.inputs.read_upf.UPF`)
    once, then resolves the per-atom mapping using
    ``data_arrays['atoms']`` and ``data_arrays['tau']``.  This is the
    Phase 3a loader for the non-local velocity correction (see
    ``TODOs/nonlocal_velocity_correction.md`` §2.5).

    Parameters
    ----------
    data_controller : DataController
        Provider of ``data_arrays`` / ``data_attributes``.  Required:

        - ``arrays['species']``: list of ``(label, pseudo_file)``.
        - ``arrays['atoms']``: per-site list of species labels.
        - ``arrays['tau']``: ``(natoms, 3)`` Cartesian positions (Bohr).
        - ``attributes['fpath']``: directory containing the UPF files
          (typically ``<prefix>.save``).

    Returns
    -------
    BetaCatalog
        Loaded species data and per-site mapping.

    Raises
    ------
    RuntimeError
        If any species' UPF is non-norm-conserving (USPP or PAW) -- Phase 3
        is NC only.
    RuntimeError
        If a species in ``arrays['atoms']`` has no entry in
        ``arrays['species']``.
    """
    arry, attr = data_controller.data_dicts()
    fpath = attr['fpath']

    # Load each unique species' UPF once.
    species_data: dict[str, BetaSpeciesData] = {}
    for label, pseudo_file in arry['species']:
        if label in species_data:
            continue
        upf_path = os.path.join(fpath, pseudo_file)
        upf = UPF(upf_path)
        _require_nc(upf, label, pseudo_file)
        lchannels = [int(b['l']) for b in upf.beta]
        species_data[label] = BetaSpeciesData(
            label=label,
            pseudo_file=pseudo_file,
            upf=upf,
            r=upf.r,
            rab=upf.rab,
            nproj=upf.nproj,
            lchannels=lchannels,
            dion=upf.dion if upf.dion is not None else np.zeros((upf.nproj, upf.nproj)),
        )

    # Resolve per-site mapping.
    atoms = arry['atoms']
    tau = np.asarray(arry['tau'], dtype=float)
    if tau.shape[0] != len(atoms):
        raise RuntimeError(
            f'tau ({tau.shape[0]} rows) does not match atoms list (length {len(atoms)}).'
        )

    sites: list[BetaSiteData] = []
    total_radial = 0
    total_lm = 0
    for i, label in enumerate(atoms):
        if label not in species_data:
            raise RuntimeError(
                f"Atom site {i} has species '{label}' but no matching entry in arrays['species']."
            )
        sp = species_data[label]
        sites.append(BetaSiteData(index=i, label=label, tau=tau[i].copy(), species=sp))
        total_radial += sp.nproj
        total_lm += sum(2 * l + 1 for l in sp.lchannels)

    return BetaCatalog(
        species=species_data,
        sites=sites,
        total_nproj_radial=total_radial,
        total_nproj_lm=total_lm,
    )


def _require_nc(upf: UPF, label: str, pseudo_file: str) -> None:
    """Raise ``RuntimeError`` unless ``upf`` is norm-conserving.

    Phase 3 of the non-local velocity correction supports NC pseudos only;
    USPP / PAW need the additional :math:`Q_{ij}` augmentation term
    (deferred to Phase 6).
    """
    # UPF v2 sets ``type`` (alias for header.pseudo_type).  UPF v1 sets
    # ``ptype`` based on the textual flag in PP_HEADER.
    kind = getattr(upf, 'type', None) or getattr(upf, 'ptype', None)
    if kind is None:
        return  # be lenient if the UPF lacks the tag entirely.
    kind = str(kind).upper()
    if kind != 'NC':
        raise RuntimeError(
            f'Non-local velocity correction (Phase 3) supports norm-conserving '
            f"pseudopotentials only; species '{label}' uses '{pseudo_file}' "
            f"of type '{kind}'. USPP/PAW augmentation will be added in Phase 6."
        )


# ---------------------------------------------------------------------------
# PAO atomic-orbital catalog (the ket / bra of <beta_I | r_alpha | phi_mu>).
# ---------------------------------------------------------------------------


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

    Phase 3b/3c loader for the non-local velocity correction.

    When ``data_arrays['atomic_basis']`` is present (PAOFLOW stashes it
    inside :meth:`PAOFLOW.projections` after the projector basis has
    been built), the catalog is reconstructed from those records so
    that ``Δp`` operates on **exactly the orbitals that produced**
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
        On non-NC pseudos, missing species, a tau/atoms shape mismatch,
        or when ``atomic_basis`` records for the same species disagree
        on the radial mesh.
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
        _require_nc(upf, label, pseudo_file)

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


# ---------------------------------------------------------------------------
# Real-space ΔR enumeration for the <beta_I | ... | phi_J(r - dR)> tables.
# ---------------------------------------------------------------------------


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


def site_beta_cutoff(site: BetaSiteData) -> float:
    r"""Largest β-projector cutoff radius on a given site."""
    cuts = [float(b['cutoff_radius']) for b in site.species.upf.beta]
    return max(cuts) if cuts else 0.0


def site_pao_cutoff(site: PAOSiteData, tol: float = 1.0e-4) -> float:
    r"""Largest PAO radial cutoff on a given site (via :func:`pao_cutoff_radius`)."""
    r = site.species.r
    cuts = [pao_cutoff_radius(ch, r, tol=tol) for ch in site.species.channels]
    return max(cuts) if cuts else 0.0


@dataclass(frozen=True)
class NLPair:
    r"""One real-space pair entering the non-local velocity tables.

    Represents the displacement :math:`\mathbf{d} = (\boldsymbol{\tau}_J +
    \Delta\mathbf{R}) - \boldsymbol{\tau}_I` between the β-projector
    center on site ``I`` (home cell) and the PAO orbital center on site
    ``J`` in the periodic image ``ΔR``.

    Attributes
    ----------
    beta_site : int
        Site index ``I`` (β projector, home cell).
    pao_site : int
        Site index ``J`` (PAO orbital).
    deltaR_lattice : tuple[int, int, int]
        Integer lattice translation ``(n1, n2, n3)`` so that
        :math:`\Delta\mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 +
        n_3 \mathbf{a}_3`.
    deltaR_cart : np.ndarray, shape ``(3,)``
        Cartesian translation (Bohr).
    displacement : np.ndarray, shape ``(3,)``
        :math:`\mathbf{d} = \boldsymbol{\tau}_J + \Delta\mathbf{R} -
        \boldsymbol{\tau}_I` (Bohr).
    distance : float
        :math:`\|\mathbf{d}\|` (Bohr).
    cutoff_used : float
        Pair cutoff :math:`r^\beta_I + r^\varphi_J + \text{pad}` that
        kept this pair in the list.
    """

    beta_site: int
    pao_site: int
    deltaR_lattice: tuple
    deltaR_cart: np.ndarray
    displacement: np.ndarray
    distance: float
    cutoff_used: float


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


def enumerate_nl_pairs(
    beta_catalog: BetaCatalog,
    pao_catalog: PAOCatalog,
    a_vectors_cart: np.ndarray,
    *,
    extra_pad: float = 0.0,
    pao_tol: float = 1.0e-4,
    distance_tol: float = 1.0e-8,
) -> list[NLPair]:
    r"""Enumerate (β-site, PAO-site, ΔR) triples within their pair cutoff.

    For every ordered pair (β-site ``I``, PAO-site ``J``) and every
    integer Bravais translation ``ΔR``, the displacement
    :math:`\mathbf{d} = \boldsymbol{\tau}_J + \Delta\mathbf{R} -
    \boldsymbol{\tau}_I` is checked against the pair cutoff
    :math:`r^\beta_I + r^\varphi_J + \text{pad}`.  Surviving triples are
    returned as :class:`NLPair`.

    The search box is automatically sized from the largest pair cutoff
    via :func:`_lattice_search_bounds`, so the caller need only supply
    the catalogs and the Cartesian lattice (Bohr).

    Parameters
    ----------
    beta_catalog : BetaCatalog
        Loaded by :func:`load_beta_projectors`.
    pao_catalog : PAOCatalog
        Loaded by :func:`load_pao_orbitals`.
    a_vectors_cart : np.ndarray, shape ``(3, 3)``
        Cartesian lattice vectors (Bohr).  Row ``i`` is :math:`\mathbf{a}_i`.
        For PAOFLOW's data layout this is ``arry['a_vectors'] * attr['alat']``.
    extra_pad : float, optional
        Extra padding added to every pair cutoff (Bohr).
    pao_tol : float, optional
        Tolerance forwarded to :func:`pao_cutoff_radius`.
    distance_tol : float, optional
        Pairs with ``|d| < distance_tol`` are still emitted (they are the
        on-site ``I == J`` and ``ΔR = 0`` contributions, important for
        the non-local velocity).

    Returns
    -------
    list[NLPair]
        Pair list (no particular order beyond the outer ``I, J`` loop).

    Raises
    ------
    RuntimeError
        If the β and PAO catalogs have differing numbers of sites
        (they must come from the same crystal).
    """
    if len(beta_catalog.sites) != len(pao_catalog.sites):
        raise RuntimeError(
            f'BetaCatalog has {len(beta_catalog.sites)} sites but PAOCatalog has '
            f'{len(pao_catalog.sites)}; both must be loaded from the same crystal.'
        )
    a_cart = np.asarray(a_vectors_cart, dtype=float)
    if a_cart.shape != (3, 3):
        raise RuntimeError(f'a_vectors_cart must have shape (3,3); got {a_cart.shape}.')

    # Per-site cutoffs.
    r_beta = [site_beta_cutoff(s) for s in beta_catalog.sites]
    r_pao = [site_pao_cutoff(s, tol=pao_tol) for s in pao_catalog.sites]

    max_pair_cut = (max(r_beta) if r_beta else 0.0) + (max(r_pao) if r_pao else 0.0) + extra_pad
    N1, N2, N3 = _lattice_search_bounds(a_cart, max_pair_cut)

    pairs: list[NLPair] = []
    for I, beta_site in enumerate(beta_catalog.sites):
        tau_I = beta_site.tau
        for J, pao_site in enumerate(pao_catalog.sites):
            tau_J = pao_site.tau
            cut = r_beta[I] + r_pao[J] + extra_pad
            if cut <= 0.0:
                continue
            cut2 = cut * cut
            for n1 in range(-N1, N1 + 1):
                for n2 in range(-N2, N2 + 1):
                    for n3 in range(-N3, N3 + 1):
                        dR = n1 * a_cart[0] + n2 * a_cart[1] + n3 * a_cart[2]
                        d = tau_J + dR - tau_I
                        d2 = float(d @ d)
                        if d2 > cut2:
                            continue
                        pairs.append(
                            NLPair(
                                beta_site=I,
                                pao_site=J,
                                deltaR_lattice=(n1, n2, n3),
                                deltaR_cart=dR.copy(),
                                displacement=d.copy(),
                                distance=float(np.sqrt(d2)),
                                cutoff_used=cut,
                            )
                        )
    return pairs


# ---------------------------------------------------------------------------
# Real-space ⟨β|φ⟩ and ⟨β|r_α|φ⟩ tables.
# ---------------------------------------------------------------------------


def n_beta_lm(site: BetaSiteData) -> int:
    r"""Number of LM-decomposed β projectors on a site (``Σ_i (2 l_i + 1)``)."""
    return sum(2 * int(b['l']) + 1 for b in site.species.upf.beta)


def iter_beta_lm(site: BetaSiteData):
    r"""Yield ``(local_lm_index, channel_index, l, m_std, qe_m)`` for the LM
    decomposition of the KB projectors on ``site``.

    Ordering matches the PAO convention used by
    :func:`load_pao_orbitals`: channel-major, with ``qe_m = 1..2l+1``
    sweeping inside each channel.  ``m_std`` is the standard real-Y_lm
    magnetic index (via :func:`qe_m_index_to_std`).
    """
    local = 0
    for ch_idx, b in enumerate(site.species.upf.beta):
        l = int(b['l'])
        for qe_m in range(1, 2 * l + 2):
            yield local, ch_idx, l, qe_m_index_to_std(qe_m, l), qe_m
            local += 1


@dataclass
class NLRealSpaceTables:
    r"""Real-space ⟨β|φ⟩ and ⟨β|r_α|φ⟩ tables on a list of :class:`NLPair`.

    For each pair ``k`` linking β-site ``I`` (home cell) to PAO-site ``J``
    in image ``ΔR``:

    * :attr:`S_bp` ``[k]`` is shape ``(n_beta_lm(I), n_pao(J))`` and
      stores :math:`S^{(k)}_{i m_\beta,\,\mu} = \langle \beta_{I,i,m_\beta}
      \,|\, \varphi_{J,\mu}(\cdot - \Delta\mathbf{R})\rangle`.
    * :attr:`S_rbp` ``[k]`` is shape ``(3, n_beta_lm(I), n_pao(J))`` and
      stores the Cartesian dipole-weighted overlap
      :math:`\langle \beta_{I,i,m_\beta} \,|\, r_\alpha \,|\,
      \varphi_{J,\mu}(\cdot - \Delta\mathbf{R})\rangle`,
      already including the geometry term
      ``M_α(d) + (τ_I)_α · S^{(k)}_{i m_\beta, \mu}``.

    Both arrays are real (everything is on the real-tesseral basis).

    Attributes
    ----------
    pairs : list[NLPair]
        Pair list this table was built on (same order).
    S_bp : list[np.ndarray]
        Per-pair overlap blocks.
    S_rbp : list[np.ndarray]
        Per-pair dipole-weighted overlap blocks (axis 0 = Cartesian α).
    beta_lm_per_site : list[int]
        ``n_beta_lm(I)`` for ``I = 0..nsites-1``.
    pao_per_site : list[int]
        ``len(pao_site.orbitals)`` for each site.
    """

    pairs: list
    S_bp: list
    S_rbp: list
    beta_lm_per_site: list
    pao_per_site: list


def build_nl_real_space_tables(
    beta_catalog: BetaCatalog,
    pao_catalog: PAOCatalog,
    pairs: list[NLPair],
    *,
    q_max: float = 20.0,
    n_q: int = 600,
    include_dipole: bool = True,
) -> NLRealSpaceTables:
    r"""Build per-pair ⟨β|φ⟩ and ⟨β|r_α|φ⟩ blocks.

    Loops over the (β-site, PAO-site, ΔR) triples in ``pairs`` and, for
    each LM-decomposed β projector and each PAO orbital on the
    corresponding sites, evaluates the two-center primitives from
    :mod:`PAOFLOW.hamiltonian._two_center`.

    Parameters
    ----------
    beta_catalog : BetaCatalog
        Loaded by :func:`load_beta_projectors`.
    pao_catalog : PAOCatalog
        Loaded by :func:`load_pao_orbitals`.
    pairs : list[NLPair]
        Output of :func:`enumerate_nl_pairs`.
    q_max, n_q : float, int
        Radial Bessel transform quadrature parameters forwarded to
        :func:`~PAOFLOW.hamiltonian._two_center.two_center_overlap` and
        :func:`~PAOFLOW.hamiltonian._two_center.two_center_dipole_overlap`.
    include_dipole : bool, optional
        If ``False``, :attr:`NLRealSpaceTables.S_rbp` is a list of
        empty arrays.  Useful for cheap diagnostic builds.

    Returns
    -------
    NLRealSpaceTables

    Notes
    -----
    Uses a precomputed radial-Bessel-transform cache: for each species
    and radial channel the transform :math:`J(q) = \int R(r)\,j_l(qr)\,r^2\,dr`
    is computed exactly once on a shared ``q_grid`` (likewise for the
    modified bra :math:`g(r) = r\,R(r)` at the parity-allowed
    :math:`L' = l \pm 1` needed by the dipole).  All subsequent pair
    evaluations reuse these cached :math:`J`'s, cutting cost by roughly
    two orders of magnitude versus naive recomputation.
    """
    beta_sites = beta_catalog.sites
    pao_sites = pao_catalog.sites

    q_grid = np.linspace(0.0, q_max, n_q)

    # --- Precompute J^β_{species,ch}(q) and the modified-bra version for
    #     the dipole on the parity-allowed L' ∈ {l-1, l+1}. ---
    beta_J: dict[tuple[str, int], np.ndarray] = {}
    beta_J_rmul: dict[tuple[str, int, int], np.ndarray] = {}
    for sp_label, sp in beta_catalog.species.items():
        r_b = sp.r
        for ch_idx, b in enumerate(sp.upf.beta):
            l = int(b['l'])
            R_beta = _upf_wfc_to_radial(r_b, np.asarray(b['wfc'], dtype=float), l)
            beta_J[(sp_label, ch_idx)] = radial_bessel_transform(r_b, R_beta, l, q_grid)
            if include_dipole:
                g = r_b * R_beta
                parity = (l + 1) % 2
                for Lp in (abs(l - 1), l + 1):
                    if Lp % 2 != parity:
                        continue
                    key = (sp_label, ch_idx, Lp)
                    if key in beta_J_rmul:
                        continue
                    beta_J_rmul[key] = radial_bessel_transform(r_b, g, Lp, q_grid)

    # --- Precompute J^φ_{species,ch}(q). ---
    pao_J: dict[tuple[str, int], np.ndarray] = {}
    for sp_label, sp in pao_catalog.species.items():
        r_p = sp.r
        for ch_idx, ch in enumerate(sp.channels):
            pao_J[(sp_label, ch_idx)] = radial_bessel_transform(r_p, ch.R_radial, ch.l, q_grid)

    beta_lm_per_site = [n_beta_lm(s) for s in beta_sites]
    pao_per_site = [len(s.orbitals) for s in pao_sites]

    S_bp_list: list[np.ndarray] = []
    S_rbp_list: list[np.ndarray] = []
    beta_lm_tuples_per_site = [list(iter_beta_lm(s)) for s in beta_sites]

    for pair in pairs:
        I = pair.beta_site
        J = pair.pao_site
        d = pair.displacement
        tau_I = beta_sites[I].tau
        sp_b_label = beta_sites[I].label
        sp_p_label = pao_sites[J].label
        n_b = beta_lm_per_site[I]
        n_p = pao_per_site[J]

        S_block = np.zeros((n_b, n_p), dtype=float)
        if include_dipole:
            Sr_block = np.zeros((3, n_b, n_p), dtype=float)
        else:
            Sr_block = np.zeros((3, 0, 0), dtype=float)

        for local_b, ch_b_idx, l_b, m_b, _qe_b in beta_lm_tuples_per_site[I]:
            J_beta = beta_J[(sp_b_label, ch_b_idx)]
            if include_dipole:
                JgA_by_Lp: dict[int, np.ndarray] = {}
                parity_b = (l_b + 1) % 2
                for Lp in (abs(l_b - 1), l_b + 1):
                    if Lp % 2 != parity_b:
                        continue
                    JgA_by_Lp[Lp] = beta_J_rmul[(sp_b_label, ch_b_idx, Lp)]
            for entry in pao_sites[J].orbitals:
                local_p = entry.basis_index - pao_sites[J].basis_offset
                ch_p_idx = entry.channel_index
                l_p = entry.l
                m_p = entry.m
                J_phi = pao_J[(sp_p_label, ch_p_idx)]
                S_val = two_center_overlap_precomputed(
                    J_beta,
                    J_phi,
                    l_b,
                    m_b,
                    l_p,
                    m_p,
                    d,
                    q_grid,
                )
                S_block[local_b, local_p] = S_val
                if include_dipole:
                    for alpha in (0, 1, 2):
                        M_alpha = two_center_dipole_overlap_precomputed(
                            JgA_by_Lp,
                            J_phi,
                            l_b,
                            m_b,
                            l_p,
                            m_p,
                            d,
                            alpha,
                            q_grid,
                        )
                        Sr_block[alpha, local_b, local_p] = M_alpha + tau_I[alpha] * S_val

        S_bp_list.append(S_block)
        S_rbp_list.append(Sr_block)

    return NLRealSpaceTables(
        pairs=pairs,
        S_bp=S_bp_list,
        S_rbp=S_rbp_list,
        beta_lm_per_site=beta_lm_per_site,
        pao_per_site=pao_per_site,
    )


def _dion_lm_expanded(site: BetaSiteData) -> np.ndarray:
    r"""Expand the radial :math:`D^I_{ij}` to the LM-decomposed basis.

    For norm-conserving KB pseudos :math:`D` is diagonal in :math:`(l, m)`:
    it couples only same-:math:`l` radial channels and is scalar in the
    :math:`m` index.  The returned matrix has shape
    ``(n_beta_lm(site), n_beta_lm(site))`` and is the block-diagonal
    expansion ``D^I_{(i,m),(j,m')} = δ_{l_i l_j} δ_{m m'} D^{rad}_{ij}``
    (in **Hartree**, the unit in which the loader stores ``dion``).
    """
    D_rad = site.species.dion
    lchan = site.species.lchannels
    offsets: list[int] = []
    cur = 0
    for l in lchan:
        offsets.append(cur)
        cur += 2 * l + 1
    n = cur
    D_lm = np.zeros((n, n), dtype=float)
    for i, li in enumerate(lchan):
        for j, lj in enumerate(lchan):
            if li != lj:
                continue
            d_ij = float(D_rad[i, j])
            if d_ij == 0.0:
                continue
            for m_idx in range(2 * li + 1):
                D_lm[offsets[i] + m_idx, offsets[j] + m_idx] = d_ij
    return D_lm


def assemble_beta_projections_k(
    beta_catalog: BetaCatalog,
    pao_catalog: PAOCatalog,
    tables: NLRealSpaceTables,
    k_points_cart: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    r"""Fourier-sum the real-space ⟨β|φ⟩ and ⟨β|r_α|φ⟩ tables to k-space.

    For each β-host site :math:`I` and each k-point, build

    .. math::
       P_I(\mathbf{k})_{i m_\beta,\,\mu}
         = \sum_{\Delta\mathbf{R}} e^{i\mathbf{k}\cdot\Delta\mathbf{R}}\,
           \langle \beta_{I,i,m_\beta}\,|\,
                   \varphi_{J(\mu),\mu}(\cdot - \Delta\mathbf{R})\rangle,

    where the sum runs over the pairs in ``tables.pairs`` that have
    ``beta_site == I``.  The dipole variant
    :math:`P^{\alpha}_I(\mathbf{k})` is built analogously from
    ``tables.S_rbp`` and includes the geometry term already baked in by
    :func:`build_nl_real_space_tables`.

    Parameters
    ----------
    beta_catalog, pao_catalog : BetaCatalog, PAOCatalog
    tables : NLRealSpaceTables
        Output of :func:`build_nl_real_space_tables`.  Must have been
        built with ``include_dipole=True``.
    k_points_cart : np.ndarray, shape ``(nk, 3)``
        k-points in Cartesian inverse Bohr (no implicit ``2\pi`` factor;
        the phase used is ``exp(i k · ΔR)`` with ``ΔR`` in Bohr).

    Returns
    -------
    P_list : list of np.ndarray
        ``P_list[I]`` has shape ``(nk, n_beta_lm(I), nawf)`` complex.
    Palpha_list : list of np.ndarray
        ``Palpha_list[I]`` has shape ``(nk, 3, n_beta_lm(I), nawf)``
        complex.
    """
    if any(Sr.size == 0 for Sr in tables.S_rbp):
        raise ValueError(
            'assemble_beta_projections_k requires tables built with include_dipole=True'
        )

    k_points_cart = np.asarray(k_points_cart, dtype=float)
    if k_points_cart.ndim != 2 or k_points_cart.shape[1] != 3:
        raise ValueError('k_points_cart must have shape (nk, 3).')

    nk = k_points_cart.shape[0]
    nsites = len(beta_catalog.sites)
    nawf = pao_catalog.total_nlm
    pao_offsets = [s.basis_offset for s in pao_catalog.sites]

    P_list: list[np.ndarray] = [
        np.zeros((nk, tables.beta_lm_per_site[I], nawf), dtype=complex) for I in range(nsites)
    ]
    Palpha_list: list[np.ndarray] = [
        np.zeros((nk, 3, tables.beta_lm_per_site[I], nawf), dtype=complex) for I in range(nsites)
    ]

    for pair, S, Sr in zip(tables.pairs, tables.S_bp, tables.S_rbp):
        I = pair.beta_site
        J = pair.pao_site
        dR = pair.deltaR_cart
        off = pao_offsets[J]
        sz = S.shape[1]
        phase = np.exp(1j * (k_points_cart @ dR))  # (nk,)
        # P[I][k, i, off:off+sz] += phase[k] * S[i, :]
        P_list[I][:, :, off : off + sz] += phase[:, None, None] * S[None, :, :]
        Palpha_list[I][:, :, :, off : off + sz] += phase[:, None, None, None] * Sr[None, :, :, :]

    return P_list, Palpha_list


def build_nonlocal_velocity_kspace(
    beta_catalog: BetaCatalog,
    pao_catalog: PAOCatalog,
    tables: NLRealSpaceTables,
    k_points_cart: np.ndarray,
    *,
    units: str = 'hartree',
) -> np.ndarray:
    r"""Assemble :math:`\Delta p_\alpha(\mathbf{k})` in the PAO basis.

    Computes

    .. math::
       \Delta p_\alpha(\mathbf{k})
         = \frac{m}{i\hbar}\,\sum_I\!\left[
              P_I^{\dagger}(\mathbf{k})\,D^I\,P^{\alpha}_I(\mathbf{k})
            - (P^{\alpha}_I(\mathbf{k}))^{\dagger}\,D^I\,P_I(\mathbf{k})
            \right]

    in the real-tesseral PAO basis (no spin).  The bracket is
    anti-Hermitian, so the result is Hermitian at every k.

    Parameters
    ----------
    beta_catalog, pao_catalog : BetaCatalog, PAOCatalog
    tables : NLRealSpaceTables
        Built with ``include_dipole=True``.
    k_points_cart : np.ndarray, shape ``(nk, 3)``
        Cartesian inverse Bohr.
    units : {'hartree', 'rydberg'}, optional
        Output units.  The stored :math:`D^I` is in Hartree, so in
        atomic Hartree units (:math:`m = \hbar = 1`) the prefactor
        :math:`m/(i\hbar) = -i`.  Selecting ``'rydberg'`` returns the
        same operator scaled by 2 to match the Rydberg-based convention
        used internally by ``do_gradient`` / ``do_momentum``.

    Returns
    -------
    dP : np.ndarray, shape ``(nk, 3, nawf, nawf)``, complex
    """
    if units not in ('hartree', 'rydberg'):
        raise ValueError(f"units must be 'hartree' or 'rydberg', got {units!r}")

    k_points_cart = np.asarray(k_points_cart, dtype=float)
    nk = k_points_cart.shape[0]
    nawf = pao_catalog.total_nlm

    P_list, Palpha_list = assemble_beta_projections_k(
        beta_catalog, pao_catalog, tables, k_points_cart
    )
    D_per_site = [_dion_lm_expanded(s) for s in beta_catalog.sites]

    dP = np.zeros((nk, 3, nawf, nawf), dtype=complex)
    for I, (P_I, Pa_I, D_I) in enumerate(zip(P_list, Palpha_list, D_per_site)):
        if D_I.size == 0:
            continue
        # P_I:   (nk, n_b, nawf)
        # Pa_I:  (nk, 3, n_b, nawf)
        # D_I:   (n_b, n_b)
        DP = np.einsum('ij,kjn->kin', D_I, P_I)  # (nk, n_b, nawf)
        DPa = np.einsum('ij,kajn->kain', D_I, Pa_I)  # (nk, 3, n_b, nawf)
        term1 = np.einsum('kim,kain->kamn', np.conj(P_I), DPa)  # P†·D·Pa
        term2 = np.einsum('kaim,kin->kamn', np.conj(Pa_I), DP)  # Pa†·D·P
        dP += term1 - term2

    dP *= -1j  # m/(iℏ) in Hartree units
    if units == 'rydberg':
        dP *= 2.0

    # PAOFLOW's projection (``calc_ylmg`` in ``do_atwfc_proj``) builds the
    # real-Y_lm angular factors with an extra ``(-1)`` for every orbital
    # with odd ``|m|`` (px, py, d_xz, d_yz, f_*).  ``real_spherical_harmonic``
    # in ``_two_center.py`` uses the standard tesseral convention (no such
    # sign).  The two extra signs cancel in ``<beta|phi>`` (squared) but not
    # in ``<beta|r_alpha|phi>``, where they survive once per PAO.  Restore
    # the PAOFLOW convention on the PAO indices of dP so it matches the
    # basis in which ``dHksp`` is expressed.
    pao_sign = np.array(
        [-1.0 if (abs(orb.m) % 2 == 1) else 1.0 for orb in pao_catalog.basis],
        dtype=float,
    )
    dP *= pao_sign[None, None, :, None] * pao_sign[None, None, None, :]

    return dP


def compute_nonlocal_velocity_on_grid(
    beta_catalog: BetaCatalog,
    pao_catalog: PAOCatalog,
    tables: NLRealSpaceTables,
    kgrid_2pi_alat: np.ndarray,
    alat: float,
    *,
    units: str = 'rydberg',
) -> np.ndarray:
    r"""Driver-facing wrapper around :func:`build_nonlocal_velocity_kspace`.

    Accepts the PAOFLOW-native k-grid representation (Cartesian in
    units of :math:`2\pi/\mathrm{alat}`, the convention written by
    :func:`PAOFLOW.utils.get_K_grid_fft.get_K_grid_fft`) and converts it
    to inverse Bohr internally before calling
    :func:`build_nonlocal_velocity_kspace`.

    Parameters
    ----------
    beta_catalog, pao_catalog, tables :
        Outputs of :func:`load_beta_projectors`,
        :func:`load_pao_orbitals`, :func:`build_nl_real_space_tables`.
    kgrid_2pi_alat : np.ndarray
        Shape ``(3, nktot)`` or ``(nktot, 3)`` — Cartesian k-points in
        units of :math:`2\pi/\mathrm{alat}` (PAOFLOW's
        ``arry['kgrid']`` is the ``(3, nktot)`` layout).
    alat : float
        Lattice parameter in Bohr.
    units : {'hartree', 'rydberg'}
        Forwarded to :func:`build_nonlocal_velocity_kspace`.

    Returns
    -------
    dP : np.ndarray, complex, shape ``(nktot, 3, nawf, nawf)``
    """
    k_arr = np.asarray(kgrid_2pi_alat, dtype=float)
    if k_arr.ndim != 2:
        raise ValueError('kgrid_2pi_alat must be 2-D')
    if k_arr.shape[0] == 3 and k_arr.shape[1] != 3:
        k_arr = k_arr.T  # (3, nk) → (nk, 3)
    elif k_arr.shape[1] != 3:
        raise ValueError('kgrid_2pi_alat must have a length-3 axis')

    k_cart = k_arr * (2.0 * np.pi / float(alat))
    return build_nonlocal_velocity_kspace(beta_catalog, pao_catalog, tables, k_cart, units=units)


# QE/CODATA Rydberg in eV (matches read_QE_xml.Hart2eV/2 = 13.60569193).
RYDBERG_IN_EV = 13.605693122994


def inject_into_dHksp(
    dHksp: np.ndarray,
    delta_pksp: np.ndarray,
    *,
    units: str = 'rydberg',
    sign: int = +1,
) -> None:
    r"""Add the non-local velocity correction into ``dHksp`` in place.

    Sign + prefactor calibrated against ``epsilon.x`` for Cu (see
    ``example17_Cu_epsilon``); after the Y_lm convention fix on the PAO
    indices of :math:`\Delta p` (see :func:`build_nonlocal_velocity_kspace`),
    cubic isotropy of :math:`\varepsilon_{\alpha\alpha}` is restored
    with ``sign=+1`` and the d→p peak position/height of the QE
    reference is recovered.  Convention:

    .. math::
       \mathrm{dHksp}[k,\alpha,n,m,\sigma]
          \mathrel{+}= \mathrm{sign} \cdot \lambda \cdot
          \Delta p_\alpha(k)_{nm}

    with :math:`\lambda = 13.605693122994` eV/Ry for ``units='rydberg'``
    or :math:`2\lambda` eV/Ha for ``units='hartree'``.

    The correction is broadcast across the spin axis (NC pseudos are
    spin-diagonal at the projector level).

    Parameters
    ----------
    dHksp : np.ndarray, shape ``(nktot, 3, nawf, nawf, nspin)``, complex
        Modified in place.
    delta_pksp : np.ndarray, shape ``(nktot, 3, nawf, nawf)``, complex
        Output of :func:`compute_nonlocal_velocity_on_grid`.
    units : {'hartree', 'rydberg'}
        Unit of ``delta_pksp``.
    sign : int
        ``+1`` (default, calibrated) or ``-1``.  Override only when
        debugging sign conventions.

    Raises
    ------
    ValueError
        On bad ``units``, bad ``sign``, mismatched shapes, or non-complex
        ``dHksp``.
    """
    if units == 'rydberg':
        lam = RYDBERG_IN_EV
    elif units == 'hartree':
        lam = 2.0 * RYDBERG_IN_EV
    else:
        raise ValueError(f"units must be 'rydberg' or 'hartree', got {units!r}")
    if sign not in (-1, 1):
        raise ValueError(f'sign must be -1 or +1, got {sign!r}')
    if dHksp.ndim != 5 or dHksp.shape[1] != 3:
        raise ValueError(f'dHksp must have shape (nktot, 3, nawf, nawf, nspin); got {dHksp.shape}')
    if delta_pksp.ndim != 4 or delta_pksp.shape[1] != 3:
        raise ValueError(
            f'delta_pksp must have shape (nktot, 3, nawf, nawf); got {delta_pksp.shape}'
        )
    nawf_dH = dHksp.shape[2]
    nawf_dP = delta_pksp.shape[2]
    if dHksp.shape[:2] != delta_pksp.shape[:2] or dHksp.shape[3] != nawf_dH:
        raise ValueError(
            f'shape mismatch: dHksp[:4]={dHksp.shape[:4]} vs delta_pksp={delta_pksp.shape}'
        )
    if nawf_dH == nawf_dP:
        spinor = False
    elif nawf_dH == 2 * nawf_dP:
        # Spin-orbit / spinor doubled basis: PAOFLOW lays out
        # [up-orbitals ; down-orbitals] globally (see do_spin_orbit.py).
        # For NC pseudopotentials V_NL is spin-diagonal at the projector
        # level, so the spinor extension is block-diagonal:
        #   Delta_p_spinor = diag(Delta_p, Delta_p)
        spinor = True
    else:
        raise ValueError(
            f'shape mismatch: dHksp[:4]={dHksp.shape[:4]} vs delta_pksp={delta_pksp.shape} '
            f'(neither equal nawf nor 2x for spinor doubling)'
        )

    scale = sign * lam
    nspin = dHksp.shape[-1]
    if not spinor:
        for ispin in range(nspin):
            dHksp[:, :, :, :, ispin] += scale * delta_pksp
    else:
        n = nawf_dP
        for ispin in range(nspin):
            dHksp[:, :, :n, :n, ispin] += scale * delta_pksp
            dHksp[:, :, n:, n:, ispin] += scale * delta_pksp
