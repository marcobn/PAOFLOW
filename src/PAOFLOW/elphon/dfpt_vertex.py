"""Coarse electron-phonon vertex from ``ph.x`` DFPT outputs (the native replacement of EPW's ``epbwrite``).

For every q of the coarse grid the Cartesian matrix elements

.. math::

    g_{mn,\\kappa\\alpha}(k, q) = \\langle \\psi_{m,k+q} | \\partial V / \\partial u_{\\kappa\\alpha}(q)
    | \\psi_{n,k} \\rangle

are assembled from the induced potential of ``ph.x`` (``dvscf``: Hartree and
exchange-correlation response), the bare local pseudopotential derivative and the
Kleinman-Bylander derivative terms, with the nscf wavefunctions of the same save directory
that provides the PAO projections.  The irreducible q of ``ph.x`` are unfolded to the full
grid by rotating the induced potential on the FFT grid (QE ``dfile_star``) and using the
true wavefunctions at ``k`` and ``k + Sq``; time-reversal partners use the conjugate
potential.  The result reproduces EPW's ``epmatq`` (Ry/bohr, no ``1/sqrt(2 M omega)``) and
feeds :func:`PAOFLOW.elphon.elph_bloch.vertex_pao_R` unchanged, or is built directly in
the projected PAO basis ``phi_j,k = sum_n A*_nj(k) psi_nk`` (identical by linearity, never
forming the band-basis array).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..inputs.read_upf import UPF
from .dfpt_io import (
    DFPTFiles,
    SpaceGroup,
    locate_dfpt_files,
    read_dvscf,
    read_irreducible_q,
    read_long_range_tensors,
    read_nscf_plane_wave_setup,
    read_patterns,
    read_space_group,
)
from .elph_bloch import kq_index_map, read_nscf
from .plane_waves import (
    PlaneWaveBlock,
    g_sphere,
    interpolate_to_grid,
    k_plus_g_cart,
    periodic_part_on_grid,
    project_on_plane_waves,
    read_plane_wave_block,
)
from .pseudopotential_terms import (
    KleinmanBylanderProjectors,
    LocalPotentialTable,
    bare_local_perturbation,
    nonlocal_perturbation,
)


@dataclass(frozen=True)
class StarMember:
    """One q of the full coarse grid and how it derives from an irreducible q.

    Attributes
    ----------
    q_cryst : ndarray ``(3,)``
        Grid representative in ``[0, 1)`` (crystal coordinates).
    iq_irr : int
        Index of the irreducible q.
    isym : int
        Operation with ``S q_irr = q_cryst - umklapp`` (``-S q_irr`` when ``time_reversed``).
    time_reversed : bool
    umklapp : ndarray ``(3,)`` int
        ``q_cryst - (+/-) S q_irr``.
    """

    q_cryst: NDArray[np.float64]
    iq_irr: int
    isym: int
    time_reversed: bool
    umklapp: NDArray[np.int_]


def cartesian_rotation(rotation: NDArray[np.int_], at: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cartesian matrix of a crystal rotation ``x -> R x`` (positions in crystal coordinates).

    Parameters
    ----------
    rotation : ndarray ``(3, 3)``
        Acts on crystal coordinates of positions.
    at : ndarray ``(3, 3)``
        Lattice vectors as rows.
    """
    a = np.asarray(at, dtype=float).T  # columns = lattice vectors
    return a @ np.asarray(rotation, dtype=float) @ np.linalg.inv(a)


def rotate_q(
    rotation: NDArray[np.int_],
    q_cryst: NDArray[np.float64],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``S q`` in crystal (reciprocal) coordinates for a crystal rotation of positions."""
    m = cartesian_rotation(rotation, at)
    q_cart = np.asarray(q_cryst, dtype=float) @ np.asarray(bg, dtype=float)
    return np.linalg.solve(np.asarray(bg, dtype=float).T, m @ q_cart)


def star_members(
    q_irr_cryst: NDArray[np.float64],
    group: SpaceGroup,
    qgrid: Sequence[int],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
    time_reversal: bool = True,
) -> list[StarMember]:
    """Unfold the irreducible q-points to the full coarse grid.

    Parameters
    ----------
    q_irr_cryst : ndarray ``(nq_irr, 3)``
    group : SpaceGroup
    qgrid : sequence of int
        Coarse q-grid ``(nq1, nq2, nq3)``.
    at, bg : ndarray ``(3, 3)``
        Lattice and reciprocal vectors (rows).
    time_reversal : bool, optional
        Also use ``q -> -q`` (non-magnetic systems).

    Returns
    -------
    list of StarMember
        One entry per grid point, ordered by irreducible q then by discovery.

    Raises
    ------
    ValueError
        If the stars do not tile the grid exactly once (symmetry mismatch between ``ph.x``
        and the nscf run, or a wrong q-grid).
    """
    ng = np.asarray(qgrid, dtype=int)
    nq = int(np.prod(ng))
    seen: dict[tuple[int, int, int], StarMember] = {}
    members: list[StarMember] = []
    for iq, q0 in enumerate(np.asarray(q_irr_cryst, dtype=float)):
        for trev in [False, True] if time_reversal else [False]:
            for isym in range(group.nsym):
                sq = rotate_q(group.rotations[isym], q0, at, bg)
                if trev:
                    sq = -sq
                lab = np.rint(sq * ng).astype(int)
                if np.abs(lab - sq * ng).max() > 1e-6:
                    raise ValueError('S q = %s is not on the %s q-grid' % (sq, tuple(ng)))
                key = tuple(int(x) for x in lab % ng)
                if key in seen:
                    continue
                q_rep = np.array(key, dtype=float) / ng
                umk = np.rint(q_rep - sq).astype(int)
                member = StarMember(q_rep, iq, isym, trev, umk)
                seen[key] = member
                members.append(member)
    if len(members) != nq:
        raise ValueError(
            'the stars of the %d irreducible q cover %d of the %d grid points; the nscf symmetries '
            'or the q-grid do not match the ph.x run' % (len(q_irr_cryst), len(members), nq)
        )
    return members


def rotate_periodic_field(
    field: NDArray[np.complex128],
    rotation: NDArray[np.int_],
    translation: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """``f'(x) = f(g^-1 x)`` on the FFT grid for the space-group operation ``g: x -> R x - t``.

    Parameters
    ----------
    field : ndarray ``(..., nr1, nr2, nr3)``
    rotation : ndarray ``(3, 3)`` int
    translation : ndarray ``(3,)``
        Crystal coordinates; must be commensurate with the grid.

    Raises
    ------
    ValueError
        If the operation does not map the grid onto itself.
    """
    nr = np.array(field.shape[-3:], dtype=int)
    rinv = np.linalg.inv(np.asarray(rotation, dtype=float))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in nr], indexing='ij'), axis=-1).reshape(-1, 3)
    x = idx / nr + np.asarray(translation, dtype=float)  # x + t
    src = (x @ rinv.T) * nr  # R^-1 (x + t) in grid units
    src_int = np.rint(src).astype(int)
    if np.abs(src - src_int).max() > 1e-6:
        raise ValueError('symmetry operation incompatible with the FFT grid %s' % (tuple(nr),))
    src_int %= nr
    flat = field.reshape(field.shape[:-3] + (-1,))
    gathered = flat[..., (src_int[:, 0] * nr[1] + src_int[:, 1]) * nr[2] + src_int[:, 2]]
    return gathered.reshape(field.shape)


def cartesian_perturbation_potentials(
    dvscf_patterns: NDArray[np.complex128],
    u: NDArray[np.complex128],
    q_irr_cryst: NDArray[np.float64],
    member: StarMember,
    group: SpaceGroup,
    tau_cryst: NDArray[np.float64],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Induced potential of every Cartesian displacement at a star member (QE ``dfile_star``).

    Parameters
    ----------
    dvscf_patterns : ndarray ``(3 nat, nr1, nr2, nr3)``
        ``dvscf`` records at the irreducible q (pattern basis, one spin component).
    u : ndarray ``(3 nat, 3 nat)``
        Displacement patterns (columns).
    q_irr_cryst : ndarray ``(3,)``
        The irreducible q.
    member : StarMember
    group : SpaceGroup
    tau_cryst : ndarray ``(nat, 3)``
        Atomic positions (crystal coordinates).
    at, bg : ndarray ``(3, 3)``

    Returns
    -------
    ndarray ``(3 nat, nr1, nr2, nr3)``, complex
        Lattice-periodic part of ``dV^Hxc / du_{kappa alpha}`` at ``member.q_cryst``.
    """
    nat = tau_cryst.shape[0]
    nr = dvscf_patterns.shape[-3:]
    dv = np.einsum('cv,v...->c...', np.conj(u), dvscf_patterns)  # Cartesian at q0
    dv = dv.reshape(nat, 3, *nr)
    identity = (
        np.array_equal(group.rotations[member.isym], np.eye(3, dtype=int))
        and np.abs(group.translations[member.isym]).max() < 1e-8
    )
    if not identity:
        q0 = np.asarray(q_irr_cryst, dtype=float)
        sq = rotate_q(group.rotations[member.isym], q0, at, bg)
        m_cart = cartesian_rotation(group.rotations[member.isym], at)
        irt = group.equivalent_atoms[member.isym]
        # covariant phases, grid rotation, atom relabel, Cartesian rotation of the direction
        phase0 = np.exp(2j * np.pi * (tau_cryst @ q0))  # (nat,)
        rotated = rotate_periodic_field(
            dv, group.rotations[member.isym], group.translations[member.isym]
        )
        out = np.empty_like(dv)
        for kappa in range(nat):
            kp = int(irt[kappa])
            out[kp] = np.einsum('ba,a...->b...', m_cart, rotated[kappa]) * (
                phase0[kappa] * np.exp(-2j * np.pi * (tau_cryst[kp] @ sq))
            )
        dv = out
    else:
        sq = np.asarray(q_irr_cryst, dtype=float)
    if member.time_reversed:
        dv = np.conj(dv)
        sq = -sq
    shift = np.asarray(member.umklapp, dtype=float)
    if np.abs(shift).max() > 0:
        idx = np.stack(np.meshgrid(*[np.arange(n) / n for n in nr], indexing='ij'), axis=-1)
        dv = dv * np.exp(-2j * np.pi * (idx @ shift))[None, None]
    if np.abs(sq + shift - member.q_cryst).max() > 1e-6:
        raise ValueError(
            'inconsistent star member: S q0 + G = %s, representative %s'
            % (sq + shift, member.q_cryst)
        )
    return dv.reshape(3 * nat, *nr)


class CoarseVertexSetup:
    """Everything that does not depend on q: wavefunction geometry, tables, symmetries.

    Parameters
    ----------
    save_dir : str
        nscf save directory (Gamma-centred k-grid, the one used for the projections): either
        the full grid or, with ``K_POINTS automatic``, its irreducible wedge.
    dfpt_dir : str
        ``ph.x`` outputs (see :func:`PAOFLOW.elphon.dfpt_io.locate_dfpt_files`).
    qgrid : sequence of int
        Coarse q-grid of the DFPT run.
    prefix : str, optional
        QE prefix (inferred when omitted).
    pseudo_dir : str, optional
        Directory of the UPF files (default: the save directory, where QE copies them).
    cache_bytes : int, optional
        Memory budget of the per-rank cache of plane-wave blocks and projectors.
    unfold_kpoints : bool, optional
        The save holds the irreducible wedge: rotate the stored Bloch functions to every
        point of the full grid (:class:`~PAOFLOW.elphon.wavefunction_symmetry.WavefunctionProvider`).
        ``None`` (default) detects it from ``data-file-schema.xml`` (a Gamma-centred
        ``monkhorst_pack`` element with fewer stored k-points than the grid); ``False`` raises
        for such a save.  The projections ``A`` must then come from
        ``pao_hamiltonian_from_save(..., unfold_kpoints=True)``.

    Attributes
    ----------
    provider : WavefunctionProvider or None
        Set when the k-points are unfolded.
    """

    def __init__(
        self,
        save_dir: str,
        dfpt_dir: str,
        qgrid: Sequence[int],
        prefix: str | None = None,
        pseudo_dir: str | None = None,
        cache_bytes: int = 256 * 1024**2,
        unfold_kpoints: bool | None = None,
    ):
        self.save_dir = save_dir
        self.files: DFPTFiles = locate_dfpt_files(dfpt_dir, prefix)
        self.setup = read_nscf_plane_wave_setup(save_dir)
        self.nscf = read_nscf(save_dir)
        self.group = read_space_group(save_dir)
        from .wavefunction_symmetry import stores_irreducible_wedge

        self.provider = None
        irreducible = stores_irreducible_wedge(save_dir)
        if unfold_kpoints is None:
            unfold_kpoints = irreducible
        elif irreducible and not unfold_kpoints:
            raise ValueError(
                '%s holds the irreducible wedge of its k-grid; pass unfold_kpoints=True (and build '
                'the projections with pao_hamiltonian_from_save(..., unfold_kpoints=True))'
                % save_dir
            )
        if unfold_kpoints:
            from .wavefunction_symmetry import provider_for

            self.provider = provider_for(save_dir, int(self.setup['nspin']))
            self.nscf = read_nscf(save_dir, unfold=True)
        if self.setup['lsda']:
            raise NotImplementedError('spin-polarised (LSDA) DFPT couplings are not supported')
        self.npol = int(self.setup['npol'])
        self.spin_orbit = bool(self.setup['spinorbit'])
        self.qgrid = tuple(int(n) for n in qgrid)
        self.at, self.bg, self.alat = self.setup['at'], self.setup['bg'], self.setup['alat']
        self.tpiba = 2.0 * np.pi / self.alat
        self.omega = self.setup['omega']
        self.tau_alat = self.setup['tau_alat']
        self.tau_cryst = self.tau_alat @ np.linalg.inv(self.at)
        self.species_of_atom = list(self.setup['atom_species'])
        self.nat = len(self.species_of_atom)
        self.fft_grid = self.setup['fft_grid']
        self.fft_smooth = self.setup['fft_smooth']
        pseudo_dir = save_dir if pseudo_dir is None else pseudo_dir
        self.upf = {
            name: UPF(os.path.join(pseudo_dir, fname))
            for name, fname in self.setup['pseudo_files'].items()
        }
        q_irr = read_irreducible_q(self.files, self.bg)
        snapped = np.rint(q_irr * np.asarray(self.qgrid)) / np.asarray(self.qgrid)
        if np.abs(snapped - q_irr).max() > 1e-6:
            raise ValueError(
                'the irreducible q of %s are not on the %s grid' % (dfpt_dir, self.qgrid)
            )
        self.q_irr = snapped  # the dyn files print q with nine digits; the grid values are exact
        self.members = star_members(self.q_irr, self.group, self.qgrid, self.at, self.bg)
        self.epsilon, self.born_charges = read_long_range_tensors(self.files)
        # smooth G sphere for the bare local term and the dense->smooth interpolation
        gcut_smooth = min(self.setup['ecutrho_ry'], 4.0 * self.setup['ecutwfc_ry']) / self.tpiba**2
        self.g_miller, self.g_cart = g_sphere(self.fft_smooth, self.bg, gcut_smooth)
        qmax_loc = np.sqrt(self.setup['ecutrho_ry']) + self.tpiba * np.sqrt(
            max(float(np.dot(m.q_cryst @ self.bg, m.q_cryst @ self.bg)) for m in self.members)
        )
        self.vloc = {
            name: LocalPotentialTable(upf, self.omega, qmax_loc) for name, upf in self.upf.items()
        }
        self.kb = KleinmanBylanderProjectors(
            self.upf,
            self.species_of_atom,
            self.omega,
            np.sqrt(self.setup['ecutwfc_ry']),
            spin_orbit=self.spin_orbit,
        )
        self.kpts_cryst = self.nscf['kpts_cryst']
        self.nk = int(self.kpts_cryst.shape[0])
        self.nbnd = int(self.setup['nbnd'])
        self.cache_bytes = int(cache_bytes)
        self._blocks: dict[int, PlaneWaveBlock] = {}
        self._beta: dict[int, NDArray[np.complex128]] = {}
        self._cached_bytes = 0

    def block(self, ik: int) -> PlaneWaveBlock:
        """Plane-wave block of k-point ``ik`` (cached within ``cache_bytes``).

        With unfolded k-points the block is rotated from its irreducible representative.
        """
        if ik not in self._blocks:
            if self.provider is not None:
                blk = self.provider.block(ik)
            else:
                blk = read_plane_wave_block(self.save_dir, ik, self.alat, self.setup['nspin'])
            size = blk.coefficients.nbytes + blk.miller.nbytes
            if self._cached_bytes + size > self.cache_bytes:
                self.clear_cache()
            self._blocks[ik] = blk
            self._cached_bytes += size
        return self._blocks[ik]

    def projectors(self, ik: int) -> NDArray[np.complex128]:
        """Kleinman-Bylander projectors on the sphere of k-point ``ik`` (cached)."""
        if ik not in self._beta:
            blk = self.block(ik)
            beta = self.kb.at(k_plus_g_cart(blk, self.bg), self.tau_alat, self.tpiba)
            if self._cached_bytes + beta.nbytes > self.cache_bytes:
                self.clear_cache()
                self.block(ik)
            self._beta[ik] = beta
            self._cached_bytes += beta.nbytes
        return self._beta[ik]

    def clear_cache(self) -> None:
        """Drop the cached blocks and projectors."""
        self._blocks.clear()
        self._beta.clear()
        self._cached_bytes = 0

    def coarse_q(self) -> NDArray[np.float64]:
        """The full coarse q-grid in the order of :attr:`members` (crystal coordinates)."""
        return np.array([m.q_cryst for m in self.members])

    def long_range_tensors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(epsilon (3, 3), born_charges (nat, 3, 3))``, zeros when ``ph.x`` had no ``epsil``."""
        eps = np.zeros((3, 3)) if self.epsilon is None else np.asarray(self.epsilon, dtype=float)
        zst = (
            np.zeros((self.nat, 3, 3))
            if self.born_charges is None
            else np.asarray(self.born_charges, dtype=float)
        )
        return eps, zst

    def pao_vertex_at_q(
        self,
        member: StarMember,
        A: NDArray[np.complex128],
        kgrid_idx: NDArray[np.integer],
        ng: tuple[int, int, int],
        basis: str = 'auto',
        bands: Sequence[int] | None = None,
    ) -> NDArray[np.complex128]:
        """PAO-gauge vertex ``g_q(R_e)`` of one coarse q, the output of :func:`vertex_pao_R`.

        Parameters
        ----------
        member : StarMember
        A : ndarray ``(nbnd, nawf, nk)``
            PAO projections ``A_ni(k) = <phi_i|psi_nk>`` of the same nscf save.
        kgrid_idx : ndarray ``(nk, 3)`` int
            Integer grid labels of the k-points.
        ng : tuple of int
            Coarse k-grid dimensions.
        basis : {'auto', 'pao', 'band'}, optional
            ``'pao'`` applies the perturbation to the projected PAO Bloch functions
            ``phi_j,k = sum_n A*_nj(k) psi_nk`` (``nawf`` functions per k); ``'band'`` to the
            ``nbnd`` bands and rotates afterwards; ``'auto'`` takes the cheaper one.  The two
            agree to rounding.
        bands : sequence of int, optional
            Band subset of the projector (see :meth:`pao_vertex_k`).

        Returns
        -------
        ndarray ``(nawf, nawf, 3 nat, n1, n2, n3)``, complex
        """
        from .elph_bloch import pao_vertex_to_cells

        gk_b = self.pao_vertex_k(member, A, ng, basis, bands)
        return pao_vertex_to_cells(
            gk_b, np.asarray(kgrid_idx, dtype=int), tuple(int(n) for n in ng)
        )

    def pao_vertex_k(
        self,
        member: StarMember,
        A: NDArray[np.complex128],
        ng: Sequence[int],
        basis: str = 'auto',
        bands: Sequence[int] | None = None,
    ) -> NDArray[np.complex128]:
        """PAO-gauge vertex ``g_q(k)_{ij,c}`` of one coarse q on the k-grid (before the ``R_e`` transform).

        Parameters
        ----------
        member, A, ng, basis
            As :meth:`pao_vertex_at_q`.
        bands : sequence of int, optional
            Restrict the band projector to these nscf bands (EPW's ``exclude_bands``
            bookkeeping, for comparisons); default: all bands of the projections.

        Returns
        -------
        ndarray ``(nk, nawf, nawf, 3 nat)``, complex
            In the k order of the nscf run.
        """
        nbnd, nawf, nk = A.shape
        if nk != self.nk or nbnd != self.nbnd:
            hint = (
                '; the save holds the irreducible wedge: build the projections with '
                'pao_hamiltonian_from_save(..., unfold_kpoints=True)'
                if self.provider is not None and nk == self.provider.kpts_irr_cryst.shape[0]
                else ''
            )
            raise ValueError(
                'projections (%d bands, %d k) do not match the save (%d, %d)%s'
                % (nbnd, nk, self.nbnd, self.nk, hint)
            )
        if bands is not None:
            keep = np.zeros(nbnd, dtype=bool)
            keep[np.asarray(bands, dtype=int)] = True
            A = np.where(keep[:, None, None], A, 0.0)
        if basis == 'auto':
            basis = 'pao' if nawf <= nbnd else 'band'
        potential = self.potential_at(member)
        ikq, g0 = kq_index_map(self.kpts_cryst, member.q_cryst, tuple(int(n) for n in ng))
        gk_b = np.empty((nk, nawf, nawf, 3 * self.nat), dtype=np.complex128)
        for ik in range(nk):
            if basis == 'pao':
                m = self.matrix_elements(
                    member,
                    potential,
                    ik,
                    int(ikq[ik]),
                    g0[ik],
                    left=A[:, :, ikq[ik]],
                    right=A[:, :, ik],
                )
            else:
                d = self.matrix_elements(member, potential, ik, int(ikq[ik]), g0[ik])  # (c, m, n)
                m = np.einsum(
                    'mi,cmn,nj->cij', A[:, :, ikq[ik]], d, np.conj(A[:, :, ik]), optimize=True
                )
            gk_b[ik] = np.moveaxis(m, 0, -1)
        return gk_b

    def potential_at(self, member: StarMember) -> NDArray[np.complex128]:
        """Total perturbing potential (induced + bare local) of every Cartesian mode at ``member``.

        Returns
        -------
        ndarray ``(3 nat, nr1, nr2, nr3)`` on the smooth grid, Ry/bohr.
        """
        nmodes = 3 * self.nat
        pat = read_patterns(self.files.patterns[member.iq_irr])
        dvscf = read_dvscf(self.files.dvscf[member.iq_irr], self.fft_grid, nmodes)[:, 0]
        if tuple(self.fft_grid) != tuple(self.fft_smooth):
            dvscf = interpolate_to_grid(dvscf, self.fft_smooth)
        dv = cartesian_perturbation_potentials(
            dvscf,
            pat['u'],
            self.q_irr[member.iq_irr],
            member,
            self.group,
            self.tau_cryst,
            self.at,
            self.bg,
        )
        q_cart = member.q_cryst @ self.bg
        dv += bare_local_perturbation(
            self.vloc,
            self.species_of_atom,
            self.tau_alat,
            q_cart,
            self.g_miller,
            self.g_cart,
            self.tpiba,
            self.fft_smooth,
        )
        return dv

    def matrix_elements(
        self,
        member: StarMember,
        potential: NDArray[np.complex128],
        ik: int,
        ikq: int,
        umklapp: NDArray[np.int_],
        left: NDArray[np.complex128] | None = None,
        right: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """``<chi_{m,k+q}| dV/du_c |chi_{n,k}>`` for one k (bands, or projected functions).

        Parameters
        ----------
        member : StarMember
        potential : ndarray ``(3 nat, nr1, nr2, nr3)``
            From :meth:`potential_at`.
        ik, ikq : int
            Indices of ``k`` and of the grid point ``k' = k + q - G0``.
        umklapp : ndarray ``(3,)`` int
            ``G0``.
        left, right : ndarray ``(nbnd, n)``, optional
            Coefficient matrices ``B`` such that the functions are ``chi_j = sum_n B*_nj psi_n``
            at ``k + q`` (left) and ``k`` (right); ``None`` uses the bands themselves.

        Returns
        -------
        ndarray ``(3 nat, n_left, n_right)``, complex, Ry/bohr
        """
        blk_k, blk_kq = self.block(ik), self.block(ikq)
        c_k = blk_k.coefficients if right is None else np.conj(right).T @ blk_k.coefficients
        c_kq = blk_kq.coefficients if left is None else np.conj(left).T @ blk_kq.coefficients
        npw_k, npw_kq, npol = blk_k.npw, blk_kq.npw, blk_k.npol
        # the (scalar, non-magnetic) potential acts on each spinor component separately
        dvpsi = np.empty((3 * self.nat, c_k.shape[0], npol * npw_kq), dtype=np.complex128)
        for spin in range(npol):
            u_r = periodic_part_on_grid(
                c_k[:, spin * npw_k : (spin + 1) * npw_k], blk_k.miller, self.fft_smooth
            )
            w = potential[:, None] * u_r[None]
            dvpsi[:, :, spin * npw_kq : (spin + 1) * npw_kq] = project_on_plane_waves(
                w, blk_kq.miller, self.fft_smooth, umklapp
            )
        # k + q + G' for the bra projectors: equal to k' + G'' of the stored block on an exact
        # grid; EPW uses the sum k + q itself, which differs when the k-points were given with
        # few digits, so the sum is used whenever the two disagree.
        kpg_kq = k_plus_g_cart(blk_kq, self.bg)
        kq_cart = blk_k.xk_cart + member.q_cryst @ self.bg
        kq_stored = blk_kq.xk_cart + np.asarray(umklapp, dtype=float) @ self.bg
        if np.abs(kq_cart - kq_stored).max() < 1e-12:
            beta_kq = self.projectors(ikq)
        else:
            kpg_kq = kpg_kq + (kq_cart - kq_stored)
            beta_kq = self.kb.at(kpg_kq, self.tau_alat, self.tpiba)
        dvpsi += nonlocal_perturbation(
            self.kb,
            c_k,
            k_plus_g_cart(blk_k, self.bg),
            kpg_kq,
            self.tau_alat,
            self.tpiba,
            beta_k=self.projectors(ik),
            beta_kq=beta_kq,
        )
        return np.einsum('mG,cnG->cmn', np.conj(c_kq), dvpsi)

    def band_vertex_at_q(
        self, member: StarMember, bands: Sequence[int] | None = None
    ) -> NDArray[np.complex128]:
        """Band-basis Cartesian vertex ``d[k, m, n, c]`` of one q (all k of the coarse grid).

        Parameters
        ----------
        member : StarMember
        bands : sequence of int, optional
            Band subset (default: all nscf bands).

        Returns
        -------
        ndarray ``(nk, nb, nb, 3 nat)``, complex
            The layout of :func:`PAOFLOW.elphon.elph_bloch.vertex_pao_R` (``m`` at ``k+q``).
        """
        potential = self.potential_at(member)
        ikq, g0 = kq_index_map(self.kpts_cryst, member.q_cryst, self.nscf_grid())
        sel = None if bands is None else np.asarray(bands, dtype=int)
        out = None
        for ik in range(self.nk):
            m = self.matrix_elements(member, potential, ik, int(ikq[ik]), g0[ik])
            if sel is not None:
                m = m[:, sel][:, :, sel]
            if out is None:
                out = np.empty((self.nk,) + m.shape[1:] + (m.shape[0],), dtype=np.complex128)
            out[ik] = np.moveaxis(m, 0, -1)
        return out

    def nscf_grid(self) -> tuple[int, int, int]:
        """The coarse k-grid of the nscf run (inferred from the k-point list)."""
        from ..inputs.read_QE_xml import uniform_grid_from_kpoints

        if self.provider is not None:
            return self.provider.grid
        (n1, n2, n3), _ = uniform_grid_from_kpoints(self.nscf['kpts_cart'], self.bg)
        return (int(n1), int(n2), int(n3))


# --------------------------------------------------------------------------- #
# Star unfolding in the PAO basis: g at S q from g at q (no further matrix elements).
# --------------------------------------------------------------------------- #


def orbital_shells(data_controller) -> tuple[dict, list[str]]:
    """``(shells_per_atom, atom_labels)`` of a PAOFLOW basis, the inputs of the orbital rotation.

    Parameters
    ----------
    data_controller : DataController
        After ``projections``.

    Returns
    -------
    shells_per_atom : dict
        Species label -> list of ``l`` of its shells (``arrays['shells']``).
    atom_labels : list of str
        Species label of every atom (``arrays['atoms']``).
    """
    arrays, _ = data_controller.data_dicts()
    return dict(arrays['shells']), list(arrays['atoms'])


def pao_vertex_rotation_operators(
    setup: CoarseVertexSetup,
    shells_per_atom: dict,
    atom_labels: Sequence[str],
    ng: Sequence[int],
) -> dict:
    """Orbital rotation matrices of every space-group operation (``pao_sym`` Wigner-D).

    Parameters
    ----------
    setup : CoarseVertexSetup
    shells_per_atom, atom_labels
        See :func:`orbital_shells`.
    ng : sequence of int
        Coarse k-grid.

    Returns
    -------
    dict
        The operator set of :func:`PAOFLOW.elphon.symmetry.build_dv_symmetry_operators`
        plus ``'rotations'`` (crystal, acting on positions), ``'translations'`` and
        ``'equivalent_atoms'`` of the group.
    """
    from .symmetry import build_dv_symmetry_operators

    ops = build_dv_symmetry_operators(
        setup.at,
        setup.tau_alat * setup.alat,
        setup.alat,
        np.transpose(setup.group.rotations, (0, 2, 1)),
        setup.group.equivalent_atoms,
        shells_per_atom,
        list(atom_labels),
        tuple(int(n) for n in ng),
    )
    ops['rotations'] = setup.group.rotations
    ops['translations'] = setup.group.translations
    ops['equivalent_atoms'] = setup.group.equivalent_atoms
    parity = []
    for label in atom_labels:
        for l in shells_per_atom[label]:
            parity += [(-1) ** int(l)] * (2 * int(l) + 1)
    ops['orbital_parity'] = np.array(parity, dtype=float)
    return ops


def rotate_pao_vertex(
    gk: NDArray[np.complex128],
    kpts_cryst: NDArray[np.float64],
    q0_cryst: NDArray[np.float64],
    member: StarMember,
    setup: CoarseVertexSetup,
    ops: dict,
    ng: Sequence[int],
    conjugate_u: bool = False,
) -> NDArray[np.complex128]:
    """``g_{S q0}(S k)`` for every k of the grid from ``g_{q0}(k)`` (PAO basis).

    Parameters
    ----------
    gk : ndarray ``(nk, nawf, nawf, 3 nat)``
        ``g_{q0}(k)_{ij,c}`` in the k order of ``kpts_cryst`` (the output layout of
        :meth:`CoarseVertexSetup.pao_vertex_at_q` before the transform to ``R_e``).
    kpts_cryst : ndarray ``(nk, 3)``
    q0_cryst : ndarray ``(3,)``
        The irreducible q of ``gk``.
    member : StarMember
        Target star member (``S q0`` or ``-S q0``, plus its umklapp vector).
    setup : CoarseVertexSetup
    ops : dict
        From :func:`pao_vertex_rotation_operators`.
    conjugate_u : bool, optional
        Use ``U^dagger`` instead of ``U`` (convention switch of the Wigner-D matrices).

    Returns
    -------
    ndarray ``(nk, nawf, nawf, 3 nat)``
        ``g_{q}(k)`` at ``q = member.q_cryst``, in the k order of ``kpts_cryst``.

    Notes
    -----
    The PAO Bloch sums carry the cell phase, ``phi_{j,k} = sum_R exp(i k.R) w_j(r - R - tau_j)``
    (structure factor ``exp(-i (k+G).tau)``), so under ``g: r -> R r + f`` with
    ``R tau_a + f = tau_a' + L_a`` they transform as ``O_g phi_{j,k} = exp(-i Sk.L_j) sum_j'
    D_j'j phi_{j',Sk}`` and the displacement operators as ``O_g dV_{q,kappa alpha} O_g^-1 =
    exp(-i Sq.L_kappa) sum_beta M_beta alpha dV_{Sq,kappa' beta}``; hence

    .. math::

        g_{Sq}(Sk)_{i'j',\\kappa'\\beta} = \\sum_{ij\\alpha} D_{i'i}\\,
        e^{-i(Sk+Sq)\\cdot L_i}\\, g_q(k)_{ij,\\kappa\\alpha}\\, e^{+i Sk\\cdot L_j}\\,
        D^*_{j'j}\\, M_{\\beta\\alpha}\\, e^{+i Sq\\cdot L_\\kappa}.

    Both ``g_q(k)`` and the stored vertex are periodic in k and q in this convention, so grid
    representatives need no extra phases; time-reversal partners are the complex conjugate
    (real orbitals, no spin-orbit coupling).  The identity holds when the band projector of
    the projections commutes with the operation (complete multiplets in ``nbnd``), which
    :func:`unfolded_pao_vertices` checks on one member.
    """
    nk, _nawf, _, ncart = gk.shape
    nat = ncart // 3
    ng_arr = np.asarray(ng, dtype=int)
    group = setup.group
    isym = member.isym
    rot = group.rotations[isym]
    t = group.translations[isym]
    irt = group.equivalent_atoms[isym]
    tau = setup.tau_cryst
    m_cart = cartesian_rotation(rot, setup.at)
    sq = rotate_q(rot, q0_cryst, setup.at, setup.bg)
    L = tau @ rot.T - t - tau[irt]  # (nat, 3) lattice shift of every atom under the operation
    if np.abs(L - np.rint(L)).max() > 1e-5:
        raise ValueError('symmetry %d does not map the atoms onto lattice translates' % isym)
    U = ops['U'][isym]
    if bool(ops['inv_flag'][isym]):
        U = U * ops['orbital_parity'][None, :]  # improper operation: w(-r) = (-1)^l w(r)
    if conjugate_u:
        U = np.conj(U.T)
    L_orb = L[ops['a_index']]  # (nawf, 3)
    kint = np.round(np.asarray(kpts_cryst) * ng_arr).astype(int) % ng_arr
    lookup = {tuple(kint[i]): i for i in range(nk)}
    sign = -1.0 if member.time_reversed else 1.0
    mode_phase = np.repeat(np.exp(2j * np.pi * (L @ sq)), 3)  # (ncart,) exp(+i Sq.L_kappa)
    out = np.empty_like(gk)
    for ik in range(nk):
        sk = rotate_q(rot, kpts_cryst[ik], setup.at, setup.bg)
        lab = np.rint(sk * sign * ng_arr).astype(int)
        if np.abs(lab - sk * sign * ng_arr).max() > 1e-6:
            raise ValueError('S k is not on the k-grid: the nscf grid is not symmetric')
        jk = lookup[tuple(lab % ng_arr)]
        ph_i = np.exp(-2j * np.pi * (L_orb @ (sk + sq)))
        ph_j = np.exp(2j * np.pi * (L_orb @ sk))
        block = gk[ik] * ph_i[:, None, None] * ph_j[None, :, None] * mode_phase[None, None, :]
        rotated = np.einsum('pi,ijc,qj->pqc', U, block, np.conj(U), optimize=True)
        modes = np.empty_like(rotated)
        for kappa in range(nat):
            kp = int(irt[kappa])
            modes[..., 3 * kp : 3 * kp + 3] = np.einsum(
                'ba,ija->ijb', m_cart, rotated[..., 3 * kappa : 3 * kappa + 3]
            )
        out[jk] = np.conj(modes) if member.time_reversed else modes
    return out


def unfolded_pao_vertices(
    setup: CoarseVertexSetup,
    A: NDArray[np.complex128],
    kpts_cryst: NDArray[np.float64],
    ng: Sequence[int],
    ops: dict,
    irreducible: Sequence[int],
    basis: str = 'auto',
    check_member: bool = True,
    tol: float = 1e-6,
    conjugate_u: bool = True,
):
    """Generate ``(member, g_q(k))`` for every star member of the given irreducible q.

    The vertex is computed once per irreducible q and rotated to its star
    (:func:`rotate_pao_vertex`); for the first star with more than one member one
    rotated member is also computed directly and compared.

    Parameters
    ----------
    setup : CoarseVertexSetup
    A : ndarray ``(nbnd, nawf, nk)``
    kpts_cryst : ndarray ``(nk, 3)``
    ng : sequence of int
    ops : dict
        From :func:`pao_vertex_rotation_operators`.
    irreducible : sequence of int
        Indices of the irreducible q to process.
    basis : str, optional
    check_member : bool, optional
        Perform the self-check (one direct computation).
    tol : float, optional
        Largest tolerated relative deviation of the self-check.
    conjugate_u : bool, optional
        Wigner-D convention (``True`` reproduces the direct vertices).

    Yields
    ------
    tuple
        ``(member, gk)`` with ``gk`` ``(nk, nawf, nawf, 3 nat)``.

    Raises
    ------
    ValueError
        If the self-check deviates by more than ``tol``: the nscf band set cuts a degenerate
        multiplet at some k-point (the projected PAO functions are then not symmetric);
        change ``nbnd`` so that the highest band is isolated everywhere, or use
        ``unfold='full'``.
    """
    checked = False
    for iq in irreducible:
        q0 = setup.q_irr[iq]
        base = setup.pao_vertex_k(StarMember(q0, iq, 0, False, np.zeros(3, int)), A, ng, basis)
        star = [m for m in setup.members if m.iq_irr == iq]
        for member in star:
            gk = rotate_pao_vertex(
                base, kpts_cryst, q0, member, setup, ops, ng, conjugate_u=conjugate_u
            )
            if check_member and not checked and (member.isym != 0 or member.time_reversed):
                direct = setup.pao_vertex_k(member, A, ng, basis)
                scale = float(np.abs(direct).max())
                dev = float(np.abs(gk - direct).max()) / scale if scale > 0 else 0.0
                per_k = np.abs(gk - direct).reshape(gk.shape[0], -1).max(1) / max(scale, 1e-300)
                bad = np.nonzero(per_k > tol)[0]
                checked = True
                print(
                    '  star unfolding self-check (irreducible q %d -> member q = %s, operation %d): '
                    'max relative deviation %.2e at %d of %d k-points'
                    % (iq, np.round(member.q_cryst, 4), member.isym, dev, bad.size, gk.shape[0]),
                    flush=True,
                )
                if dev > tol:
                    raise ValueError(
                        'star unfolding deviates by %.2e (> %.0e) at %d k-points (first: %s): the nscf band set '
                        'cuts a degenerate multiplet there, so the projected PAO functions are not symmetric. '
                        "Change nbnd so that the highest band is isolated at every k, or use unfold='full'."
                        % (dev, tol, bad.size, bad[:5].tolist())
                    )
            yield member, gk
