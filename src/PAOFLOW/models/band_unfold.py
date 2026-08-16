"""
band_unfold.py — General band-unfolding from supercell to primitive cell.

Works with any crystal symmetry.  Given the primitive-cell (PC) and
supercell (SC) lattice vectors and atomic positions, the module:

  1. Finds the integer transformation matrix  M  such that  A_SC = M · A_PC.
  2. Enumerates the  N = :math:`|\\det M|`  primitive-lattice translations inside
     the supercell.
  3. Builds the atom mapping  I(α, ℓ)  (SC atom ← PC atom α + translation ℓ).
  4. Extracts the real-space Hamiltonian from a PAOFLOW DataController,
     Fourier-transforms along an arbitrary k-path in the PC Brillouin zone,
     and computes the spectral weight  w_n(k)  for every SC eigenstate.

Public API
----------
    unfold_bands(pc_model_dict, sc_model_dict, kpath_frac, \\*,
                 nk_per_seg=80, verbose=True)

    UnfoldResult  — dataclass returned by unfold_bands()
    plot_unfolded — convenience plotting function

References
----------
  V. Popescu and A. Zunger, Phys. Rev. B 85, 085201 (2012).
  P. B. Allen et al., Phys. Rev. B 87, 085322 (2013).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import det, eigh, inv

# ═══════════════════════════════════════════════════════════════════════
#  Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class UnfoldResult:
    """Container for band-unfolding results.

    Attributes
    ----------
    kpath_cart : (nk, 3) Cartesian k-points (units of 1/alat).
    kdist      : (nk,)   cumulative k-distance along the path.
    sym_ticks  : list of (distance, label) for symmetry-point ticks.
    E_pc       : (nk, nawf_pc)  PC reference eigenvalues.
    E_sc       : (nk, nawf_sc)  SC eigenvalues.
    W          : (nk, nawf_sc)  spectral weights w_n(k).
    nawf_pc    : int   — number of orbitals in the PC.
    nawf_sc    : int   — number of orbitals in the SC.
    N          : int   — volume ratio (= :math:`|\\det M|`).
    R_translations : (N, 3)  PC lattice translations inside the SC.
    atom_map   : (n_at_pc, N)  SC atom index for each (α, ℓ).
    """

    kpath_cart: np.ndarray
    kdist: np.ndarray
    sym_ticks: list
    E_pc: np.ndarray
    E_sc: np.ndarray
    W: np.ndarray
    nawf_pc: int
    nawf_sc: int
    N: int
    R_translations: np.ndarray
    atom_map: np.ndarray
    a_pc: np.ndarray = field(repr=False)
    a_sc: np.ndarray = field(repr=False)
    M: np.ndarray = field(repr=False)

    def sum_rule_check(self) -> np.ndarray:
        """Return Σ_n W_n(k) per k-point (should ≈ nawf_pc).

        Notes
        -----
        The unfolded spectral weights obey the completeness (sum) rule

        .. math::

            \\sum_{n} w_n(\\mathbf{k}) \\approx n_{\\mathrm{awf}}^{PC},

        i.e. at every k-point the SC spectral weights add up to the number
        of orbitals in the primitive cell.
        """
        return self.W.sum(axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  Lattice & geometry helpers
# ═══════════════════════════════════════════════════════════════════════


def _find_transformation_matrix(
    a_pc: np.ndarray, a_sc: np.ndarray, tol: float = 1e-6
) -> np.ndarray:
    """Find integer matrix M such that  a_sc = M @ a_pc.

    Parameters
    ----------
    a_pc : (3, 3) — PC lattice vectors (rows).
    a_sc : (3, 3) — SC lattice vectors (rows).
    tol  : tolerance for integrality check.

    Returns
    -------
    M : (3, 3) integer array.

    Notes
    -----
    The SC lattice vectors are an integer linear combination of the PC
    lattice vectors,

    .. math::

        \\mathbf{A}_{SC} = M \\, \\mathbf{A}_{PC},
        \\qquad M = \\mathbf{A}_{SC}\\, \\mathbf{A}_{PC}^{-1},

    where rows of :math:`\\mathbf{A}_{PC}`/:math:`\\mathbf{A}_{SC}` hold the
    lattice vectors.  ``M`` is recovered by rounding the (in general
    non-integer) float solution to the nearest integer and rejecting the
    result if the residual exceeds ``tol``.
    """
    # a_sc[i] = Σ_j M[i,j] a_pc[j]   →   M = a_sc @ inv(a_pc)
    M_float = a_sc @ inv(a_pc)
    M_int = np.rint(M_float).astype(int)
    residual = np.max(np.abs(M_float - M_int))
    if residual > tol:
        raise ValueError(
            f'SC lattice vectors are not an integer multiple of PC vectors '
            f'(max residual = {residual:.2e}).  Check that a_pc and a_sc are '
            f'given in the same Cartesian frame and units.'
        )
    return M_int


def _find_translations(a_pc: np.ndarray, M: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Find the N = |det M| PC lattice translations inside the SC.

    Strategy: scan integer combinations  n1*a_pc[0] + n2*a_pc[1] + n3*a_pc[2]
    and keep those whose SC fractional coordinates lie in [0, 1)^3.

    Parameters
    ----------
    a_pc : (3, 3) PC lattice vectors (rows), in Cartesian/alat units.
    M    : (3, 3) integer transformation matrix.

    Returns
    -------
    R : (N, 3) array of translation vectors in Cartesian/alat units.

    Notes
    -----
    The number of translations equals the cell-volume ratio

    .. math::

        N = |\\det M| = \\frac{V_{SC}}{V_{PC}}.

    Each translation is a PC lattice vector

    .. math::

        \\mathbf{R}_\\ell = n_1\\, \\mathbf{a}_{PC,1}
            + n_2\\, \\mathbf{a}_{PC,2} + n_3\\, \\mathbf{a}_{PC,3},
            \\qquad n_1, n_2, n_3 \\in \\mathbb{Z},

    kept only if its SC-fractional coordinates
    :math:`\\mathbf{R}_\\ell \\, \\mathbf{A}_{SC}^{-1}` fall inside the unit
    cell :math:`[0,1)^3`, which selects exactly the :math:`N` translations
    that lie within one supercell.
    """
    N = int(abs(round(det(M.astype(float)))))
    a_sc = M.astype(float) @ a_pc
    inv_sc = inv(a_sc)  # maps Cartesian → SC fractional

    # Search range: we try all n ∈ [-Nmax, Nmax]^3.
    # N ≤ max 64 or so in practice; Nmax = max(|M|) + 1 is safe.
    Nmax = int(np.max(np.abs(M))) + 1

    translations_frac = []  # store SC fractional coords
    for n1 in range(-Nmax, Nmax + 1):
        for n2 in range(-Nmax, Nmax + 1):
            for n3 in range(-Nmax, Nmax + 1):
                R = n1 * a_pc[0] + n2 * a_pc[1] + n3 * a_pc[2]
                frac = R @ inv_sc  # SC fractional coordinates (rows = lattice vectors)
                # Reduce to [0, 1)
                frac_mod = frac - np.floor(frac + tol)
                # Accept if in [0, 1)
                if np.all(frac_mod >= -tol) and np.all(frac_mod < 1.0 - tol):
                    # Clamp tiny negatives to zero
                    frac_mod = np.where(frac_mod < 0, 0.0, frac_mod)
                    # Check for duplicates (in fractional space)
                    is_dup = False
                    for fold in translations_frac:
                        diff = frac_mod - fold
                        diff -= np.rint(diff)
                        if np.max(np.abs(diff)) < tol:
                            is_dup = True
                            break
                    if not is_dup:
                        translations_frac.append(frac_mod.copy())

    if len(translations_frac) != N:
        raise RuntimeError(
            f'Expected N={N} translations, found {len(translations_frac)}.  '
            f'This is likely a bug — please report.'
        )

    translations_frac = np.array(translations_frac)
    # Sort for reproducibility: lexicographic by fractional SC coords
    order = np.lexsort(translations_frac.T[::-1])
    translations_frac = translations_frac[order]
    # Convert back to Cartesian
    return translations_frac @ a_sc


def _build_atom_map(
    tau_pc: np.ndarray,
    tau_sc: np.ndarray,
    R_translations: np.ndarray,
    a_sc: np.ndarray,
    tol: float = 1e-4,
) -> np.ndarray:
    """Build the atom mapping I(α, ℓ) → SC atom index.

    For each PC atom α and translation R_ℓ, find the SC atom whose
    position matches  tau_pc[α] + R_ℓ  modulo SC lattice vectors.

    Parameters
    ----------
    tau_pc : (n_at_pc, 3) PC atom positions in Cartesian/alat.
    tau_sc : (n_at_sc, 3) SC atom positions in Cartesian/alat.
    R_translations : (N, 3) translation vectors.
    a_sc : (3, 3) SC lattice vectors (rows).

    Returns
    -------
    atom_map : (n_at_pc, N) integer array, atom_map[α, ℓ] = SC atom index.

    Notes
    -----
    For every PC atom :math:`\\alpha` and translation :math:`\\mathbf{R}_\\ell`
    the matching SC atom :math:`I` satisfies

    .. math::

        \\boldsymbol{\\tau}_{PC}(\\alpha) + \\mathbf{R}_\\ell
            \\equiv \\boldsymbol{\\tau}_{SC}(I) \\pmod{\\mathbf{A}_{SC}},

    i.e. the two positions coincide modulo an SC lattice translation, up to
    ``tol``.  Every SC atom must be claimed by exactly one :math:`(\\alpha,
    \\ell)` pair for the supercell to be commensurate with the primitive cell.
    """
    n_at_pc = len(tau_pc)
    N = len(R_translations)
    n_at_sc = len(tau_sc)
    inv_sc = inv(a_sc)

    atom_map = -np.ones((n_at_pc, N), dtype=int)
    used = set()

    for alpha in range(n_at_pc):
        for ell in range(N):
            target = tau_pc[alpha] + R_translations[ell]
            # Find SC atom matching target modulo SC lattice
            found = False
            for I in range(n_at_sc):
                diff = target - tau_sc[I]
                frac_diff = diff @ inv_sc
                frac_diff -= np.rint(frac_diff)
                if np.max(np.abs(frac_diff)) < tol:
                    if I in used:
                        raise ValueError(
                            f'SC atom {I} maps to multiple (α,ℓ) pairs. Check atom positions.'
                        )
                    atom_map[alpha, ell] = I
                    used.add(I)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f'No SC atom found matching PC atom {alpha} + '
                    f'R_{ell} = {target}.  Check positions/lattice vectors.'
                )

    if len(used) != n_at_sc:
        unmapped = set(range(n_at_sc)) - used
        raise ValueError(
            f'SC atoms {unmapped} are not mapped to any PC atom + translation. '
            f'The supercell may not be commensurate with the primitive cell.'
        )
    return atom_map


# ═══════════════════════════════════════════════════════════════════════
#  k-path generation
# ═══════════════════════════════════════════════════════════════════════


def make_kpath(sym_points: dict, path_str: str, a_pc: np.ndarray, nk_per_seg: int = 80) -> tuple:
    """Generate a k-path from a string specification.

    Parameters
    ----------
    sym_points : dict mapping label → (3,) fractional coords in PC basis.
    path_str   : e.g. 'Γ-X-W-K-Γ-L|U-X'.  Use '-' to connect, '|' for breaks.
    a_pc       : (3, 3) PC lattice vectors (rows), Cartesian/alat.
    nk_per_seg : number of k-points per linear segment.

    Returns
    -------
    kpath_cart : (nk, 3) Cartesian k-points.
    kdist      : (nk,) cumulative distance.
    sym_ticks  : list of (distance, label).

    Notes
    -----
    The PC reciprocal lattice vectors are defined by

    .. math::

        \\mathbf{b}_i \\cdot \\mathbf{a}_j = \\delta_{ij},
        \\qquad \\mathbf{B}_{PC} = \\mathbf{A}_{PC}^{-T},

    so a fractional point ``frac`` maps to Cartesian coordinates as
    ``frac @ b_pc``.  Each segment between two symmetry points is sampled
    by linear interpolation

    .. math::

        \\mathbf{k}(t) = \\mathbf{k}_0 + t\\,(\\mathbf{k}_1 - \\mathbf{k}_0),
        \\qquad t = 0, \\tfrac{1}{n_{\\rm seg}}, \\dots,
        \\tfrac{n_{\\rm seg}-1}{n_{\\rm seg}}.
    """
    b_pc = inv(a_pc).T  # reciprocal lattice  (a_i · b_j = δ_ij)

    pipe_segs = path_str.split('|')
    kpts_cart, kpts_dist, sym_ticks = [], [], []
    d = 0.0

    for iseg, seg_str in enumerate(pipe_segs):
        labels = seg_str.split('-')
        if iseg > 0:
            old_d, old_lbl = sym_ticks[-1]
            sym_ticks[-1] = (old_d, old_lbl + '|' + labels[0])
        else:
            sym_ticks.append((d, labels[0]))

        for i in range(len(labels) - 1):
            frac0 = np.array(sym_points[labels[i]], dtype=float)
            frac1 = np.array(sym_points[labels[i + 1]], dtype=float)
            k0 = frac0 @ b_pc
            k1 = frac1 @ b_pc
            seg_len = np.linalg.norm(k1 - k0)
            for ik in range(nk_per_seg):
                t = ik / nk_per_seg
                kpts_cart.append(k0 + t * (k1 - k0))
                kpts_dist.append(d + t * seg_len)
            d += seg_len
            sym_ticks.append((d, labels[i + 1]))

        # endpoint
        frac_end = np.array(sym_points[labels[-1]], dtype=float)
        kpts_cart.append(frac_end @ b_pc)
        kpts_dist.append(d)

    return np.array(kpts_cart), np.array(kpts_dist), sym_ticks


# ═══════════════════════════════════════════════════════════════════════
#  Hamiltonian extraction from PAOFLOW
# ═══════════════════════════════════════════════════════════════════════


def _extract_hamiltonian(model_dict: dict, outputdir: str = '_unfold_tmp', verbose: bool = False):
    """Build a PAOFLOW model and extract HRs + R-grid.

    Parameters
    ----------
    model_dict : PAOFLOW-compatible model dictionary.
    outputdir  : temporary output directory name.
    verbose    : passed to PAOFLOW.

    Returns
    -------
    HRs   : (nawf, nawf, nR, nspin) real-space Hamiltonian.
    R     : (nR, 3) lattice vectors in Cartesian/alat units.
    nawf  : number of Wannier functions.
    nspin : number of spin channels.
    norbitals : (natoms,) orbital count per atom.

    Notes
    -----
    ``HRs`` holds the real-space PAO Hamiltonian matrix elements
    :math:`H_{mn}(\\mathbf{R})` produced by PAOFLOW through the inverse
    Fourier transform of :math:`H_{mn}(\\mathbf{k})` over the DFT k-mesh,

    .. math::

        H_{mn}(\\mathbf{R}) = \\frac{1}{N_k}\\sum_{\\mathbf{k}}
            H_{mn}(\\mathbf{k})\\, e^{-2\\pi i \\mathbf{k}\\cdot\\mathbf{R}}.

    ``R`` is the corresponding real-space lattice-vector grid, later used
    to Fourier-transform back to an arbitrary k-point along the unfolding
    path (see :func:`unfold_bands`).
    """
    # Fast path: a raw-PAOFLOW model dict carries its already-built HRs and
    # R-grid under the private "_paoflow" key. Reuse them directly instead of
    # rebuilding a TB model (whose "label" would not match any builtin model).
    pao = model_dict.get('_paoflow')
    if pao is not None:
        nawf = int(pao['nawf'])
        nspin = int(pao['nspin'])
        HRs = np.asarray(pao['HRs'])
        if HRs.ndim == 6:
            HRs = HRs.reshape(nawf, nawf, -1, nspin)
        R = np.asarray(pao['R'], dtype=float)
        return HRs.copy(), R.copy(), nawf, nspin, pao.get('norbitals')

    from ._paoflow_runner import build_model_hamiltonian

    return build_model_hamiltonian(model_dict, outputdir=outputdir, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════
#  Core unfolding
# ═══════════════════════════════════════════════════════════════════════


def _compute_spectral_weights(evecs, orb_idx, uphi, nawf_sc, n_at_pc, N):
    """Compute w_n(k) for all SC bands.

    Parameters
    ----------
    evecs   : (nawf_sc, nawf_sc) eigenvectors (columns).
    orb_idx : (n_at_pc, N, norb_per_atom[α]) — list of lists for varying norb.
    uphi    : (N,) phase factors  exp(-2πi k·R_ℓ).
    nawf_sc : total SC orbitals.
    n_at_pc : number of PC atoms.
    N       : number of translations.

    Returns
    -------
    W : (nawf_sc,) spectral weights.

    Notes
    -----
    This implements the Popescu-Zunger spectral weight (V. Popescu and
    A. Zunger, Phys. Rev. B 85, 085201 (2012)), specialized to a real-space
    atomic-orbital basis: for SC eigenstate :math:`n`, the weight is

    .. math::

        w_n(\\mathbf{k}) = \\frac{1}{N} \\sum_{\\alpha}\\sum_{m}
            \\left| \\sum_{\\ell=1}^{N} e^{-2\\pi i \\mathbf{k}\\cdot\\mathbf{R}_\\ell}\\,
            C^{n}_{\\alpha, m, \\ell}(\\mathbf{k}) \\right|^2,

    where :math:`C^{n}_{\\alpha, m, \\ell}(\\mathbf{k})` is the SC
    eigenvector coefficient of band :math:`n` on orbital :math:`m` of the
    atom obtained by translating PC atom :math:`\\alpha` by
    :math:`\\mathbf{R}_\\ell` (``uphi`` supplies the phase factors, ``psi``
    the corresponding eigenvector components).  Summed over all SC bands
    at fixed :math:`\\mathbf{k}`, the weights satisfy the sum rule
    :math:`\\sum_n w_n(\\mathbf{k}) = n_{\\mathrm{awf}}^{PC}`.
    """
    W = np.zeros(nawf_sc)
    for alpha in range(n_at_pc):
        # orb_idx[alpha] is (N, norb_alpha)
        idx = orb_idx[alpha]  # shape (N, norb_alpha)
        psi = evecs[idx, :]  # shape (N, norb_alpha, nawf_sc)
        # Project: sum over ℓ with phase
        proj = np.einsum('r,rmn->mn', uphi, psi)  # (norb_alpha, nawf_sc)
        W += np.sum(np.abs(proj) ** 2, axis=0)
    return W / N


def unfold_bands(
    pc_model_dict: dict,
    sc_model_dict: dict,
    sym_points: dict,
    path_str: str,
    *,
    nk_per_seg: int = 80,
    verbose: bool = True,
) -> UnfoldResult:
    """Unfold supercell bands onto the primitive-cell Brillouin zone.

    Parameters
    ----------
    pc_model_dict : PAOFLOW model dict for the **primitive cell**.
    sc_model_dict : PAOFLOW model dict for the **supercell**.
    sym_points    : dict of high-symmetry point labels → fractional coords
                    in the **PC reciprocal basis**. Example:
                    ``{'Γ': [0,0,0], 'X': [0.5, 0, 0.5], ...}``
    path_str      : band-path string, e.g. ``'Γ-X-W-K-Γ-L|U-X'``.
    nk_per_seg    : k-points per linear sub-segment.
    verbose       : print progress information.

    Returns
    -------
    UnfoldResult  dataclass with eigenvalues, spectral weights, and metadata.

    Notes
    -----
    Both model dicts must use the **same** alat.  The PC and SC lattice
    vectors (``a_vectors``) must be given in Cartesian units of alat (the
    PAOFLOW / EDTB convention).

    At every k-point along the path, the PC and SC Bloch Hamiltonians are
    built from the real-space Hamiltonians by Fourier transform,

    .. math::

        H(\\mathbf{k}) = \\sum_{\\mathbf{R}} H(\\mathbf{R})\\,
            e^{2\\pi i \\mathbf{k}\\cdot\\mathbf{R}},

    Hermitized as :math:`H(\\mathbf{k}) \\to \\tfrac{1}{2}\\left(H(\\mathbf{k})
    + H(\\mathbf{k})^\\dagger\\right)` to remove round-off asymmetry, and
    diagonalized,

    .. math::

        H(\\mathbf{k})\\, C_n(\\mathbf{k}) = E_n(\\mathbf{k})\\, C_n(\\mathbf{k}),

    giving the PC reference eigenvalues ``E_pc`` and the SC eigenvalues
    ``E_sc``/eigenvectors used by :func:`_compute_spectral_weights` to
    obtain the spectral weights ``W`` (see that function's ``Notes`` for
    the unfolding formula).  The completeness sum rule
    :math:`\\sum_n W_n(\\mathbf{k}) \\approx n_{\\mathrm{awf}}^{PC}` is checked
    at every k-point and a warning is issued if the deviation exceeds 0.01.
    """
    # Diagnostics are collective; only the root rank should print them.
    if verbose:
        try:
            from mpi4py import MPI

            verbose = MPI.COMM_WORLD.Get_rank() == 0
        except ImportError:
            pass

    # ── 1. Lattice geometry ──────────────────────────────────────
    a_pc = np.array(pc_model_dict['model']['a_vectors'], dtype=float)
    a_sc = np.array(sc_model_dict['model']['a_vectors'], dtype=float)

    M = _find_transformation_matrix(a_pc, a_sc)
    N = int(abs(round(det(M.astype(float)))))
    R_translations = _find_translations(a_pc, M)

    if verbose:
        print(f'Transformation matrix M (det = {N}):')
        for row in M:
            print(f'  {row}')
        print(f'Found {N} translations inside SC.')

    # ── 2. Atom mapping ──────────────────────────────────────────
    pc_atoms = pc_model_dict['model']['atoms']
    sc_atoms = sc_model_dict['model']['atoms']
    n_at_pc = len(pc_atoms)
    n_at_sc = len(sc_atoms)

    tau_pc = np.array([pc_atoms[str(i)]['tau'] for i in range(n_at_pc)])
    tau_sc = np.array([sc_atoms[str(i)]['tau'] for i in range(n_at_sc)])

    atom_map = _build_atom_map(tau_pc, tau_sc, R_translations, a_sc)

    if verbose:
        print('\nAtom mapping (PC atom α → SC atom I):')
        for alpha in range(n_at_pc):
            row = ', '.join(str(atom_map[alpha, ell]) for ell in range(N))
            print(f'  α={alpha}: SC atoms [{row}]')

    # ── 3. Extract Hamiltonians ──────────────────────────────────
    if verbose:
        print('\nBuilding PC Hamiltonian...')
    HRs_pc, R_pc, nawf_pc, nspin, norb_pc = _extract_hamiltonian(
        pc_model_dict, '_unfold_pc', verbose=False
    )

    if verbose:
        print('Building SC Hamiltonian...')
    HRs_sc, R_sc, nawf_sc, _, norb_sc = _extract_hamiltonian(
        sc_model_dict, '_unfold_sc', verbose=False
    )

    # ── 4. Build orbital index table ─────────────────────────────
    # norb_sc[I] = number of orbitals on SC atom I.
    # atom_block_start[I] = starting orbital index for SC atom I.
    atom_block_start_sc = np.zeros(n_at_sc, dtype=int)
    for I in range(1, n_at_sc):
        atom_block_start_sc[I] = atom_block_start_sc[I - 1] + norb_sc[I - 1]

    # For each PC atom α, all its SC copies should have the same norb.
    # orb_idx[alpha] has shape (N, norb_alpha).
    orb_idx = []
    for alpha in range(n_at_pc):
        norb_alpha = norb_sc[atom_map[alpha, 0]]
        idx = np.zeros((N, norb_alpha), dtype=int)
        for ell in range(N):
            I = atom_map[alpha, ell]
            if norb_sc[I] != norb_alpha:
                raise ValueError(
                    f'Orbital count mismatch: SC atom {I} has {norb_sc[I]} '
                    f'orbitals but SC atom {atom_map[alpha, 0]} has {norb_alpha}.'
                )
            idx[ell, :] = atom_block_start_sc[I] + np.arange(norb_alpha)
        orb_idx.append(idx)

    if verbose:
        total_orb_pc = sum(norb_sc[atom_map[a, 0]] for a in range(n_at_pc))
        print(f'\nPC: {nawf_pc} orbitals,  SC: {nawf_sc} orbitals,  N={N}')
        print(f'Expected sum rule: Σ_n W_n(k) = {total_orb_pc}')

    # ── 5. Generate k-path ───────────────────────────────────────
    kpath, kdist, sym_ticks = make_kpath(sym_points, path_str, a_pc, nk_per_seg)
    nk = len(kpath)
    if verbose:
        print(f'k-path: {nk} points, {len(sym_ticks)} symmetry ticks')

    # ── 6. Fourier transform & unfold ────────────────────────────
    E_pc = np.zeros((nk, nawf_pc))
    E_sc = np.zeros((nk, nawf_sc))
    W = np.zeros((nk, nawf_sc))

    for ik in range(nk):
        k = kpath[ik]

        # PC reference
        phi_pc = np.exp(2j * np.pi * R_pc @ k)
        Hk_pc = HRs_pc[:, :, :, 0] @ phi_pc
        Hk_pc = 0.5 * (Hk_pc + Hk_pc.conj().T)
        E_pc[ik] = eigh(Hk_pc)[0]

        # SC
        phi_sc = np.exp(2j * np.pi * R_sc @ k)
        Hk_sc = HRs_sc[:, :, :, 0] @ phi_sc
        Hk_sc = 0.5 * (Hk_sc + Hk_sc.conj().T)
        evals, evecs = eigh(Hk_sc)
        E_sc[ik] = evals

        # Spectral weights
        uphi = np.exp(-2j * np.pi * R_translations @ k)
        W[ik] = _compute_spectral_weights(evecs, orb_idx, uphi, nawf_sc, n_at_pc, N)

    # ── 7. Sanity checks ────────────────────────────────────────
    wsum = W.sum(axis=1)
    expected = nawf_pc
    deviation = np.max(np.abs(wsum - expected))
    if verbose:
        print(
            f'\nSum rule:  Σ_n W_n(k)  mean = {wsum.mean():.6f}  '
            f'(expected {expected}),  max deviation = {deviation:.2e}'
        )
    if deviation > 0.01:
        warnings.warn(
            f'Sum-rule deviation {deviation:.4f} exceeds tolerance.  '
            f'Check lattice vectors and atom positions.'
        )

    return UnfoldResult(
        kpath_cart=kpath,
        kdist=kdist,
        sym_ticks=sym_ticks,
        E_pc=E_pc,
        E_sc=E_sc,
        W=W,
        nawf_pc=nawf_pc,
        nawf_sc=nawf_sc,
        N=N,
        R_translations=R_translations,
        atom_map=atom_map,
        a_pc=a_pc,
        a_sc=a_sc,
        M=M,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: unfold directly from EDTBModel objects
# ═══════════════════════════════════════════════════════════════════════


def unfold_from_models(
    pc_model,  # EDTBModel
    sc_model,  # EDTBModel
    sym_points: dict,
    path_str: str,
    *,
    nk_per_seg: int = 80,
    verbose: bool = True,
) -> UnfoldResult:
    """Unfold bands using EDTBModel objects directly.

    Parameters
    ----------
    pc_model   : EDTBModel for the primitive cell.
    sc_model   : EDTBModel for the supercell.
    sym_points : high-symmetry points in PC fractional coords.
    path_str   : band-path string.
    nk_per_seg : k-points per segment.
    verbose    : print progress.

    Returns
    -------
    UnfoldResult
    """
    return unfold_bands(
        pc_model.to_model_dict(),
        sc_model.to_model_dict(),
        sym_points,
        path_str,
        nk_per_seg=nk_per_seg,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════


def plot_unfolded(
    result: UnfoldResult,
    *,
    y_lim: tuple = (-12, 6),
    w_thresh: float = 0.02,
    cmap: str = 'Reds',
    figsize: tuple = (10, 6),
    title: str | None = None,
    show: bool = True,
    ax=None,
):
    """Plot unfolded band structure.

    Parameters
    ----------
    result   : UnfoldResult from unfold_bands().
    y_lim    : energy window.
    w_thresh : minimum spectral weight to display.
    cmap     : matplotlib colormap for scatter points.
    figsize  : figure size if creating a new figure.
    title    : plot title.
    show     : call plt.show().
    ax       : existing matplotlib Axes (optional).

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # PC reference bands (black lines)
    for n in range(result.nawf_pc):
        ax.plot(
            result.kdist,
            result.E_pc[:, n],
            'k-',
            lw=1.5,
            alpha=0.5,
            label='PC reference' if n == 0 else None,
        )

    # Unfolded SC bands (scatter)
    for nu in range(result.nawf_sc):
        mask = result.W[:, nu] > w_thresh
        if mask.any():
            ax.scatter(
                result.kdist[mask],
                result.E_sc[mask, nu],
                c=result.W[mask, nu],
                cmap=cmap,
                s=5,
                vmin=0,
                vmax=1,
                zorder=5,
                rasterized=True,
            )

    ax.scatter([], [], c='red', s=20, label='SC unfolded')

    # Symmetry ticks
    tick_pos = [t[0] for t in result.sym_ticks]
    tick_lbl = [t[1] for t in result.sym_ticks]
    for x in tick_pos:
        ax.axvline(x, color='gray', lw=0.5, alpha=0.5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=11)

    ax.set_xlim(result.kdist[0], result.kdist[-1])
    ax.set_ylim(*y_lim)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='lower right')

    if ax is None:
        plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════
#  Standard high-symmetry paths (convenience)
# ═══════════════════════════════════════════════════════════════════════

# Fractional coordinates in the primitive-cell reciprocal basis.
# These match PAOFLOW / Setyawan-Curtarolo conventions.

FCC_SYM_POINTS = {
    'Γ': [0.0, 0.0, 0.0],
    'X': [0.5, 0.0, 0.5],
    'W': [0.5, 0.25, 0.75],
    'K': [0.375, 0.375, 0.75],
    'L': [0.5, 0.5, 0.5],
    'U': [0.625, 0.25, 0.625],
}
FCC_PATH = 'Γ-X-W-K-Γ-L-U-W-L-K|U-X'

BCC_SYM_POINTS = {
    'Γ': [0.0, 0.0, 0.0],
    'H': [0.5, -0.5, 0.5],
    'P': [0.25, 0.25, 0.25],
    'N': [0.0, 0.0, 0.5],
}
BCC_PATH = 'Γ-H-N-Γ-P-H|P-N'

HEX_SYM_POINTS = {
    'Γ': [0.0, 0.0, 0.0],
    'A': [0.0, 0.0, 0.5],
    'H': [1 / 3, 1 / 3, 0.5],
    'K': [1 / 3, 1 / 3, 0.0],
    'L': [0.5, 0.0, 0.5],
    'M': [0.5, 0.0, 0.0],
}
HEX_PATH = 'Γ-M-K-Γ-A-L-H-A|L-M|K-H'

CUB_SYM_POINTS = {
    'Γ': [0.0, 0.0, 0.0],
    'X': [0.0, 0.5, 0.0],
    'M': [0.5, 0.5, 0.0],
    'R': [0.5, 0.5, 0.5],
}
CUB_PATH = 'Γ-X-M-Γ-R-X|M-R'
