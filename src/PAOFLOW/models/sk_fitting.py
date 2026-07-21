# sk_fitting.py
#
# Slater-Koster two-center tight-binding fitting engine.
#
# Extracts SK parameters from a PAOFLOW PAO Hamiltonian (HRs) via
# eigenvalue-based least-squares fitting with analytic (Hellmann-Feynman)
# Jacobian.  Supports single-shell and multi-shell configurations.
#
# Usage
# -----
#   from sk_fitting import SKFitter
#
#   fitter = SKFitter(arryp, attrp, n_shells=2, nkfit=6)
#   result = fitter.fit(n_trials=20, seed=123)
#
#   # Optimized parameter vector & RMSE
#   p_opt  = result['p_opt']
#   rmse   = result['rmse']
#
#   # Ready-to-use PAOFLOW model dict
#   model  = fitter.build_model_dict(p_opt)
#
# References
# ----------
# J.C. Slater & G.F. Koster, Phys. Rev. 94, 1498 (1954), Table I.

"""sk_fitting — Slater-Koster tight-binding fitting engine.

This module implements a hierarchy of three fitting classes that extract
Slater-Koster (SK) tight-binding parameters from a PAOFLOW PAO Hamiltonian
:math:`H(\\mathbf{R})` via eigenvalue-based least-squares optimisation with
an analytic (Hellmann-Feynman) Jacobian.

Physics background
------------------
Slater-Koster theory expresses every hopping matrix element between
two orbitals in terms of a small set of two-centre integrals
:math:`V_{ss\\sigma}, V_{sp\\sigma}, V_{pp\\sigma}, V_{pp\\pi}, \\ldots, V_{dd\\delta}`
and the bond direction cosines :math:`(l_x, l_y, l_z)`.  The function
:func:`sk_element` implements the full Slater-Koster table for
:math:`s/p/d` orbitals, including all :math:`\\sqrt{3}` prefactors.
:func:`sk_design_row` returns the linear coefficient vector of each SK
parameter for a given pair of orbitals and bond direction, enabling
vectorised Jacobian computation.

Class hierarchy
---------------

:class:`SKFitter`
    Base two-centre Slater-Koster fitter.

    * Enumerates all bonds up to *n_shells* neighbor shells from the
      :math:`H(\\mathbf{R})` grid.
    * Builds precomputed design tensors and block-sparse
      :math:`\\partial H/\\partial V` arrays (``_dHk_shells``).
    * Fits on-site energies :math:`\\varepsilon` and hopping parameters
      :math:`V_\\lambda` by minimising the eigenvalue RMSE

      .. math::

          \\mathcal{L} = \\sum_{\\mathbf{k},n} \\bigl(E_{n\\mathbf{k}}^\\text{SK}(p)
              - E_{n\\mathbf{k}}^\\text{PAO}\\bigr)^2

      using ``scipy.optimize.least_squares`` with a Hellmann-Feynman
      Jacobian and multi-start random initialisation.
    * Outputs a species-pair-keyed model dict compatible with
      ``edtb_params``.

:class:`SKFitterEDTB` *(extends SKFitter)*
    Environment-dependent TB extension.  Each hopping is screened by a
    bond-environment sum :math:`S_{ij}`:

    .. math::

        V_\\lambda^{\\text{eff}}(i,j) = V_\\lambda^{(2c)}
            \\exp\\!\\bigl(-\\gamma_\\lambda\\,S_{ij}\\bigr), \\qquad
        S_{ij} = \\sum_{k \\neq i,j} f_c(d_{ik})\\,f_c(d_{jk})

    where :math:`f_c` is a smooth cosine cutoff tapering to zero at
    ``r_cut``.  The screening strength :math:`\\gamma` can be shared
    globally, per angular-momentum pair, or per SK channel
    (``gamma_mode``).  Optionally fits environment-dependent on-site
    shifts :math:`\\varepsilon_\\alpha \\to \\varepsilon_\\alpha + \\eta_\\alpha
    \\sum_k f_c(d_{ik})`.

:class:`MultiGeomEDTB`
    Multi-geometry EDTB fitter.  Fits a **single shared** parameter set
    :math:`(\\varepsilon, V_\\lambda, \\gamma, \\eta)` to DFT band structures
    from multiple atomic configurations simultaneously — essential for
    learning physically meaningful :math:`\\gamma` values.  Geometries
    may differ in lattice parameter, strain, or surface termination
    (same species and orbital basis required).  Evaluations across
    geometries are parallelised with ``ThreadPoolExecutor`` (``eigh``
    releases the GIL).  Supports species harmonisation when not all
    geometries contain the same species.

Module-level helpers
--------------------
:func:`sk_element`
    Full SK table for an :math:`s/p/d` orbital pair and bond cosines.
:func:`sk_design_row`
    Length-10 linear coefficient vector of the SK parameters for a
    given orbital pair.

Key module-level constants
--------------------------
``SK_PARAM_NAMES``, ``SK_LABELS``
    Canonical names and labels for the 10 SK integrals
    (:math:`V_{ss\\sigma}` through :math:`V_{dd\\delta}`).
``ORBITAL_NAMES``
    Chemistry real-orbital names ``('s', 'px', ..., 'dz2')``.
``LPAIR_ACTIVE_NAMES``, ``LPAIR_ACTIVE_INDICES``
    Active SK parameters for each angular-momentum pair
    :math:`(l, l')`.
``SHELL_TO_ORBITALS``
    Orbital name list for each angular-momentum shell.

Typical usage
-------------
::

    from PAOFLOW.models.sk_fitting import SKFitter, SKFitterEDTB, MultiGeomEDTB

    # 1. Plain SK fit
    fitter = SKFitter(arryp, attrp, n_shells=2, nkfit=6)
    result = fitter.fit(n_trials=20, seed=123)
    model  = fitter.build_model_dict(result['p_opt'])

    # 2. Staged EDTB: fix SK then fit γ
    p_sk = result['p_opt']
    edtb = SKFitterEDTB(arryp, attrp, n_shells=2, r_cut=8.0, gamma_mode='per_lpair')
    result_edtb = edtb.fit(p0_sk=p_sk, n_trials=10)
    model_edtb  = edtb.build_model_dict(result_edtb['p_opt'])

    # 3. Multi-geometry EDTB
    geoms = [(arry_eq, attr_eq), (arry_p5, attr_p5), (arry_m5, attr_m5)]
    mg = MultiGeomEDTB(geoms, n_shells=2, r_cut=8.0, gamma_mode='global')
    result_mg = mg.fit(p0_sk=p_sk, n_trials=10, n_jobs=-1)

Reference
---------
J.C. Slater & G.F. Koster, *Phys. Rev.* **94**, 1498 (1954).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from PAOFLOW.spectrum.kpnts_interpolation_mesh import get_path as _get_path

# ═══════════════════════════════════════════════════════════════
#  1. Slater-Koster two-center integrals (standard √3 convention)
# ═══════════════════════════════════════════════════════════════

ORBITAL_NAMES = ('s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2')
SK_PARAM_NAMES = ['sss', 'sps', 'pps', 'ppp', 'sds', 'pds', 'pdp', 'dds', 'ddp', 'ddd']
SK_LABELS = [
    'Vssσ',
    'Vspσ',
    'Vppσ',
    'Vppπ',
    'Vsdσ',
    'Vpdσ',
    'Vpdπ',
    'Vddσ',
    'Vddπ',
    'Vddδ',
]

_SQ3 = np.sqrt(3.0)
_HSQ3 = _SQ3 / 2.0

SHELL_TO_ORBITALS = {
    0: ['s'],
    1: ['px', 'py', 'pz'],
    2: ['dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2'],
}

# Active SK parameter names for each angular-momentum pair (l_lo, l_hi)
LPAIR_ACTIVE_NAMES = {
    (0, 0): ['sss'],
    (0, 1): ['sps'],
    (0, 2): ['sds'],
    (1, 1): ['pps', 'ppp'],
    (1, 2): ['pds', 'pdp'],
    (2, 2): ['dds', 'ddp', 'ddd'],
}
LPAIR_ACTIVE_INDICES = {
    lpair: [SK_PARAM_NAMES.index(n) for n in names] for lpair, names in LPAIR_ACTIVE_NAMES.items()
}
CHANNEL_LABELS = {
    (0, 0): ['σ'],
    (0, 1): ['σ'],
    (0, 2): ['σ'],
    (1, 1): ['σ', 'π'],
    (1, 2): ['σ', 'π'],
    (2, 2): ['σ', 'π', 'δ'],
}

# Map each SK channel name to its (l, l') angular-momentum pair
CHANNEL_L_MAP = {name: lpair for lpair, names in LPAIR_ACTIVE_NAMES.items() for name in names}


def sk_element(orb_a: str, orb_b: str, lx: float, ly: float, lz: float, sh: dict) -> float:
    """Slater-Koster two-center hopping matrix element H(orb_a, orb_b).

    Parameters
    ----------
    orb_a, orb_b : str
        Orbital names from ``ORBITAL_NAMES``.
    lx, ly, lz : float
        Direction cosines of the bond vector.
    sh : dict
        SK parameter dict with keys from ``SK_PARAM_NAMES``.

    Returns
    -------
    float
        The matrix element value.
    """
    sq3, hsq3 = _SQ3, _HSQ3
    l2, m2, n2 = lx * lx, ly * ly, lz * lz
    lm, ln, mn = lx * ly, lx * lz, ly * lz
    pd = {'px': lx, 'py': ly, 'pz': lz}
    do = {'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2'}

    # s-s
    if orb_a == 's' and orb_b == 's':
        return sh['sss']

    # s-p / p-s
    if orb_a == 's' and orb_b in pd:
        return pd[orb_b] * sh['sps']
    if orb_b == 's' and orb_a in pd:
        return -pd[orb_a] * sh['sps']

    # p-p
    if orb_a in pd and orb_b in pd:
        if orb_a == orb_b:
            ll = pd[orb_a]
            return ll**2 * sh['pps'] + (1 - ll**2) * sh['ppp']
        return pd[orb_a] * pd[orb_b] * (sh['pps'] - sh['ppp'])

    # s-d / d-s
    def _sd(d):
        if d == 'dxy':
            return sq3 * lm * sh['sds']
        if d == 'dyz':
            return sq3 * mn * sh['sds']
        if d == 'dzx':
            return sq3 * ln * sh['sds']
        if d == 'dx2-y2':
            return hsq3 * (l2 - m2) * sh['sds']
        if d == 'dz2':
            return (n2 - 0.5 * (l2 + m2)) * sh['sds']

    if orb_a == 's' and orb_b in do:
        return _sd(orb_b)
    if orb_b == 's' and orb_a in do:
        return _sd(orb_a)

    # p-d / d-p
    def _pd(p, d):
        S, P = sh['pds'], sh['pdp']
        if p == 'px':
            if d == 'dxy':
                return sq3 * l2 * ly * S + ly * (1 - 2 * l2) * P
            if d == 'dyz':
                return sq3 * lm * lz * S - 2 * lm * lz * P
            if d == 'dzx':
                return sq3 * l2 * lz * S + lz * (1 - 2 * l2) * P
            if d == 'dx2-y2':
                return hsq3 * lx * (l2 - m2) * S + lx * (1 - l2 + m2) * P
            if d == 'dz2':
                return lx * (n2 - 0.5 * (l2 + m2)) * S - sq3 * lx * n2 * P
        if p == 'py':
            if d == 'dxy':
                return sq3 * m2 * lx * S + lx * (1 - 2 * m2) * P
            if d == 'dyz':
                return sq3 * m2 * lz * S + lz * (1 - 2 * m2) * P
            if d == 'dzx':
                return sq3 * lm * lz * S - 2 * lm * lz * P
            if d == 'dx2-y2':
                return hsq3 * ly * (l2 - m2) * S - ly * (1 + l2 - m2) * P
            if d == 'dz2':
                return ly * (n2 - 0.5 * (l2 + m2)) * S - sq3 * ly * n2 * P
        if p == 'pz':
            if d == 'dxy':
                return sq3 * lm * lz * S - 2 * lm * lz * P
            if d == 'dyz':
                return sq3 * n2 * ly * S + ly * (1 - 2 * n2) * P
            if d == 'dzx':
                return sq3 * n2 * lx * S + lx * (1 - 2 * n2) * P
            if d == 'dx2-y2':
                return hsq3 * lz * (l2 - m2) * S - lz * (l2 - m2) * P
            if d == 'dz2':
                return lz * (n2 - 0.5 * (l2 + m2)) * S + sq3 * lz * (l2 + m2) * P
        return 0.0

    if orb_a in pd and orb_b in do:
        return _pd(orb_a, orb_b)
    if orb_b in pd and orb_a in do:
        return -_pd(orb_b, orb_a)

    # d-d
    if orb_a in do and orb_b in do:
        S, P, D = sh['dds'], sh['ddp'], sh['ddd']
        l2m2, l2n2, m2n2 = l2 * m2, l2 * n2, m2 * n2
        df = l2 - m2
        da, db = orb_a, orb_b

        # diagonal
        if da == db == 'dxy':
            return 3 * l2m2 * S + (l2 + m2 - 4 * l2m2) * P + (n2 + l2m2) * D
        if da == db == 'dyz':
            return 3 * m2n2 * S + (m2 + n2 - 4 * m2n2) * P + (l2 + m2n2) * D
        if da == db == 'dzx':
            return 3 * l2n2 * S + (l2 + n2 - 4 * l2n2) * P + (m2 + l2n2) * D
        if da == db == 'dx2-y2':
            return 0.75 * df**2 * S + (l2 + m2 - df**2) * P + (n2 + 0.25 * df**2) * D
        if da == db == 'dz2':
            t = n2 - 0.5 * (l2 + m2)
            return t**2 * S + 3 * n2 * (l2 + m2) * P + 0.75 * (l2 + m2) ** 2 * D

        # off-diagonal
        p = frozenset([da, db])
        if p == frozenset(['dxy', 'dyz']):
            return 3 * lx * m2 * lz * S + ln * (1 - 4 * m2) * P + ln * (m2 - 1) * D
        if p == frozenset(['dxy', 'dzx']):
            return 3 * l2 * ly * lz * S + mn * (1 - 4 * l2) * P + mn * (l2 - 1) * D
        if p == frozenset(['dyz', 'dzx']):
            return 3 * ly * n2 * lx * S + lm * (1 - 4 * n2) * P + lm * (n2 - 1) * D
        if p == frozenset(['dxy', 'dx2-y2']):
            return 1.5 * lm * df * S + 2 * lm * (m2 - l2) * P + 0.5 * lm * df * D
        if p == frozenset(['dyz', 'dx2-y2']):
            return 1.5 * mn * df * S - mn * (1 + 2 * df) * P + mn * (1 + 0.5 * df) * D
        if p == frozenset(['dzx', 'dx2-y2']):
            return 1.5 * ln * df * S + ln * (1 - 2 * df) * P - ln * (1 - 0.5 * df) * D
        # couplings to dz² (√3 factors)
        t = n2 - 0.5 * (l2 + m2)
        if p == frozenset(['dxy', 'dz2']):
            return sq3 * (lm * t * S - 2 * lm * n2 * P + 0.5 * lm * (1 + n2) * D)
        if p == frozenset(['dyz', 'dz2']):
            return sq3 * (mn * t * S + mn * (l2 + m2 - n2) * P - 0.5 * mn * (l2 + m2) * D)
        if p == frozenset(['dzx', 'dz2']):
            return sq3 * (ln * t * S + ln * (l2 + m2 - n2) * P - 0.5 * ln * (l2 + m2) * D)
        if p == frozenset(['dx2-y2', 'dz2']):
            return sq3 * (0.5 * df * t * S + n2 * (m2 - l2) * P + 0.25 * (1 + n2) * df * D)
    return 0.0


def sk_design_row(orb_a: str, orb_b: str, lx: float, ly: float, lz: float) -> np.ndarray:
    """Coefficient of each SK parameter for matrix element H(orb_a, orb_b).

    Returns
    -------
    np.ndarray
        Length-10 array: entry *k* is the coefficient of ``SK_PARAM_NAMES[k]``.
    """
    row = np.zeros(len(SK_PARAM_NAMES))
    for k, name in enumerate(SK_PARAM_NAMES):
        unit = {n: (1.0 if n == name else 0.0) for n in SK_PARAM_NAMES}
        row[k] = sk_element(orb_a, orb_b, lx, ly, lz, unit)
    return row


# ═══════════════════════════════════════════════════════════════
#  2. SKFitter — main fitting engine
# ═══════════════════════════════════════════════════════════════


class SKFitter:
    """Slater-Koster eigenvalue-based fitter.

    Constructs design tensors from a PAO Hamiltonian and fits SK parameters
    by minimising the eigenvalue RMSE on a uniform k-mesh.

    Parameters
    ----------
    arryp : dict
        PAOFLOW ``arrays`` dict (needs ``a_vectors``, ``b_vectors``, ``tau``,
        ``atoms``, ``shells``, ``HRs``; optionally ``configuration``).
    attrp : dict
        PAOFLOW ``attributes`` dict (needs ``alat``, ``natoms``).
    n_shells : int
        Number of neighbor shells to include (default 2 → NN + NNN).
    nkfit : int
        Subdivisions along each reciprocal axis for the fitting k-mesh
        (total k-points = ``nkfit**3``).
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        arryp: dict,
        attrp: dict,
        *,
        n_shells: int = 2,
        nkfit: int = 6,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self._setup_system(arryp, attrp)
        self._enumerate_bonds()
        self._select_shells(n_shells)
        self._build_parameter_registry()
        self._build_reference_eigenvalues(nkfit)
        self._build_design_tensors()
        self._build_onsite_map()
        self._precompute_dHk()
        self._build_regularization_weights()

    # ── 2a. System setup ──────────────────────────────────────

    def _setup_system(self, arryp, attrp):
        HRs = arryp['HRs']
        self.a_vecs = arryp['a_vectors']
        self.b_vecs = arryp['b_vectors']
        self.alat = attrp['alat']
        self.nat = attrp['natoms']
        self.nawf = HRs.shape[0]
        self.nk1 = HRs.shape[2]
        self.nk2 = HRs.shape[3]
        self.nk3 = HRs.shape[4]
        self.HRs = HRs

        self.atoms_list = arryp['atoms']
        self.tau_bohr = arryp['tau']
        self.tau_alat = self.tau_bohr / self.alat
        self.shells_dict = arryp['shells']
        self.config_dict = arryp.get('configuration', None)
        self.unique_species = list(dict.fromkeys(self.atoms_list))

        # Canonical shell-group list keyed by (l, radial_rank).
        #
        # Each *radial* shell of a given angular momentum becomes its own
        # group, enabling multi-configuration fits: a Si 'standard' basis
        # [3S, 3P, 3D, 4S, 4P] yields five groups (3S, 3P, 3D, 4S, 4P) rather
        # than collapsing 3S/4S (and 3P/4P) onto a single s/p channel.
        # ``radial_rank`` is the 0-based order of appearance of a shell among
        # all shells of the same l for a species (first s-shell → (0, 0),
        # second s-shell → (0, 1), …).  A minimal basis with one shell per l
        # reduces to the previous angular-momentum-only grouping.
        #
        # Group order follows the *configuration* order of the first species
        # (first-seen insertion order) rather than a sorted (l, rank) order,
        # so that shell-pair keys emitted in ``build_model_dict`` match the
        # canonicalisation used by ``models.Slater_Koster`` (which orders
        # pairs by configuration-list index).
        self._species_shell_keys = {
            sp: self._shell_keys(self.shells_dict[sp]) for sp in self.unique_species
        }
        canonical_keys = []
        for sp in self.unique_species:
            for k in self._species_shell_keys[sp]:
                if k not in canonical_keys:
                    canonical_keys.append(k)
        self.n_groups = len(canonical_keys)
        self.group_key = list(canonical_keys)  # list of (l, radial_rank)
        self.group_l = [k[0] for k in canonical_keys]
        self.is_multiconfig = any(rank > 0 for (_l, rank) in canonical_keys)
        _key_to_group = {k: g for g, k in enumerate(canonical_keys)}

        # Per-atom orbital structure
        self.atom_orbitals = []
        self.atom_orbital_group = []
        self.atom_block_start = []
        self.norb_per_atom = []

        idx = 0
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            shell_keys = self._species_shell_keys[sp]
            orb_list, grp_list = [], []
            for shell_idx, shell_l in enumerate(self.shells_dict[sp]):
                cg = _key_to_group[shell_keys[shell_idx]]
                for orb_name in SHELL_TO_ORBITALS[shell_l]:
                    orb_list.append(orb_name)
                    grp_list.append(cg)
            self.atom_orbitals.append(orb_list)
            self.atom_orbital_group.append(grp_list)
            self.atom_block_start.append(idx)
            self.norb_per_atom.append(len(orb_list))
            idx += len(orb_list)
        assert idx == self.nawf, f'Total orbitals ({idx}) != nawf ({self.nawf})'

        # Representative label per group (used for parameter names and the
        # exported model dict).  Prefer the species configuration label
        # (e.g. '3S', '4S'); fall back to a synthetic 'l{l}#{rank}' tag.
        self.cfg_names = [self._group_label(g) for g in range(self.n_groups)]

        # R-vectors and H(R) blocks
        R_list, HR_list = [], []
        for i1 in range(self.nk1):
            for i2 in range(self.nk2):
                for i3 in range(self.nk3):
                    r1 = i1 if 2 * i1 <= self.nk1 else i1 - self.nk1
                    r2 = i2 if 2 * i2 <= self.nk2 else i2 - self.nk2
                    r3 = i3 if 2 * i3 <= self.nk3 else i3 - self.nk3
                    R_cart = r1 * self.a_vecs[0] + r2 * self.a_vecs[1] + r3 * self.a_vecs[2]
                    R_list.append(R_cart)
                    HR_list.append(HRs[:, :, i1, i2, i3, 0])
        self.R_arr = np.array(R_list)
        self.HR_arr = np.array(HR_list)

        if self.verbose:
            print(f'SKFitter: {self.nat} atoms, nawf={self.nawf}, alat={self.alat:.4f} Bohr')
            print(f'  Shell groups: {self.cfg_names} (l = {self.group_l})')

    @staticmethod
    def _shell_keys(shell_l_list):
        """Assign each shell a ``(l, radial_rank)`` key.

        ``radial_rank`` counts shells of the same angular momentum in their
        order of appearance, so multiple radial shells with the same l
        (e.g. 3S and 4S) receive distinct keys ``(0, 0)`` and ``(0, 1)``.
        This is what lets the fitter treat each configuration shell as an
        independent SK channel.
        """
        rank_counter = {}
        keys = []
        for l_val in shell_l_list:
            r = rank_counter.get(l_val, 0)
            keys.append((l_val, r))
            rank_counter[l_val] = r + 1
        return keys

    def _group_label(self, g):
        """Return a representative label for shell-group ``g``.

        Uses the species configuration label (e.g. ``'3S'``, ``'4S'``) from
        the first species carrying that ``(l, radial_rank)`` key when a
        ``configuration`` dict is available; otherwise a synthetic
        ``'l{l}#{rank}'`` tag.
        """
        key = self.group_key[g]
        for sp in self.unique_species:
            keys = self._species_shell_keys[sp]
            if key in keys:
                if self.config_dict is not None:
                    return list(self.config_dict[sp])[keys.index(key)]
                break
        l_val, rank = key
        return f'l{l_val}#{rank}'

    # ── 2b. Bond enumeration ──────────────────────────────────

    def _enumerate_bonds(self, shell_tol: float = 1e-3):
        """Enumerate all bonds on the DFT k-grid and group into shells.

        Parameters
        ----------
        shell_tol : float
            Relative tolerance for merging shells.  Two distance keys
            *d1* and *d2* are merged when ``|d2 - d1| / d1 < shell_tol``.
            Default 1e-3 (0.1 %).
        """
        raw_bonds = defaultdict(list)
        for i1 in range(self.nk1):
            for i2 in range(self.nk2):
                for i3 in range(self.nk3):
                    r1 = i1 if 2 * i1 <= self.nk1 else i1 - self.nk1
                    r2 = i2 if 2 * i2 <= self.nk2 else i2 - self.nk2
                    r3 = i3 if 2 * i3 <= self.nk3 else i3 - self.nk3
                    R_cart = r1 * self.a_vecs[0] + r2 * self.a_vecs[1] + r3 * self.a_vecs[2]
                    for iat in range(self.nat):
                        for jat in range(self.nat):
                            d_vec = R_cart + self.tau_alat[jat] - self.tau_alat[iat]
                            d_norm = np.linalg.norm(d_vec)
                            if d_norm < 1e-8:
                                continue
                            raw_bonds[round(d_norm, 5)].append(
                                (R_cart, iat, jat, d_vec, d_norm, i1, i2, i3)
                            )

        # Merge shells whose distance keys differ by less than shell_tol
        sorted_keys = sorted(raw_bonds.keys())
        shell_bonds = defaultdict(list)
        canonical = {}  # map raw key → merged representative key
        for dk in sorted_keys:
            merged = False
            for ck in canonical.values():
                if abs(dk - ck) / ck < shell_tol:
                    canonical[dk] = ck
                    merged = True
                    break
            if not merged:
                canonical[dk] = dk
        for dk in sorted_keys:
            shell_bonds[canonical[dk]].extend(raw_bonds[dk])

        self._shell_bonds = shell_bonds
        self.shell_dists = sorted(shell_bonds.keys())

        if self.verbose:
            n_show = min(5, len(self.shell_dists))
            print(f'  Neighbor shells (first {n_show} of {len(self.shell_dists)}):')
            for s, d in enumerate(self.shell_dists[:n_show]):
                bonds = shell_bonds[d]
                pairs = set((b[1], b[2]) for b in bonds)
                pair_str = ', '.join(
                    f'{self.atoms_list[i]}({i})→{self.atoms_list[j]}({j})' for i, j in sorted(pairs)
                )
                print(f'    Shell {s + 1}: d={d:.5f}, {len(bonds):>3d} bonds  ({pair_str})')

    def _select_shells(self, n_shells: int):
        """Pick the neighbor shells to include in the fit."""
        self.n_shells = min(n_shells, len(self.shell_dists))
        self.shell_bonds_list = [
            self._shell_bonds[self.shell_dists[i]] for i in range(self.n_shells)
        ]
        shell_tags = ['nn', 'nnn', 'nnnn', 'nnnnn']
        self.shell_tags = shell_tags[: self.n_shells]
        if self.verbose:
            for i, tag in enumerate(self.shell_tags):
                print(f'  Included: {tag} ({len(self.shell_bonds_list[i])} bonds)')

    # ── 2c. Parameter registry ────────────────────────────────

    def _build_parameter_registry(self):
        """Register hopping parameters for each shell-group pair."""
        self.hop_pair_list = []
        self.hop_pair_start = {}
        self.hop_pair_active = {}
        self.n_hop = 0
        self.hop_param_labels = []

        for ga in range(self.n_groups):
            for gb in range(ga, self.n_groups):
                la, lb = self.group_l[ga], self.group_l[gb]
                lpair = (min(la, lb), max(la, lb))
                active = LPAIR_ACTIVE_INDICES[lpair]
                labels = CHANNEL_LABELS[lpair]

                self.hop_pair_list.append((ga, gb))
                self.hop_pair_start[(ga, gb)] = self.n_hop
                self.hop_pair_active[(ga, gb)] = active

                pair_tag = f'{self.cfg_names[ga]}-{self.cfg_names[gb]}'
                for lab in labels:
                    self.hop_param_labels.append(f'V({pair_tag}){lab}')
                self.n_hop += len(active)

        # On-site parameters
        self.species_onsite_groups = {}
        self.species_param_start = {}
        self.n_onsite = 0
        self.onsite_param_names = []

        for sp in self.unique_species:
            self.species_param_start[sp] = self.n_onsite
            cfg = (
                list(self.config_dict[sp])
                if self.config_dict
                else [f'g{i}' for i in range(len(self.shells_dict[sp]))]
            )
            groups = self._get_onsite_groups(self.shells_dict[sp], cfg)
            self.species_onsite_groups[sp] = groups
            for pname, _ in groups:
                self.onsite_param_names.append(pname)
            self.n_onsite += len(groups)

        self.n_params = self.n_onsite + self.n_shells * self.n_hop
        self.param_labels = list(self.onsite_param_names)
        for i, tag in enumerate(self.shell_tags):
            self.param_labels += [f'{tag.upper()}_{l}' for l in self.hop_param_labels]

        if self.verbose:
            print(
                f'  Parameters: {self.n_params} total '
                f'({self.n_onsite} on-site + '
                f'{self.n_shells}×{self.n_hop} hopping)'
            )

    @staticmethod
    def _get_onsite_groups(shell_l_list, cfg_labels):
        """Map each configuration group to its orbital indices, splitting d into t2g/eg."""
        groups = []
        idx = 0
        for ig, (l_val, cfg) in enumerate(zip(shell_l_list, cfg_labels)):
            norb_l = 2 * l_val + 1
            if l_val <= 1:
                groups.append((f'ε({cfg})', list(range(idx, idx + norb_l))))
            elif l_val == 2:
                groups.append((f'ε({cfg}_t2g)', [idx, idx + 1, idx + 2]))
                groups.append((f'ε({cfg}_eg)', [idx + 3, idx + 4]))
            idx += norb_l
        return groups

    # ── 2d. Reference eigenvalues ─────────────────────────────

    def _build_reference_eigenvalues(self, nkfit):
        # Accept nkfit as int (uniform) or tuple/list of 3 ints (anisotropic)
        if isinstance(nkfit, (tuple, list)):
            nk1, nk2, nk3 = int(nkfit[0]), int(nkfit[1]), int(nkfit[2])
        else:
            nk1 = nk2 = nk3 = int(nkfit)
        self._nkfit_grid = (nk1, nk2, nk3)

        kpts = []
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                for ik3 in range(nk3):
                    kfrac = np.array(
                        [
                            ik1 / max(nk1, 1),
                            ik2 / max(nk2, 1),
                            ik3 / max(nk3, 1),
                        ]
                    )
                    kpts.append(kfrac @ self.b_vecs)
        self.kpts = np.array(kpts)
        self.Nk = len(kpts)

        phases_all = np.exp(2j * np.pi * (self.kpts @ self.R_arr.T))
        self.E_pao = np.zeros((self.Nk, self.nawf))
        for ik in range(self.Nk):
            Hk = np.einsum('r,rij->ij', phases_all[ik], self.HR_arr)
            self.E_pao[ik] = np.sort(np.linalg.eigvalsh(Hk).real)

        if self.verbose:
            grid_str = f'{nk1}×{nk2}×{nk3}' if (nk1 != nk2 or nk2 != nk3) else str(nk1)
            print(
                f'  Reference: {self.Nk} k-points (grid {grid_str}), {self.nawf} bands, '
                f'E ∈ [{self.E_pao.min():.3f}, {self.E_pao.max():.3f}] eV'
            )

    # ── 2e. Design tensors ────────────────────────────────────

    def _precompute_design_tensor(self, bonds):
        nbonds = len(bonds)
        M = np.zeros((nbonds, self.n_hop, self.nawf, self.nawf))
        R_bond = np.zeros((nbonds, 3))

        for ib, (R_cart, iat, jat, d_vec, d_norm, *_) in enumerate(bonds):
            lx, ly, lz = d_vec / d_norm
            R_bond[ib] = R_cart

            bi = self.atom_block_start[iat]
            bj = self.atom_block_start[jat]
            oi = self.atom_orbitals[iat]
            oj = self.atom_orbitals[jat]
            gi = self.atom_orbital_group[iat]
            gj = self.atom_orbital_group[jat]

            for ai, orb_a in enumerate(oi):
                for aj, orb_b in enumerate(oj):
                    ga_loc, gb_loc = gi[ai], gj[aj]
                    canonical = (min(ga_loc, gb_loc), max(ga_loc, gb_loc))
                    start = self.hop_pair_start[canonical]
                    active = self.hop_pair_active[canonical]
                    design = sk_design_row(orb_a, orb_b, lx, ly, lz)
                    for local_k, sk_k in enumerate(active):
                        M[ib, start + local_k, bi + ai, bj + aj] = design[sk_k]
        return M, R_bond

    @staticmethod
    def _build_bond_groups(bonds, M, atom_block_start, atom_orbitals):
        """Group bonds by (iat, jat) and extract compact sub-blocks."""
        from collections import defaultdict

        bg = defaultdict(list)
        for ib, (_, iat, jat, *_rest) in enumerate(bonds):
            bg[(iat, jat)].append(ib)
        groups = []
        for (iat, jat), blist in bg.items():
            idx = np.array(blist)
            bi = atom_block_start[iat]
            bj = atom_block_start[jat]
            no_i = len(atom_orbitals[iat])
            no_j = len(atom_orbitals[jat])
            M_sub = M[idx][:, :, bi : bi + no_i, bj : bj + no_j].copy()
            groups.append((idx, bi, bj, no_i, no_j, M_sub))
        return groups

    def _build_design_tensors(self):
        self._M_shells = []
        self._R_shells = []
        self._bond_groups_shells = []
        for i, bonds in enumerate(self.shell_bonds_list):
            M, R = self._precompute_design_tensor(bonds)
            self._M_shells.append(M)
            self._R_shells.append(R)
            self._bond_groups_shells.append(
                self._build_bond_groups(bonds, M, self.atom_block_start, self.atom_orbitals)
            )
            if self.verbose:
                print(f'  Design tensor ({self.shell_tags[i]}): {M.shape}')

    # ── 2f. On-site map ──────────────────────────────────────

    def _build_onsite_map(self):
        self._onsite_map = np.zeros((self.n_onsite, self.nawf, self.nawf))
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            bi = self.atom_block_start[iat]
            pstart = self.species_param_start[sp]
            for ig, (_, local_indices) in enumerate(self.species_onsite_groups[sp]):
                for li in local_indices:
                    self._onsite_map[pstart + ig, bi + li, bi + li] = 1.0
        self._onsite_diag = np.array([np.diag(self._onsite_map[p]) for p in range(self.n_onsite)])

    # ── 2g. Precompute dH/dk arrays ──────────────────────────

    def _precompute_dHk(self):
        self._phases_shells = []
        self._dHk_shells = []
        Nk = self.Nk
        nawf = self.nawf
        n_hop = self.n_hop
        for s, (M, R) in enumerate(zip(self._M_shells, self._R_shells)):
            phases = np.exp(2j * np.pi * (self.kpts @ R.T))
            self._phases_shells.append(phases)
            # Block-sparse dHk construction
            dHk = np.zeros((Nk, n_hop, nawf, nawf), dtype=complex)
            for idx, bi, bj, no_i, no_j, M_sub in self._bond_groups_shells[s]:
                dHk[:, :, bi : bi + no_i, bj : bj + no_j] += np.einsum(
                    'kb,bpij->kpij', phases[:, idx], M_sub
                )
            self._dHk_shells.append(dHk)

        mem_MB = sum(d.nbytes for d in self._dHk_shells) / 1e6
        if self.verbose:
            print(f'  Precomputed dHk arrays: {mem_MB:.1f} MB')

    # ── 2g′. Regularization weights ──────────────────────────

    def _build_regularization_weights(self):
        """Build per-parameter Tikhonov weights.

        On-site parameters get weight 0 (not penalized).
        Hopping parameters get weight proportional to the shell distance
        (farther neighbors are penalized more strongly).
        """
        w = np.zeros(self.n_params)
        for s in range(self.n_shells):
            d_s = self.shell_dists[s]  # distance in alat units
            i0 = self.n_onsite + s * self.n_hop
            w[i0 : i0 + self.n_hop] = d_s
        # Normalise so the mean hopping weight is 1
        hop_w = w[self.n_onsite :]
        if hop_w.sum() > 0:
            w[self.n_onsite :] = hop_w / hop_w.mean()
        self._reg_weights = w

    # ── 2h. Forward model (eigenvalues + Jacobian) ────────────

    def _eigenvalues_and_jacobian(self, p):
        """Compute SK eigenvalues and Hellmann-Feynman Jacobian for all k-points."""
        Nk, nawf = self.Nk, self.nawf
        n_onsite, n_hop = self.n_onsite, self.n_hop

        # On-site contribution
        H_onsite = np.einsum('p,pij->ij', p[:n_onsite], self._onsite_map)

        # Build H(k) via block-sparse groups
        Hk_all = np.broadcast_to(H_onsite, (Nk, nawf, nawf)).astype(complex).copy()

        for s in range(self.n_shells):
            i0 = n_onsite + s * n_hop
            V_s = p[i0 : i0 + n_hop]
            for idx, bi, bj, no_i, no_j, M_sub in self._bond_groups_shells[s]:
                wM = np.einsum('p,bpij->bij', V_s, M_sub)
                Hk_all[:, bi : bi + no_i, bj : bj + no_j] += np.einsum(
                    'kb,bij->kij', self._phases_shells[s][:, idx], wM
                )

        # Batch eigendecomposition
        evals_all, evecs_all = np.linalg.eigh(Hk_all)
        E_sk = evals_all.real

        # Hellmann-Feynman Jacobian
        dE_dp = np.zeros((Nk, nawf, self.n_params))
        psi_sq = np.abs(evecs_all) ** 2
        dE_dp[:, :, :n_onsite] = np.einsum('kin,pi->knp', psi_sq, self._onsite_diag)

        evecs_bcast = evecs_all[:, np.newaxis, :, :]
        for s in range(self.n_shells):
            i0 = n_onsite + s * n_hop
            i1 = i0 + n_hop
            tmp = np.matmul(self._dHk_shells[s], evecs_bcast)
            dE_dp[:, :, i0:i1] = np.real(np.einsum('kin,kpin->knp', evecs_all.conj(), tmp))

        return E_sk, dE_dp

    # ── 2i. Initial parameter guess ───────────────────────────

    def extract_onsite_from_HR0(self) -> np.ndarray:
        """Extract initial on-site energies from H(R=0) diagonal blocks.

        Uses QE orbital ordering (m = 0, +1, -1, …) for t2g/eg splitting.

        Returns
        -------
        np.ndarray
            On-site parameter vector of length ``n_onsite``.
        """
        H_R0 = self.HRs[:, :, 0, 0, 0, 0]

        species_onsites_avg = {}
        for sp in self.unique_species:
            cfg = (
                list(self.config_dict[sp])
                if self.config_dict
                else [f'g{i}' for i in range(len(self.shells_dict[sp]))]
            )
            sp_onsites = defaultdict(list)
            for iat in range(self.nat):
                if self.atoms_list[iat] != sp:
                    continue
                bi = self.atom_block_start[iat]
                ni = self.norb_per_atom[iat]
                diag = np.real(np.diag(H_R0)[bi : bi + ni])
                onsites = self._extract_onsite_block(diag, self.shells_dict[sp], cfg)
                for k, v in onsites.items():
                    sp_onsites[k].append(v)
            species_onsites_avg[sp] = {k: np.mean(v) for k, v in sp_onsites.items()}

        p0 = np.zeros(self.n_onsite)
        for sp in self.unique_species:
            pstart = self.species_param_start[sp]
            for ig, (pname, _) in enumerate(self.species_onsite_groups[sp]):
                if pname in species_onsites_avg[sp]:
                    p0[pstart + ig] = species_onsites_avg[sp][pname]
        return p0

    @staticmethod
    def _extract_onsite_block(diag_block, shell_l_list, cfg_labels):
        """Extract on-site energies from one atom's H(R=0) diagonal (QE ordering)."""
        onsites = {}
        idx = 0
        for ig, (l_val, cfg) in enumerate(zip(shell_l_list, cfg_labels)):
            norb_l = 2 * l_val + 1
            sub = diag_block[idx : idx + norb_l]
            if l_val == 0:
                onsites[f'ε({cfg})'] = sub[0]
            elif l_val == 1:
                onsites[f'ε({cfg})'] = np.mean(sub)
            elif l_val == 2:
                t2g_local = [1, 2, 4]  # dzx, dyz, dxy in QE ordering
                eg_local = [0, 3]  # dz2, dx2-y2
                onsites[f'ε({cfg}_t2g)'] = np.mean(sub[t2g_local])
                onsites[f'ε({cfg}_eg)'] = np.mean(sub[eg_local])
            idx += norb_l
        return onsites

    # ── 2j. Fitting ───────────────────────────────────────────

    def _run_single_trial(self, p_init, alpha, n_data, max_nfev, ftol, xtol, gtol):
        """Run one least-squares trial from a given initial point.

        Returns ``(rmse, p_opt, OptimizeResult)``.
        """
        use_reg = alpha > 0.0
        reg_w = self._reg_weights
        last_jac = [None]

        def fun(p):
            E_sk, dE_dp = self._eigenvalues_and_jacobian(p)
            res_data = (E_sk - self.E_pao).ravel()
            J_data = dE_dp.reshape(n_data, self.n_params)
            if use_reg:
                last_jac[0] = np.vstack([J_data, np.diag(alpha * reg_w)])
                return np.concatenate([res_data, alpha * reg_w * p])
            last_jac[0] = J_data
            return res_data

        def jac(p):
            return last_jac[0]

        res = least_squares(
            fun,
            p_init,
            jac=jac,
            method='lm',
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        rmse = np.sqrt(np.mean(res.fun[:n_data] ** 2))
        return rmse, res.x.copy(), res

    def fit(
        self,
        n_trials: int = 10,
        seed: int | None = 123,
        max_nfev: int = 1000,
        ftol: float = 1e-12,
        xtol: float = 1e-12,
        gtol: float = 1e-12,
        alpha: float = 0.0,
        n_jobs: int = 1,
    ) -> dict:
        """Run multi-start least-squares optimisation.

        Parameters
        ----------
        n_trials : int
            Number of random restarts.
        seed : int or None
            Random seed for reproducibility.
        max_nfev : int
            Max function evaluations per trial.
        ftol, xtol, gtol : float
            Tolerances for ``scipy.optimize.least_squares``.
        alpha : float
            Tikhonov regularization strength (default 0 = no penalty).
            Adds ``alpha * w_i * p_i`` penalty rows to the residual, where
            ``w_i`` is proportional to the neighbor-shell distance (farther
            shells are penalized more).  On-site parameters are not penalized.
            Typical values: 0.01–1.0 (start small, increase if far-neighbor
            hoppings blow up).
        n_jobs : int
            Number of parallel workers for multi-start trials
            (default 1 = sequential).  Use ``-1`` for all available cores.
            Requires ``joblib`` when ``n_jobs != 1``.

        Returns
        -------
        dict
            ``p_opt`` : best parameter vector.
            ``rmse`` : best RMSE (eV) (data-only, excluding penalty).
            ``max_err`` : max absolute error (eV).
            ``all_results`` : list of ``(rmse, p, OptimizeResult)`` sorted by RMSE.
            ``param_labels`` : parameter names.
        """
        Nk, nawf = self.Nk, self.nawf
        n_data = Nk * nawf

        # Hopping scale for initialisation
        E_half = 0.5 * (self.E_pao.max() - self.E_pao.min())
        hop_scales = [E_half / np.sqrt(len(bonds)) for bonds in self.shell_bonds_list]
        p0_onsite = self.extract_onsite_from_HR0()

        rng = np.random.RandomState(seed)

        # Pre-generate all initial points (use the RNG sequentially
        # so results are reproducible regardless of n_jobs)
        p_inits = []
        for trial in range(n_trials):
            p_init = np.zeros(self.n_params)
            p_init[: self.n_onsite] = p0_onsite
            for s, sc in enumerate(hop_scales):
                i0 = self.n_onsite + s * self.n_hop
                p_init[i0 : i0 + self.n_hop] = rng.uniform(-sc, sc, self.n_hop)
            p_inits.append(p_init)

        # ── Run trials ──
        use_parallel = n_jobs != 1 and n_trials > 1
        common_kw = dict(
            alpha=alpha,
            n_data=n_data,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        if self.verbose:
            print(f"\n{'=' * 65}")
            par_tag = f', n_jobs={n_jobs}' if use_parallel else ''
            print(f'Multi-start optimisation: {n_trials} trials{par_tag}')

        if use_parallel:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_single_trial)(p, **common_kw) for p in p_inits
            )
            all_results = [(r, p, res) for r, p, res in results]
        else:
            if self.verbose:
                print(
                    f"{'Trial':>5s}  {'Init RMSE (meV)':>15s}  "
                    f"{'Final RMSE (meV)':>16s}  {'nfev':>5s}"
                )
                print('-' * 50)
            all_results = []
            best_so_far = np.inf
            for trial, p_init in enumerate(p_inits):
                rmse, p_opt, res = self._run_single_trial(p_init, **common_kw)
                all_results.append((rmse, p_opt, res))
                tag = ' *' if rmse < best_so_far else ''
                if rmse < best_so_far:
                    best_so_far = rmse
                if self.verbose:
                    rmse_init = np.sqrt(
                        np.mean((self.eigenvalues(p_init) - self.E_pao).ravel() ** 2)
                    )
                    print(
                        f'{trial + 1:5d}  {rmse_init * 1000:15.2f}  '
                        f'{rmse * 1000:16.2f}  {res.nfev:5d}{tag}'
                    )

        # ── Collect results ──
        all_results.sort(key=lambda x: x[0])
        best_rmse, best_p, best_res = all_results[0]
        best_data_res = best_res.fun[:n_data]

        if self.verbose:
            if use_parallel:
                print(f'  Completed {n_trials} trials in parallel')
            print(f"{'=' * 65}")
            msg = (
                f'Best RMSE = {best_rmse * 1000:.2f} meV, '
                f'max|δ| = {np.max(np.abs(best_data_res)) * 1000:.2f} meV'
            )
            if alpha > 0:
                msg += f'  (α = {alpha:.4g})'
            print(msg)
            print(f"\n{'Parameter':<30s}  {'Value':>10s}")
            print('-' * 43)
            for i, name in enumerate(self.param_labels):
                print(f'{name:<30s}  {best_p[i]: .5f}')

        return {
            'p_opt': best_p,
            'rmse': best_rmse,
            'max_err': np.max(np.abs(best_data_res)),
            'all_results': all_results,
            'param_labels': list(self.param_labels),
        }

    # ── 2k. Build PAOFLOW model dict ─────────────────────────

    def build_model_dict(self, p: np.ndarray) -> dict:
        """Convert a fitted parameter vector into a model dict.

        The output uses species-pair-keyed hoppings with explicit
        shell reference distances (Bohr), following the ``edtb_params``
        schema::

            "hoppings": {
              "Pt-Pt": [
                {"r_ref": 5.247, "params": {"sss": ..., "sps": ...}},
                ...
              ]
            }

        Parameters
        ----------
        p : np.ndarray
            Parameter vector (length ``n_params``).

        Returns
        -------
        dict
            Model dict with species-pair-keyed hoppings.
        """

        # ── SK-param dict for one shell ──
        def _build_hop(p_hop):
            # Single configuration (one shell per l): flat SK dict, keyed by
            # SK integral name ('sss', 'sps', ...) — the legacy format.
            if not self.is_multiconfig:
                d = {}
                for ga, gb in self.hop_pair_list:
                    start = self.hop_pair_start[(ga, gb)]
                    active = self.hop_pair_active[(ga, gb)]
                    for lk, sk_k in enumerate(active):
                        d[SK_PARAM_NAMES[sk_k]] = float(p_hop[start + lk])
                return d
            # Multi-configuration: one SK sub-dict per shell-pair, keyed by
            # configuration labels ordered by group index (e.g. '3S-4S',
            # '3S-3P').  This matches the canonicalisation used by
            # models.Slater_Koster, which orders pairs by config-list index.
            d = {}
            for ga, gb in self.hop_pair_list:
                start = self.hop_pair_start[(ga, gb)]
                active = self.hop_pair_active[(ga, gb)]
                pair_key = f'{self.cfg_names[ga]}-{self.cfg_names[gb]}'
                sub = {}
                for lk, sk_k in enumerate(active):
                    sub[SK_PARAM_NAMES[sk_k]] = float(p_hop[start + lk])
                d[pair_key] = sub
            return d

        # ── Species-pair-keyed hoppings ──
        from .edtb_params import compute_pair_shell_distances, species_pair_key

        # Build per-shell SK params (species-blind)
        shell_params = []
        for s in range(len(self.shell_tags)):
            i0 = self.n_onsite + s * self.n_hop
            shell_params.append(_build_hop(p[i0 : i0 + self.n_hop]))

        # Build atom list in Bohr for per-pair distance computation
        atoms_bohr = [
            {
                'species': self.atoms_list[iat],
                'tau': (self.tau_alat[iat] * self.alat).tolist(),
            }
            for iat in range(self.nat)
        ]
        a_vecs_bohr = self.a_vecs * self.alat

        sorted_species = sorted(set(self.unique_species))
        hoppings = {}
        for i, sp1 in enumerate(sorted_species):
            for sp2 in sorted_species[i:]:
                key = species_pair_key(sp1, sp2)
                pair_dists = compute_pair_shell_distances(
                    a_vecs_bohr,
                    atoms_bohr,
                    sp1,
                    sp2,
                    n_shells=len(self.shell_tags),
                )
                shells = []
                for s in range(len(self.shell_tags)):
                    r_ref = (
                        round(pair_dists[s], 6)
                        if s < len(pair_dists)
                        else round(float(self.shell_dists[s] * self.alat), 6)
                    )
                    shells.append({'r_ref': r_ref, 'params': dict(shell_params[s])})
                hoppings[key] = shells

        # ── Atom dicts ──
        model_atoms = {}
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            pstart = self.species_param_start[sp]
            groups = self.species_onsite_groups[sp]
            atom_d = {'name': sp, 'tau': self.tau_alat[iat].tolist()}

            if self.config_dict:
                atom_d['configuration'] = list(self.config_dict[sp])
                for ig, (pname, _) in enumerate(groups):
                    key = pname[2:-1]  # 'ε(3S)' → '3S'
                    atom_d[key] = float(p[pstart + ig])
            else:
                orb_list = []
                for shell_l in self.shells_dict[sp]:
                    orb_list.extend(SHELL_TO_ORBITALS[shell_l])
                atom_d['orbitals'] = list(self.atom_orbitals[iat])
                for ig, (_, local_indices) in enumerate(groups):
                    val = float(p[pstart + ig])
                    for li in local_indices:
                        atom_d[orb_list[li]] = val

            model_atoms[str(iat)] = atom_d

        return {
            'label': 'Slater_Koster',
            'alat': float(self.alat),
            'model': {
                'a_vectors': self.a_vecs.tolist(),
                'atoms': model_atoms,
                'hoppings': hoppings,
            },
        }

    # ── 2l. Convenience: eigenvalues for given p ──────────────

    def eigenvalues(self, p: np.ndarray) -> np.ndarray:
        """Compute SK eigenvalues for parameter vector *p* on the fitting k-mesh.

        Returns
        -------
        np.ndarray
            Shape ``(Nk, nawf)`` eigenvalues in eV.
        """
        E_sk, _ = self._eigenvalues_and_jacobian(p)
        return E_sk


# ═══════════════════════════════════════════════════════════════
#  3. SKFitterEDTB — environment-dependent tight-binding
# ═══════════════════════════════════════════════════════════════


class SKFitterEDTB(SKFitter):
    r"""Environment-dependent tight-binding (EDTB) extension of SKFitter.

    Augments the two-center SK hopping integrals with an
    environment-dependent screening factor:

    .. math::

        V_\lambda^{\text{eff}}(i,j) =
            V_\lambda^{(2c)} \exp\!\bigl(-\gamma_\lambda\,S_{ij}\bigr)

    where :math:`S_{ij} = \sum_{k \neq i,j} f_c(d_{ik})\,f_c(d_{jk})` is
    a bond screening sum and :math:`f_c` is a smooth cosine cutoff
    that tapers to zero between :math:`0.8\,r_\text{cut}` and
    :math:`r_\text{cut}`.

    Optionally fits environment-dependent on-site shifts:

    .. math::

        \varepsilon_\alpha \;\to\;
            \varepsilon_\alpha + \eta_\alpha \sum_k f_c(d_{ik})

    The screening strengths :math:`\gamma` can be parametrised at three
    granularity levels (``gamma_mode``):

    * ``'global'`` — one :math:`\gamma` for all channels (1 parameter).
    * ``'per_lpair'`` — one per angular-momentum pair
      (ss, sp, pp, …; up to 6).
    * ``'per_channel'`` — one per SK integral
      (ssσ, spσ, ppσ, ppπ, …; up to 10).

    Parameters
    ----------
    arryp, attrp : dict
        PAOFLOW data dicts (same as :class:`SKFitter`).
    n_shells : int
        Number of neighbor shells (default 2).
    nkfit : int
        k-mesh subdivision (default 6).
    r_cut : float
        Screening cutoff radius **in Bohr**.
    gamma_mode : {'global', 'per_lpair', 'per_channel'}
        Granularity of screening parameters (default ``'global'``).
    fit_onsite_shift : bool
        Whether to fit :math:`\eta` on-site shift parameters (default False).
    verbose : bool
        Print progress information.

    Notes
    -----
    For a single crystal structure the screening parameters are partially
    redundant with the shell-dependent hoppings.  Meaningful :math:`\gamma`
    values typically require multi-structure training data or external
    constraints on the two-center integrals (e.g. supply ``p0_sk`` to
    :meth:`fit` so that only :math:`\gamma` and :math:`\eta` are free to
    adjust).

    Usage
    -----
    >>> fitter = SKFitterEDTB(arry, attr, n_shells=2, nkfit=6, r_cut=8.0)
    >>> result = fitter.fit(n_trials=10, seed=42)
    >>> model  = fitter.build_model_dict(result['p_opt'])

    Staged fitting (recommended):

    >>> fitter_sk = SKFitter(arry, attr, n_shells=2)
    >>> p_sk = fitter_sk.fit(n_trials=20)['p_opt']
    >>> fitter_edtb = SKFitterEDTB(arry, attr, n_shells=2, r_cut=8.0)
    >>> result = fitter_edtb.fit(p0_sk=p_sk, n_trials=10)
    """

    _LPAIR_LABELS = {
        (0, 0): 'ss',
        (0, 1): 'sp',
        (0, 2): 'sd',
        (1, 1): 'pp',
        (1, 2): 'pd',
        (2, 2): 'dd',
    }

    def __init__(
        self,
        arryp: dict,
        attrp: dict,
        *,
        n_shells: int = 2,
        nkfit: int = 6,
        r_cut: float,
        gamma_mode: str = 'global',
        fit_onsite_shift: bool = False,
        verbose: bool = True,
    ):
        # Base SKFitter setup (bonds, design tensors, onsite map, etc.)
        super().__init__(arryp, attrp, n_shells=n_shells, nkfit=nkfit, verbose=verbose)
        self.r_cut_bohr = float(r_cut)
        self.r_cut_alat = self.r_cut_bohr / self.alat
        self.gamma_mode = gamma_mode
        self.fit_onsite_shift = fit_onsite_shift

        self._build_screening_geometry()
        self._build_edtb_parameters()
        self._build_regularization_weights()  # rebuild with EDTB params

    # ── 3a. Screening geometry ────────────────────────────────

    def _build_screening_geometry(self):
        """Precompute screening sums S_ij for all bonds and coordination numbers."""
        r_cut = self.r_cut_alat
        r_taper = 0.8 * r_cut

        # Supercell for the screening neighbourhood
        min_a = min(np.linalg.norm(v) for v in self.a_vecs)
        sc_range = int(np.ceil(r_cut / min_a)) + 1
        sc_pos = []
        for i1 in range(-sc_range, sc_range + 1):
            for i2 in range(-sc_range, sc_range + 1):
                for i3 in range(-sc_range, sc_range + 1):
                    R = i1 * self.a_vecs[0] + i2 * self.a_vecs[1] + i3 * self.a_vecs[2]
                    for iat in range(self.nat):
                        sc_pos.append(R + self.tau_alat[iat])
        sc_pos = np.array(sc_pos)

        def _fc_vec(d):
            """Vectorised smooth cosine cutoff (zero at self-distance)."""
            fc = np.where(
                d <= r_taper,
                1.0,
                np.where(
                    d >= r_cut,
                    0.0,
                    0.5 * (1.0 + np.cos(np.pi * (d - r_taper) / (r_cut - r_taper))),
                ),
            )
            fc[d < 1e-10] = 0.0  # exclude self
            return fc

        # f_c(d_{ia,k}) for home atoms → all supercell atoms
        fc_home = np.zeros((self.nat, len(sc_pos)))
        for ia in range(self.nat):
            fc_home[ia] = _fc_vec(np.linalg.norm(sc_pos - self.tau_alat[ia], axis=1))

        self.coord_i = np.sum(fc_home, axis=1)

        # S_ij for every bond in each shell
        self.S_bonds = []
        for s in range(self.n_shells):
            bonds = self.shell_bonds_list[s]
            S = np.empty(len(bonds))
            for ib, (R_cart, iat, jat, *_rest) in enumerate(bonds):
                pos_j = R_cart + self.tau_alat[jat]
                d_jk = np.linalg.norm(sc_pos - pos_j, axis=1)
                S[ib] = np.dot(fc_home[iat], _fc_vec(d_jk))
            self.S_bonds.append(S)

        if self.verbose:
            for s in range(self.n_shells):
                sv = self.S_bonds[s]
                print(
                    f'  Screening ({self.shell_tags[s]}): '
                    f'S̄={sv.mean():.3f}, '
                    f'range=[{sv.min():.3f}, {sv.max():.3f}]'
                )
            print('  Coordination: ' + ', '.join(f'{c:.2f}' for c in self.coord_i))

    # ── 3b. EDTB parameter registry ──────────────────────────

    def _build_edtb_parameters(self):
        """Register γ (and optional η) parameters."""
        gm = self.gamma_mode

        # Active l-pairs and SK channels
        active_lp, active_ch = set(), set()
        for ga, gb in self.hop_pair_list:
            la, lb = self.group_l[ga], self.group_l[gb]
            active_lp.add((min(la, lb), max(la, lb)))
            for sk_idx in self.hop_pair_active[(ga, gb)]:
                active_ch.add(SK_PARAM_NAMES[sk_idx])
        self.active_lpairs = sorted(active_lp)
        self.active_channels = sorted(active_ch, key=lambda x: SK_PARAM_NAMES.index(x))

        # hop param index → γ index (same mapping for every shell)
        self._hop_to_gamma = np.zeros(self.n_hop, dtype=int)

        if gm == 'global':
            self.n_gamma = 1
            self.gamma_labels = ['γ']
        elif gm == 'per_lpair':
            lp2i = {lp: i for i, lp in enumerate(self.active_lpairs)}
            self.n_gamma = len(self.active_lpairs)
            self.gamma_labels = [f'γ_{self._LPAIR_LABELS[lp]}' for lp in self.active_lpairs]
            for ga, gb in self.hop_pair_list:
                la, lb = self.group_l[ga], self.group_l[gb]
                gidx = lp2i[(min(la, lb), max(la, lb))]
                st = self.hop_pair_start[(ga, gb)]
                for lk in range(len(self.hop_pair_active[(ga, gb)])):
                    self._hop_to_gamma[st + lk] = gidx
        elif gm == 'per_channel':
            ch2i = {ch: i for i, ch in enumerate(self.active_channels)}
            self.n_gamma = len(self.active_channels)
            self.gamma_labels = [f'γ_{ch}' for ch in self.active_channels]
            for ga, gb in self.hop_pair_list:
                st = self.hop_pair_start[(ga, gb)]
                for lk, sk_idx in enumerate(self.hop_pair_active[(ga, gb)]):
                    self._hop_to_gamma[st + lk] = ch2i[SK_PARAM_NAMES[sk_idx]]
        else:
            raise ValueError(f'Unknown gamma_mode: {gm!r}')

        # On-site shift η
        if self.fit_onsite_shift:
            present = set()
            for sp in self.unique_species:
                for l_val in self.shells_dict[sp]:
                    present.add({0: 's', 1: 'p', 2: 'd'}[l_val])
            self.eta_orb_types = sorted(present, key='spd'.index)
            self.n_eta = len(self.eta_orb_types)
            self.eta_labels = [f'η_{t}' for t in self.eta_orb_types]
            _otype = {
                's': 's',
                'px': 'p',
                'py': 'p',
                'pz': 'p',
                'dxy': 'd',
                'dyz': 'd',
                'dzx': 'd',
                'dx2-y2': 'd',
                'dz2': 'd',
            }
            self._eta_diag = np.zeros((self.n_eta, self.nawf))
            for iat in range(self.nat):
                bi = self.atom_block_start[iat]
                for io, orb in enumerate(self.atom_orbitals[iat]):
                    q = self.eta_orb_types.index(_otype[orb])
                    self._eta_diag[q, bi + io] = self.coord_i[iat]
        else:
            self.n_eta = 0
            self.eta_labels = []

        # Parameter bookkeeping
        n_sk = self.n_onsite + self.n_shells * self.n_hop
        self.n_sk = n_sk
        self.n_gamma_start = n_sk
        self.n_eta_start = n_sk + self.n_gamma
        self.n_params = n_sk + self.n_gamma + self.n_eta

        self.param_labels = self.param_labels[:n_sk]
        self.param_labels.extend(self.gamma_labels)
        self.param_labels.extend(self.eta_labels)

        if self.verbose:
            print(f'  EDTB screening: {self.n_gamma} γ ({self.gamma_mode}), {self.n_eta} η')
            print(f'  Total parameters: {self.n_params}')

    # ── 3c. Regularization weights ────────────────────────────

    def _build_regularization_weights(self):
        """Extend Tikhonov weights to cover γ and η parameters."""
        if not hasattr(self, 'n_gamma_start'):
            # Called from super().__init__() before EDTB setup
            super()._build_regularization_weights()
            return
        w = np.zeros(self.n_params)
        for s in range(self.n_shells):
            d_s = self.shell_dists[s]
            i0 = self.n_onsite + s * self.n_hop
            w[i0 : i0 + self.n_hop] = d_s
        hw = w[self.n_onsite : self.n_sk]
        if hw.sum() > 0:
            w[self.n_onsite : self.n_sk] = hw / hw.mean()
        w[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = 1.0
        if self.n_eta > 0:
            w[self.n_eta_start : self.n_eta_start + self.n_eta] = 1.0
        self._reg_weights = w

    # ── 3d. Forward model (screened eigenvalues + Jacobian) ───

    def _eigenvalues_and_jacobian(self, p):
        """Eigenvalues and Hellmann-Feynman Jacobian with EDTB screening.

        Uses block-sparse operations via bond_groups for sub-block efficiency.
        """
        # Guard: during super().__init__(), EDTB attrs are not set yet
        if not hasattr(self, 'n_gamma_start'):
            return super()._eigenvalues_and_jacobian(p)

        Nk, nawf = self.Nk, self.nawf
        n_onsite, n_hop = self.n_onsite, self.n_hop
        gamma = p[self.n_gamma_start : self.n_gamma_start + self.n_gamma]

        # ── on-site Hamiltonian ──
        H0 = np.einsum('p,pij->ij', p[:n_onsite], self._onsite_map)
        if self.n_eta > 0:
            eta = p[self.n_eta_start : self.n_eta_start + self.n_eta]
            shift = np.einsum('q,qi->i', eta, self._eta_diag)
            H0[np.arange(nawf), np.arange(nawf)] += shift

        Hk = np.broadcast_to(H0, (Nk, nawf, nawf)).astype(complex).copy()

        # ── screening scales: scale[b,p] = exp(-γ_{q(p)} · S_b) ──
        gamma_per_hop = gamma[self._hop_to_gamma]  # (n_hop,)
        scales = []
        for s in range(self.n_shells):
            scales.append(np.exp(-gamma_per_hop[None, :] * self.S_bonds[s][:, None]))

        # ── screened dH/dV (block-sparse) ──
        screened_dHk = []
        for s in range(self.n_shells):
            sc_s = scales[s]  # (n_bonds_s, n_hop)
            dHk_s = np.zeros((Nk, n_hop, nawf, nawf), dtype=complex)
            for idx, bi, bj, no_i, no_j, M_sub in self._bond_groups_shells[s]:
                M_sc = sc_s[idx][:, :, None, None] * M_sub
                dHk_s[:, :, bi : bi + no_i, bj : bj + no_j] += np.einsum(
                    'kb,bpij->kpij', self._phases_shells[s][:, idx], M_sc
                )
            screened_dHk.append(dHk_s)

        # ── hopping contribution to H(k) ──
        for s in range(self.n_shells):
            i0 = n_onsite + s * n_hop
            Hk += np.einsum('p,kpij->kij', p[i0 : i0 + n_hop], screened_dHk[s])

        # ── eigendecomposition ──
        evals, evecs = np.linalg.eigh(Hk)
        E_sk = evals.real

        # ── Jacobian (Hellmann-Feynman) ──
        dE_dp = np.zeros((Nk, nawf, self.n_params))
        psi2 = np.abs(evecs) ** 2

        # ∂E/∂ε  (on-site)
        dE_dp[:, :, :n_onsite] = np.einsum('kin,pi->knp', psi2, self._onsite_diag)

        # ∂E/∂V  (hopping — through screened dHk)
        evecs_bc = evecs[:, np.newaxis, :, :]
        for s in range(self.n_shells):
            i0 = n_onsite + s * n_hop
            tmp = np.matmul(screened_dHk[s], evecs_bc)
            dE_dp[:, :, i0 : i0 + n_hop] = np.real(np.einsum('kin,kpin->knp', evecs.conj(), tmp))

        # ∂E/∂γ_q  (block-sparse)
        for q in range(self.n_gamma):
            mask = self._hop_to_gamma == q
            if not np.any(mask):
                continue
            dH_dg = np.zeros((Nk, nawf, nawf), dtype=complex)
            for s in range(self.n_shells):
                i0 = n_onsite + s * n_hop
                V = p[i0 : i0 + n_hop]
                sc_s = scales[s]
                for idx, bi, bj, no_i, no_j, M_sub in self._bond_groups_shells[s]:
                    VS = V[mask] * sc_s[idx][:, mask]
                    VS *= self.S_bonds[s][idx, None]
                    wM = np.einsum('bp,bpij->bij', VS, M_sub[:, mask, :, :])
                    dH_dg[:, bi : bi + no_i, bj : bj + no_j] -= np.einsum(
                        'kb,bij->kij', self._phases_shells[s][:, idx], wM
                    )
            HFq = np.matmul(dH_dg, evecs)
            dE_dp[:, :, self.n_gamma_start + q] = np.real(
                np.einsum('kin,kin->kn', evecs.conj(), HFq)
            )

        # ∂E/∂η_q  (on-site shift)
        if self.n_eta > 0:
            for q in range(self.n_eta):
                dE_dp[:, :, self.n_eta_start + q] = np.einsum('kin,i->kn', psi2, self._eta_diag[q])

        return E_sk, dE_dp

    # ── 3e. Fitting ───────────────────────────────────────────

    def fit(
        self,
        *,
        p0_sk: np.ndarray | None = None,
        n_trials: int = 10,
        seed: int | None = 123,
        max_nfev: int = 1000,
        ftol: float = 1e-12,
        xtol: float = 1e-12,
        gtol: float = 1e-12,
        alpha: float = 0.0,
        n_jobs: int = 1,
    ) -> dict:
        """Multi-start least-squares fit including screening parameters.

        Parameters
        ----------
        p0_sk : np.ndarray, optional
            Initial SK parameter vector (length ``n_sk``).  If provided the
            first trial uses these values directly; subsequent trials add a
            small random perturbation to the hopping part.  If *None*,
            on-site energies are extracted from H(R=0) and hoppings are
            randomised (same behaviour as :class:`SKFitter`).
        n_trials, seed, max_nfev, ftol, xtol, gtol, alpha
            Same as :meth:`SKFitter.fit`.
        n_jobs : int
            Number of parallel workers for multi-start trials
            (default 1 = sequential).  Use ``-1`` for all available cores.
            Requires ``joblib`` when ``n_jobs != 1``.

        Returns
        -------
        dict
            ``p_opt`` : best parameter vector (length ``n_params``).
            ``rmse`` : best RMSE (eV, data-only).
            ``max_err`` : max absolute error (eV).
            ``all_results`` : sorted list of ``(rmse, p, OptimizeResult)``.
            ``param_labels`` : parameter names.
        """
        Nk, nawf = self.Nk, self.nawf
        n_data = Nk * nawf

        # SK initialisation
        if p0_sk is not None:
            p0_sk = np.asarray(p0_sk, dtype=float)
            if p0_sk.shape[0] != self.n_sk:
                raise ValueError(f'p0_sk length {p0_sk.shape[0]} != n_sk {self.n_sk}')
        p0_onsite = self.extract_onsite_from_HR0()
        E_half = 0.5 * (self.E_pao.max() - self.E_pao.min())
        hop_scales = [E_half / np.sqrt(len(b)) for b in self.shell_bonds_list]

        rng = np.random.RandomState(seed)

        # Pre-generate all initial points
        p_inits = []
        for trial in range(n_trials):
            p_init = np.zeros(self.n_params)
            if p0_sk is not None:
                p_init[: self.n_sk] = p0_sk
                if trial > 0:
                    for s in range(self.n_shells):
                        i0 = self.n_onsite + s * self.n_hop
                        i1 = i0 + self.n_hop
                        p_init[i0:i1] *= 1.0 + 0.05 * rng.randn(self.n_hop)
            else:
                p_init[: self.n_onsite] = p0_onsite
                for s, sc in enumerate(hop_scales):
                    i0 = self.n_onsite + s * self.n_hop
                    p_init[i0 : i0 + self.n_hop] = rng.uniform(-sc, sc, self.n_hop)
            # γ: small positive random initialisation
            p_init[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = rng.uniform(
                0.0, 0.01, self.n_gamma
            )
            p_inits.append(p_init)

        # ── Run trials ──
        use_parallel = n_jobs != 1 and n_trials > 1
        common_kw = dict(
            alpha=alpha,
            n_data=n_data,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        if self.verbose:
            print(f"\n{'=' * 65}")
            par_tag = f', n_jobs={n_jobs}' if use_parallel else ''
            print(
                f'EDTB multi-start optimisation: {n_trials} trials, '
                f'γ_mode={self.gamma_mode}{par_tag}'
            )

        if use_parallel:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_single_trial)(p, **common_kw) for p in p_inits
            )
            all_results = [(r, p, res) for r, p, res in results]
        else:
            if self.verbose:
                print(
                    f"{'Trial':>5s}  {'Init RMSE (meV)':>15s}  "
                    f"{'Final RMSE (meV)':>16s}  {'nfev':>5s}"
                )
                print('-' * 50)
            all_results = []
            best_so_far = np.inf
            for trial, p_init in enumerate(p_inits):
                rmse, p_opt, res = self._run_single_trial(p_init, **common_kw)
                all_results.append((rmse, p_opt, res))
                tag = ' *' if rmse < best_so_far else ''
                if rmse < best_so_far:
                    best_so_far = rmse
                if self.verbose:
                    rmse_init = np.sqrt(
                        np.mean((self.eigenvalues(p_init[: self.n_sk]) - self.E_pao).ravel() ** 2)
                    )
                    print(
                        f'{trial + 1:5d}  {rmse_init * 1000:15.2f}  '
                        f'{rmse * 1000:16.2f}  {res.nfev:5d}{tag}'
                    )

        # ── Collect results ──
        all_results.sort(key=lambda x: x[0])
        best_rmse, best_p, best_res = all_results[0]
        best_data_res = best_res.fun[:n_data]

        if self.verbose:
            if use_parallel:
                print(f'  Completed {n_trials} trials in parallel')
            print(f"{'=' * 65}")
            msg = (
                f'Best RMSE = {best_rmse * 1000:.2f} meV, '
                f'max|δ| = {np.max(np.abs(best_data_res)) * 1000:.2f} meV'
            )
            if alpha > 0:
                msg += f'  (α = {alpha:.4g})'
            print(msg)
            print(f"\n{'Parameter':<30s}  {'Value':>10s}")
            print('-' * 43)
            for i, name in enumerate(self.param_labels):
                print(f'{name:<30s}  {best_p[i]: .5f}')

        return {
            'p_opt': best_p,
            'rmse': best_rmse,
            'max_err': np.max(np.abs(best_data_res)),
            'all_results': all_results,
            'param_labels': list(self.param_labels),
        }

    # ── 3f. Build PAOFLOW model dict ─────────────────────────

    def build_model_dict(self, p: np.ndarray) -> dict:
        """Convert fitted parameters to an ``SK_EDTB`` model dict.

        The screening ``gamma`` is wrapped in a species-pair key,
        consistent with the species-pair-keyed hoppings from the
        base class.

        Parameters
        ----------
        p : np.ndarray
            Full parameter vector (length ``n_params``).

        Returns
        -------
        dict
            Model dict with ``label='SK_EDTB'`` and ``screening`` block.
        """
        # Base SK dict (already species-pair-keyed hoppings)
        base = super().build_model_dict(p[: self.n_sk])
        base['label'] = 'SK_EDTB'

        # Screening block — gamma keyed by species pair
        from .edtb_params import species_pair_key

        gamma = p[self.n_gamma_start : self.n_gamma_start + self.n_gamma]
        if self.gamma_mode == 'global':
            gamma_val = float(gamma[0])
        elif self.gamma_mode == 'per_lpair':
            gamma_val = {
                self._LPAIR_LABELS[lp]: float(gamma[i]) for i, lp in enumerate(self.active_lpairs)
            }
        elif self.gamma_mode == 'per_channel':
            gamma_val = {ch: float(gamma[i]) for i, ch in enumerate(self.active_channels)}

        sorted_species = sorted(set(self.unique_species))
        gamma_dict = {}
        for i, sp1 in enumerate(sorted_species):
            for sp2 in sorted_species[i:]:
                key = species_pair_key(sp1, sp2)
                if isinstance(gamma_val, dict):
                    gamma_dict[key] = dict(gamma_val)
                else:
                    gamma_dict[key] = gamma_val

        screening = {'r_cut': self.r_cut_bohr, 'gamma': gamma_dict}

        if self.n_eta > 0:
            eta = p[self.n_eta_start : self.n_eta_start + self.n_eta]
            screening['onsite_shift'] = {
                self.eta_orb_types[i]: float(eta[i]) for i in range(self.n_eta)
            }

        base['model']['screening'] = screening
        return base


# ═══════════════════════════════════════════════════════════════
#  4. MultiGeomEDTB — multi-geometry EDTB fitting
# ═══════════════════════════════════════════════════════════════


class MultiGeomEDTB:
    r"""Multi-geometry environment-dependent tight-binding fitter.

    Fits a **single shared** set of parameters
    :math:`(\varepsilon, V_\lambda, \gamma, \eta)` to DFT band structures
    from **multiple atomic configurations** simultaneously, which is
    essential for learning physically meaningful screening strengths
    :math:`\gamma` that capture the environment dependence of hopping
    integrals.

    Each geometry is represented internally by an independent
    :class:`SKFitterEDTB` instance (with its own screening sums
    :math:`S_{ij}`, design tensors, and reference eigenvalues).  The
    combined objective is the (optionally weighted) concatenation of
    eigenvalue residuals over all geometries.

    .. note::

        All geometries must share the **same species, orbital basis, and
        shell structure** so that the hopping parameter vector is
        identical.  This is automatically satisfied when the training set
        consists of the same material at different lattice parameters,
        strains, surfaces, or defect configurations (with the same
        pseudopotential and projection basis).

    Typical training sets
    ---------------------
    * **Volume scan**: equilibrium ± 2%, ± 5% isotropic expansion
      (easiest to generate, same symmetry).
    * **Tetragonal distortion**: c/a ≠ 1 strains that break cubic
      symmetry.
    * **Surface slab**: 5–7 layer slab with vacuum; atoms at the
      surface have reduced coordination → very different
      :math:`S_{ij}`.

    Parameters
    ----------
    geometries : list of (arryp, attrp) tuples
        PAOFLOW data-dict pairs, one per configuration.
    n_shells : int
        Number of neighbor shells (default 2).
    nkfit : int
        k-mesh subdivision (default 6).
    r_cut : float
        Screening cutoff radius **in Bohr**.
    gamma_mode : {'global', 'per_lpair', 'per_channel'}
        Screening parameter granularity.
    fit_onsite_shift : bool
        Whether to fit η on-site shift parameters (default False).
    nkfit : int or list of int/tuple
        Subdivisions along each reciprocal axis for the fitting k-mesh.
        If an int, applies uniformly to all geometries.  If a list,
        must have one entry per geometry; each entry can be an int
        (uniform) or a 3-tuple ``(n1, n2, n3)`` for an anisotropic grid.
        Use ``'auto'`` (default) to automatically detect slab geometries
        and reduce the k-grid to 1 along the vacuum direction.
    weights : list of float, optional
        Per-geometry weights for the loss function.  Default: uniform
        (all 1.0).  Increase weight on geometries that should be
        reproduced more accurately (e.g. equilibrium bulk).
    verbose : bool
        Print progress information.

    Usage
    -----
    >>> from sk_fitting import SKFitter, MultiGeomEDTB
    >>> # Pre-fit SK on equilibrium geometry
    >>> fitter_sk = SKFitter(arry_eq, attr_eq, n_shells=3)
    >>> p_sk = fitter_sk.fit(n_trials=20)['p_opt']
    >>> # Multi-geometry EDTB
    >>> geoms = [(arry_eq, attr_eq), (arry_p5, attr_p5), (arry_m5, attr_m5)]
    >>> mg = MultiGeomEDTB(geoms, n_shells=3, r_cut=8.0, gamma_mode='per_lpair')
    >>> result = mg.fit(p0_sk=p_sk, n_trials=10, n_jobs=-1)
    >>> model  = mg.build_model_dict(result['p_opt'])
    """

    @staticmethod
    def _detect_nkfit(arry, nkfit_base):
        """Detect slab dimensionality and return an appropriate (n1, n2, n3).

        Heuristic: if the length of a lattice vector is > 2× the geometric
        mean of the other two, that direction is vacuum → use nkfit=1.
        """
        a_vecs = arry['a_vectors']
        lengths = np.array([np.linalg.norm(a_vecs[i]) for i in range(3)])
        # geo_mean = np.cbrt(np.prod(lengths))
        nk = [nkfit_base, nkfit_base, nkfit_base]
        for i in range(3):
            other = [lengths[j] for j in range(3) if j != i]
            mean_other = 0.5 * (other[0] + other[1])
            if lengths[i] > 2.0 * mean_other:
                nk[i] = 1
        return tuple(nk)

    def __init__(
        self,
        geometries: list[tuple[dict, dict]],
        *,
        n_shells: int = 2,
        nkfit: int | str | list = 'auto',
        r_cut: float,
        gamma_mode: str = 'global',
        fit_onsite_shift: bool = False,
        weights: list[float] | None = None,
        verbose: bool = True,
    ):
        if len(geometries) < 2:
            raise ValueError('MultiGeomEDTB requires at least 2 geometries')

        self.n_geom = len(geometries)
        self.verbose = verbose

        # ── Resolve per-geometry nkfit ──
        nkfit_base = 6  # default base grid
        if isinstance(nkfit, str) and nkfit == 'auto':
            nkfit_per_geom = [self._detect_nkfit(arry, nkfit_base) for arry, _ in geometries]
        elif isinstance(nkfit, (int, np.integer)):
            nkfit_base = int(nkfit)
            nkfit_per_geom = [self._detect_nkfit(arry, nkfit_base) for arry, _ in geometries]
        elif isinstance(nkfit, (list, tuple)) and len(nkfit) == len(geometries):
            nkfit_per_geom = []
            for nk in nkfit:
                if isinstance(nk, (int, np.integer)):
                    nkfit_per_geom.append((int(nk), int(nk), int(nk)))
                else:
                    nkfit_per_geom.append(tuple(nk))
        else:
            raise ValueError(f"nkfit must be 'auto', an int, or a list of length {len(geometries)}")

        # ── Build one SKFitterEDTB per geometry ──
        if verbose:
            print(f'MultiGeomEDTB: building {self.n_geom} sub-fitters …')

        self.fitters: list[SKFitterEDTB] = []
        for ig, (arry, attr) in enumerate(geometries):
            nk = nkfit_per_geom[ig]
            if verbose:
                print(f'\n── Geometry {ig} (nkfit={nk[0]}×{nk[1]}×{nk[2]}) ──')
            f = SKFitterEDTB(
                arry,
                attr,
                n_shells=n_shells,
                nkfit=nk,
                r_cut=r_cut,
                gamma_mode=gamma_mode,
                fit_onsite_shift=fit_onsite_shift,
                verbose=verbose,
            )
            self.fitters.append(f)

        # ── Harmonize species across sub-fitters ──
        # When geometries have different species sets (e.g. bulk Si,
        # bulk Ge, SiGe interface), pad each fitter with dummy onsite
        # parameters for missing species (zero design tensor → zero
        # Jacobian → no effect on that geometry's cost).
        self._harmonize_species()

        # ── Validate compatibility ──
        # All geometries must share the same parameter vector — same
        # species, orbital basis, shell count, and screening mode.
        # System sizes (nawf, Nk) may differ (e.g. bulk + slab).
        ref = self.fitters[0]
        for ig, f in enumerate(self.fitters[1:], 1):
            if f.n_params != ref.n_params:
                raise ValueError(
                    f'Geometry {ig}: n_params={f.n_params} != reference {ref.n_params}'
                )
            if f.n_hop != ref.n_hop:
                raise ValueError(f'Geometry {ig}: n_hop={f.n_hop} != reference {ref.n_hop}')
            if f.param_labels != ref.param_labels:
                raise ValueError(f'Geometry {ig}: param_labels mismatch with reference')

        # ── Mirror key attributes from the reference fitter ──
        self.nawf_per_geom = [f.nawf for f in self.fitters]
        self.nawf = ref.nawf  # backward compat (reference geometry)
        self.n_params = ref.n_params
        self.n_onsite = ref.n_onsite
        self.n_hop = ref.n_hop
        self.n_shells = ref.n_shells
        self.n_sk = ref.n_sk
        self.n_gamma = ref.n_gamma
        self.n_gamma_start = ref.n_gamma_start
        self.n_eta = ref.n_eta
        self.n_eta_start = ref.n_eta_start
        self.param_labels = list(ref.param_labels)
        self.gamma_mode = gamma_mode

        # ── Per-geometry weights ──
        if weights is None:
            self.weights = np.ones(self.n_geom)
        else:
            self.weights = np.asarray(weights, dtype=float)
            if len(self.weights) != self.n_geom:
                raise ValueError(f'len(weights)={len(self.weights)} != n_geom={self.n_geom}')

        # ── Per-geometry data size ──
        self.n_data_per_geom = [f.Nk * f.nawf for f in self.fitters]
        self.n_data_total = sum(self.n_data_per_geom)

        # Regularization weights from the reference geometry
        self._reg_weights = ref._reg_weights.copy()

        if verbose:
            print(f"\n{'=' * 65}")
            print('MultiGeomEDTB summary:')
            print(f'  {self.n_geom} geometries, {self.n_params} shared parameters')
            print(f'  n_data_total = {self.n_data_total}')
            for ig, f in enumerate(self.fitters):
                nk_str = (
                    f"grid={'×'.join(str(x) for x in f._nkfit_grid)}"
                    if hasattr(f, '_nkfit_grid')
                    else ''
                )
                print(f'  Geom {ig}: Nk={f.Nk}, nawf={f.nawf}, alat={f.alat:.4f} Bohr  {nk_str}')

    # ── 4a-pre. Species harmonisation ────────────────────────

    def _harmonize_species(self):
        """Pad sub-fitters so all share the same parameter structure.

        Collects the union of species across all geometries.  For each
        sub-fitter that lacks a species, dummy onsite parameters are
        injected with a zero design tensor (zero Jacobian ⟹ no effect
        on that geometry's cost).  ``cfg_names`` and
        ``hop_param_labels`` are standardised so that ``param_labels``
        match across all fitters.
        """
        # ── collect union of species (first-appearance order) ──
        all_species: list[str] = []
        all_shells: dict[str, list[int]] = {}
        all_config: dict[str, list[str]] = {}
        has_config = False
        for f in self.fitters:
            if f.config_dict:
                has_config = True
            for sp in f.unique_species:
                if sp not in all_species:
                    all_species.append(sp)
                    all_shells[sp] = list(f.shells_dict[sp])
                    if f.config_dict and sp in f.config_dict:
                        all_config[sp] = list(f.config_dict[sp])

        # ── check whether harmonisation is needed ──
        species_sets = [set(f.unique_species) for f in self.fitters]
        if all(s == set(all_species) for s in species_sets):
            # All fitters already have all species — only fix cfg_names
            ref_cfg = (
                list(all_config[all_species[0]])
                if has_config and all_species[0] in all_config
                else [
                    f'g{i}(l={self.fitters[0].group_l[i]})' for i in range(self.fitters[0].n_groups)
                ]
            )
            labels_ok = all(f.cfg_names == ref_cfg for f in self.fitters)
            if labels_ok:
                return  # nothing to do
            # Fall through to fix cfg_names / hop_param_labels

        if self.verbose:
            print(
                f'\n  Harmonising species: '
                f'{[f.unique_species for f in self.fitters]} → {all_species}'
            )

        # ── canonical cfg_names from the first species ──
        sp0 = all_species[0]
        ref_cfg_names = (
            list(all_config[sp0])
            if has_config and sp0 in all_config
            else [f'g{i}(l={self.fitters[0].group_l[i]})' for i in range(self.fitters[0].n_groups)]
        )

        # ── patch each sub-fitter ──
        for f in self.fitters:
            # a. register missing species in shells_dict / config_dict
            for sp in all_species:
                if sp not in f.shells_dict:
                    f.shells_dict[sp] = list(all_shells[sp])
                if has_config:
                    if f.config_dict is None:
                        f.config_dict = {}
                    if sp not in f.config_dict and sp in all_config:
                        f.config_dict[sp] = list(all_config[sp])

            # b. set canonical species order and cfg_names
            f.unique_species = list(all_species)
            f.cfg_names = list(ref_cfg_names)

            # c. rebuild hopping labels (same n_hop, new cfg_names)
            f.hop_param_labels = []
            for ga in range(f.n_groups):
                for gb in range(ga, f.n_groups):
                    la, lb = f.group_l[ga], f.group_l[gb]
                    lpair = (min(la, lb), max(la, lb))
                    labels = CHANNEL_LABELS[lpair]
                    pair_tag = f'{f.cfg_names[ga]}-{f.cfg_names[gb]}'
                    for lab in labels:
                        f.hop_param_labels.append(f'V({pair_tag}){lab}')

            # d. rebuild onsite parameters for the full species set
            f.n_onsite = 0
            f.onsite_param_names = []
            f.species_param_start = {}
            f.species_onsite_groups = {}

            for sp in all_species:
                f.species_param_start[sp] = f.n_onsite
                cfg = (
                    list(f.config_dict[sp])
                    if f.config_dict and sp in f.config_dict
                    else [f'g{i}' for i in range(len(f.shells_dict[sp]))]
                )
                groups = f._get_onsite_groups(f.shells_dict[sp], cfg)
                f.species_onsite_groups[sp] = groups
                for pname, _ in groups:
                    f.onsite_param_names.append(pname)
                f.n_onsite += len(groups)

            # e. update parameter counts
            n_sk = f.n_onsite + f.n_shells * f.n_hop
            f.n_sk = n_sk
            f.n_gamma_start = n_sk
            f.n_eta_start = n_sk + f.n_gamma
            f.n_params = n_sk + f.n_gamma + f.n_eta

            # f. rebuild param_labels
            f.param_labels = list(f.onsite_param_names)
            for tag in f.shell_tags:
                f.param_labels += [f'{tag.upper()}_{l}' for l in f.hop_param_labels]
            f.param_labels.extend(f.gamma_labels)
            f.param_labels.extend(f.eta_labels)

            # g. rebuild _onsite_map and _onsite_diag
            f._onsite_map = np.zeros((f.n_onsite, f.nawf, f.nawf))
            for iat in range(f.nat):
                sp = f.atoms_list[iat]
                bi = f.atom_block_start[iat]
                pstart = f.species_param_start[sp]
                for ig, (_, local_indices) in enumerate(f.species_onsite_groups[sp]):
                    for li in local_indices:
                        f._onsite_map[pstart + ig, bi + li, bi + li] = 1.0
            f._onsite_diag = np.array([np.diag(f._onsite_map[p]) for p in range(f.n_onsite)])

            # h. rebuild regularization weights
            f._build_regularization_weights()

        # ── remember which geometry has which real species ──
        self._geom_real_species = species_sets

    # ── 4a. Combined forward model ───────────────────────────

    def _eval_single_geometry(self, ig, p):
        """Evaluate eigenvalues and Jacobian for one geometry (thread-safe)."""
        f = self.fitters[ig]
        E_sk, dE_dp = f._eigenvalues_and_jacobian(p)
        res = (E_sk - f.E_pao).ravel()
        J = dE_dp.reshape(self.n_data_per_geom[ig], self.n_params)
        w = self.weights[ig]
        return w * res, w * J

    def _eigenvalues_and_jacobian_all(self, p):
        """Concatenated weighted residuals and Jacobian over all geometries.

        When ``n_geom >= 3``, evaluations are run in parallel using
        threads (``np.linalg.eigh`` releases the GIL).

        Returns
        -------
        res_all : np.ndarray, shape (n_data_total,)
            Concatenated weighted eigenvalue residuals.
        J_all : np.ndarray, shape (n_data_total, n_params)
            Vertically stacked weighted Jacobians.
        """
        if self.n_geom >= 3:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.n_geom) as pool:
                futures = [
                    pool.submit(self._eval_single_geometry, ig, p) for ig in range(self.n_geom)
                ]
                results = [fut.result() for fut in futures]
            residuals = [r for r, _ in results]
            jacobians = [j for _, j in results]
        else:
            residuals = []
            jacobians = []
            for ig in range(self.n_geom):
                r, j = self._eval_single_geometry(ig, p)
                residuals.append(r)
                jacobians.append(j)
        return np.concatenate(residuals), np.vstack(jacobians)

    # ── 4b. Single trial ─────────────────────────────────────

    def _run_single_trial(self, p_init, alpha, max_nfev, ftol, xtol, gtol, fixed_onsite=None):
        """Run one least-squares trial on the multi-geometry objective.

        Parameters
        ----------
        fixed_onsite : np.ndarray or None
            If given, array of length ``n_onsite`` with fixed on-site
            energies.  Only the hopping, screening, and shift parameters
            are optimised.

        Returns ``(rmse, p_opt, OptimizeResult)``.
        """
        use_reg = alpha > 0.0
        reg_w = self._reg_weights
        n_data = self.n_data_total
        last_jac = [None]

        if fixed_onsite is not None:
            n_on = self.n_onsite
            p_init_red = p_init[n_on:].copy()
            reg_w_red = reg_w[n_on:]

            def _expand(p_red):
                p_full = np.empty(self.n_params)
                p_full[:n_on] = fixed_onsite
                p_full[n_on:] = p_red
                return p_full

            def fun(p_red):
                p_full = _expand(p_red)
                res_data, J_full = self._eigenvalues_and_jacobian_all(p_full)
                J_red = J_full[:, n_on:]
                if use_reg:
                    last_jac[0] = np.vstack([J_red, np.diag(alpha * reg_w_red)])
                    return np.concatenate([res_data, alpha * reg_w_red * p_red])
                last_jac[0] = J_red
                return res_data

            def jac(p_red):
                return last_jac[0]

            res = least_squares(
                fun,
                p_init_red,
                jac=jac,
                method='lm',
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                max_nfev=max_nfev,
            )
            p_opt_full = _expand(res.x)
            rmse = np.sqrt(np.mean(res.fun[:n_data] ** 2))
            return rmse, p_opt_full, res

        def fun(p):
            res_data, J_data = self._eigenvalues_and_jacobian_all(p)
            if use_reg:
                last_jac[0] = np.vstack([J_data, np.diag(alpha * reg_w)])
                return np.concatenate([res_data, alpha * reg_w * p])
            last_jac[0] = J_data
            return res_data

        def jac(p):
            return last_jac[0]

        res = least_squares(
            fun,
            p_init,
            jac=jac,
            method='lm',
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        rmse = np.sqrt(np.mean(res.fun[:n_data] ** 2))
        return rmse, res.x.copy(), res

    # ── 4c. Initial guesses ──────────────────────────────────

    def extract_onsite_from_HR0(self) -> np.ndarray:
        """Extract initial on-site energies, merging across geometries.

        After species harmonisation each fitter has the full species
        set, but dummy-species slots return zero.  This method
        collects real-species values from whichever geometry actually
        contains each species.
        """
        p0 = np.zeros(self.n_onsite)
        filled = np.zeros(self.n_onsite, dtype=bool)
        # ref = self.fitters[0]
        for ig, f in enumerate(self.fitters):
            p_ig = f.extract_onsite_from_HR0()
            real_sp = (
                self._geom_real_species[ig]
                if hasattr(self, '_geom_real_species')
                else set(f.unique_species)
            )
            for sp in real_sp:
                pstart = f.species_param_start[sp]
                n_sp = len(f.species_onsite_groups[sp])
                idx = slice(pstart, pstart + n_sp)
                if not filled[idx].any():
                    p0[idx] = p_ig[idx]
                    filled[idx] = True
        return p0

    # ── 4d. Fitting ──────────────────────────────────────────

    def fit(
        self,
        *,
        p0_sk: np.ndarray | None = None,
        n_trials: int = 10,
        seed: int | None = 123,
        max_nfev: int = 2000,
        ftol: float = 1e-12,
        xtol: float = 1e-12,
        gtol: float = 1e-12,
        alpha: float = 0.0,
        n_jobs: int = 1,
        fix_onsite: dict | None = None,
    ) -> dict:
        """Multi-start least-squares fit across all geometries.

        Parameters
        ----------
        p0_sk : np.ndarray, optional
            Initial SK parameter vector (length ``n_sk``).  Typically from
            a single-geometry :meth:`SKFitter.fit` on the equilibrium
            configuration.
        n_trials : int
            Number of random restarts.
        seed : int or None
            Random seed for reproducibility.
        max_nfev : int
            Max function evaluations per trial (default 2000, larger than
            single-geometry because the combined landscape is harder).
        ftol, xtol, gtol : float
            Tolerances for ``scipy.optimize.least_squares``.
        alpha : float
            Tikhonov regularization strength.
        n_jobs : int
            Parallel workers for multi-start trials (``-1`` = all cores).
        fix_onsite : dict, optional
            Fix on-site energies to given values instead of fitting them.
            Dict mapping species name to an on-site dict, e.g.
            ``{'Si': {'s': -3.64, 'p': 2.14, 't2g': 6.30, 'eg': 6.37}, 'Ge': {'s': -5.30, 'p': 1.68, ...}}``.
            Typically obtained from independently fitted bulk models via
            ``model.params['onsite']['Si']``.

        Returns
        -------
        dict
            ``p_opt`` : best parameter vector (length ``n_params``).\n
            ``rmse`` : best combined RMSE (eV).\n
            ``per_geom_rmse`` : list of per-geometry RMSE values (eV).\n
            ``max_err`` : max absolute eigenvalue error (eV).\n
            ``all_results`` : sorted list of ``(rmse, p, OptimizeResult)``.\n
            ``param_labels`` : parameter names.
        """
        ref = self.fitters[0]

        # ── Parse fix_onsite ──
        fixed_onsite_vals = None
        if fix_onsite is not None:
            fixed_onsite_vals = np.zeros(self.n_onsite)
            for sp, on_dict in fix_onsite.items():
                if sp not in ref.species_param_start:
                    raise ValueError(f"fix_onsite: unknown species '{sp}'")
                pstart = ref.species_param_start[sp]
                for ig, (pname, orb_idx) in enumerate(ref.species_onsite_groups[sp]):
                    # Determine orbital type from group name suffix and size
                    if pname.endswith('_t2g)'):
                        key = 't2g'
                    elif pname.endswith('_eg)'):
                        key = 'eg'
                    elif len(orb_idx) == 1:
                        key = 's'
                    elif len(orb_idx) == 3:
                        key = 'p'
                    else:
                        raise ValueError(
                            f"fix_onsite['{sp}']: cannot determine orbital "
                            f"type for group '{pname}' with {len(orb_idx)} orbitals"
                        )
                    if key not in on_dict:
                        raise ValueError(
                            f"fix_onsite['{sp}']: missing key '{key}' "
                            f'(available: {sorted(on_dict.keys())})'
                        )
                    fixed_onsite_vals[pstart + ig] = on_dict[key]
            if self.verbose:
                print('\n  Fixing on-site energies (not fitted):')
                for i, name in enumerate(self.param_labels[: self.n_onsite]):
                    print(f'    {name} = {fixed_onsite_vals[i]:.6f}')

        # ── SK initialisation ──
        if p0_sk is not None:
            p0_sk = np.asarray(p0_sk, dtype=float)
            if p0_sk.shape[0] != self.n_sk:
                raise ValueError(f'p0_sk length {p0_sk.shape[0]} != n_sk {self.n_sk}')
        p0_onsite = self.extract_onsite_from_HR0()
        E_half = 0.5 * (ref.E_pao.max() - ref.E_pao.min())
        hop_scales = [E_half / np.sqrt(len(b)) for b in ref.shell_bonds_list]

        rng = np.random.RandomState(seed)

        # ── Pre-generate initial points ──
        p_inits = []
        for trial in range(n_trials):
            p_init = np.zeros(self.n_params)
            if p0_sk is not None:
                p_init[: self.n_sk] = p0_sk
                if trial > 0:
                    for s in range(self.n_shells):
                        i0 = self.n_onsite + s * self.n_hop
                        i1 = i0 + self.n_hop
                        p_init[i0:i1] *= 1.0 + 0.05 * rng.randn(self.n_hop)
            else:
                p_init[: self.n_onsite] = p0_onsite
                for s, sc in enumerate(hop_scales):
                    i0 = self.n_onsite + s * self.n_hop
                    p_init[i0 : i0 + self.n_hop] = rng.uniform(-sc, sc, self.n_hop)
            # γ: small positive random initialisation
            p_init[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = rng.uniform(
                0.0, 0.01, self.n_gamma
            )
            if fixed_onsite_vals is not None:
                p_init[: self.n_onsite] = fixed_onsite_vals
            p_inits.append(p_init)

        # ── Run trials ──
        use_parallel = n_jobs != 1 and n_trials > 1
        common_kw = dict(
            alpha=alpha,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            fixed_onsite=fixed_onsite_vals,
        )

        if self.verbose:
            import os as _os

            n_cpu = _os.cpu_count() or 1
            # Intra-trial threads: geometry-level ThreadPoolExecutor
            # kicks in when n_geom >= 3
            geom_threads = self.n_geom if self.n_geom >= 3 else 1
            effective_jobs = (
                min(n_jobs, n_trials)
                if n_jobs > 0
                else min(n_cpu, n_trials)
                if n_jobs == -1
                else n_trials
            )
            total_threads = effective_jobs * geom_threads

            print(f"\n{'=' * 65}")
            par_tag = f', n_jobs={n_jobs}' if use_parallel else ''
            print(
                f'Multi-geometry EDTB optimisation: {n_trials} trials, '
                f'{self.n_geom} geometries{par_tag}'
            )

            if use_parallel:
                print('\n  Parallelism diagnostics:')
                print(f'    CPU cores available          : {n_cpu}')
                print(f'    Joblib worker processes       : {effective_jobs}')
                print(f'    Geometry threads per process  : {geom_threads}')
                print(f'    Total concurrent threads      : {total_threads}')
                if total_threads > n_cpu:
                    import warnings

                    rec = max(1, n_cpu // geom_threads)
                    msg = (
                        f'Thread oversubscription detected: {total_threads} '
                        f'threads on {n_cpu} cores. '
                        f'Each trial spawns {geom_threads} geometry threads '
                        f'(ThreadPoolExecutor for {self.n_geom} geometries), '
                        f'and joblib adds {effective_jobs} worker processes on '
                        f'top. This causes cores to context-switch and thrash '
                        f'caches, often making the fit *slower* than sequential. '
                        f'Recommended: n_jobs={rec} (= {n_cpu} cores / '
                        f'{geom_threads} geometry threads), or n_jobs=1 for '
                        f'sequential trials with per-trial progress output.'
                    )
                    warnings.warn(msg, stacklevel=2)
                    print(f'    ⚠ Recommended n_jobs ≤ {rec}  (cores / geometry_threads)')
                else:
                    print('    ✓ Good: threads ≤ cores, no oversubscription')

        if use_parallel:
            import os

            from joblib import Parallel, delayed

            # Prevent OMP/MKL thread oversubscription when using
            # process-level parallelism via joblib.
            old_omp = os.environ.get('OMP_NUM_THREADS')
            old_mkl = os.environ.get('MKL_NUM_THREADS')
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            try:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self._run_single_trial)(p, **common_kw) for p in p_inits
                )
            finally:
                # Restore original thread settings
                if old_omp is None:
                    os.environ.pop('OMP_NUM_THREADS', None)
                else:
                    os.environ['OMP_NUM_THREADS'] = old_omp
                if old_mkl is None:
                    os.environ.pop('MKL_NUM_THREADS', None)
                else:
                    os.environ['MKL_NUM_THREADS'] = old_mkl
            all_results = [(r, p, res) for r, p, res in results]
        else:
            if self.verbose:
                print(
                    f"{'Trial':>5s}  {'Init RMSE (meV)':>15s}  "
                    f"{'Final RMSE (meV)':>16s}  {'nfev':>5s}"
                )
                print('-' * 50)
            all_results = []
            best_so_far = np.inf
            for trial, p_init in enumerate(p_inits):
                rmse, p_opt, res = self._run_single_trial(p_init, **common_kw)
                all_results.append((rmse, p_opt, res))
                tag = ' *' if rmse < best_so_far else ''
                if rmse < best_so_far:
                    best_so_far = rmse
                if self.verbose:
                    rmse_init = np.sqrt(np.mean((ref.eigenvalues(p_init) - ref.E_pao).ravel() ** 2))
                    print(
                        f'{trial + 1:5d}  {rmse_init * 1000:15.2f}  '
                        f'{rmse * 1000:16.2f}  {res.nfev:5d}{tag}'
                    )

        # ── Collect results ──
        all_results.sort(key=lambda x: x[0])
        best_rmse, best_p, best_res = all_results[0]

        # Per-geometry RMSE breakdown (un-weighted)
        per_geom_rmse = []
        offset = 0
        for ig, f in enumerate(self.fitters):
            nd = self.n_data_per_geom[ig]
            r_uw = best_res.fun[offset : offset + nd] / self.weights[ig]
            per_geom_rmse.append(float(np.sqrt(np.mean(r_uw**2))))
            offset += nd

        if self.verbose:
            if use_parallel:
                print(f'  Completed {n_trials} trials in parallel')
            print(f"{'=' * 65}")
            print(f'Combined RMSE = {best_rmse * 1000:.2f} meV')
            for ig, r in enumerate(per_geom_rmse):
                print(f'  Geom {ig}: RMSE = {r * 1000:.2f} meV (w={self.weights[ig]:.2f})')
            if alpha > 0:
                print(f'  (α = {alpha:.4g})')
            n_fitted = (
                self.n_params - self.n_onsite if fixed_onsite_vals is not None else self.n_params
            )
            print(
                f'  Parameters: {n_fitted} fitted'
                + (f', {self.n_onsite} on-site fixed' if fixed_onsite_vals is not None else '')
            )
            print(f"\n{'Parameter':<30s}  {'Value':>10s}")
            print('-' * 43)
            for i, name in enumerate(self.param_labels):
                tag = ' (fixed)' if fixed_onsite_vals is not None and i < self.n_onsite else ''
                print(f'{name:<30s}  {best_p[i]: .5f}{tag}')

        return {
            'p_opt': best_p,
            'rmse': best_rmse,
            'per_geom_rmse': per_geom_rmse,
            'max_err': float(np.max(np.abs(best_res.fun[: self.n_data_total]))),
            'all_results': all_results,
            'param_labels': list(self.param_labels),
        }

    # ── 4e. Build model dict ─────────────────────────────────

    def build_model_dict(self, p: np.ndarray, geom_idx: int | None = None) -> dict:
        """Convert fitted parameters to a PAOFLOW ``SK_EDTB`` model dict.

        Parameters
        ----------
        p : np.ndarray
            Full parameter vector (length ``n_params``).
        geom_idx : int or None
            Which geometry to use for lattice vectors and atom positions.
            If ``None`` (default), the first geometry that contains all
            harmonised species is chosen automatically; this avoids
            ``compute_pair_shell_distances`` failing on dummy-species
            pairs that have no atoms.

        Returns
        -------
        dict
            Model dict with ``label='SK_EDTB'``.
        """
        if geom_idx is None:
            # Pick the first geometry that has all species
            all_sp = set(self.fitters[0].unique_species)
            geom_idx = 0
            if hasattr(self, '_geom_real_species'):
                for ig, sp_set in enumerate(self._geom_real_species):
                    if sp_set >= all_sp:
                        geom_idx = ig
                        break
        return self.fitters[geom_idx].build_model_dict(p)

    # ── 4f. Convenience: eigenvalues for a specific geometry ─

    def eigenvalues(self, p: np.ndarray, geom_idx: int = 0) -> np.ndarray:
        """Compute eigenvalues on the fitting k-mesh for geometry *geom_idx*.

        Parameters
        ----------
        p : np.ndarray
            Full parameter vector.
        geom_idx : int
            Geometry index (default 0).

        Returns
        -------
        np.ndarray
            Shape ``(Nk, nawf)`` eigenvalues in eV.
        """
        E, _ = self.fitters[geom_idx]._eigenvalues_and_jacobian(p)
        return E


# ═══════════════════════════════════════════════════════════════
#  5. Distance-dependent EDTB fitter (Goodwin-style hoppings)
# ═══════════════════════════════════════════════════════════════
#
# Hopping function:
#   V_λ(r) = V0_λ · (r_0/r)^n_λ · exp(n_λ · [-(r/r_c)^n_c + (r_0/r_c)^n_c])
#
# Parameter vector layout:
#   [ε_onsite | V0_channels | n_channels | n_c | γ_screening | η_shift]
#


class MultiGeomEDTB_DD:
    """Multi-geometry distance-dependent EDTB fitter.

    Fits Goodwin-style hopping parameters shared across multiple
    geometries (e.g. bulk + slabs at different interlayer distances).

    Parameters
    ----------
    geometry_data : list of (arryp, attrp) tuples
        PAOFLOW arrays/attributes for each geometry.
    r_0 : float
        Reference NN distance (Bohr).  Fixed, not fitted.
    r_c : float
        Hopping cutoff (Bohr).  Fixed, used in Goodwin exponent and
        as maximum bond distance.
    r_cut : float
        Screening cutoff radius (Bohr).
    gamma_mode : str
        ``'global'``, ``'per_lpair'``, or ``'per_channel'``.
    fit_onsite_shift : bool
        Whether to fit η on-site coordination-shift parameters.
    nkfit : int or list of int
        k-grid density for fitting.  If a list, one per geometry.
    weights : list of float, optional
        Per-geometry weights (default: uniform).
    verbose : bool
    """

    _LPAIR_LABELS = {
        (0, 0): 'ss',
        (0, 1): 'sp',
        (0, 2): 'sd',
        (1, 1): 'pp',
        (1, 2): 'pd',
        (2, 2): 'dd',
    }

    def __init__(
        self,
        geometry_data,
        *,
        r_0,
        r_c,
        r_cut,
        gamma_mode='global',
        fit_onsite_shift=False,
        nkfit=6,
        weights=None,
        verbose=True,
    ):
        self.verbose = verbose
        self.n_geom = len(geometry_data)

        # ── Physical constants ───────────────────────────────
        arryp0, attrp0 = geometry_data[0]
        self.alat = float(attrp0['alat'])
        self.r_0_bohr = float(r_0)
        self.r_c_bohr = float(r_c)
        self.r_cut_bohr = float(r_cut)
        self.r_0_alat = self.r_0_bohr / self.alat
        self.r_c_alat = self.r_c_bohr / self.alat
        self.r_cut_alat = self.r_cut_bohr / self.alat
        self.gamma_mode = gamma_mode
        self.fit_onsite_shift = fit_onsite_shift

        # ── Shared atomic structure (from first geometry) ────
        self.atoms_list = list(arryp0['atoms'])
        self.unique_species = list(dict.fromkeys(self.atoms_list))
        self.nat = int(attrp0['natoms'])
        self.nawf = int(arryp0['HRs'].shape[0])
        self.shells_dict = arryp0['shells']
        self.config_dict = arryp0.get('configuration', None)
        self._setup_orbital_structure()

        # ── Parameter layout ─────────────────────────────────
        self._build_group_pair_info()
        self._build_onsite_info()
        self._build_dd_param_layout()

        if self.verbose:
            print(
                f'\nMultiGeomEDTB_DD: {self.n_geom} geometries, {self.nat} atoms, nawf={self.nawf}'
            )
            print(
                f'  r_0={self.r_0_bohr:.3f} Bohr, '
                f'r_c={self.r_c_bohr:.3f} Bohr, '
                f'r_cut={self.r_cut_bohr:.3f} Bohr'
            )
            print(f'  Parameters: {self.n_params} total')
            print(
                f'    {self.n_onsite} on-site, '
                f'{self.n_ch} V0, {self.n_ch} n, 1 n_c, '
                f'{self.n_gamma} γ ({gamma_mode}), '
                f'{self.n_eta} η'
            )

        # ── Per-geometry setup ───────────────────────────────
        if isinstance(nkfit, (list, tuple)) and len(nkfit) == self.n_geom:
            nkfit_list = list(nkfit)
        else:
            nkfit_list = [nkfit] * self.n_geom

        self.weights = list(weights) if weights is not None else [1.0] * self.n_geom
        self._geom = []
        for ig, (arryp, attrp) in enumerate(geometry_data):
            gd = self._setup_one_geometry(arryp, attrp, nkfit_list[ig], ig)
            self._geom.append(gd)

        self.n_data_per_geom = [g['Nk'] * g['nawf'] for g in self._geom]
        self.n_data_total = sum(self.n_data_per_geom)

        self._build_regularization_weights()

    # ── Orbital structure ─────────────────────────────────────

    def _setup_orbital_structure(self):
        sp0 = self.unique_species[0]
        self.group_l = list(self.shells_dict[sp0])
        self.n_groups = len(self.group_l)
        self.cfg_names = (
            list(self.config_dict[sp0])
            if self.config_dict
            else [f'g{i}(l={self.group_l[i]})' for i in range(self.n_groups)]
        )
        self.atom_orbitals = []
        self.atom_orbital_group = []
        self.atom_block_start = []
        self.norb_per_atom = []
        idx = 0
        for iat in range(self.nat):
            sp = self.atoms_list[iat]
            orbs, grps = [], []
            for ig, l_val in enumerate(self.shells_dict[sp]):
                for orb in SHELL_TO_ORBITALS[l_val]:
                    orbs.append(orb)
                    grps.append(ig)
            self.atom_orbitals.append(orbs)
            self.atom_orbital_group.append(grps)
            self.atom_block_start.append(idx)
            self.norb_per_atom.append(len(orbs))
            idx += len(orbs)

    # ── Group-pair hopping structure ──────────────────────────

    def _build_group_pair_info(self):
        self.hop_pair_list = []
        self.hop_pair_start = {}
        self.hop_pair_active = {}
        self.n_hop = 0
        self.hop_labels = []
        for ga in range(self.n_groups):
            for gb in range(ga, self.n_groups):
                la, lb = self.group_l[ga], self.group_l[gb]
                lpair = (min(la, lb), max(la, lb))
                active = LPAIR_ACTIVE_INDICES[lpair]
                labels = CHANNEL_LABELS[lpair]
                self.hop_pair_list.append((ga, gb))
                self.hop_pair_start[(ga, gb)] = self.n_hop
                self.hop_pair_active[(ga, gb)] = active
                tag = f'{self.cfg_names[ga]}-{self.cfg_names[gb]}'
                for lab in labels:
                    self.hop_labels.append(f'V({tag}){lab}')
                self.n_hop += len(active)
        self.n_ch = self.n_hop  # alias for clarity

    # ── On-site parameter structure ───────────────────────────

    def _build_onsite_info(self):
        self.species_onsite_groups = {}
        self.species_param_start = {}
        self.n_onsite = 0
        self.onsite_labels = []
        for sp in self.unique_species:
            self.species_param_start[sp] = self.n_onsite
            cfg = (
                list(self.config_dict[sp])
                if self.config_dict
                else [f'g{i}' for i in range(len(self.shells_dict[sp]))]
            )
            groups = SKFitter._get_onsite_groups(self.shells_dict[sp], cfg)
            self.species_onsite_groups[sp] = groups
            for pname, _ in groups:
                self.onsite_labels.append(pname)
            self.n_onsite += len(groups)

    # ── DD parameter layout ───────────────────────────────────

    def _build_dd_param_layout(self):
        """Build the parameter vector layout for distance-dependent mode.

        Layout: [onsite | V0 | n_exp | n_c | gamma | eta]
        """
        # Active channels for gamma mapping
        active_lp, active_ch = set(), set()
        for ga, gb in self.hop_pair_list:
            la, lb = self.group_l[ga], self.group_l[gb]
            active_lp.add((min(la, lb), max(la, lb)))
            for sk_idx in self.hop_pair_active[(ga, gb)]:
                active_ch.add(SK_PARAM_NAMES[sk_idx])
        self.active_lpairs = sorted(active_lp)
        self.active_channels = sorted(active_ch, key=lambda x: SK_PARAM_NAMES.index(x))

        # hop index → gamma index
        self._hop_to_gamma = np.zeros(self.n_ch, dtype=int)
        gm = self.gamma_mode
        if gm == 'global':
            self.n_gamma = 1
            self.gamma_labels = ['γ']
        elif gm == 'per_lpair':
            lp2i = {lp: i for i, lp in enumerate(self.active_lpairs)}
            self.n_gamma = len(self.active_lpairs)
            self.gamma_labels = [f'γ_{self._LPAIR_LABELS[lp]}' for lp in self.active_lpairs]
            for ga, gb in self.hop_pair_list:
                la, lb = self.group_l[ga], self.group_l[gb]
                gidx = lp2i[(min(la, lb), max(la, lb))]
                st = self.hop_pair_start[(ga, gb)]
                for lk in range(len(self.hop_pair_active[(ga, gb)])):
                    self._hop_to_gamma[st + lk] = gidx
        elif gm == 'per_channel':
            ch2i = {ch: i for i, ch in enumerate(self.active_channels)}
            self.n_gamma = len(self.active_channels)
            self.gamma_labels = [f'γ_{ch}' for ch in self.active_channels]
            for ga, gb in self.hop_pair_list:
                st = self.hop_pair_start[(ga, gb)]
                for lk, sk_idx in enumerate(self.hop_pair_active[(ga, gb)]):
                    self._hop_to_gamma[st + lk] = ch2i[SK_PARAM_NAMES[sk_idx]]
        else:
            raise ValueError(f'Unknown gamma_mode: {gm!r}')

        # η (on-site shift)
        if self.fit_onsite_shift:
            present = set()
            for sp in self.unique_species:
                for l_val in self.shells_dict[sp]:
                    present.add({0: 's', 1: 'p', 2: 'd'}[l_val])
            self.eta_orb_types = sorted(present, key='spd'.index)
            self.n_eta = len(self.eta_orb_types)
            self.eta_labels = [f'η_{t}' for t in self.eta_orb_types]
        else:
            self.eta_orb_types = []
            self.n_eta = 0
            self.eta_labels = []

        # Index bookkeeping
        self.V0_start = self.n_onsite
        self.n_start = self.V0_start + self.n_ch
        self.nc_idx = self.n_start + self.n_ch
        self.n_gamma_start = self.nc_idx + 1
        self.n_eta_start = self.n_gamma_start + self.n_gamma
        self.n_params = self.n_eta_start + self.n_eta

        # Labels
        self.param_labels = list(self.onsite_labels)
        self.param_labels += [f'V0_{l}' for l in self.hop_labels]
        self.param_labels += [f'n_{l}' for l in self.hop_labels]
        self.param_labels.append('n_c')
        self.param_labels += self.gamma_labels
        self.param_labels += self.eta_labels

    # ── Per-geometry setup ────────────────────────────────────

    def _setup_one_geometry(self, arryp, attrp, nkfit, ig):
        """Pre-compute bonds, design tensors, screening, phases, ref eigenvalues."""
        a_vecs = arryp['a_vectors']
        tau_bohr = arryp['tau']
        alat = float(attrp['alat'])
        tau_alat = tau_bohr / alat
        nat = int(attrp['natoms'])
        nawf = int(arryp['HRs'].shape[0])
        b_vecs = arryp['b_vectors']

        # ── Reference eigenvalues ──
        HRs = arryp['HRs']
        nk_dft = (HRs.shape[2], HRs.shape[3], HRs.shape[4])
        R_list, HR_list = [], []
        for i1 in range(nk_dft[0]):
            for i2 in range(nk_dft[1]):
                for i3 in range(nk_dft[2]):
                    r1 = i1 if 2 * i1 <= nk_dft[0] else i1 - nk_dft[0]
                    r2 = i2 if 2 * i2 <= nk_dft[1] else i2 - nk_dft[1]
                    r3 = i3 if 2 * i3 <= nk_dft[2] else i3 - nk_dft[2]
                    R_list.append(r1 * a_vecs[0] + r2 * a_vecs[1] + r3 * a_vecs[2])
                    HR_list.append(HRs[:, :, i1, i2, i3, 0])
        R_arr = np.array(R_list)
        HR_arr = np.array(HR_list)

        if isinstance(nkfit, (tuple, list)):
            nk1, nk2, nk3 = int(nkfit[0]), int(nkfit[1]), int(nkfit[2])
        else:
            nk1 = nk2 = nk3 = int(nkfit)
        kpts = []
        for ik1 in range(nk1):
            for ik2 in range(nk2):
                for ik3 in range(nk3):
                    kpts.append(
                        np.array([ik1 / max(nk1, 1), ik2 / max(nk2, 1), ik3 / max(nk3, 1)]) @ b_vecs
                    )
        kpts = np.array(kpts)
        Nk = len(kpts)
        phases_dft = np.exp(2j * np.pi * (kpts @ R_arr.T))
        E_pao = np.zeros((Nk, nawf))
        for ik in range(Nk):
            Hk = np.einsum('r,rij->ij', phases_dft[ik], HR_arr)
            E_pao[ik] = np.sort(np.linalg.eigvalsh(Hk).real)

        # ── Per-geometry atom orbital structure ──
        atoms_list_g = list(arryp['atoms'])
        atom_orbitals_g = []
        atom_orbital_group_g = []
        atom_block_start_g = []
        idx = 0
        for iat in range(nat):
            sp = atoms_list_g[iat]
            orbs, grps = [], []
            for ig_l, l_val in enumerate(self.shells_dict[sp]):
                for orb in SHELL_TO_ORBITALS[l_val]:
                    orbs.append(orb)
                    grps.append(ig_l)
            atom_orbitals_g.append(orbs)
            atom_orbital_group_g.append(grps)
            atom_block_start_g.append(idx)
            idx += len(orbs)

        # ── Enumerate all bonds within r_c ──
        r_c_alat = self.r_c_bohr / alat
        r_cut_alat = self.r_cut_bohr / alat
        # r_0_alat = self.r_0_bohr / alat
        min_a = min(np.linalg.norm(v) for v in a_vecs)
        cell_range = int(np.ceil(r_c_alat / min_a)) + 1

        bonds = []
        for i1 in range(-cell_range, cell_range + 1):
            for i2 in range(-cell_range, cell_range + 1):
                for i3 in range(-cell_range, cell_range + 1):
                    R = i1 * a_vecs[0] + i2 * a_vecs[1] + i3 * a_vecs[2]
                    for iat in range(nat):
                        for jat in range(nat):
                            d_vec = R + tau_alat[jat] - tau_alat[iat]
                            d_norm = np.linalg.norm(d_vec)
                            if d_norm < 1e-8 or d_norm > r_c_alat:
                                continue
                            bonds.append((R, iat, jat, d_vec, d_norm))
        n_bonds = len(bonds)

        # ── Design tensors: M[b, p, m, n] ──
        M = np.zeros((n_bonds, self.n_ch, nawf, nawf))
        R_bond = np.zeros((n_bonds, 3))
        d_bonds = np.zeros(n_bonds)
        for ib, (R_cart, iat, jat, d_vec, d_norm) in enumerate(bonds):
            lx, ly, lz = d_vec / d_norm
            R_bond[ib] = R_cart
            d_bonds[ib] = d_norm
            bi = atom_block_start_g[iat]
            bj = atom_block_start_g[jat]
            oi = atom_orbitals_g[iat]
            oj = atom_orbitals_g[jat]
            gi = atom_orbital_group_g[iat]
            gj = atom_orbital_group_g[jat]
            for ai, orb_a in enumerate(oi):
                for aj, orb_b in enumerate(oj):
                    ga_loc, gb_loc = gi[ai], gj[aj]
                    canonical = (min(ga_loc, gb_loc), max(ga_loc, gb_loc))
                    start = self.hop_pair_start[canonical]
                    active = self.hop_pair_active[canonical]
                    design = sk_design_row(orb_a, orb_b, lx, ly, lz)
                    for lk, sk_k in enumerate(active):
                        M[ib, start + lk, bi + ai, bj + aj] = design[sk_k]

        # ── Screening sums ──
        sc_range = int(np.ceil(r_cut_alat / min_a)) + 1
        sc_pos = []
        for i1 in range(-sc_range, sc_range + 1):
            for i2 in range(-sc_range, sc_range + 1):
                for i3 in range(-sc_range, sc_range + 1):
                    R = i1 * a_vecs[0] + i2 * a_vecs[1] + i3 * a_vecs[2]
                    for iat in range(nat):
                        sc_pos.append(R + tau_alat[iat])
        sc_pos = np.array(sc_pos)
        r_taper = 0.8 * r_cut_alat

        def _fc_vec(d):
            fc = np.where(
                d <= r_taper,
                1.0,
                np.where(
                    d >= r_cut_alat,
                    0.0,
                    0.5 * (1.0 + np.cos(np.pi * (d - r_taper) / (r_cut_alat - r_taper))),
                ),
            )
            fc[d < 1e-10] = 0.0
            return fc

        fc_home = np.zeros((nat, len(sc_pos)))
        for ia in range(nat):
            fc_home[ia] = _fc_vec(np.linalg.norm(sc_pos - tau_alat[ia], axis=1))
        coord_i = np.sum(fc_home, axis=1)

        S_bonds = np.empty(n_bonds)
        for ib, (R_cart, iat, jat, *_) in enumerate(bonds):
            pos_j = R_cart + tau_alat[jat]
            d_jk = np.linalg.norm(sc_pos - pos_j, axis=1)
            S_bonds[ib] = np.dot(fc_home[iat], _fc_vec(d_jk))

        # ── On-site map ──
        onsite_map = np.zeros((self.n_onsite, nawf, nawf))
        for iat in range(nat):
            sp = atoms_list_g[iat]
            bi = atom_block_start_g[iat]
            pstart = self.species_param_start[sp]
            for ig_param, (_, local_indices) in enumerate(self.species_onsite_groups[sp]):
                for li in local_indices:
                    onsite_map[pstart + ig_param, bi + li, bi + li] = 1.0
        onsite_diag = np.array([np.diag(onsite_map[p]) for p in range(self.n_onsite)])

        # ── η diag (on-site shift) ──
        eta_diag = None
        if self.n_eta > 0:
            _otype = {
                's': 's',
                'px': 'p',
                'py': 'p',
                'pz': 'p',
                'dxy': 'd',
                'dyz': 'd',
                'dzx': 'd',
                'dx2-y2': 'd',
                'dz2': 'd',
            }
            eta_diag = np.zeros((self.n_eta, nawf))
            for iat in range(nat):
                bi = atom_block_start_g[iat]
                for io, orb in enumerate(atom_orbitals_g[iat]):
                    q = self.eta_orb_types.index(_otype[orb])
                    eta_diag[q, bi + io] = coord_i[iat]

        # ── Phases ──
        phases = np.exp(2j * np.pi * (kpts @ R_bond.T))  # (Nk, n_bonds)

        # ── Bond groups for block-sparse Jacobian ──
        from collections import defaultdict

        _bg = defaultdict(list)
        for ib, (_, iat, jat, _, _) in enumerate(bonds):
            _bg[(iat, jat)].append(ib)
        bond_groups = []
        for (iat, jat), blist in _bg.items():
            idx = np.array(blist)
            bi = atom_block_start_g[iat]
            bj = atom_block_start_g[jat]
            no_i = len(atom_orbitals_g[iat])
            no_j = len(atom_orbitals_g[jat])
            M_sub = M[idx][:, :, bi : bi + no_i, bj : bj + no_j].copy()
            bond_groups.append((idx, bi, bj, no_i, no_j, M_sub))

        if self.verbose:
            print(
                f'  Geom {ig}: {n_bonds} bonds, '
                f'{Nk} k-pts (grid {nk1}×{nk2}×{nk3}), '
                f'E ∈ [{E_pao.min():.3f}, {E_pao.max():.3f}] eV'
            )

        return {
            'a_vecs': a_vecs,
            'tau_alat': tau_alat,
            'alat': alat,
            'nawf': nawf,
            'nat': nat,
            'Nk': Nk,
            'kpts': kpts,
            'E_pao': E_pao,
            'n_bonds': n_bonds,
            'M': M,  # (n_bonds, n_ch, nawf, nawf)
            'R_bond': R_bond,
            'd_bonds': d_bonds,
            'S_bonds': S_bonds,
            'coord_i': coord_i,
            'phases': phases,  # (Nk, n_bonds)
            'bond_groups': bond_groups,
            'onsite_map': onsite_map,
            'onsite_diag': onsite_diag,
            'eta_diag': eta_diag,
            'b_vecs': arryp['b_vectors'],
            'atoms_list': atoms_list_g,
            'atom_orbitals': atom_orbitals_g,
            'atom_block_start': atom_block_start_g,
        }

    # ── Forward model ─────────────────────────────────────────

    def _eigenvalues_and_jacobian(self, p, gd):
        """Compute eigenvalues and Hellmann-Feynman Jacobian for one geometry.

        Uses block-sparse operations: each bond only touches a small orbital
        sub-block (e.g. 9×9 for C-C), giving up to (nawf/n_orb)² speed-up
        over dense (nawf×nawf) operations.
        """
        Nk = gd['Nk']
        nawf = gd['nawf']
        d_bonds = gd['d_bonds']
        S_bonds = gd['S_bonds']
        phases = gd['phases']
        # n_bonds = gd['n_bonds']
        bond_groups = gd['bond_groups']

        n_ch = self.n_ch
        n_onsite = self.n_onsite
        r_0 = self.r_0_bohr / gd['alat']
        r_c = self.r_c_bohr / gd['alat']

        # Extract parameters
        V0 = p[self.V0_start : self.V0_start + n_ch]
        n_exp = p[self.n_start : self.n_start + n_ch]
        n_c = p[self.nc_idx]
        gamma = p[self.n_gamma_start : self.n_gamma_start + self.n_gamma]

        # ── Goodwin function evaluation (vectorized) ──
        ratio = r_0 / d_bonds  # (n_bonds,)
        dr_rc = d_bonds / r_c
        r0_rc = r_0 / r_c
        g_bonds = -(dr_rc**n_c) + (r0_rc**n_c)  # (n_bonds,)

        # Screening per channel
        gamma_per_ch = gamma[self._hop_to_gamma]  # (n_ch,)
        _scr_arg = -gamma_per_ch[None, :] * S_bonds[:, None]
        np.clip(_scr_arg, -30.0, 30.0, out=_scr_arg)
        scale = np.exp(_scr_arg)  # (n_bonds, n_ch)

        # base[b, p] -- vectorised over channels (no Python loop)
        ln_ratio = np.log(ratio)  # (n_bonds,)
        arg = n_exp[None, :] * (ln_ratio[:, None] + g_bonds[:, None])
        np.clip(arg, -30.0, 30.0, out=arg)
        base = np.exp(arg)  # (n_bonds, n_ch)
        h = V0[None, :] * base * scale  # (n_bonds, n_ch)

        # ── H(k) via block-sparse groups ──
        H_onsite = np.einsum('p,pij->ij', p[:n_onsite], gd['onsite_map'])
        if self.n_eta > 0:
            eta = p[self.n_eta_start : self.n_eta_start + self.n_eta]
            shift = np.einsum('q,qi->i', eta, gd['eta_diag'])
            H_onsite[np.arange(nawf), np.arange(nawf)] += shift
        Hk = np.broadcast_to(H_onsite, (Nk, nawf, nawf)).astype(complex).copy()
        for idx, bi, bj, no_i, no_j, M_sub in bond_groups:
            h_grp = h[idx]  # (nb, n_ch)
            wM = np.einsum('bp,bpij->bij', h_grp, M_sub)  # (nb, no_i, no_j)
            Hk[:, bi : bi + no_i, bj : bj + no_j] += np.einsum('kb,bij->kij', phases[:, idx], wM)

        # ── Eigendecomposition ──
        evals, evecs = np.linalg.eigh(Hk)
        E_sk = evals.real

        # ── Jacobian via Hellmann-Feynman ──
        dE = np.zeros((Nk, nawf, self.n_params))
        psi2 = np.abs(evecs) ** 2

        # ∂E/∂ε
        dE[:, :, :n_onsite] = np.einsum('kin,pi->knp', psi2, gd['onsite_diag'])

        # Block-sparse dHk builder (only populates nonzero sub-blocks)
        def _dHk_ch(w):
            dHk = np.zeros((Nk, n_ch, nawf, nawf), dtype=complex)
            for idx, bi, bj, no_i, no_j, M_sub in bond_groups:
                w_grp = w[idx]  # (nb, n_ch)
                wM = M_sub * w_grp[:, :, None, None]  # (nb, n_ch, no_i, no_j)
                dHk[:, :, bi : bi + no_i, bj : bj + no_j] += np.einsum(
                    'kb,bpij->kpij', phases[:, idx], wM
                )
            return dHk

        # ∂E/∂V0_p
        dHk_V0 = _dHk_ch(base * scale)
        evecs_bc = evecs[:, np.newaxis, :, :]  # (Nk, 1, nawf, nawf)
        tmp = np.matmul(dHk_V0, evecs_bc)
        dE[:, :, self.V0_start : self.V0_start + n_ch] = np.real(
            np.einsum('kin,kpin->knp', evecs.conj(), tmp)
        )

        # ∂E/∂n_p
        dn_factor = ln_ratio + g_bonds  # (n_bonds,)
        w_n = h * dn_factor[:, None]  # (n_bonds, n_ch)
        dHk_n = _dHk_ch(w_n)
        tmp = np.matmul(dHk_n, evecs_bc)
        dE[:, :, self.n_start : self.n_start + n_ch] = np.real(
            np.einsum('kin,kpin->knp', evecs.conj(), tmp)
        )

        # ∂E/∂n_c  (block-sparse, channel-contracted)
        ln_dr_rc = np.log(np.maximum(dr_rc, 1e-30))
        ln_r0_rc = np.log(max(r0_rc, 1e-30))
        dg_dnc = -(dr_rc**n_c) * ln_dr_rc + (r0_rc**n_c) * ln_r0_rc
        w_nc = h * (n_exp[None, :] * dg_dnc[:, None])  # (n_bonds, n_ch)
        dHk_nc = np.zeros((Nk, nawf, nawf), dtype=complex)
        for idx, bi, bj, no_i, no_j, M_sub in bond_groups:
            F_sub = np.einsum('bp,bpij->bij', w_nc[idx], M_sub)
            dHk_nc[:, bi : bi + no_i, bj : bj + no_j] += np.einsum(
                'kb,bij->kij', phases[:, idx], F_sub
            )
        tmp_nc = np.matmul(dHk_nc, evecs)
        dE[:, :, self.nc_idx] = np.real(np.einsum('kin,kin->kn', evecs.conj(), tmp_nc))

        # ∂E/∂γ_q  (block-sparse, channel-contracted)
        for q in range(self.n_gamma):
            mask = self._hop_to_gamma == q
            if not np.any(mask):
                continue
            w_g = np.zeros_like(h)
            w_g[:, mask] = -S_bonds[:, None] * h[:, mask]
            dHk_g = np.zeros((Nk, nawf, nawf), dtype=complex)
            for idx, bi, bj, no_i, no_j, M_sub in bond_groups:
                F_sub = np.einsum('bp,bpij->bij', w_g[idx], M_sub)
                dHk_g[:, bi : bi + no_i, bj : bj + no_j] += np.einsum(
                    'kb,bij->kij', phases[:, idx], F_sub
                )
            tmp_g = np.matmul(dHk_g, evecs)
            dE[:, :, self.n_gamma_start + q] = np.real(
                np.einsum('kin,kin->kn', evecs.conj(), tmp_g)
            )

        # ∂E/∂η_q
        if self.n_eta > 0:
            eta_diag = gd['eta_diag']
            for q in range(self.n_eta):
                dE[:, :, self.n_eta_start + q] = np.einsum('kin,i->kn', psi2, eta_diag[q])

        return E_sk, dE

    # ── Multi-geometry aggregation ────────────────────────────

    def _eval_single_geometry(self, ig, p):
        gd = self._geom[ig]
        E_sk, dE = self._eigenvalues_and_jacobian(p, gd)
        res = (E_sk - gd['E_pao']).ravel()
        J = dE.reshape(-1, self.n_params)
        w = self.weights[ig]
        return w * res, w * J

    def _eigenvalues_and_jacobian_all(self, p):
        if self.n_geom >= 3:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.n_geom) as pool:
                futures = [
                    pool.submit(self._eval_single_geometry, ig, p) for ig in range(self.n_geom)
                ]
                results = [f.result() for f in futures]
        else:
            results = [self._eval_single_geometry(ig, p) for ig in range(self.n_geom)]
        return np.concatenate([r for r, _ in results]), np.vstack([j for _, j in results])

    # ── Regularization ────────────────────────────────────────

    def _build_regularization_weights(self):
        w = np.zeros(self.n_params)
        w[self.V0_start : self.V0_start + self.n_ch] = 1.0
        w[self.n_start : self.n_start + self.n_ch] = 0.1
        w[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = 1.0
        if self.n_eta > 0:
            w[self.n_eta_start : self.n_eta_start + self.n_eta] = 1.0
        self._reg_weights = w

    # ── Single trial helper ─────────────────────────────────

    def _run_single_trial(self, p_init, alpha, max_nfev, ftol, xtol, gtol, bounds):
        """Run one least-squares trial.  Returns ``(rmse, p_opt, OptimizeResult)``."""
        use_reg = alpha > 0.0
        reg_w = self._reg_weights
        n_data = self.n_data_total
        use_bounds = bounds is not None
        last_jac = [None]

        def fun(p):
            res_data, J_data = self._eigenvalues_and_jacobian_all(p)
            if use_reg:
                last_jac[0] = np.vstack([J_data, np.diag(alpha * reg_w)])
                return np.concatenate([res_data, alpha * reg_w * p])
            last_jac[0] = J_data
            return res_data

        def jac(p):
            return last_jac[0]

        kw = dict(
            jac=jac,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        if use_bounds:
            kw['bounds'] = bounds
            kw['method'] = 'trf'
        else:
            kw['method'] = 'lm'

        res = least_squares(fun, p_init, **kw)
        rmse = np.sqrt(np.mean(res.fun[:n_data] ** 2))
        return rmse, res.x.copy(), res

    # ── Fitting ───────────────────────────────────────────────

    def fit(
        self,
        *,
        p0=None,
        n_trials=10,
        seed=123,
        max_nfev=2000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        alpha=0.0,
        bounds=None,
        n_jobs=1,
    ):
        """Multi-start least-squares fit.

        Parameters
        ----------
        p0 : np.ndarray, optional
            Initial parameter vector.
        n_trials : int
            Number of random restarts.
        seed : int or None
        max_nfev : int
        ftol, xtol, gtol : float
        alpha : float
            Tikhonov regularization.
        bounds : tuple of (lower, upper), optional
            Bounds for the parameter vector. Each is array of length n_params
            or ``-np.inf`` / ``np.inf``.
        n_jobs : int
            Parallel workers for multi-start trials (``-1`` = all cores).
            Requires ``joblib`` when ``n_jobs != 1``.

        Returns
        -------
        dict
        """
        rng = np.random.RandomState(seed)
        n_data = self.n_data_total
        # reg_w = self._reg_weights

        # ── Default initial guess ──
        if p0 is None:
            p0 = np.zeros(self.n_params)
            # Onsite: from HR0, geometry 0
            gd = self._geom[0]
            for iat in range(self.nat):
                sp = self.atoms_list[iat]
                bi = self.atom_block_start[iat]
                pstart = self.species_param_start[sp]
                for ig_param, (_, local_indices) in enumerate(self.species_onsite_groups[sp]):
                    li0 = local_indices[0]
                    p0[pstart + ig_param] = (
                        float(gd['E_pao'][0, bi + li0]) if gd['E_pao'].shape[1] > bi + li0 else 0.0
                    )
            # V0: small random, n_exp: ~2, n_c: ~6
            p0[self.V0_start : self.V0_start + self.n_ch] = rng.uniform(-1.0, 1.0, self.n_ch)
            p0[self.n_start : self.n_start + self.n_ch] = 2.0
            p0[self.nc_idx] = 6.5
            p0[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = 0.005
        else:
            p0 = np.asarray(p0, dtype=float)

        # ── Generate trials ──
        p_inits = []
        for trial in range(n_trials):
            pi = p0.copy()
            if trial > 0:
                pi[self.V0_start : self.V0_start + self.n_ch] *= 1.0 + 0.3 * rng.randn(self.n_ch)
                pi[self.n_start : self.n_start + self.n_ch] += 0.5 * rng.randn(self.n_ch)
                pi[self.nc_idx] += 1.0 * rng.randn()
                pi[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = rng.uniform(
                    0.0, 0.02, self.n_gamma
                )
            p_inits.append(pi)

        # ── Run trials ──
        use_parallel = n_jobs != 1 and n_trials > 1
        common_kw = dict(
            alpha=alpha,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            bounds=bounds,
        )

        if self.verbose:
            import os as _os

            n_cpu = _os.cpu_count() or 1
            geom_threads = self.n_geom if self.n_geom >= 3 else 1
            effective_jobs = (
                min(n_jobs, n_trials)
                if n_jobs > 0
                else min(n_cpu, n_trials)
                if n_jobs == -1
                else n_trials
            )
            total_threads = effective_jobs * geom_threads

            print(f"\n{'=' * 65}")
            par_tag = f', n_jobs={n_jobs}' if use_parallel else ''
            print(f'DD EDTB optimisation: {n_trials} trials, {self.n_geom} geometries{par_tag}')

            if use_parallel:
                print('\n  Parallelism diagnostics:')
                print(f'    CPU cores available          : {n_cpu}')
                print(f'    Joblib worker processes       : {effective_jobs}')
                print(f'    Geometry threads per process  : {geom_threads}')
                print(f'    Total concurrent threads      : {total_threads}')
                if total_threads > n_cpu:
                    import warnings

                    rec = max(1, n_cpu // geom_threads)
                    msg = (
                        f'Thread oversubscription detected: {total_threads} '
                        f'threads on {n_cpu} cores. '
                        f'Each trial spawns {geom_threads} geometry threads '
                        f'(ThreadPoolExecutor for {self.n_geom} geometries), '
                        f'and joblib adds {effective_jobs} worker processes on '
                        f'top. This causes cores to context-switch and thrash '
                        f'caches, often making the fit *slower* than sequential. '
                        f'Recommended: n_jobs={rec} (= {n_cpu} cores / '
                        f'{geom_threads} geometry threads), or n_jobs=1 for '
                        f'sequential trials with per-trial progress output.'
                    )
                    warnings.warn(msg, stacklevel=2)
                    print(f'    \u26a0 Recommended n_jobs \u2264 {rec}  (cores / geometry_threads)')
                else:
                    print('    \u2713 Good: threads \u2264 cores, no oversubscription')

        if use_parallel:
            import os

            from joblib import Parallel, delayed

            old_omp = os.environ.get('OMP_NUM_THREADS')
            old_mkl = os.environ.get('MKL_NUM_THREADS')
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            try:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self._run_single_trial)(p, **common_kw) for p in p_inits
                )
            finally:
                if old_omp is None:
                    os.environ.pop('OMP_NUM_THREADS', None)
                else:
                    os.environ['OMP_NUM_THREADS'] = old_omp
                if old_mkl is None:
                    os.environ.pop('MKL_NUM_THREADS', None)
                else:
                    os.environ['MKL_NUM_THREADS'] = old_mkl
            all_results = [(r, p, res) for r, p, res in results]
        else:
            if self.verbose:
                print(
                    f"{'Trial':>5s}  {'Init RMSE (meV)':>15s}  "
                    f"{'Final RMSE (meV)':>16s}  {'nfev':>5s}"
                )
                print('-' * 50)
            all_results = []
            best_so_far = np.inf
            for trial, p_init in enumerate(p_inits):
                rmse, p_opt, res = self._run_single_trial(p_init, **common_kw)
                all_results.append((rmse, p_opt, res))
                tag = ' *' if rmse < best_so_far else ''
                if rmse < best_so_far:
                    best_so_far = rmse
                if self.verbose:
                    E0, _ = self._eigenvalues_and_jacobian(p_init, self._geom[0])
                    rmse_init = np.sqrt(np.mean((E0 - self._geom[0]['E_pao']).ravel() ** 2))
                    print(
                        f'{trial + 1:5d}  {rmse_init * 1000:15.2f}  '
                        f'{rmse * 1000:16.2f}  {res.nfev:5d}{tag}'
                    )

        all_results.sort(key=lambda x: x[0])
        best_rmse, best_p, best_res = all_results[0]

        # Per-geometry RMSE
        per_geom_rmse = []
        offset = 0
        for ig in range(self.n_geom):
            nd = self.n_data_per_geom[ig]
            r_uw = best_res.fun[offset : offset + nd] / self.weights[ig]
            per_geom_rmse.append(float(np.sqrt(np.mean(r_uw**2))))
            offset += nd

        if self.verbose:
            print(f"{'=' * 65}")
            print(f'Combined RMSE = {best_rmse * 1000:.2f} meV')
            for ig, r in enumerate(per_geom_rmse):
                print(f'  Geom {ig}: RMSE = {r * 1000:.2f} meV (w={self.weights[ig]:.2f})')
            print(f"\n{'Parameter':<30s}  {'Value':>10s}")
            print('-' * 43)
            for i, name in enumerate(self.param_labels):
                print(f'{name:<30s}  {best_p[i]: .5f}')

        return {
            'p_opt': best_p,
            'rmse': best_rmse,
            'per_geom_rmse': per_geom_rmse,
            'max_err': float(np.max(np.abs(best_res.fun[:n_data]))),
            'all_results': all_results,
            'param_labels': list(self.param_labels),
        }

    # ── Build model dict ──────────────────────────────────────

    def build_model_dict(self, p, geom_idx=0):
        """Convert parameter vector to a PAOFLOW model dict.

        Parameters
        ----------
        p : np.ndarray
            Full parameter vector.
        geom_idx : int
            Which geometry to use for lattice/positions.

        Returns
        -------
        dict
        """
        p = np.asarray(p)
        gd = self._geom[geom_idx]

        # Atoms dict
        atoms_list_g = gd['atoms_list']
        atom_orbitals_g = gd['atom_orbitals']
        atoms_dict = {}
        for iat in range(gd['nat']):
            sp = atoms_list_g[iat]
            pstart = self.species_param_start[sp]
            atom_d = {
                'name': sp,
                'tau': gd['tau_alat'][iat].tolist(),
                'orbitals': list(atom_orbitals_g[iat]),
            }
            for ig_param, (pname, local_indices) in enumerate(self.species_onsite_groups[sp]):
                e_val = float(p[pstart + ig_param])
                for li in local_indices:
                    orb = atom_orbitals_g[iat][li]
                    atom_d[orb] = e_val
            atoms_dict[str(iat)] = atom_d

        # Hoppings (DD format)
        V0 = p[self.V0_start : self.V0_start + self.n_ch]
        n_exp = p[self.n_start : self.n_start + self.n_ch]
        n_c = float(p[self.nc_idx])

        channels = {}
        idx = 0
        for ga, gb in self.hop_pair_list:
            active = self.hop_pair_active[(ga, gb)]
            for lk, sk_k in enumerate(active):
                ch_name = SK_PARAM_NAMES[sk_k]
                channels[ch_name] = {
                    'V0': float(V0[idx]),
                    'n': float(n_exp[idx]),
                }
                idx += 1

        from .edtb_params import species_pair_key

        sorted_species = sorted(set(self.unique_species))
        hoppings = {}
        dd_entry = {
            'type': 'distance_dependent',
            'r_0': self.r_0_bohr,
            'r_c': self.r_c_bohr,
            'n_c': n_c,
            'channels': channels,
        }
        for i, sp1 in enumerate(sorted_species):
            for sp2 in sorted_species[i:]:
                key = species_pair_key(sp1, sp2)
                hoppings[key] = {
                    'type': dd_entry['type'],
                    'r_0': dd_entry['r_0'],
                    'r_c': dd_entry['r_c'],
                    'n_c': dd_entry['n_c'],
                    'channels': {k: dict(v) for k, v in dd_entry['channels'].items()},
                }

        # Screening
        gamma = p[self.n_gamma_start : self.n_gamma_start + self.n_gamma]
        gm = self.gamma_mode
        if gm == 'global':
            gamma_val = float(gamma[0])
        elif gm == 'per_lpair':
            gamma_val = {
                self._LPAIR_LABELS[lp]: float(gamma[i]) for i, lp in enumerate(self.active_lpairs)
            }
        elif gm == 'per_channel':
            gamma_val = {ch: float(gamma[i]) for i, ch in enumerate(self.active_channels)}
        else:
            gamma_val = float(gamma[0])

        gamma_dict = {}
        for i, sp1 in enumerate(sorted_species):
            for sp2 in sorted_species[i:]:
                key = species_pair_key(sp1, sp2)
                if isinstance(gamma_val, dict):
                    gamma_dict[key] = dict(gamma_val)
                else:
                    gamma_dict[key] = gamma_val

        screening = {'r_cut': self.r_cut_bohr, 'gamma': gamma_dict}
        if self.n_eta > 0:
            eta = p[self.n_eta_start : self.n_eta_start + self.n_eta]
            screening['onsite_shift'] = {
                self.eta_orb_types[i]: float(eta[i]) for i in range(self.n_eta)
            }

        return {
            'label': 'SK_EDTB',
            'alat': float(gd['alat']),
            'model': {
                'a_vectors': gd['a_vecs'].tolist(),
                'atoms': atoms_dict,
                'hoppings': hoppings,
                'screening': screening,
            },
        }

    # ── Convenience ───────────────────────────────────────────

    def eigenvalues(self, p, geom_idx=0):
        """Eigenvalues on the fitting k-mesh for a given geometry."""
        E, _ = self._eigenvalues_and_jacobian(p, self._geom[geom_idx])
        return E

    def p0_from_discrete_params(self, params, *, n_c_init=6.5, n_default=2.0):
        """Build an initial parameter vector from discrete-shell EDTB params.

        Parameters
        ----------
        params : dict
            Loaded from a discrete-shell ``*_EDTB_params.json`` file.
        n_c_init : float
            Initial value for the Goodwin cutoff exponent n_c.
        n_default : float
            Fallback power-law exponent when shell ratio estimation fails.

        Returns
        -------
        np.ndarray
            Parameter vector of length ``n_params``.
        """
        p0 = np.zeros(self.n_params)

        # ── On-site energies ──
        for sp in self.unique_species:
            pstart = self.species_param_start[sp]
            onsite_vals = params['onsite'][sp]
            for ig_param, (pname, _) in enumerate(self.species_onsite_groups[sp]):
                # pname is like "ε(2S)" — extract the orbital key
                # Map to the JSON keys: s, p, t2g, eg
                key = pname.split('(')[-1].rstrip(')')
                # config names like 2S → look up in onsite dict
                _key_map = {
                    '2S': 's',
                    '3S': 's',
                    's': 's',
                    '2P': 'p',
                    '3P': 'p',
                    'p': 'p',
                    '3D_t2g': 't2g',
                    '3D_eg': 'eg',
                    '4D_t2g': 't2g',
                    '4D_eg': 'eg',
                    '5D_t2g': 't2g',
                    '5D_eg': 'eg',
                    't2g': 't2g',
                    'eg': 'eg',
                    'd': 't2g',
                }
                json_key = _key_map.get(key, key.lower())
                if json_key in onsite_vals:
                    p0[pstart + ig_param] = onsite_vals[json_key]

        # ── Hopping channels: V0 and n ──
        sp0 = self.unique_species[0]
        pair_key = f'{sp0}-{sp0}'
        shells = params['hoppings'][pair_key]
        r_ref = [s['r_ref'] for s in shells]
        r0 = self.r_0_bohr

        idx = 0
        for ga, gb in self.hop_pair_list:
            active = self.hop_pair_active[(ga, gb)]
            for sk_k in active:
                ch_name = SK_PARAM_NAMES[sk_k]
                # V0 from nearest shell to r_0
                v_1nn = shells[0]['params'].get(ch_name, 0.0)
                p0[self.V0_start + idx] = v_1nn

                # Estimate n from 1NN/2NN ratio
                n_est = n_default
                if len(shells) >= 2:
                    v_2nn = shells[1]['params'].get(ch_name, 0.0)
                    if abs(v_1nn) > 1e-6 and abs(v_2nn) > 1e-6:
                        # Goodwin at r_0 → V0, at r1 → V0*(r0/r1)^n * exp(n*g)
                        # With n_c_init and r_c, solve for n:
                        r1 = r_ref[1]
                        rc = self.r_c_bohr
                        g = -((r1 / rc) ** n_c_init) + (r0 / rc) ** n_c_init
                        ln_ratio = np.log(r0 / r1)
                        denom = ln_ratio + g
                        if abs(denom) > 1e-10:
                            n_est = np.log(abs(v_2nn / v_1nn)) / denom
                            n_est = np.clip(n_est, 0.5, 15.0)
                        else:
                            n_est = n_default
                p0[self.n_start + idx] = n_est
                idx += 1

        # ── n_c ──
        p0[self.nc_idx] = n_c_init

        # ── Gamma (screening) ──
        gamma_dict = params.get('screening', {}).get('gamma', {}).get(pair_key, {})
        if isinstance(gamma_dict, dict):
            gm = self.gamma_mode
            if gm == 'per_lpair':
                for i, lp in enumerate(self.active_lpairs):
                    lp_label = self._LPAIR_LABELS[lp]
                    if lp_label in gamma_dict:
                        p0[self.n_gamma_start + i] = gamma_dict[lp_label]
            elif gm == 'per_channel':
                for i, ch in enumerate(self.active_channels):
                    # Map channel to lpair label to look up gamma
                    for lp, label in self._LPAIR_LABELS.items():
                        if ch.startswith(label) or ch in [
                            c
                            for c in SK_PARAM_NAMES
                            if (
                                min(CHANNEL_L_MAP.get(c, (0, 0))),
                                max(CHANNEL_L_MAP.get(c, (0, 0))),
                            )
                            == lp
                        ]:
                            if label in gamma_dict:
                                p0[self.n_gamma_start + i] = gamma_dict[label]
                                break
            elif gm == 'global':
                vals = list(gamma_dict.values())
                if vals:
                    p0[self.n_gamma_start] = np.mean(vals)
        elif isinstance(gamma_dict, (int, float)):
            p0[self.n_gamma_start : self.n_gamma_start + self.n_gamma] = gamma_dict

        if self.verbose:
            print(f'\np0 from discrete-shell params ({len(p0)} parameters):')
            for i, name in enumerate(self.param_labels):
                print(f'  {name:<30s}  {p0[i]: .5f}')

        return p0


# ═══════════════════════════════════════════════════════════════════════
#  SKFitterEDTBHSP — EDTB fitter with high-symmetry-path k-points
# ═══════════════════════════════════════════════════════════════════════


class SKFitterEDTBHSP(SKFitterEDTB):
    """SKFitterEDTB augmented with high-symmetry-path k-points and per-k weights.

    Subclass of :class:`SKFitterEDTB` that augments the fitting k-point pool
    with points sampled along the canonical high-symmetry band path.
    Each HSP k-point receives a configurable weight *w_hsp* relative
    to the uniform BZ-grid points (weight 1).

    Usage
    -----
    >>> fitter = SKFitterEDTBHSP(arry, attr, n_shells=3, nkfit=4, ...)
    >>> nk_added, path_str = fitter.augment_with_bands_path(nk=500, w_hsp=2.0, ibrav=2)
    >>> result = fitter.fit(n_trials=10, seed=123, alpha=0.1, n_jobs=-1)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Uniform weight=1 for initial BZ-grid k-points
        self._kpt_weights = np.ones(self.Nk)

    def augment_with_bands_path(
        self, nk=500, w_hsp=1.0, ibrav=2, band_path=None, special_points=None
    ):
        """Add high-symmetry-path k-points to the fitting pool.

        Parameters
        ----------
        nk : int
            Approximate number of HSP k-points to add.
        w_hsp : float
            Weight assigned to each HSP k-point (BZ-grid points have weight 1).
        ibrav : int
            PAOFLOW lattice index (2 = FCC). Auto-path when *band_path* is None.
        band_path : str or None
            Band path string (e.g. ``'G-L-G-X-W-X-K-G'``). None → auto from ibrav.
        special_points : dict or None
            Mapping label → (kx, ky, kz) in fractional coordinates.
            None → auto-determined from ibrav.

        Returns
        -------
        nk_added : int
            Number of k-points actually added.
        path_str : str
            Band-path string returned by :func:`get_path`.
        """
        # If nk <= 0, skip augmentation entirely to recover the original fit
        if nk <= 0:
            print(f'  HSP k-points added: 0  (nk=0 → no augmentation); total Nk = {self.Nk}')
            return 0, ''

        # 2-pass dk scaling: first pass gets nk_trial, then rescale dk to hit ~nk
        dk_trial = 0.00001
        kq_trial, _ = _get_path(
            ibrav,
            self.alat,
            self.a_vecs,
            dk_trial,
            self.b_vecs,
            band_path,
            special_points,
        )
        nk_trial = kq_trial.shape[1]
        scaled_dk = dk_trial * (nk_trial / max(nk, 1))
        kq_frac, path_str = _get_path(
            ibrav,
            self.alat,
            self.a_vecs,
            scaled_dk,
            self.b_vecs,
            band_path,
            special_points,
        )

        # Fractional → Cartesian (PAOFLOW convention: kfrac @ b_vecs)
        kpts_hsp = kq_frac.T @ self.b_vecs  # (nk_new, 3)
        nk_new = kpts_hsp.shape[0]

        # PAO reference eigenvalues at the new k-points
        phases_hsp = np.exp(2j * np.pi * (kpts_hsp @ self.R_arr.T))
        E_hsp = np.zeros((nk_new, self.nawf))
        for ik in range(nk_new):
            Hk = np.einsum('r,rij->ij', phases_hsp[ik], self.HR_arr)
            E_hsp[ik] = np.sort(np.linalg.eigvalsh(Hk).real)

        # Augment fitting arrays
        self.kpts = np.vstack([self.kpts, kpts_hsp])
        self.E_pao = np.vstack([self.E_pao, E_hsp])
        self._kpt_weights = np.concatenate([self._kpt_weights, w_hsp * np.ones(nk_new)])
        self.Nk += nk_new
        self._precompute_dHk()  # rebuild k-dependent Hamiltonian tensors

        print(f'  HSP k-points added: {nk_new}  (target ~{nk}); total Nk = {self.Nk}')
        return nk_new, path_str

    # ── weighted least-squares trial ──────────────────────────────────

    def _run_single_trial(self, p_init, alpha, n_data, max_nfev, ftol, xtol, gtol):
        """Weighted least-squares trial: rows scaled by sqrt(w_k).

        Minimises  Σ_k  w_k · ||E_sk[k] − E_ref[k]||²
        by scaling residuals / Jacobian rows by √w_k.
        The reported RMSE is the *weighted* RMSE.
        """
        use_reg = alpha > 0.0
        reg_w = self._reg_weights
        w_sqrt = np.sqrt(self._kpt_weights)  # (Nk,)
        last_jac = [None]

        def fun(p):
            E_sk, dE_dp = self._eigenvalues_and_jacobian(p)
            res_data = (w_sqrt[:, None] * (E_sk - self.E_pao)).ravel()
            J_data = (w_sqrt[:, None, None] * dE_dp).reshape(n_data, self.n_params)
            if use_reg:
                last_jac[0] = np.vstack([J_data, np.diag(alpha * reg_w)])
                return np.concatenate([res_data, alpha * reg_w * p])
            last_jac[0] = J_data
            return res_data

        def jac(p):
            return last_jac[0]

        res = least_squares(
            fun,
            p_init,
            jac=jac,
            method='lm',
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        rmse = np.sqrt(np.mean(res.fun[:n_data] ** 2))  # weighted RMSE
        return rmse, res.x.copy(), res

    # ── extract initial SK vector from a saved params dict ────────────

    def extract_p0_sk(self, params_dict):
        """Extract the SK parameter sub-vector from a saved EDTB params dict.

        Reverses ``build_model_dict`` for the SK part (on-site + hoppings),
        so the returned vector can be fed to ``fit(p0_sk=...)``.

        Parameters
        ----------
        params_dict : dict
            EDTB parameter dict (as stored in ``*_params.json``).

        Returns
        -------
        np.ndarray
            SK parameter vector of length ``n_sk``.
        """
        L_TO_KEY = {0: 's', 1: 'p', 2: 'd'}

        p0 = np.zeros(self.n_sk)

        # ── On-site energies ──
        for sp in sorted(set(self.unique_species)):
            pstart = self.species_param_start[sp]
            groups = self.species_onsite_groups[sp]
            onsite = params_dict['onsite'][sp]
            for ig, (pname, _) in enumerate(groups):
                inner = pname[2:-1]  # 'g0', 'g1', 'g2_t2g', ...
                if '_t2g' in inner:
                    key = 't2g'
                elif '_eg' in inner:
                    key = 'eg'
                else:
                    # Use shells_dict (actual l-values) rather than group_l
                    # (which is just canonical indices 0,1,2,...)
                    gidx = int(inner[1:])  # group index
                    actual_l = self.shells_dict[sp][gidx]
                    key = L_TO_KEY[actual_l]
                p0[pstart + ig] = onsite[key]

        # ── Hopping integrals ──
        pair_key = list(params_dict['hoppings'].keys())[0]
        for s in range(self.n_shells):
            hop_params = params_dict['hoppings'][pair_key][s]['params']
            for ga, gb in self.hop_pair_list:
                start = self.hop_pair_start[(ga, gb)]
                active = self.hop_pair_active[(ga, gb)]
                for lk, sk_k in enumerate(active):
                    idx = self.n_onsite + s * self.n_hop + start + lk
                    p0[idx] = hop_params[SK_PARAM_NAMES[sk_k]]

        return p0
