"""Long-range (dipole and quadrupole) part of the electron-phonon coupling and of the dynamical matrix.

The Fourier interpolation of the coupling and of the force constants from the coarse grids to the
dense ones cannot represent the non-analytic long-range fields of a polar crystal: the Fröhlich vertex
diverges as :math:`1/|q|` and the dipole-dipole term of the dynamical matrix sets the LO-TO splitting.
Both are therefore subtracted on the coarse grid, the short-range remainder is interpolated, and the
analytic model is added back at every dense q (Verdi and Giustino, PRL 115, 176401 (2015); Sjakste
et al., PRB 92, 054307 (2015); quadrupoles: Brunin et al., PRL 125, 136601 (2020)).  The model follows
EPW's ``rgd_blk_epw`` and ``rgd_blk`` (``EPW/src/longrange.f90``, QE 7.5), so that the two codes agree
term by term:

.. math::

    g^{\\mathcal L}_{\\kappa\\alpha}(q) = \\frac{4\\pi e^2}{\\Omega}\\sum_{G\\ne -q}
    \\frac{e^{-(q+G)\\cdot\\varepsilon^\\infty\\cdot(q+G)/4\\alpha}}{(q+G)\\cdot\\varepsilon^\\infty\\cdot(q+G)}
    \\, e^{-i(q+G)\\cdot\\tau_\\kappa}\\Big[\\, i\\,(q+G)\\cdot Z^*_\\kappa\\,\\big|_\\alpha
    + \\tfrac12 (q+G)\\cdot Q^{\\kappa}_{\\alpha}\\cdot(q+G)\\Big],

in the PAO gauge diagonal in the orbital indices (EPW: the identity in the Wannier gauge), and the
dipole-dipole force constants with the same Ewald sum and the q = 0 self term that enforces the
acoustic sum rule.  Rydberg units (``e^2 = 2``); q and G in ``2 pi / alat`` on input, bohr^-1 inside.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

E2_RY = 2.0
DEFAULT_ALPHA = 1.0  # Ewald parameter, units of (2 pi / alat)^2 (EPW ``alph``)
DEFAULT_GMAX = 14.0  # EPW ``gmax``: terms with (q+G).eps.(q+G) / (4 alpha) > gmax are dropped


def read_quadrupole_fmt(path: str, nat: int) -> NDArray[np.float64]:
    """Read EPW's ``quadrupole.fmt`` (one header line, then ``atom dir Qxx Qyy Qzz Qyz Qxz Qxy``).

    Parameters
    ----------
    path : str
    nat : int
        Number of atoms (``3 nat`` data lines are expected).

    Returns
    -------
    ndarray ``(nat, 3, 3, 3)``
        ``Q[atom, displacement, p, q]`` (symmetric in ``p, q``), the convention of EPW ``Qmat``.
    """
    quad = np.zeros((nat, 3, 3, 3))
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines()[1:] if ln.strip()]
    if len(lines) < 3 * nat:
        raise ValueError('%s: expected %d data lines, found %d' % (path, 3 * nat, len(lines)))
    for ln in lines[: 3 * nat]:
        tok = ln.split()
        na, idir = int(tok[0]) - 1, int(tok[1]) - 1
        qxx, qyy, qzz, qyz, qxz, qxy = (float(t) for t in tok[2:8])
        quad[na, idir] = [[qxx, qxy, qxz], [qxy, qyy, qyz], [qxz, qyz, qzz]]
    return quad


@dataclass
class LongRangeModel:
    """Analytic long-range model of a crystal (dielectric tensor, Born charges, quadrupoles).

    Parameters
    ----------
    epsilon : ndarray ``(3, 3)``
        High-frequency dielectric tensor.
    born_charges : ndarray ``(nat, 3, 3)``
        ``Z*[atom, field, displacement]`` (QE ``zeu(field, displacement, atom)``).
    tau_alat : ndarray ``(nat, 3)``
        Atomic positions, Cartesian, units of ``alat``.
    alat : float
        Lattice parameter (bohr).
    at, bg : ndarray ``(3, 3)``
        Direct (rows, ``alat``) and reciprocal (rows, ``2 pi / alat``) lattice vectors.
    quadrupoles : ndarray ``(nat, 3, 3, 3)``, optional
        ``Q[atom, displacement, p, q]`` (:func:`read_quadrupole_fmt`); ``None`` for dipoles only.
    alpha : float, optional
        Ewald parameter in ``(2 pi / alat)^2`` (EPW ``alph = 1``).
    gmax : float, optional
        Cutoff of the G sum (EPW ``gmax = 14``).

    Notes
    -----
    The Gaussian damping uses the dielectric norm ``(q+G).eps.(q+G)`` as EPW does (its documentation
    writes ``|q+G|^2``).  The G box is centred on the vector that minimises ``|q+G|``.
    """

    epsilon: NDArray[np.float64]
    born_charges: NDArray[np.float64]
    tau_alat: NDArray[np.float64]
    alat: float
    at: NDArray[np.float64]
    bg: NDArray[np.float64]
    quadrupoles: NDArray[np.float64] | None = None
    alpha: float = DEFAULT_ALPHA
    gmax: float = DEFAULT_GMAX

    def __post_init__(self) -> None:
        self.epsilon = np.asarray(self.epsilon, dtype=float).reshape(3, 3)
        self.born_charges = np.asarray(self.born_charges, dtype=float)
        self.tau_alat = np.asarray(self.tau_alat, dtype=float)
        self.at = np.asarray(self.at, dtype=float)
        self.bg = np.asarray(self.bg, dtype=float)
        nat = self.tau_alat.shape[0]
        if self.born_charges.shape != (nat, 3, 3):
            raise ValueError(
                'born_charges must be (nat, 3, 3), got %s' % (self.born_charges.shape,)
            )
        if self.quadrupoles is not None:
            self.quadrupoles = np.asarray(self.quadrupoles, dtype=float)
            if self.quadrupoles.shape != (nat, 3, 3, 3):
                raise ValueError('quadrupoles must be (nat, 3, 3, 3)')
        self.alat = float(self.alat)
        self.alpha = float(self.alpha)
        self.gmax = float(self.gmax)

    # ----------------------------------------------------------------- geometry
    @property
    def nat(self) -> int:
        """Number of atoms."""
        return int(self.tau_alat.shape[0])

    @property
    def tpiba(self) -> float:
        """``2 pi / alat`` (bohr^-1)."""
        return 2.0 * np.pi / self.alat

    @property
    def omega(self) -> float:
        """Cell volume (bohr^3)."""
        return abs(float(np.linalg.det(self.at))) * self.alat**3

    @property
    def tau_cart(self) -> NDArray[np.float64]:
        """Atomic positions in bohr."""
        return self.tau_alat * self.alat

    @property
    def has_quadrupoles(self) -> bool:
        """Whether a quadrupole tensor is present (and not identically zero)."""
        return self.quadrupoles is not None and bool(np.any(self.quadrupoles != 0.0))

    def is_polar(self, tolerance: float = 0.1) -> bool:
        """``max |Z*| > tolerance``."""
        return bool(np.abs(self.born_charges).max() > tolerance)

    def signature(self) -> str:
        """Short hash of the tensors and parameters (vertex-cache key component)."""
        payload = {
            'eps': np.round(self.epsilon, 8).tolist(),
            'zstar': np.round(self.born_charges, 8).tolist(),
            'quad': None if self.quadrupoles is None else np.round(self.quadrupoles, 8).tolist(),
            'tau': np.round(self.tau_alat, 10).tolist(),
            'alat': round(self.alat, 10),
            'alpha': self.alpha,
            'gmax': self.gmax,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    # ----------------------------------------------------------------- constructors
    @classmethod
    def from_dfpt(cls, setup, quadrupoles=None, **kwargs) -> LongRangeModel:
        """From a :class:`~PAOFLOW.elphon.dfpt_vertex.CoarseVertexSetup` (Γ dynamical matrix tensors).

        Raises
        ------
        ValueError
            If ``ph.x`` ran without ``epsil`` (no dielectric tensor in the dyn file).
        """
        if setup.epsilon is None or setup.born_charges is None:
            raise ValueError(
                'the DFPT run has no dielectric tensor / Born charges (ph.x without epsil=.true.)'
            )
        return cls(
            setup.epsilon, setup.born_charges, setup.tau_alat, setup.alat, setup.at, setup.bg,
            quadrupoles=quadrupoles, **kwargs,
        )  # fmt: skip

    @classmethod
    def from_coupling(
        cls, coupling_dir: str, save_dir: str, prefix: str | None = None, quadrupoles=None, **kwargs
    ) -> LongRangeModel:
        """From the ``ph.x`` outputs (``_ph0/`` or ``pp.py`` ``save/``) and the nscf save.

        Parameters
        ----------
        coupling_dir : str
            Directory with the dynamical matrices (:func:`~PAOFLOW.elphon.dfpt_io.locate_dfpt_files`).
        save_dir : str
            QE save for the lattice and atomic positions.
        prefix : str, optional
        quadrupoles : ndarray ``(nat, 3, 3, 3)`` or str, optional
            Tensor or path of an EPW ``quadrupole.fmt`` file.
        """
        from .dfpt_io import locate_dfpt_files, read_long_range_tensors
        from .elph_bloch import read_nscf

        files = locate_dfpt_files(coupling_dir, prefix)
        epsilon, zstar = read_long_range_tensors(files)
        if epsilon is None or zstar is None:
            raise ValueError(
                '%s: the dynamical matrix at Gamma carries no dielectric tensor / Born charges '
                '(run ph.x with epsil=.true.)' % coupling_dir
            )
        nscf = read_nscf(save_dir)
        tau_alat = np.asarray(nscf['tau_cryst']) @ nscf['at']
        if isinstance(quadrupoles, str):
            quadrupoles = read_quadrupole_fmt(quadrupoles, tau_alat.shape[0])
        return cls(
            epsilon, zstar, tau_alat, nscf['alat'], nscf['at'], nscf['bg'],
            quadrupoles=quadrupoles, **kwargs,
        )  # fmt: skip

    @classmethod
    def from_epw(cls, epw: dict, nscf: dict, quadrupoles=None, **kwargs) -> LongRangeModel:
        """From :func:`~PAOFLOW.elphon.do_pao_eph.load_epw_coupling` (``epsi``, ``zstar``) and ``read_nscf``."""
        tau_alat = np.asarray(nscf['tau_cryst']) @ nscf['at']
        if isinstance(quadrupoles, str):
            quadrupoles = read_quadrupole_fmt(quadrupoles, tau_alat.shape[0])
        return cls(
            epw['epsi'], epw['zstar'], tau_alat, nscf['alat'], nscf['at'], nscf['bg'],
            quadrupoles=quadrupoles, **kwargs,
        )  # fmt: skip

    # ----------------------------------------------------------------- Ewald sums
    def _box(self, q_cryst, around_q: bool):
        """``(gg (n, 3) bohr^-1, weight (n,))`` of the damped Coulomb kernel over the G box.

        ``around_q``: the box is centred on the G minimising ``|q+G|`` and the vectors are
        ``q + G`` (vertex and q-dependent force constants); otherwise the box is centred on 0 and
        the vectors are ``G`` with ``G = 0`` excluded (self term).
        """
        q_cryst = np.asarray(q_cryst, dtype=float)
        nr = [int(np.sqrt(4.0 * self.gmax * self.alpha) / np.linalg.norm(b)) + 1 for b in self.bg]
        if around_q:
            base = -np.rint(q_cryst).astype(int)
            cand = base[None, :] + np.array(
                [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
            )
            norms = np.linalg.norm((q_cryst + cand) @ self.bg, axis=1)
            centre = cand[int(np.argmin(norms))]
            offset = q_cryst
        else:
            centre = np.zeros(3, dtype=int)
            offset = np.zeros(3)
        m = np.stack(
            np.meshgrid(*[np.arange(-n, n + 1) + c for n, c in zip(nr, centre)], indexing='ij'),
            axis=-1,
        ).reshape(-1, 3)
        gg = ((m + offset) @ self.bg) * self.tpiba  # (n, 3) bohr^-1
        qeq = np.einsum('gi,ij,gj->g', gg, self.epsilon, gg)
        metric = self.tpiba**2
        keep = (qeq > 0.0) & (qeq / (metric * self.alpha * 4.0) < self.gmax)
        gg, qeq = gg[keep], qeq[keep]
        weight = (
            (E2_RY * 4.0 * np.pi / self.omega) * np.exp(-qeq / (metric * self.alpha * 4.0)) / qeq
        )
        return gg, weight

    def _moments(self, gg):
        """``zag (n, nat, 3)`` = ``(q+G).Z*_kappa`` per displacement and ``qag`` the quadrupole analogue."""
        zag = np.einsum('gp,kpa->gka', gg, self.born_charges)
        if self.quadrupoles is None:
            qag = np.zeros_like(zag)
        else:
            qag = np.einsum('gp,gq,kapq->gka', gg, gg, self.quadrupoles)
        return zag, qag

    def coefficients(self, q_cryst) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]:
        """Long-range vertex per Cartesian displacement at ``q`` (Ry/bohr).

        Parameters
        ----------
        q_cryst : array_like ``(3,)``
            q in crystal coordinates.

        Returns
        -------
        coeff_delta : ndarray ``(3 nat,)``, complex
            Coefficient of the orbital identity (EPW ``coeff_delta``), mode index ``3 kappa + alpha``.
        coeff_r : ndarray ``(3, 3 nat)``, complex or None
            Coefficient of the position operator (EPW ``coeff_r``), present with quadrupoles only.
        """
        gg, weight = self._box(q_cryst, around_q=True)
        nat = self.nat
        if gg.shape[0] == 0:
            return np.zeros(3 * nat, dtype=complex), (
                np.zeros((3, 3 * nat), dtype=complex) if self.has_quadrupoles else None
            )
        phase = np.exp(-1j * (gg @ self.tau_cart.T))  # (n, nat): e^{-i (q+G).tau_kappa}
        zag, qag = self._moments(gg)
        facq = weight[:, None] * phase  # (n, nat)
        coeff_delta = np.einsum('gk,gka->ka', facq, 1j * zag + 0.5 * qag).reshape(3 * nat)
        coeff_r = None
        if self.has_quadrupoles:
            coeff_r = -np.einsum('gk,gka,gd->dka', facq, zag, gg).reshape(3, 3 * nat)
        return coeff_delta, coeff_r

    def vertex_pao(self, q_cryst, nawf: int, orbital_positions=None) -> NDArray[np.complex128]:
        """The long-range vertex in the PAO gauge, ``(nawf, nawf, 3 nat)`` (Ry/bohr).

        Parameters
        ----------
        q_cryst : array_like ``(3,)``
        nawf : int
            Number of PAO orbitals.
        orbital_positions : ndarray ``(nawf, 3)``, optional
            Orbital centres (crystal coordinates), used for the position-operator term of the
            quadrupole model (EPW's Wannier position matrix); ignored for dipoles only.

        Notes
        -----
        Diagonal in the orbitals, as EPW's identity in the Wannier gauge; the same matrix is
        subtracted from every coarse k and added at every dense k of this q.
        """
        coeff_delta, coeff_r = self.coefficients(q_cryst)
        lr = np.zeros((nawf, nawf, 3 * self.nat), dtype=complex)
        idx = np.arange(nawf)
        lr[idx, idx, :] = coeff_delta[None, :]
        if coeff_r is not None and orbital_positions is not None:
            r_cart = (np.asarray(orbital_positions, dtype=float) @ self.at) * self.alat  # (nawf, 3)
            lr[idx, idx, :] += r_cart @ coeff_r  # (nawf, 3 nat)
        return lr

    def dynamical_matrix(self, q_cryst) -> NDArray[np.complex128]:
        """Long-range force constants at ``q`` (Ry/bohr^2, ``(3 nat, 3 nat)``, not mass-weighted).

        The ``q``-dependent dipole, dipole-quadrupole and quadrupole-quadrupole terms plus the
        ``q = 0`` self term on the atom-diagonal blocks (EPW ``rgd_blk``, ``signe = +1``).  Subtract
        it from the coarse ``C(q)`` before the ``q -> R_p`` transform and add it back at every dense q.
        """
        nat = self.nat
        tau = self.tau_cart
        dyn = np.zeros((nat, 3, nat, 3), dtype=complex)
        # self term: G != 0 at q = 0, atom-diagonal blocks
        gg, weight = self._box(np.zeros(3), around_q=False)
        if gg.shape[0]:
            zag, qag = self._moments(gg)
            # phase[g, a, b] = e^{i G.(tau_a - tau_b)}
            phase = np.exp(
                1j * (gg @ (tau[:, None, :] - tau[None, :, :]).reshape(-1, 3).T)
            ).reshape(gg.shape[0], nat, nat)
            fnat = np.einsum('gab,gbj->gaj', phase, zag)
            qnat = np.einsum('gab,gbj->gaj', phase, qag)
            self_term = (
                np.einsum('g,gai,gaj->aij', weight, zag, fnat)
                + 0.5j
                * (
                    np.einsum('g,gai,gaj->aij', weight, qag, fnat)
                    - np.einsum('g,gai,gaj->aij', weight, zag, qnat)
                )
                + 0.25 * np.einsum('g,gai,gaj->aij', weight, qag, qnat)
            )
            for a in range(nat):
                dyn[a, :, a, :] -= self_term[a]
        # q-dependent term
        gg, weight = self._box(q_cryst, around_q=True)
        if gg.shape[0]:
            zag, qag = self._moments(gg)
            phase = np.exp(
                1j * (gg @ (tau[:, None, :] - tau[None, :, :]).reshape(-1, 3).T)
            ).reshape(gg.shape[0], nat, nat)  # e^{i (q+G).(tau_a - tau_b)}
            facg = weight[:, None, None] * phase  # (g, a, b)
            dyn += (
                np.einsum('gab,gai,gbj->aibj', facg, zag, zag)
                + 0.5j
                * (
                    np.einsum('gab,gai,gbj->aibj', facg, qag, zag)
                    - np.einsum('gab,gai,gbj->aibj', facg, zag, qag)
                )
                + 0.25 * np.einsum('gab,gai,gbj->aibj', facg, qag, qag)
            )
        return dyn.reshape(3 * nat, 3 * nat)
