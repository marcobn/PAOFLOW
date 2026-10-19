"""Screening correction of the electron-phonon coupling (Poncé, Margine and Giustino 2018).

DFT underestimates band gaps, so DFPT overscreens the phonon perturbation: the
induced potential, and with it the coupling ``g``, is too weak at long
wavelength.  Recomputing the phonons with a better screening is prohibitive;
Poncé *et al.* [1]_ instead rescale every matrix element by the ratio of two
model dielectric functions,

.. math::

    g'_{mn\\nu}(\\mathbf k, \\mathbf q) = g_{mn\\nu}(\\mathbf k, \\mathbf q)\\,
        \\frac{\\epsilon^{\\rm DFT}(|\\mathbf q|)}{\\epsilon^{\\rm target}(|\\mathbf q|)},

both given by Resta's Thomas-Fermi model [2]_ with the respective static
dielectric constant (the head of the dielectric matrix) as the only input.  The
target is "the most accurate description of the screening that we can afford",
the measured value in [1]_ (Si: 12.89 in DFT, 11.94 measured).  Local-field
effects are neglected (a few per cent).

References
----------
.. [1] S. Poncé, E. R. Margine and F. Giustino, Phys. Rev. B 97, 121201(R) (2018).
.. [2] R. Resta, Phys. Rev. B 16, 2717 (1977).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


# integer offsets -1..1 along each reciprocal axis (shortest images, shortest G)
_NEIGHBOUR_CELLS = np.stack(np.meshgrid(*(np.arange(-1, 2),) * 3, indexing='ij'), -1).reshape(-1, 3)


def thomas_fermi_wavevector(valence_electrons: float, cell_volume: float) -> float:
    """Thomas-Fermi wave vector of the valence electron gas (bohr^-1).

    Parameters
    ----------
    valence_electrons : float
        Valence electrons per cell.
    cell_volume : float
        Unit-cell volume (bohr^3).

    Returns
    -------
    float
        :math:`q_{TF} = (4 k_F/\\pi)^{1/2}` in atomic units, with
        :math:`k_F = (3\\pi^2 n)^{1/3}`; about 1.10 bohr^-1 for Si.

    Raises
    ------
    ValueError
        If the electron count or the volume is not positive.
    """
    if valence_electrons <= 0.0 or cell_volume <= 0.0:
        raise ValueError('the valence electron count and the cell volume must be positive.')
    fermi_wavevector = (3.0 * np.pi**2 * valence_electrons / cell_volume) ** (1.0 / 3.0)
    return float(np.sqrt(4.0 * fermi_wavevector / np.pi))


def resta_screening_radius(eps_static: float, thomas_fermi_q: float) -> float:
    """Screening radius ``R`` of Resta's model (bohr).

    Parameters
    ----------
    eps_static : float
        Static dielectric constant :math:`\\epsilon(0) > 1`.
    thomas_fermi_q : float
        Thomas-Fermi wave vector (bohr^-1, :func:`thomas_fermi_wavevector`).

    Returns
    -------
    float
        The root of :math:`\\sinh(q_{TF} R)/(q_{TF} R) = \\epsilon(0)`; about
        4.3 bohr for Si.

    Raises
    ------
    ValueError
        If ``eps_static <= 1``.
    """
    from scipy.optimize import brentq

    if eps_static <= 1.0:
        raise ValueError(
            'Resta screening needs a dielectric constant above 1, got %g.' % eps_static
        )
    upper = 1.0
    while np.sinh(upper) / upper < eps_static:
        upper *= 2.0
    root = brentq(lambda x: np.sinh(x) / x - eps_static, 1.0e-8, upper, xtol=1.0e-14)
    return float(root / thomas_fermi_q)


def resta_dielectric(
    q_norm: ArrayLike, eps_static: float, thomas_fermi_q: float
) -> NDArray[np.float64]:
    """Resta's static dielectric function of a semiconductor.

    Parameters
    ----------
    q_norm : array_like
        Wave-vector moduli (bohr^-1).
    eps_static : float
        Static dielectric constant :math:`\\epsilon(0)`.
    thomas_fermi_q : float
        Thomas-Fermi wave vector (bohr^-1).

    Returns
    -------
    ndarray, shape of ``q_norm``
        :math:`\\epsilon(q)`, equal to ``eps_static`` at ``q = 0`` and tending to
        1 at large ``q``.

    Notes
    -----
    Thomas-Fermi screening of the valence charge inside a sphere of radius
    :math:`R` (:func:`resta_screening_radius`) gives

    .. math::

        \\epsilon(q) = \\frac{q_{TF}^2 + q^2}
            {q_{TF}^2\\, \\sin(qR)/(\\epsilon(0)\\, qR) + q^2}.
    """
    q = np.abs(np.asarray(q_norm, dtype=float))
    radius = resta_screening_radius(eps_static, thomas_fermi_q)
    sinc = np.sinc(q * radius / np.pi)  # sin(qR)/(qR), 1 at q = 0
    q_tf2 = thomas_fermi_q**2
    return (q_tf2 + q**2) / (q_tf2 * sinc / eps_static + q**2)


def folded_q_norm(q_cryst: ArrayLike, bg: NDArray[np.float64], alat: float) -> NDArray[np.float64]:
    """Modulus of the shortest ``q + G`` (bohr^-1).

    Parameters
    ----------
    q_cryst : array_like, shape ``(3,)`` or ``(nq, 3)``
        Wave vectors in crystal coordinates of ``bg``.
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, units of 2 pi/alat).
    alat : float
        Lattice parameter (bohr).

    Returns
    -------
    ndarray, shape ``()`` or ``(nq,)``
        :math:`\\min_{\\mathbf G} |\\mathbf q + \\mathbf G|`, the same for every
        lattice-equivalent ``q``.
    """
    q = np.asarray(q_cryst, dtype=float)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    reduced = q - np.round(q)  # in [-1/2, 1/2]
    cart = (reduced[:, None, :] + _NEIGHBOUR_CELLS[None]) @ np.asarray(bg, dtype=float)  # 2 pi/alat
    norm = np.linalg.norm(cart, axis=-1).min(axis=1) * 2.0 * np.pi / alat
    return norm[0] if single else norm


@dataclass(frozen=True)
class CouplingScreening:
    """Poncé-Resta screening correction of the coupling.

    Attributes
    ----------
    eps_dft : float
        Static dielectric constant of the calculation that screened the
        coupling (DFPT).
    eps_target : float
        Dielectric constant that should have screened it (e.g. the measured one).
    thomas_fermi_q : float
        Thomas-Fermi wave vector of the valence electrons (bohr^-1).
    """

    eps_dft: float
    eps_target: float
    thomas_fermi_q: float

    def ratio(self, q_norm: ArrayLike) -> NDArray[np.float64]:
        """Coupling factor :math:`\\epsilon^{DFT}(q)/\\epsilon^{target}(q)` at ``|q|`` (bohr^-1)."""
        return resta_dielectric(q_norm, self.eps_dft, self.thomas_fermi_q) / resta_dielectric(
            q_norm, self.eps_target, self.thomas_fermi_q
        )

    def factor(self, q_cryst: ArrayLike, bg: NDArray[np.float64], alat: float) -> float:
        """Coupling factor at the phonon wave vector ``q_cryst`` (crystal coordinates)."""
        return float(self.ratio(folded_q_norm(q_cryst, bg, alat)))

    def summary(self, bg: NDArray[np.float64], alat: float) -> dict:
        """Provenance for the output files: the constants and the factor at Gamma and at
        the nearest zone face (half the shortest reciprocal-lattice vector)."""
        lengths = np.linalg.norm(_NEIGHBOUR_CELLS @ np.asarray(bg, dtype=float), axis=1)
        q_edge = 0.5 * float(lengths[lengths > 1.0e-12].min()) * 2.0 * np.pi / alat
        return {
            'coupling_screening': 'ponce-resta',
            'screening_eps_dft': self.eps_dft,
            'screening_eps_target': self.eps_target,
            'screening_thomas_fermi_q': self.thomas_fermi_q,
            'screening_ratio_gamma': float(self.ratio(0.0)),
            'screening_ratio_boundary': float(self.ratio(q_edge)),
            'screening_q_boundary': q_edge,
        }


def resolve_coupling_screening(
    eps_target: float | None,
    eps_dft: float | None,
    coupling_epsi: NDArray[np.float64] | None,
    valence_electrons: float,
    cell_volume: float,
) -> CouplingScreening | None:
    """Build the screening correction of a dense-q run, or ``None`` when it is off.

    Parameters
    ----------
    eps_target : float or None
        Target static dielectric constant; ``None`` disables the correction.
    eps_dft : float or None
        Dielectric constant of the coupling; ``None`` takes the isotropic average
        of ``coupling_epsi``.
    coupling_epsi : ndarray, shape ``(3, 3)``, optional
        Dielectric tensor stored with the coupling (EPW ``.epb``; zero when ph.x
        did not compute it).
    valence_electrons : float
        Valence electrons per cell (Thomas-Fermi wave vector).
    cell_volume : float
        Unit-cell volume (bohr^3).

    Returns
    -------
    CouplingScreening or None

    Raises
    ------
    ValueError
        If ``eps_dft`` is given without ``eps_target``, or ``eps_dft`` is missing
        and the coupling carries no dielectric tensor, or a constant is not
        above 1.
    """
    if eps_target is None:
        if eps_dft is not None:
            raise ValueError(
                'screening_eps_dft needs screening_eps (the target dielectric constant).'
            )
        return None
    if eps_dft is None:
        epsi = np.zeros((3, 3)) if coupling_epsi is None else np.asarray(coupling_epsi, dtype=float)
        if not np.any(epsi != 0.0):
            raise ValueError(
                'the coupling files carry no dielectric tensor (ph.x without epsil, or a '
                'metal): pass screening_eps_dft, the DFT dielectric constant of the coupling.'
            )
        eps_dft = float(np.trace(epsi) / 3.0)
    for name, value in (('screening_eps', eps_target), ('screening_eps_dft', eps_dft)):
        if value <= 1.0:
            raise ValueError('%s must be above 1, got %g.' % (name, value))
    return CouplingScreening(
        float(eps_dft), float(eps_target), thomas_fermi_wavevector(valence_electrons, cell_volume)
    )


def screening_report(info: dict) -> str:
    """One-line description of a screening correction (``CouplingScreening.summary``).

    Parameters
    ----------
    info : dict
        Output of :meth:`CouplingScreening.summary`, or ``{'coupling_screening': 'none'}``.

    Returns
    -------
    str
        For example ``coupling screening (Ponce 2018, Resta): eps 12.97 -> 11.94;
        |g|^2 x1.180 at Gamma, x1.031 at the zone face``.
    """
    if info.get('coupling_screening', 'none') == 'none':
        return '  coupling screening: DFPT (no correction)'
    return (
        '  coupling screening (Ponce 2018, Resta): eps %.2f -> %.2f; |g|^2 x%.3f at Gamma, '
        'x%.3f at the zone face (|q| = %.3f bohr^-1)'
        % (
            info['screening_eps_dft'],
            info['screening_eps_target'],
            info['screening_ratio_gamma'] ** 2,
            info['screening_ratio_boundary'] ** 2,
            info['screening_q_boundary'],
        )
    )
