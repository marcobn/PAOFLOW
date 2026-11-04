r"""Molecular (0D) reading of the second-order optical responses of a box.

For a molecule in a vacuum box evaluated at the Gamma point only (``nk = (1, 1, 1)``, a
``1 x 1 x 1`` real-space Hamiltonian), the SHG susceptibility of the box is the first
hyperpolarizability per volume, the shift "current" is the rate of change of the molecular dipole
under illumination, and the injection currents vanish (no band velocities).  This module converts
the box quantities to molecular ones, computes the orientational averages measured by EFISH and
hyper-Rayleigh scattering, the ground-state dipole of the tight-binding model and the
minimum-image check of the box.

Conventions
-----------
``nonlinear_optical_spectra`` writes :math:`P^a(2\\omega) = \\epsilon_0\\chi^{abc}E^bE^c` for a field
:math:`E(t) = E(\\omega)e^{-i\\omega t} + c.c.` with the charge entering as :math:`e^3`, :math:`e > 0`.
For electrons the dipole :math:`\\mu = -e\\sum x` and the coupling :math:`+e\\,x\\cdot E` give
:math:`(-e)^3`, so the molecular hyperpolarizability of the "B" (perturbation) convention,
:math:`\\mu^a(2\\omega) = \\beta^{abc}_BE^bE^c`, is :math:`\\beta_B = -\\epsilon_0\\Omega\\chi` and the Taylor
("T") one, :math:`\\mu = \\mu_0 + \\alpha F + \\tfrac12\\beta_T F F + \\dots` for a static field
:math:`F`, is :math:`\\beta_T = 2\\beta_B`.  Both were pinned against finite-field and real-time
references of a tight-binding model (tests ``test_nonlinear_optics_molecule``).  In atomic units
(:math:`e^3a_0^3/E_h^2 = 3.2064\\times10^{-53}\\,\\mathrm{C^3m^3J^{-2}} = 8.639\\times10^{-33}` esu)

.. math::

    \\beta_T\\,[\\mathrm{a.u.}] = -2\\times 0.040919\;\\Omega[\\mathrm{bohr}^3]\;\\chi[\\mathrm{pm/V}].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .nonlinear_optics import ELECTRON_CHARGE_SI, EPSILON0_SI, component_label

BOHR_M = 0.529177210903e-10
HARTREE_J = 4.3597447222071e-18
#: one atomic unit of beta, e^3 a0^3 / E_h^2 (C^3 m^3 J^-2)
BETA_AU_SI = ELECTRON_CHARGE_SI**3 * BOHR_M**3 / HARTREE_J**2
#: beta_B (a.u.) per (bohr^3 x pm/V): eps0 a0^3 1e-12 / (1 a.u. of beta) = 0.040919
BETA_AU_PER_BOHR3_PMV = EPSILON0_SI * BOHR_M**3 * 1.0e-12 / BETA_AU_SI
#: one atomic unit of beta in esu (cm^5 esu^-1)
BETA_AU_TO_ESU = BETA_AU_SI * (2.99792458e9) ** 3 * 1.0e6 / 1.0e14


def hyperpolarizability(chi_pm_v: NDArray, volume_bohr3: float) -> NDArray[np.complex128]:
    """Taylor hyperpolarizability of the molecule (a.u., electrons) from the box susceptibility.

    Parameters
    ----------
    chi_pm_v : array_like
        :math:`\\chi^{abc}(-2\\omega;\\omega,\\omega)` of the box in pm/V (any shape).
    volume_bohr3 : float
        Box volume (bohr^3).

    Returns
    -------
    ndarray, complex
        :math:`\\beta_T = -2\\epsilon_0\\Omega\\chi` in atomic units; :math:`\\beta_B = \\beta_T/2`.
    """
    return -2.0 * BETA_AU_PER_BOHR3_PMV * float(volume_bohr3) * np.asarray(chi_pm_v, dtype=complex)


def tensor_from_components(values: NDArray, labels) -> NDArray:
    """Arrange per-component spectra ``(ncomp, ...)`` into a ``(3, 3, 3, ...)`` tensor.

    Parameters
    ----------
    values : ndarray ``(ncomp, ...)``
    labels : sequence of str
        Component labels (``'xyz'``), all 27 required.

    Raises
    ------
    ValueError
        If a component is missing.
    """
    values = np.asarray(values)
    index = {label: i for i, label in enumerate(labels)}
    out = np.zeros((3, 3, 3) + values.shape[1:], dtype=values.dtype)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                label = component_label((a, b, c))
                if label not in index:
                    raise ValueError("component '%s' missing: compute components='all'" % label)
                out[a, b, c] = values[index[label]]
    return out


def ground_state_dipole(
    H: NDArray,
    positions_bohr: NDArray[np.float64],
    nocc: int,
    spin_factor: float,
    ionic: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Ground-state dipole of the tight-binding model, :math:`\\mu = \\sum_I Z_I\\mathbf R_I - g_s\\sum_{n\\,\\mathrm{occ}}\\langle n|x|n\\rangle`.

    Parameters
    ----------
    H : ndarray ``(n, n)``
        Gamma-point Hamiltonian (Hermitian).
    positions_bohr : ndarray ``(n, 3)``
        Orbital centres (bohr); the position operator is diagonal in the orbital basis, so the
        electronic part is the Mulliken-type sum of the orbital populations times their centres.
    nocc : int
        Occupied states of this spin channel.
    spin_factor : float
        2 for a spin-degenerate channel, 1 otherwise.
    ionic : tuple of ndarray, optional
        ``(positions (nat, 3) bohr, charges (nat,))`` of the ionic cores (valence charges of the
        pseudopotentials).  Without it only the electronic part is returned, which depends on the
        origin unless the orbital populations sum to zero: pass it for a neutral molecule.

    Returns
    -------
    ndarray ``(3,)``
        In units of :math:`e\\cdot\\mathrm{bohr}` (physics sign, from negative to positive charge;
        1 e bohr = 2.5417 debye).
    """
    _, U = np.linalg.eigh(np.asarray(H))
    weight = np.sum(np.abs(U[:, : int(nocc)]) ** 2, axis=1)  # (n,) electrons per orbital
    dipole = -float(spin_factor) * weight @ np.asarray(positions_bohr, dtype=float)
    if ionic is not None:
        R = np.asarray(ionic[0], dtype=float).reshape(-1, 3)
        Z = np.asarray(ionic[1], dtype=float)
        dipole = dipole + Z @ R
    return dipole


def ionic_charges_from_data(data_controller):
    """Atomic positions (bohr) and valence charges of a PAOFLOW run from its basis list and UPF files.

    Parameters
    ----------
    data_controller : DataController
        After ``projections``: ``arrays['basis']`` (orbital centres and element labels),
        ``arrays['species']`` (label, UPF file) and ``attr['fpath']`` (the save directory holding
        the UPF files).

    Returns
    -------
    tuple of ndarray or None
        ``(positions (nat, 3), charges (nat,))``, or ``None`` when the information is missing.
    """
    arrays, attr = data_controller.data_dicts()
    basis = arrays.get('basis')
    species = arrays.get('species')
    if not basis or not species or 'tau' not in basis[0] or 'atom' not in basis[0]:
        return None
    import os

    from ..inputs.read_upf import UPF

    charges: dict = {}
    for label, upf_file in species:
        path = os.path.join(str(attr.get('fpath', '')), str(upf_file))
        if not os.path.isfile(path):
            return None
        charges[str(label)] = float(UPF(path).val)
    seen: dict = {}
    positions, zs = [], []
    for orbital in basis:
        key = tuple(np.round(np.asarray(orbital['tau'], dtype=float), 6))
        if key not in seen:
            seen[key] = len(positions)
            positions.append(np.asarray(orbital['tau'], dtype=float))
            zs.append(charges.get(str(orbital['atom']), np.nan))
    if np.isnan(zs).any():
        return None
    return np.array(positions), np.array(zs)


def beta_parallel(beta: NDArray, dipole: NDArray[np.float64]) -> NDArray:
    """EFISH projection :math:`\\beta_\\parallel = \\tfrac15\\sum_{ij}(\\beta_{iij}+\\beta_{iji}+\\beta_{jii})\\hat\\mu_j`.

    Parameters
    ----------
    beta : ndarray ``(3, 3, 3, ...)``
    dipole : ndarray ``(3,)``
        Direction of the ground-state dipole.

    Returns
    -------
    ndarray ``(...)``
    """
    mu = np.asarray(dipole, dtype=float)
    norm = np.linalg.norm(mu)
    if norm == 0.0:
        return np.zeros(np.asarray(beta).shape[3:], dtype=np.asarray(beta).dtype)
    u = mu / norm
    beta = np.asarray(beta)
    vec = (
        np.einsum('iij...->j...', beta)
        + np.einsum('iji...->j...', beta)
        + np.einsum('jii...->j...', beta)
    )
    return np.tensordot(u, vec, axes=(0, 0)) / 5.0


def so3_quadrature(n_angle: int = 7, n_beta: int = 4):
    """Rotation matrices and weights that average polynomials of degree <= 6 exactly over SO(3).

    Euler angles ``z-y-z``: uniform grids in ``alpha`` and ``gamma`` (``n_angle`` points, exact
    for trigonometric degree ``< n_angle``) and Gauss-Legendre in ``cos beta`` (``n_beta`` points,
    exact to degree ``2 n_beta - 1``).

    Returns
    -------
    rotations : ndarray ``(nq, 3, 3)``
    weights : ndarray ``(nq,)``, summing to 1
    """
    nodes, wts = np.polynomial.legendre.leggauss(int(n_beta))
    alphas = 2.0 * np.pi * np.arange(n_angle) / n_angle
    rots, weights = [], []
    for alpha in alphas:
        ca, sa = np.cos(alpha), np.sin(alpha)
        rz_a = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        for cb, wb in zip(nodes, wts):
            sb = np.sqrt(max(0.0, 1.0 - cb * cb))
            ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
            for gamma in alphas:
                cg, sg = np.cos(gamma), np.sin(gamma)
                rz_g = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
                rots.append(rz_a @ ry @ rz_g)
                weights.append(wb / (2.0 * n_angle * n_angle))
    return np.array(rots), np.array(weights)


def hrs_averages(beta: NDArray) -> tuple[NDArray, NDArray]:
    """Orientational averages :math:`\\langle|\\beta_{ZZZ}|^2\\rangle` and :math:`\\langle|\\beta_{XZZ}|^2\\rangle`.

    Parameters
    ----------
    beta : ndarray ``(3, 3, 3, ...)``
        Laboratory-frame tensor is :math:`\\beta'_{ijk} = R_{ia}R_{jb}R_{kc}\\beta_{abc}`; the averages
        are exact (degree-6 quadrature on SO(3)) for any tensor, Kleinman-symmetric or not.

    Returns
    -------
    zzz, xzz : ndarray ``(...)``
    """
    beta = np.asarray(beta)
    rots, weights = so3_quadrature()
    zzz = np.zeros(beta.shape[3:], dtype=float)
    xzz = np.zeros(beta.shape[3:], dtype=float)
    for R, w in zip(rots, weights):
        lab = np.einsum('ia,jb,kc,abc...->ijk...', R, R, R, beta)
        zzz += w * np.abs(lab[2, 2, 2]) ** 2
        xzz += w * np.abs(lab[0, 2, 2]) ** 2
    return zzz, xzz


def hrs_quantities(beta: NDArray) -> dict:
    """``beta_hrs`` = sqrt(<|b_ZZZ|^2> + <|b_XZZ|^2>) and the depolarisation ratio <|b_ZZZ|^2>/<|b_XZZ|^2>."""
    zzz, xzz = hrs_averages(beta)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(xzz > 0.0, zzz / np.where(xzz > 0.0, xzz, 1.0), np.nan)
    return {'beta_hrs': np.sqrt(zzz + xzz), 'depolarization_ratio': ratio, 'zzz2': zzz, 'xzz2': xzz}


def reshift_complement(
    H_gamma: NDArray,
    nbnd: int,
    nocc_channels,
    shift_above_homo_ev: float = 30.0,
    tol: float = 1.0e-6,
) -> tuple[NDArray, list[tuple[float, float]]]:
    """Move the shifted complement of a Gamma PAO Hamiltonian to ``shift_above_homo_ev`` above the HOMO.

    Parameters
    ----------
    H_gamma : ndarray ``(n, n)`` or ``(n, n, nspin)``
        Gamma-point PAO Hamiltonian (eV) whose ``n - nbnd`` complement states are degenerate at
        the shift energy of ``pao_hamiltonian``.
    nbnd : int
        Projected (physical) states.
    nocc_channels : sequence of int
        Occupied states per channel (the HOMO of each channel sets the reference).
    shift_above_homo_ev : float, optional
        New energy of the complement, relative to the HOMO of the channel (eV).
    tol : float, optional
        Relative tolerance of the degeneracy check of the complement.

    Returns
    -------
    H : ndarray
        Same shape as ``H_gamma``, the projected states untouched.
    moved : list of (old, new)
        Complement energy before and after, per channel (eV).

    Raises
    ------
    ValueError
        If the top ``n - nbnd`` eigenvalues are not degenerate (no shifted complement).

    Notes
    -----
    The automatic shift of ``projectability`` puts the complement just above the last projected
    band, which is inert for band structures but acts, in a response calculation, as a collective
    manifold of large dipole strength at low energy: it over-screens the hyperpolarizability of a
    molecule and inflates its polarizability. At 20–30 eV above the HOMO, where the σ* and
    continuum channels of the real molecule lie, the same states give the atomic polarizability
    correctly (PNA, `Workspace/PAOFLOW/validation/nlo_molecules/coupled/REPORT.md`).
    """
    H = np.array(H_gamma, dtype=complex)
    squeeze = H.ndim == 2
    if squeeze:
        H = H[:, :, None]
    n = H.shape[0]
    ncomp = n - int(nbnd)
    moved = []
    for s_ in range(H.shape[2]):
        e, C = np.linalg.eigh(H[:, :, s_])
        if ncomp <= 0:
            moved.append((np.nan, np.nan))
            continue
        top = e[int(nbnd) :]
        scale = max(abs(float(top[0])), 1.0)
        if float(top.max() - top.min()) > tol * scale:
            raise ValueError(
                'the top %d eigenvalues spread over %.3e eV: no degenerate complement to re-shift'
                % (ncomp, float(top.max() - top.min()))
            )
        homo = float(e[int(nocc_channels[s_]) - 1])
        new = homo + float(shift_above_homo_ev)
        moved.append((float(top[0]), new))
        e = e.copy()
        e[int(nbnd) :] = new
        H[:, :, s_] = (C * e[None, :]) @ C.conj().T
    return (H[:, :, 0] if squeeze else H), moved


def minimum_image_violations(
    H: NDArray,
    positions_cryst: NDArray[np.float64],
    tol_ev: float = 1.0e-2,
    fraction: float = 0.45,
) -> list[tuple[int, int, float, float]]:
    """Orbital pairs that couple across more than ``fraction`` of the box along some axis.

    The Gamma-point Hamiltonian is the sum over all images; the minimum-image operators are exact
    only when every coupled pair lies inside the Wigner-Seitz cell of the box.  Pairs with
    ``|H_ij| > tol_ev`` whose minimum-image separation exceeds ``fraction`` of a lattice vector
    along any axis are returned as ``(i, j, |H_ij| eV, max fractional separation)``.

    Parameters
    ----------
    H : ndarray ``(n, n)``
    positions_cryst : ndarray ``(n, 3)``
        Orbital centres in crystal coordinates of the box.
    tol_ev, fraction : float
    """
    H = np.asarray(H)
    pos = np.asarray(positions_cryst, dtype=float)
    n = H.shape[0]
    out = []
    for i in range(n):
        strong = np.nonzero(np.abs(H[i]) > float(tol_ev))[0]
        if strong.size == 0:
            continue
        d = pos[strong] - pos[i][None, :]
        d -= np.rint(d)  # minimum image in [-0.5, 0.5)
        worst = np.max(np.abs(d), axis=1)
        for j, w in zip(strong, worst):
            if w > float(fraction) and j > i:
                out.append((int(i), int(j), float(np.abs(H[i, j])), float(w)))
    return out


def molecular_summary(
    results: dict,
    H_gamma: NDArray,
    positions_bohr: NDArray[np.float64],
    positions_cryst: NDArray[np.float64],
    volume_bohr3: float,
    nocc_channels,
    spin_factor: float,
    ionic=None,
) -> dict:
    """Molecular quantities from the box results of one or two spin channels.

    Parameters
    ----------
    results : dict
        Total results of :func:`~PAOFLOW.response.nonlinear_optics.nonlinear_optical_spectra`
        (``components='all'``); ``'shg'`` and/or ``'shift'`` present.
    H_gamma : ndarray ``(n, n, nspin)``
        Gamma Hamiltonians of the channels (eV).
    positions_bohr, positions_cryst : ndarray ``(n, 3)``
    volume_bohr3 : float
    nocc_channels : sequence of int
        Occupied states per channel.
    spin_factor : float
    ionic : tuple of ndarray, optional
        ``(positions, charges)`` of the ionic cores (:func:`ionic_charges_from_data`); without it
        the dipole is the electronic part only, origin-dependent for a neutral molecule.

    Returns
    -------
    dict
        ``'beta_T_au'`` ``(ncomp, ne)`` complex (Taylor, a.u.), ``'beta_parallel_au'``,
        ``'beta_hrs_au'``, ``'depolarization_ratio'`` ``(ne,)``, ``'dipole_ebohr'`` ``(3,)``,
        ``'dmudt_si'`` ``(ncomp, ne)`` (shift response per molecule, A m^3 V^-2),
        ``'minimum_image_violations'`` (list).
    """
    out = {}
    H_gamma = np.asarray(H_gamma)
    if H_gamma.ndim == 2:
        H_gamma = H_gamma[:, :, None]
    dipole = np.zeros(3)
    for s in range(H_gamma.shape[2]):
        dipole += ground_state_dipole(
            H_gamma[:, :, s], positions_bohr, int(nocc_channels[s]), spin_factor
        )
    if ionic is not None:
        dipole = dipole + np.asarray(ionic[1], dtype=float) @ np.asarray(
            ionic[0], dtype=float
        ).reshape(-1, 3)
    out['dipole_ebohr'] = dipole
    out['dipole_debye'] = dipole * 2.541746
    if 'shg' in results:
        beta_c = hyperpolarizability(results['shg'], volume_bohr3)  # (ncomp, ne)
        out['beta_T_au'] = beta_c
        tensor = tensor_from_components(beta_c, results['components'])  # (3, 3, 3, ne)
        out['beta_parallel_au'] = beta_parallel(tensor, dipole)
        hrs = hrs_quantities(tensor)
        out['beta_hrs_au'] = hrs['beta_hrs']
        out['depolarization_ratio'] = hrs['depolarization_ratio']
    if 'shift' in results:
        out['dmudt_si'] = np.asarray(results['shift']) * float(volume_bohr3) * BOHR_M**3
    H_sum = H_gamma.sum(axis=2) / H_gamma.shape[2]
    out['minimum_image_violations'] = minimum_image_violations(H_sum, positions_cryst)
    return out
