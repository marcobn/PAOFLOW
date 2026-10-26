"""Dynamical quadrupoles from the long-wavelength expansion of the DFPT charge response.

The charge induced by a monochromatic displacement of atom ``kappa`` along ``beta`` has the
multipole expansion (Royo and Stengel, PRX 9, 021050 (2019))

.. math::

    \\varepsilon_L(\\hat q)\\Big[-\\Omega\\,\\langle \\tilde n_{\\kappa\\beta}(q)\\rangle\\,
    e^{i q\\cdot\\tau_\\kappa} - i\\, q_\\beta Z^{\\rm ion}_\\kappa\\Big]
    = -i\\, q\\cdot Z^*_{\\kappa\\beta} - \\tfrac12\\, q\\cdot Q^{\\kappa}_{\\beta}\\cdot q + O(q^3),

where :math:`\\langle\\tilde n\\rangle` is the cell average of the lattice-periodic part of the
induced electron density per unit displacement (``ph.x`` ``fildrho``), :math:`-iq_\\beta Z^{\\rm
ion}` the point-ion charge moved with the atom (no higher multipoles), :math:`\\varepsilon_L =
\\hat q\\cdot\\varepsilon^\\infty\\cdot\\hat q` removes the screening of the macroscopic field,
which a finite-q DFPT run applies to the total (ion plus electron) charge, :math:`Z^*` the Born
charge and :math:`Q^\\kappa_{\\beta\\alpha\\gamma}` the dynamical quadrupole in e bohr (second moment of
the induced charge about the displaced atom, EPW's ``Qmat(kappa, beta, alpha, gamma)``).  The
response is real in real space, so the odd orders are imaginary and the even ones real: the
quadrupole is the real part at order :math:`q^2`, uncontaminated by the :math:`q^2` dependence of
the dielectric function and by the ionic term, which are imaginary at orders :math:`q^3` and
:math:`q`.

Twelve single-q ``ph.x`` runs (six independent directions, two magnitudes) determine the
18 components per atom; the linear term reproduces the Born charges of the Gamma-point run,
which checks the extraction and fixes the sign convention of the density file.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .dfpt_io import read_dvscf, read_patterns

DIRECTIONS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))  # xx yy zz yz xz xy


def single_q_phonon_inputs(
    prefix: str,
    magnitudes: Sequence[float] = (0.02, 0.04),
    directions: Sequence[Sequence[float]] = DIRECTIONS,
    tr2_ph: float = 1.0e-17,
    outdir: str = './',
) -> list[tuple[str, NDArray[np.float64], str]]:
    """``ph.x`` inputs of the single-q runs (``fildrho``) of the long-wavelength expansion.

    Parameters
    ----------
    prefix : str
        QE prefix.
    magnitudes : sequence of float, optional
        ``|q|`` in ``2 pi / alat`` (two values allow the extrapolation to q -> 0).
    directions : sequence of 3-vectors, optional
        Cartesian directions; six independent ones determine the symmetric quadratic forms.
    tr2_ph : float, optional
        DFPT threshold (the q^2 term is small: keep it tight).
    outdir : str, optional
        ``outdir`` of the run (each q in its own directory with a copy of the scf save).

    Returns
    -------
    list of ``(name, q_cart, text)``
        Run name, q in ``2 pi / alat`` and the input text.
    """
    runs = []
    for i, d in enumerate(directions):
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        for s in magnitudes:
            q = s * d
            name = 'q%d_%s' % (i + 1, ('%g' % s).replace('.', 'p'))
            text = (
                'single q for the quadrupole expansion\n&INPUTPH\n'
                "  prefix = '%s', outdir = '%s', fildyn = '%s.dynq', fildrho = 'drho',\n"
                '  trans = .true., ldisp = .false., tr2_ph = %.1e\n/\n%.10f %.10f %.10f\n'
                % (prefix, outdir, prefix, tr2_ph, q[0], q[1], q[2])
            )
            runs.append((name, q, text))
    return runs


def find_density_response(phx_dir: str, prefix: str, drho_name: str = 'drho') -> tuple[str, str]:
    """``(drho file, patterns.1.xml)`` of a single-q ``ph.x`` run (``_ph0/`` layout).

    Raises
    ------
    FileNotFoundError
        If the density file or the patterns are missing.
    """
    ph0 = os.path.join(phx_dir, '_ph0')
    candidates = [
        os.path.join(ph0, '%s.%s1' % (prefix, drho_name)),
        os.path.join(ph0, '%s.%s' % (prefix, drho_name)),
        os.path.join(ph0, '%s.%s.u1' % (prefix, drho_name)),
    ]
    files = [c for c in candidates if os.path.isfile(c)]
    if not files:
        found = [f for f in os.listdir(ph0)] if os.path.isdir(ph0) else []
        raise FileNotFoundError(
            'no %s file of prefix %r in %s (found %s)' % (drho_name, prefix, ph0, found[:8])
        )
    patterns = os.path.join(ph0, '%s.phsave' % prefix, 'patterns.1.xml')
    if not os.path.isfile(patterns):
        raise FileNotFoundError('patterns.1.xml not found in %s' % ph0)
    return files[0], patterns


def cartesian_density_response(
    drho_patterns: NDArray[np.complex128], u: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Induced density of every Cartesian displacement from the pattern-basis records.

    Parameters
    ----------
    drho_patterns : ndarray ``(3 nat, nr1, nr2, nr3)``
        One spin component of :func:`~PAOFLOW.elphon.dfpt_io.read_dvscf` applied to ``fildrho``.
    u : ndarray ``(3 nat, 3 nat)``
        Displacement patterns (columns), :func:`~PAOFLOW.elphon.dfpt_io.read_patterns`.

    Returns
    -------
    ndarray ``(3 nat, nr1, nr2, nr3)``, complex
        Mode index ``3 kappa + beta`` (the contraction of the induced potentials).
    """
    return np.einsum('cv,v...->c...', np.conj(u), drho_patterns)


def expansion_coefficients(
    drho_cart: NDArray[np.complex128],
    q_cart: NDArray[np.float64],
    tau_cart: NDArray[np.float64],
    alat: float,
    omega: float,
    sign: float = 1.0,
) -> NDArray[np.complex128]:
    """Electronic charge response ``M_{kappa beta}(q) = -Omega <n~> e^{i q.tau}`` per Cartesian mode.

    Parameters
    ----------
    drho_cart : ndarray ``(3 nat, nr1, nr2, nr3)``
        Lattice-periodic induced electron density per unit displacement (bohr^-3 / bohr).
    q_cart : ndarray ``(3,)``
        q in ``2 pi / alat``.
    tau_cart : ndarray ``(nat, 3)``
        Atomic positions (bohr).
    alat : float
        bohr.
    omega : float
        Cell volume (bohr^3).
    sign : float, optional
        ``+1`` when the file holds the induced electron (number) density; ``-1`` for a charge
        density (:func:`calibrate_sign`).

    Returns
    -------
    ndarray ``(nat, 3)``, complex
        The screened electronic charge response (e per bohr of displacement, phase of the
        displaced atom removed); :func:`screened_to_bare` adds the ionic term and the screening.
    """
    nat = tau_cart.shape[0]
    q_bohr = np.asarray(q_cart, dtype=float) * 2.0 * np.pi / float(alat)
    moments = omega * drho_cart.reshape(3 * nat, -1).mean(axis=1)  # Omega <n~> per mode
    phase = np.exp(1j * (np.asarray(tau_cart, dtype=float) @ q_bohr))  # (nat,)
    return -sign * moments.reshape(nat, 3) * phase[:, None]


def screened_to_bare(
    response: NDArray[np.complex128],
    q_cart: NDArray[np.float64],
    alat: float,
    epsilon: NDArray[np.float64],
    zion: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """``A = eps_L (M - i q_beta Z_ion)``: the bare total charge response, ``-iq.Z* - q.Q.q/2 + ...``.

    Parameters
    ----------
    response : ndarray ``(nat, 3)``
        :func:`expansion_coefficients`.
    q_cart : ndarray ``(3,)``
        q in ``2 pi / alat``.
    alat : float
    epsilon : ndarray ``(3, 3)``
        High-frequency dielectric tensor.
    zion : ndarray ``(nat,)``
        Valence charge of each atom's pseudopotential.
    """
    q_bohr = np.asarray(q_cart, dtype=float) * 2.0 * np.pi / float(alat)
    qhat = q_bohr / np.linalg.norm(q_bohr)
    eps_l = float(qhat @ np.asarray(epsilon, dtype=float) @ qhat)
    ionic = -1j * q_bohr[None, :] * np.asarray(zion, dtype=float)[:, None]  # (nat, 3)
    return eps_l * (np.asarray(response, dtype=complex) + ionic)


def fit_multipoles(
    samples: Sequence[tuple[NDArray[np.float64], NDArray[np.complex128]]],
    alat: float,
    epsilon: NDArray[np.float64] | None = None,
    zion: NDArray[np.float64] | None = None,
) -> dict:
    """Born charges and quadrupoles from the expansion coefficients.

    Parameters
    ----------
    samples : sequence of ``(q_cart, M)``
        q in ``2 pi / alat`` and :func:`expansion_coefficients` ``(nat, 3)``; six independent
        directions are needed, two magnitudes allow the ``q -> 0`` extrapolation.
    alat : float
        bohr.
    epsilon : ndarray ``(3, 3)``, optional
    zion : ndarray ``(nat,)``, optional
        With both, the samples are converted with :func:`screened_to_bare` and the linear term is
        the Born charge ``Z*``; without, the samples are fitted as given (``zstar`` is then the
        raw linear coefficient, ``Z*/eps_L - Z_ion`` for a DFPT density).

    Returns
    -------
    dict
        ``zstar`` ``(nat, 3, 3)`` ``[atom, field, displacement]`` from the imaginary part;
        ``quadrupoles`` ``(nat, 3, 3, 3)`` ``[atom, displacement, alpha, gamma]`` in e bohr (symmetric in
        the last two) extrapolated from the two smallest magnitudes (Richardson, O(q^2)
        corrections), ``quadrupoles_by_magnitude`` ``{|q|: Q}``, ``residual_zstar`` and
        ``residual_quadrupole`` (relative least-squares residuals).
    """
    if (epsilon is None) != (zion is None):
        raise ValueError('epsilon and zion must be given together')
    if epsilon is not None:
        samples = [(q, screened_to_bare(a, q, alat, epsilon, zion)) for q, a in samples]
    qs = np.array([np.asarray(q, dtype=float) * 2.0 * np.pi / float(alat) for q, _ in samples])
    coeff = np.array([np.asarray(a, dtype=complex) for _, a in samples])  # (ns, nat, 3)
    ns, nat, _ = coeff.shape
    # linear term: Im A_{kb} = - sum_a q_a Z[k, a, b]
    zstar = np.zeros((nat, 3, 3))
    res_z = 0.0
    for k in range(nat):
        for b in range(3):
            sol, res, _, _ = np.linalg.lstsq(-qs, coeff[:, k, b].imag, rcond=None)
            zstar[k, :, b] = sol
            scale = max(np.abs(coeff[:, k, b].imag).max(), 1e-30)
            res_z = max(res_z, float(np.sqrt(res[0]) if res.size else 0.0) / scale)
    # quadratic term, per magnitude: Re A_{kb} = -1/2 sum_{ag} q_a q_g Q[k, b, a, g]
    magnitudes = np.round(np.linalg.norm(qs, axis=1) * float(alat) / (2.0 * np.pi), 10)
    by_mag = {}
    res_q = 0.0
    for mag in sorted(set(magnitudes)):
        sel = magnitudes == mag
        design = np.array(
            [[q[a] * q[g] * (1.0 if a == g else 2.0) for a, g in _PAIRS] for q in qs[sel]]
        )  # the six symmetric components
        quad = np.zeros((nat, 3, 3, 3))
        for k in range(nat):
            for b in range(3):
                sol, res, rank, _ = np.linalg.lstsq(
                    -0.5 * design, coeff[sel, k, b].real, rcond=None
                )
                if rank < 6:
                    raise ValueError('six independent q directions are needed per magnitude')
                for (a, g), value in zip(_PAIRS, sol):
                    quad[k, b, a, g] = quad[k, b, g, a] = value
                scale = max(np.abs(coeff[sel, k, b].real).max(), 1e-30)
                res_q = max(res_q, float(np.sqrt(res[0]) if res.size else 0.0) / scale)
        by_mag[float(mag)] = quad
    mags = sorted(by_mag)
    if len(mags) >= 2:
        s1, s2 = mags[0], mags[1]
        # Q(s) = Q0 + c s^2  ->  Q0 = (s2^2 Q(s1) - s1^2 Q(s2)) / (s2^2 - s1^2)
        quad = (s2**2 * by_mag[s1] - s1**2 * by_mag[s2]) / (s2**2 - s1**2)
    else:
        quad = by_mag[mags[0]]
    return {
        'zstar': zstar,
        'quadrupoles': quad,
        'quadrupoles_by_magnitude': by_mag,
        'residual_zstar': res_z,
        'residual_quadrupole': res_q,
    }


def calibrate_sign(
    raw_linear: NDArray[np.float64],
    zstar: NDArray[np.float64],
    epsilon: NDArray[np.float64],
    zion: NDArray[np.float64],
) -> float:
    """``+1`` or ``-1``: the sign that makes the raw linear coefficient of the samples physical.

    Parameters
    ----------
    raw_linear : ndarray ``(nat, 3, 3)``
        ``zstar`` of :func:`fit_multipoles` without ``epsilon``/``zion`` (the screened electronic
        response alone), expected to be ``Z*/eps_L - Z_ion``; the ionic term dominates, so the
        test is robust even for ``Z* = 0``.
    zstar : ndarray ``(nat, 3, 3)``
        Born charges of the Gamma-point DFPT run.
    epsilon : ndarray ``(3, 3)``
    zion : ndarray ``(nat,)``
    """
    eps_iso = float(np.trace(np.asarray(epsilon, dtype=float)) / 3.0)
    expected = (
        np.asarray(zstar, dtype=float) / eps_iso
        - np.asarray(zion, dtype=float)[:, None, None] * np.eye(3)[None]
    )
    fit, ref = np.asarray(raw_linear).ravel(), expected.ravel()
    return 1.0 if np.dot(fit, ref) >= 0.0 else -1.0


def write_quadrupole_fmt(path: str, quadrupoles: NDArray[np.float64]) -> str:
    """Write EPW's ``quadrupole.fmt`` (``atom dir Qxx Qyy Qzz Qyz Qxz Qxy`` per atom and direction)."""
    quad = np.asarray(quadrupoles, dtype=float)
    with open(path, 'w') as fh:
        fh.write('atom   dir        Qxx       Qyy       Qzz       Qyz       Qxz       Qxy\n')
        for na in range(quad.shape[0]):
            for idir in range(3):
                b = quad[na, idir]
                fh.write(
                    '%4d %5d %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n'
                    % (na + 1, idir + 1, b[0, 0], b[1, 1], b[2, 2], b[1, 2], b[0, 2], b[0, 1])
                )
    return path


def quadrupoles_from_runs(
    runs: Sequence[tuple[NDArray[np.float64], str]],
    prefix: str,
    fft_grid: Sequence[int],
    tau_cart: NDArray[np.float64],
    alat: float,
    omega: float,
    epsilon: NDArray[np.float64],
    zion: NDArray[np.float64],
    zstar_ref: NDArray[np.float64] | None = None,
    nspin_mag: int = 1,
    drho_name: str = 'drho',
) -> dict:
    """Read the single-q ``ph.x`` runs and fit the multipoles.

    Parameters
    ----------
    runs : sequence of ``(q_cart, directory)``
        q in ``2 pi / alat`` and the run directory holding ``_ph0/``.
    prefix : str
    fft_grid : sequence of int
        Dense FFT grid of the runs (``read_nscf_plane_wave_setup(save)['fft_grid']``).
    tau_cart, alat, omega
        As :func:`expansion_coefficients`.
    epsilon : ndarray ``(3, 3)``
        High-frequency dielectric tensor of the Gamma run.
    zion : ndarray ``(nat,)``
        Valence charge of each atom's pseudopotential.
    zstar_ref : ndarray ``(nat, 3, 3)``, optional
        Born charges of the Gamma run: fixes the sign convention (:func:`calibrate_sign`) and is
        reported against the fitted value (``zstar_ref``).
    nspin_mag, drho_name : optional

    Returns
    -------
    dict
        :func:`fit_multipoles` (``zstar`` is the Born charge) plus ``sign`` and ``samples``.
    """
    nat = np.asarray(tau_cart).shape[0]
    samples = []
    for q_cart, directory in runs:
        drho_path, pattern_path = find_density_response(directory, prefix, drho_name)
        u = read_patterns(pattern_path)['u']
        drho = read_dvscf(drho_path, tuple(int(n) for n in fft_grid), 3 * nat, nspin_mag)[:, 0]
        cart = cartesian_density_response(drho, u)
        samples.append(
            (
                np.asarray(q_cart, dtype=float),
                expansion_coefficients(cart, q_cart, tau_cart, alat, omega),
            )
        )
    sign = 1.0
    if zstar_ref is not None:
        raw = fit_multipoles(samples, alat)['zstar']
        sign = calibrate_sign(raw, zstar_ref, epsilon, zion)
        if sign < 0:
            samples = [(q, -a) for q, a in samples]
    fit = fit_multipoles(samples, alat, epsilon, zion)
    if zstar_ref is not None:
        fit['zstar_ref'] = np.asarray(zstar_ref, dtype=float)
    fit['sign'] = sign
    fit['samples'] = samples
    return fit
