"""Phonon ingredients and the ``g_mn^v(k, q)`` contraction (P2).

The electron-phonon matrix element is

    g_mn^v(k, q) = sqrt(hbar / (2 omega_qv))
                   * sum_{kappa,alpha} e_{kappa alpha, v}(q) / sqrt(M_kappa)
                     * <psi_{m, k+q} | dH/du_{kappa alpha} | psi_{n, k}>,

where ``e_{kappa alpha, v}(q)`` is the phonopy phonon eigenvector, ``M_kappa``
the atomic mass and ``dH/du_{kappa alpha}`` the (folded, Fourier-transformed)
Cartesian derivative of the PAO Hamiltonian produced by
:mod:`PAOFLOW.elphon.dvscf_fd`.

This module provides the phonon-side ingredients (frequencies, eigenvectors,
masses at arbitrary ``q``) and the electronic contraction that turns a Cartesian
derivative into ``g``.  It is unit-tested directly against phonopy and with
fabricated derivatives; the full pipeline is wired once the symmetry expansion
and the supercell -> primitive fold provide ``dH/du(k, q)``.
"""

import numpy as np

# 1 THz -> eV (Planck constant times 1e12 Hz, divided by the elementary charge).
THZ_TO_EV = 4.135667696e-3


def phonon_modes(phonon, qpoints, with_eigenvectors=True):
    """Phonon frequencies, eigenvectors and masses at the given q-points.

    Parameters
    ----------
    phonon : phonopy.Phonopy
        Object with force constants already produced.
    qpoints : array_like
        ``(nq, 3)`` q-points in fractional reciprocal coordinates.
    with_eigenvectors : bool
        Also return the mode eigenvectors.

    Returns
    -------
    frequencies : ndarray
        ``(nq, nmode)`` phonon frequencies in THz.
    eigenvectors : ndarray or None
        ``(nq, natom, 3, nmode)`` complex mode eigenvectors; the atomic
        displacement of mode ``v`` is ``e[.., kappa, :, v] / sqrt(M_kappa)``.
    masses : ndarray
        ``(natom,)`` atomic masses (amu).
    """
    phonon.run_qpoints(np.asarray(qpoints, dtype=float), with_eigenvectors=with_eigenvectors)
    qp = phonon.qpoints

    frequencies = np.asarray(qp.frequencies)  # (nq, nmode)
    masses = np.asarray(phonon.primitive.masses, dtype=float)
    natom = len(masses)
    nmode = 3 * natom

    eigenvectors = None
    if with_eigenvectors:
        ev = np.asarray(qp.eigenvectors)  # (nq, 3*natom, nmode)
        eigenvectors = ev.reshape(frequencies.shape[0], natom, 3, nmode)

    return frequencies, eigenvectors, masses


def mode_displacement_pattern(eigvec_q, masses):
    """Mass-weighted Cartesian displacement pattern ``e_{kappa alpha,v}/sqrt(M)``.

    Parameters
    ----------
    eigvec_q : ndarray
        ``(natom, 3, nmode)`` phonon eigenvectors at one q.
    masses : ndarray
        ``(natom,)`` atomic masses.

    Returns
    -------
    ndarray
        ``(nmode, natom*3)`` per-mode displacement pattern (Cartesian index
        ``kappa*3 + alpha`` fastest).
    """
    eigvec_q = np.asarray(eigvec_q)
    natom, three, nmode = eigvec_q.shape
    inv_sqrt_m = 1.0 / np.sqrt(np.asarray(masses, dtype=float))
    pattern = eigvec_q * inv_sqrt_m[:, None, None]  # (natom, 3, nmode)
    return np.moveaxis(pattern.reshape(natom * three, nmode), 0, 1)  # (nmode, natom*3)


def frequency_prefactor(omega_q, units='THz', omega_floor_ev=1.0e-6):
    """``sqrt(1 / (2 omega))`` with sub-``omega_floor`` modes zeroed (acoustic Gamma)."""
    omega = np.asarray(omega_q, dtype=float)
    omega_ev = omega * THZ_TO_EV if str(units).lower() in ('thz', 'thz-1') else omega
    pref = np.zeros_like(omega_ev)
    mask = np.abs(omega_ev) > omega_floor_ev
    pref[mask] = np.sqrt(1.0 / (2.0 * np.abs(omega_ev[mask])))
    return pref


def assemble_g_kq(
    dHdu_kq,
    v_k,
    v_kq,
    eigvec_q,
    omega_q,
    masses,
    omega_units='THz',
    omega_floor_ev=1.0e-6,
):
    """Assemble the electron-phonon matrix ``g_mn^v`` for one ``(k, q)``.

    Parameters
    ----------
    dHdu_kq : ndarray
        ``(natom*3, nawf, nawf)`` Cartesian derivative of ``H`` in the PAO Bloch
        basis for this transition, ``dHdu_kq[kappa*3+alpha] = dH/du_{kappa alpha}``
        connecting ``k`` to ``k+q``.
    v_k, v_kq : ndarray
        ``(nawf, nbnd)`` electronic eigenvectors at ``k`` and ``k+q`` (columns are
        bands).
    eigvec_q : ndarray
        ``(natom, 3, nmode)`` phonon eigenvectors at ``q``.
    omega_q : ndarray
        ``(nmode,)`` phonon frequencies (in ``omega_units``).
    masses : ndarray
        ``(natom,)`` atomic masses (amu).

    Returns
    -------
    ndarray
        ``(nmode, nbnd_kq, nbnd_k)`` complex electron-phonon matrix elements.
    """
    dHdu_kq = np.asarray(dHdu_kq)
    v_k = np.asarray(v_k)
    v_kq = np.asarray(v_kq)

    if dHdu_kq.shape[0] != np.asarray(eigvec_q).shape[0] * 3:
        raise ValueError('dHdu_kq Cartesian dimension does not match 3 * natom.')

    # Rotate every Cartesian derivative into the electronic band basis.
    dH_band = np.einsum('ai,cab,bj->cij', v_kq.conj(), dHdu_kq, v_k)  # (ncart, nbnd_kq, nbnd_k)

    # Mass-weighted mode pattern and the sum over Cartesian displacements.
    pattern = mode_displacement_pattern(eigvec_q, masses)  # (nmode, ncart)
    dH_mode = np.einsum('vc,cij->vij', pattern, dH_band)  # (nmode, nbnd_kq, nbnd_k)

    pref = frequency_prefactor(omega_q, units=omega_units, omega_floor_ev=omega_floor_ev)
    return pref[:, None, None] * dH_mode
