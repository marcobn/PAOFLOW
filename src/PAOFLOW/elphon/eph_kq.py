"""Bloch-basis electron-phonon matrix ``g_mn^v(k, q)`` and Eliashberg ``a2F``/``lambda`` (P3/P4).

The real-space tensor ``g_R`` from :func:`PAOFLOW.elphon.do_gkq.assemble_eph_tensor`
is Fourier-transformed to the electron momentum ``k`` and the (commensurate)
phonon momentum ``q``, projected onto the primitive PAO Bloch states and combined
with the phonon polarisations to give

.. math::

    g_{mn}^{v}(k, q) = \\sum_{\\kappa\\alpha}
        \\sqrt{\\tfrac{\\hbar}{2 M_\\kappa \\omega_{qv}}}\\,
        e_{\\kappa\\alpha}^{v}(q)\\,
        \\langle m, k+q | \\partial H / \\partial u_{\\kappa\\alpha} | n, k \\rangle .

Both Fourier transforms use the folded ``fftfreq`` real-space grid convention of
PAOFLOW (:func:`PAOFLOW.utils.get_R_grid_fft.get_R_grid_fft`), so the electronic
states (from an ``ifftn`` of the primitive ``HRs``) and ``dH/du`` share the same
k-grid and the ``k+q`` transition is a pure index shift.  The 2x2x2 supercell
supports the ``N_p = 8`` commensurate q-points; ``k`` runs over the full primitive
FFT grid (22^3 for Al).
"""

import numpy as np

# Physical constants (SI) for the zero-point displacement amplitude.
_HBAR_JS = 1.054571817e-34
_AMU_KG = 1.66053906660e-27
_BOHR_M = 5.29177210903e-11
# 1 THz -> eV.
THZ_TO_EV = 4.135667696e-3


def bloch_hamiltonian(HR):
    """``H(k)`` on the native FFT grid from a folded-``fftfreq`` real-space ``HR``.

    ``H(k) = sum_R H(R) e^{2 pi i k.R} = N * ifftn(H(R))`` (the fftfreq folding
    leaves the phase unchanged).  ``HR`` has shape
    ``(nawf, nawf, n1, n2, n3, nspin)``; the returned ``H(k)`` is indexed by the
    integer k-grid point ``m`` with ``k_frac = m / N``.
    """
    HR = np.asarray(HR)
    n1, n2, n3 = HR.shape[2:5]
    return np.fft.ifftn(HR, axes=(2, 3, 4)) * (n1 * n2 * n3)


def primitive_eigenstates(HR):
    """Diagonalise ``H(k)`` on the primitive FFT grid.

    Returns
    -------
    E : ndarray, shape ``(n1, n2, n3, nspin, nawf)``
        Ascending eigenvalues (eV).
    V : ndarray, shape ``(n1, n2, n3, nspin, nawf, nawf)``
        Eigenvectors as columns (``V[..., :, b]`` is band ``b``).
    """
    Hk = bloch_hamiltonian(HR)  # (nawf, nawf, n1, n2, n3, nspin)
    H = np.moveaxis(Hk, (0, 1), (-2, -1))  # (n1, n2, n3, nspin, nawf, nawf)
    E, V = np.linalg.eigh(H, UPLO='U')
    return E, V


def fourier_dHdu(g_R):
    """Fourier transform the real-space e-ph tensor to ``dH/du(k, q)``.

    ``g_R`` has shape ``(ncart, nawf, nawf, e1, e2, e3, p1, p2, p3, nspin)`` with
    the electron cells ``R_e`` on axes 3-5 and the phonon cells ``R_p`` on axes
    6-8 (both folded-``fftfreq``).  Returns ``dHdu`` of shape
    ``(ncart, nawf, nawf, k1, k2, k3, q1, q2, q3, nspin)`` with
    ``k_frac = m_k / N_e`` and ``q_frac = m_q / N_p``.
    """
    g_R = np.asarray(g_R)
    ne = int(np.prod(g_R.shape[3:6]))
    npq = int(np.prod(g_R.shape[6:9]))
    d = np.fft.ifftn(g_R, axes=(3, 4, 5)) * ne
    d = np.fft.ifftn(d, axes=(6, 7, 8)) * npq
    return d


def _embed_fftfreq(A, axes, new_sizes):
    """Zero-pad a folded-``fftfreq`` array to larger axis sizes (Fourier interpolation).

    Each listed axis of ``A`` is indexed in ``numpy`` ``fftfreq`` order; the
    entries are placed at the matching frequencies of a larger, zero-filled axis.
    Transforming the padded array to k-space then interpolates onto a denser
    k-grid (the standard band-limited Wannier/PAO interpolation).
    """
    for ax, nt in zip(axes, new_sizes):
        nr = A.shape[ax]
        if nt == nr:
            continue
        if nt < nr:
            raise ValueError('target size %d smaller than source %d on axis %d' % (nt, nr, ax))
        npos = (nr + 1) // 2  # non-negative frequencies: indices 0 .. npos-1
        nneg = nr // 2  # negative frequencies: last nneg indices
        shape = list(A.shape)
        shape[ax] = nt
        B = np.zeros(shape, dtype=A.dtype)
        src_pos = [slice(None)] * A.ndim
        src_pos[ax] = slice(0, npos)
        dst_pos = [slice(None)] * A.ndim
        dst_pos[ax] = slice(0, npos)
        B[tuple(dst_pos)] = A[tuple(src_pos)]
        if nneg:
            src_neg = [slice(None)] * A.ndim
            src_neg[ax] = slice(nr - nneg, nr)
            dst_neg = [slice(None)] * A.ndim
            dst_neg[ax] = slice(nt - nneg, nt)
            B[tuple(dst_neg)] = A[tuple(src_neg)]
        A = B
    return A


def estates_on_grid(HR, nk):
    """Diagonalise ``H(k)`` on an ``nk^3`` grid, Fourier-interpolated from ``HR``.

    ``HR`` (native grid ``n^3``) is zero-padded in R-space to ``nk^3`` and
    transformed, giving band-limited PAO interpolation onto the denser grid.

    Returns
    -------
    E : ndarray, shape ``(nk, nk, nk, nspin, nawf)``
    V : ndarray, shape ``(nk, nk, nk, nspin, nawf, nawf)``
    """
    HR = np.asarray(HR)
    HRp = _embed_fftfreq(HR, (2, 3, 4), (nk, nk, nk))
    Hk = np.fft.ifftn(HRp, axes=(2, 3, 4)) * (nk**3)
    H = np.moveaxis(Hk, (0, 1), (-2, -1))  # (nk, nk, nk, nspin, nawf, nawf)
    E, V = np.linalg.eigh(H, UPLO='U')
    return E, V


def fourier_dHdu_on_grid(g_R, nk):
    """``dH/du(k, q)`` with the electron k-grid Fourier-interpolated to ``nk^3``.

    The electron cells ``R_e`` (axes 3-5) are zero-padded to ``nk`` before the
    transform (k-interpolation); the phonon cells ``R_p`` (axes 6-8) are
    transformed on their native grid, giving the ``S^3`` commensurate q-points.
    Returns ``dHdu`` of shape ``(ncart, nawf, nawf, nk, nk, nk, q1, q2, q3, nspin)``.
    """
    g_R = np.asarray(g_R)
    ge = _embed_fftfreq(g_R, (3, 4, 5), (nk, nk, nk))
    d = np.fft.ifftn(ge, axes=(3, 4, 5)) * (nk**3)
    npq = int(np.prod(g_R.shape[6:9]))
    d = np.fft.ifftn(d, axes=(6, 7, 8)) * npq
    return d


def zero_point_amplitude(mass_amu, omega_thz, omega_floor_thz=1.0e-3):
    """Zero-point displacement amplitude ``sqrt(hbar / (2 M omega))`` in Bohr.

    ``mass_amu`` is a scalar (or array) atomic mass in amu, ``omega_thz`` the mode
    frequency in THz.  Modes below ``omega_floor_thz`` (acoustic Gamma) return 0.
    """
    mass_amu, omega_thz = np.broadcast_arrays(
        np.asarray(mass_amu, dtype=float), np.asarray(omega_thz, dtype=float)
    )
    omega_rad = 2.0 * np.pi * np.abs(omega_thz) * 1.0e12
    m_kg = mass_amu * _AMU_KG
    mask = np.abs(omega_thz) > omega_floor_thz
    denom = np.where(mask, 2.0 * m_kg * omega_rad, 1.0)
    a_m = np.sqrt(_HBAR_JS / denom)
    return np.where(mask, a_m / _BOHR_M, 0.0)


def _commensurate_qgrid(sq):
    """List of commensurate q-points ``(m/sq)`` and their integer indices."""
    q_frac, q_idx = [], []
    for i in range(sq[0]):
        for j in range(sq[1]):
            for k in range(sq[2]):
                q_frac.append((i / sq[0], j / sq[1], k / sq[2]))
                q_idx.append((i, j, k))
    return np.array(q_frac, dtype=float), q_idx


def assemble_g_bloch(dHdu_kq, v_k, v_kq, eigvec_q, omega_thz, masses):
    """Physical ``g_mn^v`` (eV) for one ``(k, q)`` transition.

    Parameters
    ----------
    dHdu_kq : ndarray, ``(ncart, nawf, nawf)``
        ``dH/du_{kappa alpha}`` (eV/Bohr) connecting ``k`` and ``k+q``.
    v_k, v_kq : ndarray, ``(nawf, nbnd)``
        Electronic eigenvectors (columns) at ``k`` and ``k+q``.
    eigvec_q : ndarray, ``(natom, 3, nmode)``
        Phonon polarisation vectors at ``q``.
    omega_thz : ndarray, ``(nmode,)``
        Phonon frequencies (THz).
    masses : ndarray, ``(natom,)``
        Atomic masses (amu).

    Returns
    -------
    ndarray, ``(nmode, nbnd_kq, nbnd_k)`` complex
        ``g_mn^v(k, q)`` in eV.
    """
    dHdu_kq = np.asarray(dHdu_kq)
    v_k = np.asarray(v_k)
    v_kq = np.asarray(v_kq)
    eigvec_q = np.asarray(eigvec_q)
    natom, three, nmode = eigvec_q.shape

    # Cartesian derivatives in the electronic band basis: (ncart, nbnd_kq, nbnd_k).
    dH_band = np.einsum('ai,cab,bj->cij', v_kq.conj(), dHdu_kq, v_k)

    # Zero-point amplitude sqrt(hbar/2 M omega) (Bohr), per (atom, mode).
    amp = zero_point_amplitude(masses[:, None], omega_thz[None, :])  # (natom, nmode)
    # Cartesian displacement pattern per mode weighted by the amplitude: (nmode, ncart).
    pattern = eigvec_q * amp[:, None, :]  # (natom, 3, nmode)
    pattern = np.moveaxis(pattern.reshape(natom * three, nmode), 0, 1)  # (nmode, ncart)

    return np.einsum('vc,cij->vij', pattern, dH_band)  # (nmode, nbnd_kq, nbnd_k)


def _gaussian_delta(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def eliashberg(
    data_controller,
    g_R,
    phonon,
    ispin=0,
    smearing_ev=0.30,
    nk_electron=None,
    nomega=400,
    omega_pad=1.2,
    return_gkq=False,
):
    """Assemble ``g_mn^v(k, q)`` and the isotropic ``a2F(omega)`` / ``lambda``.

    Uses the primitive PAO Hamiltonian (``arry['HRs']``) for the electronic
    states and a phonopy object carrying the second-order force constants for the
    phonon frequencies / eigenvectors at the commensurate q-points.

    Parameters
    ----------
    nk_electron : int, optional
        Electronic k-grid size (``nk_electron^3``); the electronic states and
        ``dH/du`` are Fourier-interpolated to it, decoupling the k-sampling from
        the supercell's ``R_e`` grid.  Must be divisible by every supercell size.
        Defaults to the native ``HRs`` grid.
    return_gkq : bool, optional
        If ``True``, also return the full ``g_kq`` array (``(nk, nq, nmode, nawf,
        nawf)``).  Off by default -- at large grids this is many GB.

    Returns
    -------
    dict
        ``{'omega', 'a2F', 'lambda', 'lambda_qv', 'N_EF', 'q_frac', 'omega_q',
        'gamma_acoustic', 'nk_electron'[, 'g_kq']}``.  ``gamma_acoustic`` is
        ``max |g|`` over the three acoustic modes at ``q = Gamma`` (a small value
        confirms the acoustic sum rule).
    """
    arry, attr = data_controller.data_dicts()
    HR = np.asarray(arry['HRs'])
    nawf = HR.shape[0]
    EF = float(attr['Efermi'])

    sq = tuple(int(s) for s in g_R.shape[6:9])
    Nk = int(nk_electron) if nk_electron is not None else int(HR.shape[2])
    for s in sq:
        if Nk % s != 0:
            raise ValueError(
                'nk_electron=%d must be divisible by every supercell size %s.' % (Nk, sq)
            )
    nk = Nk**3
    shift = (Nk // sq[0], Nk // sq[1], Nk // sq[2])

    E, V = estates_on_grid(HR, Nk)  # (Nk,Nk,Nk,nspin,nawf), (...,nawf,nawf)
    dHdu = fourier_dHdu_on_grid(g_R, Nk)  # (ncart,nawf,nawf,Nk,Nk,Nk,q1,q2,q3,nspin)
    ncart = dHdu.shape[0]

    q_frac, q_idx = _commensurate_qgrid(sq)
    nq = len(q_idx)

    from .gkq import phonon_modes

    freqs, eigvecs, masses = phonon_modes(phonon, q_frac)  # (nq,nmode),(nq,natom,3,nmode)
    nmode = freqs.shape[1]
    natom = masses.shape[0]

    delta_grid = _gaussian_delta(E[:, :, :, ispin, :] - EF, smearing_ev)  # (Nk,Nk,Nk,nawf)
    N_EF = float(delta_grid.sum() / nk)  # DOS at E_F per spin (1/eV)
    delta_flat = delta_grid.reshape(nk, nawf)

    Vgrid = V[:, :, :, ispin, :, :]  # (Nk,Nk,Nk,nawf,nawf)
    Vk_flat = Vgrid.reshape(nk, nawf, nawf)

    lam_qv = np.zeros((nq, nmode), dtype=float)
    omega_all_ev = freqs * THZ_TO_EV
    gamma_acoustic = 0.0
    g_kq = np.zeros((nk, nq, nmode, nawf, nawf), dtype=complex) if return_gkq else None

    for iq, (qi, qj, qk) in enumerate(q_idx):
        roll = (-qi * shift[0], -qj * shift[1], -qk * shift[2])
        Vkq_flat = np.roll(Vgrid, roll, axis=(0, 1, 2)).reshape(nk, nawf, nawf)
        dkq_flat = np.roll(delta_grid, roll, axis=(0, 1, 2)).reshape(nk, nawf)

        # dH/du at this q, over the full k-grid: (nk, ncart, nawf, nawf).
        dHq = np.moveaxis(
            dHdu[:, :, :, :, :, :, qi, qj, qk, ispin].reshape(ncart, nawf, nawf, nk), 3, 0
        )
        # Rotate into the electronic band basis (batched over k).
        tmp = np.einsum('kai,kcab->kcib', Vkq_flat.conj(), dHq)
        dH_band = np.einsum('kcib,kbj->kcij', tmp, Vk_flat)  # (nk, ncart, nbnd_kq, nbnd_k)

        # Mass-weighted, amplitude-scaled phonon pattern: (nmode, ncart).
        amp = zero_point_amplitude(masses[:, None], freqs[iq][None, :])  # (natom, nmode)
        pattern = np.moveaxis((eigvecs[iq] * amp[:, None, :]).reshape(natom * 3, nmode), 0, 1)
        g = np.einsum('vc,kcij->kvij', pattern, dH_band)  # (nk, nmode, nbnd_kq, nbnd_k)

        if return_gkq:
            g_kq[:, iq] = g

        # Fermi-surface double-delta: sum_{k,m,n} |g|^2 delta(e_{k+q,m}) delta(e_{k,n}).
        w = np.abs(g) ** 2  # (nk, nmode, m, n)
        num = np.einsum('kvmn,km,kn->v', w, dkq_flat, delta_flat)  # (nmode,)
        oev = omega_all_ev[iq]
        safe = oev > 1.0e-6
        lam_qv[iq, safe] = (2.0 / (N_EF * nk)) * num[safe] / oev[safe]

        if (qi, qj, qk) == (0, 0, 0):
            gamma_acoustic = float(np.max(np.abs(g[:, 0:3])))

    lam = float(lam_qv.sum() / nq)

    # a2F(omega): distribute lambda_qv * omega_qv / 2 as delta(omega - omega_qv).
    wmax = float(omega_all_ev.max()) * omega_pad
    omega = np.linspace(0.0, max(wmax, 1e-3), nomega)
    a2F = np.zeros(nomega)
    sig_w = 0.05 * max(omega_all_ev.max(), 1e-3)
    for iq in range(nq):
        for v in range(nmode):
            if omega_all_ev[iq, v] <= 1.0e-6:
                continue
            weight = 0.5 * lam_qv[iq, v] * omega_all_ev[iq, v]
            a2F += weight * _gaussian_delta(omega - omega_all_ev[iq, v], sig_w)
    a2F /= nq

    out = {
        'omega': omega,
        'a2F': a2F,
        'lambda': lam,
        'lambda_qv': lam_qv,
        'N_EF': N_EF,
        'q_frac': q_frac,
        'omega_q': freqs,
        'gamma_acoustic': gamma_acoustic,
        'nk_electron': Nk,
    }
    if return_gkq:
        out['g_kq'] = g_kq
    return out
