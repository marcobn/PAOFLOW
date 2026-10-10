"""Unit tests for the phonon-assisted optical absorption of the PAO e-ph route."""

import numpy as np
import pytest

pytest.importorskip('scipy')

from PAOFLOW.elphon.elph_bloch import (
    RY_TO_EV,
    band_velocities,
    band_vertex,
    kq_index_map,
    precompute_dense_electrons,
    vertex_pao_R,
    vertex_ws_coefficients,
)
from PAOFLOW.elphon.phonon_assisted_absorption import (
    HBAR_C_EV_CM,
    absorption_coefficient,
    direct_absorption_kernel,
    indirect_absorption_kernel,
    write_absorption_outputs,
)

# --------------------------------------------------------------------------- #
# Exactly solvable two-site tight-binding model (simple cubic, alat = 1 bohr)
# --------------------------------------------------------------------------- #
TAU = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
NAWF = TAU.shape[0]
CUTOFF = 1.0  # every hopping fits strictly inside the Wigner-Seitz cell of a 4^3 grid


def _model_terms(seed):
    """Hoppings ``h_ij(R)`` (eV) and a vertex ``G_ij(R)`` between orbital i at 0 and j at R.

    ``h`` obeys ``h_ji(-R) = conj(h_ij(R))``; ``G`` (a k -> k+q operator) is generic.
    """
    rng = np.random.default_rng(seed)
    sym = rng.standard_normal((NAWF, NAWF))
    sym = sym + sym.T
    anti = rng.standard_normal((NAWF, NAWF))
    anti = anti + anti.T
    onsite = np.diag([-2.0, 1.5, -1.0, 2.5])
    vertex = rng.standard_normal((NAWF, NAWF)) + 1j * rng.standard_normal((NAWF, NAWF))
    hops, verts = [], []
    for cell in np.ndindex(5, 5, 5):
        R = np.array(cell) - 2
        for i in range(NAWF):
            for j in range(NAWF):
                d = R + TAU[j] - TAU[i]
                r2 = float(d @ d)
                if r2 > CUTOFF**2 + 1e-9:
                    continue
                envelope = np.exp(-r2)
                h = sym[i, j] * envelope + 1j * anti[i, j] * d[0] * envelope
                if not np.any(R) and i == j:
                    h = h + onsite[i, i]
                hops.append((i, j, R, h))
                verts.append((i, j, R, vertex[i, j] * envelope * (1.0 + 0.3 * d[1])))
    return hops, verts


def _bloch(terms, k_cryst):
    """``O_ij(k) = sum_R exp(2 pi i k.R) O_ij(R)`` (lattice gauge)."""
    out = np.zeros((NAWF, NAWF), dtype=complex)
    for i, j, R, value in terms:
        out[i, j] += np.exp(2j * np.pi * (k_cryst @ R)) * value
    return out


def _exact_velocity(hops, k_cryst):
    """Exact ``i [H, r]`` with the position operator diagonal on the orbital centres (eV bohr)."""
    out = np.zeros((3, NAWF, NAWF), dtype=complex)
    for i, j, R, value in hops:
        d = R + TAU[j] - TAU[i]
        out += (
            1j
            * d[:, None, None]
            * (np.exp(2j * np.pi * (k_cryst @ R)) * value)
            * (np.eye(NAWF)[i][:, None] * np.eye(NAWF)[j][None, :])
        )
    return out


def _paoflow_HRs(hops, ng):
    """PAOFLOW layout: ``H(k) = sum_n exp(-2 pi i k.n) HRs[..., n]``, i.e. ``HRs[n] = h(-n)``."""
    HRs = np.zeros((NAWF, NAWF, ng, ng, ng, 1), dtype=complex)
    for i, j, R, value in hops:
        n = (-R) % ng
        HRs[i, j, n[0], n[1], n[2], 0] += value
    return HRs


def _grid(n):
    axes = [np.arange(n) / n] * 3
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def test_velocity_and_vertex_phase_consistency_against_exact_model():
    """The full PAO chain reproduces the exact gauge-invariant |S1 + S2|^2 spectrum.

    Coarse 4^3 Kohn-Sham data with random band phases go through
    ``vertex_pao_R``; the dense band velocities and couplings at 8^3-grid points off
    the coarse grid must give the same phonon-assisted spectrum as the exact model,
    which fails if g and v were in mutually conjugated or shifted gauges.
    """
    ng, nk_dense = 4, 8
    hops, verts = _model_terms(seed=3)
    q = np.array([0.25, 0.0, 0.5])
    rng = np.random.default_rng(7)

    kpts = _grid(ng)
    states = []
    for k in kpts:
        e, c = np.linalg.eigh(_bloch(hops, k))
        states.append(c * np.exp(2j * np.pi * rng.random(NAWF))[None, :])  # random phases
    states = np.array(states)  # (nk, orbital, band)
    ikq, _ = kq_index_map(kpts, q, (ng, ng, ng))
    d = np.einsum(
        'kim,kij,kjn->kmn', states[ikq].conj(), np.array([_bloch(verts, k) for k in kpts]), states
    )[..., None]  # (nk, m, n, ncart = 1)
    A = np.transpose(states, (2, 1, 0))  # A_{n i}(k) = <phi_i|psi_n> = c_{i n}
    kgrid_idx = np.round(kpts * ng).astype(int) % ng
    gR = vertex_pao_R(d, A, ikq, kgrid_idx, (ng, ng, ng))

    electrons = precompute_dense_electrons(
        _paoflow_HRs(hops, ng), np.eye(3), nk_dense, [0.01], None, (ng, ng, ng),
        orbital_positions=TAU, velocities=True, alat=1.0,
    )  # fmt: skip
    K = electrons['K']
    shift = np.round(q * nk_dense).astype(int)
    idx3 = np.round(K * nk_dense).astype(int)
    ikq_dense = np.ravel_multi_index(((idx3 + shift) % nk_dense).T, (nk_dense,) * 3)
    sel = np.array([1, 7, 30, 77, 200, 311, 450])  # off the coarse 4^3 grid
    V = electrons['V']
    coupling = band_vertex(
        vertex_ws_coefficients(gR, electrons), electrons['phkg'][sel], V[ikq_dense[sel]],
        V[sel], np.ones((1, 1)),
    )  # fmt: skip
    vk = band_velocities(electrons, sel)
    vkq = band_velocities(electrons, ikq_dense[sel])

    # exact quantities in an independent eigenvector gauge
    e_ex, e_ex_q, v_ex, v_ex_q, g_ex = [], [], [], [], []
    for k in K[sel]:
        ek, ck = np.linalg.eigh(_bloch(hops, k))
        ekq, ckq = np.linalg.eigh(_bloch(hops, k + q))
        e_ex.append(ek / RY_TO_EV)
        e_ex_q.append(ekq / RY_TO_EV)
        v_ex.append(np.einsum('im,aij,jn->amn', ck.conj(), _exact_velocity(hops, k), ck))
        v_ex_q.append(np.einsum('im,aij,jn->amn', ckq.conj(), _exact_velocity(hops, k + q), ckq))
        g_ex.append((ckq.conj().T @ _bloch(verts, k) @ ck)[None])
    np.testing.assert_allclose(electrons['E'][sel], np.array(e_ex), atol=1e-10)

    def spectrum(ek, ekq, v1, v2, g):
        return indirect_absorption_kernel(
            ek, ekq, v1, v2, g, np.array([0.02]), [0.01], np.linspace(0.05, 0.6, 12),
            [0.005, 0.05], 0.03, 10.0, 1e-6, occupation_tol=0.0,
        )  # fmt: skip

    ef = 0.0
    ours = spectrum(
        electrons['E'][sel] - ef, electrons['E'][ikq_dense[sel]] - ef, vk, vkq, coupling
    )
    exact = spectrum(
        np.array(e_ex) - ef, np.array(e_ex_q) - ef, np.array(v_ex) / RY_TO_EV,
        np.array(v_ex_q) / RY_TO_EV, np.array(g_ex),
    )  # fmt: skip
    assert np.max(exact[0]) > 0.0
    np.testing.assert_allclose(ours[0], exact[0], rtol=1e-7, atol=1e-12 * np.max(exact[0]))
    np.testing.assert_allclose(ours[1], exact[1], rtol=1e-7, atol=1e-12 * np.max(exact[1]))


def test_band_velocities_finite_difference_and_correction():
    """Without orbital separation, v is dH/dk; a PAO-basis correction is added before rotation."""
    hops, _ = _model_terms(seed=5)
    single_site = [(i, j, R, h) for i, j, R, h in hops if i < 2 and j < 2]
    HRs = _paoflow_HRs(single_site, 4)[:2, :2]
    electrons = precompute_dense_electrons(
        HRs, np.eye(3), 6, [0.01], None, (4, 4, 4), velocities=True, alat=1.0
    )
    sel = np.array([0, 5, 43, 100])
    v = band_velocities(electrons, sel)
    step = 1e-5
    for b, ik in enumerate(sel):
        k = electrons['K'][ik]
        Vk = electrons['V'][ik]
        for a in range(3):
            dk = np.zeros(3)
            dk[a] = step / (2.0 * np.pi)  # k_cart = 2 pi k_cryst for alat = 1, at = I
            H_plus = _bloch([(i, j, R, h) for i, j, R, h in single_site], k + dk)[:2, :2]
            H_minus = _bloch([(i, j, R, h) for i, j, R, h in single_site], k - dk)[:2, :2]
            fd = Vk.conj().T @ ((H_plus - H_minus) / (2.0 * step)) @ Vk / RY_TO_EV
            np.testing.assert_allclose(v[b, a], fd, atol=1e-8)
    extra = np.zeros((3, 2, 2), dtype=complex)
    extra[0] = [[0.2, 0.1j], [-0.1j, -0.3]]
    corrected = band_velocities(
        electrons, sel, nl_correction=lambda K: np.repeat(extra[None], len(K), 0)
    )
    for b, ik in enumerate(sel):
        Vk = electrons['V'][ik]
        np.testing.assert_allclose(
            corrected[b, 0] - v[b, 0], Vk.conj().T @ extra[0] @ Vk, atol=1e-12
        )
        np.testing.assert_allclose(corrected[b], np.conj(np.transpose(corrected[b], (0, 2, 1))))


# --------------------------------------------------------------------------- #
# Literal transliteration of EPW indabs.f90 / dirabs (reference loops)
# --------------------------------------------------------------------------- #
def _w0gauss(x):
    return np.exp(-(x**2)) / np.sqrt(np.pi)


def _wgauss_fd(x):
    return 1.0 / (1.0 + np.exp(-x))


def _epw_indabs(ek, ekq, vk, vkq, epf, wq, temps, omegas, etas, degaussw, fsthick, eps_ac):
    nk, nb = ek.shape
    nmodes = wq.size
    eps = np.zeros((len(temps), len(etas), 3, len(omegas)))
    epsl = np.zeros_like(eps)
    for it, T in enumerate(temps):
        nqv = np.zeros(nmodes)
        for im in range(nmodes):
            if wq[im] > eps_ac:
                f = _wgauss_fd(-wq[im] / T)
                nqv[im] = f / (1.0 - 2.0 * f)
        for ik in range(nk):
            for ib in range(nb):
                ekk = ek[ik, ib]
                if abs(ekk) >= fsthick:
                    continue
                wgkk = _wgauss_fd(-ekk / T)
                for jb in range(nb):
                    ekq_ = ekq[ik, jb]
                    if not (
                        abs(ekq_) < fsthick and ekq_ < ekk + wq[-1] + omegas[-1] + 6 * degaussw
                    ):
                        continue
                    wgkq = _wgauss_fd(-ekq_ / T)
                    if ekq_ - ekk - wq[-1] - omegas[-1] > 6 * degaussw:
                        continue
                    if ekq_ - ekk + wq[-1] - omegas[0] < -6 * degaussw:
                        continue
                    for im in range(nmodes):
                        if wq[im] <= eps_ac:
                            continue
                        for m, eta in enumerate(etas):
                            s1a = s1e = s2a = s2e = np.zeros(3, dtype=complex)
                            for mb in range(nb):
                                ekmk, ekmq = ek[ik, mb], ekq[ik, mb]
                                s1a = s1a + epf[ik, im, jb, mb] * vk[ik, :, mb, ib] / (
                                    ekmk - ekq_ + wq[im] + 1j * eta
                                )
                                s1e = s1e + epf[ik, im, jb, mb] * vk[ik, :, mb, ib] / (
                                    ekmk - ekq_ - wq[im] + 1j * eta
                                )
                                s2a = s2a + epf[ik, im, mb, ib] * vkq[ik, :, jb, mb] / (
                                    ekmq - ekk - wq[im] + 1j * eta
                                )
                                s2e = s2e + epf[ik, im, mb, ib] * vkq[ik, :, jb, mb] / (
                                    ekmq - ekk + wq[im] + 1j * eta
                                )
                            pfac = nqv[im] * wgkk * (1 - wgkq) - (nqv[im] + 1) * (1 - wgkk) * wgkq
                            pface = (nqv[im] + 1) * wgkk * (1 - wgkq) - nqv[im] * (1 - wgkk) * wgkq
                            for iw, w in enumerate(omegas):
                                if (
                                    abs(ekq_ - ekk - wq[im] - w) > 6 * degaussw
                                    and abs(ekq_ - ekk + wq[im] - w) > 6 * degaussw
                                ):
                                    continue
                                we = _w0gauss((ekq_ - ekk - w + wq[im]) / degaussw) / degaussw
                                wa = _w0gauss((ekq_ - ekk - w - wq[im]) / degaussw) / degaussw
                                la = (
                                    degaussw
                                    / (degaussw**2 + (ekq_ - ekk - w - wq[im]) ** 2)
                                    / np.pi
                                )
                                le = (
                                    degaussw
                                    / (degaussw**2 + (ekq_ - ekk - w + wq[im]) ** 2)
                                    / np.pi
                                )
                                c = 16 * np.pi**2 / w**2 / (2 * wq[im])
                                eps[it, m, :, iw] += c * (
                                    pfac * wa * abs(s1a + s2a) ** 2
                                    + pface * we * abs(s1e + s2e) ** 2
                                )
                                epsl[it, m, :, iw] += c * (
                                    pfac * la * abs(s1a + s2a) ** 2
                                    + pface * le * abs(s1e + s2e) ** 2
                                )
    return eps, epsl


def _random_case(seed, nk=2, nb=4, nmode=2):
    rng = np.random.default_rng(seed)
    ek = np.sort(rng.uniform(-0.06, 0.06, (nk, nb)), axis=1)
    ekq = np.sort(rng.uniform(-0.06, 0.06, (nk, nb)), axis=1)

    def herm(shape):
        a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))

    vk, vkq = herm((nk, 3, nb, nb)), herm((nk, 3, nb, nb))
    epf = rng.standard_normal((nk, nmode, nb, nb)) + 1j * rng.standard_normal((nk, nmode, nb, nb))
    wq = np.array([0.002, 0.004])[:nmode]
    return ek, ekq, vk, vkq, epf, wq


@pytest.mark.parametrize('seed', [0, 1])
def test_indirect_kernel_matches_epw_loops(seed):
    ek, ekq, vk, vkq, epf, wq = _random_case(seed)
    temps, etas = [0.002, 0.004], [0.001, 0.01]
    omegas = np.linspace(0.01, 0.12, 15)
    degauss, fsthick = 0.004, 0.055
    ref = _epw_indabs(ek, ekq, vk, vkq, epf, wq, temps, omegas, etas, degauss, fsthick, 1e-5)
    got = indirect_absorption_kernel(
        ek, ekq, vk, vkq, epf, wq, temps, omegas, etas, degauss, fsthick, 1e-5,
        occupation_tol=0.0, pair_block=3,
    )  # fmt: skip
    assert np.max(ref[0]) > 0.0
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-10, atol=1e-14 * np.max(ref[0]))
    np.testing.assert_allclose(got[1], ref[1], rtol=1e-10, atol=1e-14 * np.max(ref[1]))


def test_indirect_kernel_skips_acoustic_modes():
    ek, ekq, vk, vkq, epf, wq = _random_case(4)
    args = ([0.003], np.linspace(0.01, 0.1, 5), [0.01], 0.004, 0.1)
    full = indirect_absorption_kernel(ek, ekq, vk, vkq, epf, wq, *args, 1e-5)[0]
    soft = indirect_absorption_kernel(ek, ekq, vk, vkq, epf, np.array([1e-7, wq[1]]), *args, 1e-5)[
        0
    ]
    only_second = indirect_absorption_kernel(ek, ekq, vk, vkq, epf[:, 1:], wq[1:], *args, 1e-5)[0]
    np.testing.assert_allclose(soft, only_second, rtol=1e-12)
    assert not np.allclose(full, only_second)


def test_direct_kernel_matches_epw_loops():
    ek, _, vk, _, _, _ = _random_case(2, nk=3, nb=5)
    temps, omegas, degauss, fsthick = [0.002], np.linspace(0.01, 0.12, 12), 0.004, 0.055
    ref = np.zeros((1, 3, omegas.size))
    for ik in range(ek.shape[0]):
        for ib in range(ek.shape[1]):
            if abs(ek[ik, ib]) >= fsthick:
                continue
            for jb in range(ek.shape[1]):
                if abs(ek[ik, jb]) >= fsthick:
                    continue
                pfac = _wgauss_fd(-ek[ik, ib] / temps[0]) - _wgauss_fd(-ek[ik, jb] / temps[0])
                for iw, w in enumerate(omegas):
                    x = ek[ik, jb] - ek[ik, ib] - w
                    if abs(x) > 6 * degauss:
                        continue
                    ref[0, :, iw] += (
                        16
                        * np.pi**2
                        / w**2
                        * pfac
                        * _w0gauss(x / degauss)
                        / degauss
                        * np.abs(vk[ik, :, jb, ib]) ** 2
                    )
    got = direct_absorption_kernel(ek, vk, temps, omegas, degauss, fsthick)[0]
    assert np.max(ref) > 0.0
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-14 * np.max(ref))


def test_absorption_coefficient_and_outputs(tmp_path):
    omega = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        absorption_coefficient(omega, np.array([0.5, 1.0]), 2.0),
        omega * np.array([0.5, 1.0]) / (2.0 * HBAR_C_EV_CM),
    )
    results = {
        'omega_ev': omega,
        'eps2_indirect': np.ones((1, 2, 3, 2)),
        'eps2_indirect_lorentz': np.ones((1, 2, 3, 2)),
        'eps2_direct': np.ones((1, 3, 2)),
        'eps2_direct_lorentz': np.ones((1, 3, 2)),
        'alpha_cm': np.ones((1, 2, 2)),
        'etas_ev': np.array([0.01, 0.1]),
        'temps_k': np.array([300.0]),
    }
    written = write_absorption_outputs(results, str(tmp_path))
    names = sorted(p.split('/')[-1] for p in written)
    assert names == sorted(
        [
            'epsilon2_indabs_300.0K.dat',
            'epsilon2_indabs_lorenz300.0K.dat',
            'epsilon2_dirabs_300.0K.dat',
            'alpha_300.0K.dat',
            'absorption.npz',
        ]
    )
    assert np.loadtxt(tmp_path / 'epsilon2_indabs_300.0K.dat').shape == (2, 3)
    assert np.loadtxt(tmp_path / 'epsilon2_dirabs_300.0K.dat').shape == (2, 9)
