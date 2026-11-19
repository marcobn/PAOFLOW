"""Static uniform spin susceptibility on the crystal grid (response.spin_response).

* a synthetic metallic channel: the GMRES solution equals the dense (1 - chi0 f_s)^-1 chi0 h, and without
  kernel the susceptibility is the Pauli term -sum_k w f';
* the pseudo-core density on the grid holds the charge of the radial rho_atc of every atom.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from PAOFLOW.hamiltonian.stoner import occupation, occupation_derivative
from PAOFLOW.response import coupled_crystal as cc
from PAOFLOW.response.coupled_response import HARTREE_EV
from PAOFLOW.response.spin_response import core_density_on_grid, solve_static_spin_response

pytestmark = pytest.mark.unit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PSEUDOS = os.path.join(ROOT, 'PSEUDOS', 'nc-sr-05_pbe_standard_upf')


def _metallic_channel(nk: int = 2, n: int = 4, width: float = 0.05, seed: int = 3):
    """Random bands crossing the Fermi level (0) with orthonormal periodic parts on a 3^3 grid."""
    rng = np.random.default_rng(seed)
    nkd = nk**3
    e = np.sort(rng.uniform(-0.15, 0.15, (nkd, n)), axis=1)
    U = np.stack(
        [
            np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
            for _ in range(nkd)
        ]
    )
    mill = cc.miller_sphere(np.zeros(3), np.eye(3), 2.0)
    coeff = [
        np.linalg.qr(
            rng.standard_normal((mill.shape[1], n)) + 1j * rng.standard_normal((mill.shape[1], n))
        )[0].T
        for _ in range(nkd)
    ]
    ch = cc.CrystalChannel(
        energies=e, eigenvectors=U, position=np.zeros((nkd, 3, n, n), complex),
        occupations=2.0 * occupation(e, 0.0, width, 'fermi-dirac'), weights=np.full(nkd, 1.0 / nkd),
        coeff=coeff, mill=[mill] * nkd, nr=cc.grid_for_sphere(mill), omega=50.0,
        at_bohr=50.0 ** (1 / 3) * np.eye(3),
    )  # fmt: skip
    ch.occupation_derivative = 2.0 * occupation_derivative(e, 0.0, width, 'fermi-dirac')
    return ch


def _dense_chi0(ch) -> np.ndarray:
    """chi0(r, r') of the channel: sum_k w sum_nm c_nm psi_n(r) psi_m*(r) psi_n*(r') psi_m(r') dv."""
    n_r = int(np.prod(ch.nr))
    dv = ch.omega / n_r
    f, fp, e = ch.occupations, ch.occupation_derivative, ch.energies
    chi0 = np.zeros((n_r, n_r), dtype=complex)
    for ik in range(ch.nk_local):
        psi = ch.periodic_table(ik) @ ch.eigenvectors[ik]  # (N_r, n)
        for a in range(ch.n):
            for b in range(ch.n):
                gap = e[ik, b] - e[ik, a]
                c = (
                    0.5 * (fp[ik, a] + fp[ik, b])
                    if abs(gap) < 1e-7
                    else (f[ik, b] - f[ik, a]) / gap
                )
                pair = psi[:, a] * psi[:, b].conj()
                chi0 += ch.weights[ik] * c * np.outer(pair, pair.conj()) * dv
    return np.real(chi0)


def test_grid_solution_equals_the_dense_solve():
    ch = _metallic_channel()
    n_r = int(np.prod(ch.nr))
    dv = ch.omega / n_r
    kernel = -1.0 * np.random.default_rng(7).uniform(
        0.5, 1.0, n_r
    )  # Hartree bohr^3, ferromagnetic sign
    chi0 = _dense_chi0(ch)
    bare = chi0 @ np.ones(n_r)
    dm = np.linalg.solve(np.eye(n_r) - chi0 * kernel[None, :], bare)
    out = solve_static_spin_response(ch, kernel, tol=1e-12)
    assert out['chi_bare'] == pytest.approx(-bare.sum() * dv / HARTREE_EV, rel=1e-10)
    assert out['chi'] == pytest.approx(-dm.sum() * dv / HARTREE_EV, rel=1e-8)
    assert out['enhancement'] > 1.0


def test_without_kernel_the_susceptibility_is_the_pauli_term():
    ch = _metallic_channel()
    out = solve_static_spin_response(ch, np.zeros(int(np.prod(ch.nr))))
    pauli = -np.sum(ch.weights[:, None] * ch.occupation_derivative) / HARTREE_EV
    assert out['chi'] == pytest.approx(pauli, rel=1e-10)
    assert out['chi_bare'] == pytest.approx(pauli, rel=1e-10)
    assert out['density_of_states'] == pytest.approx(0.5 * pauli, rel=1e-10)


class _Controller:
    def __init__(self, arrays, attributes):
        self.data_arrays, self.data_attributes = arrays, attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


@pytest.mark.skipif(not os.path.exists(os.path.join(PSEUDOS, 'Fe.upf')), reason='no PSEUDOS folder')
def test_core_density_holds_the_core_charge():
    from PAOFLOW.inputs.read_upf import UPF

    alat = 5.4169
    at = alat * np.eye(3)
    controller = _Controller(
        {'species': [('Fe', 'Fe.upf')], 'atoms': ['Fe', 'Fe'], 'tau': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) * alat},
        {'fpath': PSEUDOS},
    )  # fmt: skip
    nr = (48, 48, 48)
    core = core_density_on_grid(controller, nr, at)
    upf = UPF(os.path.join(PSEUDOS, 'Fe.upf'))
    r = np.asarray(upf.r)
    charge = 4 * np.pi * np.trapezoid(r**2 * np.asarray(upf.rho_atc)[: r.size], r)
    assert core.sum() * alat**3 / np.prod(nr) == pytest.approx(2 * charge, rel=1e-8)
    # the density peaks on the atoms and is the same on both (bcc: equivalent sites)
    box = core.reshape(nr)
    assert box[0, 0, 0] == pytest.approx(box[24, 24, 24], rel=1e-8)
    assert box[0, 0, 0] == pytest.approx(core.max(), rel=1e-12)
