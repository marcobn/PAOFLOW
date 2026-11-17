"""Self-consistent spin response: the LSDA spin kernel, the Stoner integrals and the Stoner mean field.

* the spin kernel is the second derivative of the full PW92 LSDA energy in the magnetization;
* the smeared occupations and their derivatives agree;
* a one-band cubic model: chi = 2N / (1 - I N) (spinor and collinear), the ferromagnetic solution beyond
  I N = 1 is self-consistent and stable, and the linear mode gives the negative (unstable) chi;
* the Stoner integral of a Gaussian shell in a uniform density has its closed form;
* the PAOFLOW methods on a model.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from PAOFLOW.basis_gen.xc import lsda_spin_kernel
from PAOFLOW.hamiltonian.stoner import (
    HARTREE_EV,
    chemical_potential,
    occupation,
    occupation_derivative,
    solve_spin_response,
    stoner_matrix,
)

pytestmark = pytest.mark.unit

# PW92 (Perdew & Wang, PRB 45, 13244 (1992)): eps_c(rs, 0), eps_c(rs, 1) and -alpha_c(rs)
_PARAMETERS = (
    (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
)


def _g(rs, A, a1, b1, b2, b3, b4):
    q1 = 2.0 * A * (b1 * np.sqrt(rs) + b2 * rs + b3 * rs**1.5 + b4 * rs**2)
    return -2.0 * A * (1.0 + a1 * rs) * np.log(1.0 + 1.0 / q1)


def _lsda_energy_density(n_up: float, n_down: float) -> float:
    """n eps_xc(n, zeta), Hartree bohr^-3: Slater exchange and PW92 correlation."""
    n = n_up + n_down
    zeta = (n_up - n_down) / n
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    eps_x = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    eps_x *= 0.5 * ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3))
    fz = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2.0) / (2 ** (4 / 3) - 2.0)
    ec0, ec1, minus_alpha = (_g(rs, *p) for p in _PARAMETERS)
    eps_c = ec0 - minus_alpha * fz / 1.709921 * (1 - zeta**4) + (ec1 - ec0) * fz * zeta**4
    return n * (eps_x + eps_c)


@pytest.mark.parametrize('n', [1.0e-3, 1.0e-2, 0.1, 1.0])
def test_spin_kernel_is_the_second_derivative_of_the_lsda_energy(n):
    dm = 1.0e-4 * n
    second = (
        _lsda_energy_density(0.5 * (n + dm), 0.5 * (n - dm))
        - 2.0 * _lsda_energy_density(0.5 * n, 0.5 * n)
        + _lsda_energy_density(0.5 * (n - dm), 0.5 * (n + dm))
    ) / dm**2
    assert lsda_spin_kernel(np.array([n]))[0] == pytest.approx(second, rel=1e-5)


@pytest.mark.parametrize('kind', ['marzari-vanderbilt', 'gaussian', 'fermi-dirac'])
def test_occupation_derivative_matches_the_occupations(kind):
    e = np.linspace(-1.0, 1.0, 41)
    step = 1.0e-6
    numeric = (occupation(e + step, 0.1, 0.2, kind) - occupation(e - step, 0.1, 0.2, kind)) / (
        2 * step
    )
    np.testing.assert_allclose(occupation_derivative(e, 0.1, 0.2, kind), numeric, atol=1e-7)


# --------------------------------------------------------------------------- one-band cubic model
NK, WIDTH, ELECTRONS = 16, 0.1, 0.8  # electrons per cell (both spins)


def _band() -> np.ndarray:
    k = 2 * np.pi * np.arange(NK) / NK
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    return (-2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz))).reshape(-1)


def _density_of_states() -> float:
    e = _band()[:, None]
    mu = chemical_potential(e, 0.5 * ELECTRONS, WIDTH, 'fermi-dirac')
    return -float(np.mean(occupation_derivative(e, mu, WIDTH, 'fermi-dirac')))


def _spinor_model():
    e = _band()
    hk = np.zeros((e.size, 2, 2, 1), dtype=complex)
    hk[:, 0, 0, 0] = hk[:, 1, 1, 0] = e
    spin = 0.5 * np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    return hk, spin, [{'orbitals': np.array([0, 1])}]


def _collinear_model():
    e = _band()
    hk = np.zeros((e.size, 1, 1, 2), dtype=complex)
    hk[:, 0, 0, 0] = hk[:, 0, 0, 1] = e
    return hk, None, [{'orbitals': np.array([0])}]


def _zeeman(h: float, nspin: int) -> np.ndarray:
    """Splitting +-h of spin up/down (a field along +z with the electron's sign)."""
    if nspin == 2:
        return np.array([[[h, -h]]], dtype=complex)
    return np.array([[[h], [0]], [[0], [-h]]], dtype=complex)


def _chi(model, stoner: float, h: float = 1.0e-4, **options) -> float:
    hk, spin, groups = model
    moments = [
        solve_spin_response(
            hk, _zeeman(s * h, hk.shape[-1]), groups, np.array([[stoner]]), spin, ELECTRONS, WIDTH,
            'fermi-dirac', tol=1e-11, **options,
        )['moments'][0, 2]
        for s in (1.0, -1.0)
    ]  # fmt: skip
    return -(moments[0] - moments[1]) / (2 * h)


@pytest.mark.parametrize('product', [0.0, 0.5, 0.9])
def test_linear_response_is_the_rpa_stoner_enhancement(product):
    n = _density_of_states()
    expected = 2 * n / (1 - product)
    assert _chi(_spinor_model(), product / n) == pytest.approx(expected, rel=2e-3)
    assert _chi(_collinear_model(), product / n) == pytest.approx(expected, rel=2e-3)


def test_linear_mode_gives_the_negative_susceptibility_of_an_unstable_state():
    n = _density_of_states()
    product = 1.2
    chi = _chi(_spinor_model(), product / n, check_stability=False, warmup=0)
    assert chi == pytest.approx(2 * n / (1 - product), rel=5e-3)
    assert chi < 0


def test_ferromagnetic_solution_is_self_consistent_and_stable():
    n = _density_of_states()
    stoner = 1.3 / n
    hk, spin, groups = _spinor_model()
    out = solve_spin_response(
        hk, np.zeros((2, 2, 1), complex), groups, np.array([[stoner]]), spin, ELECTRONS, WIDTH,
        'fermi-dirac', tol=1e-10,
    )  # fmt: skip
    m = out['moments'][0, 2]
    assert out['converged'] and out['stable'] and abs(m) > 0.1
    # bands split by I m: the moment they hold is m again
    e = _band()[:, None]
    split = np.concatenate([e - 0.5 * stoner * m, e + 0.5 * stoner * m], axis=1)
    mu = chemical_potential(split, ELECTRONS, WIDTH, 'fermi-dirac')
    f = occupation(split, mu, WIDTH, 'fermi-dirac')
    assert m == pytest.approx(f[:, 0].mean() - f[:, 1].mean(), abs=1e-8)
    # below the threshold the paramagnetic state is the solution
    weak = solve_spin_response(
        hk, np.zeros((2, 2, 1), complex), groups, np.array([[0.8 / n]]), spin, ELECTRONS, WIDTH,
        'fermi-dirac', tol=1e-10,
    )  # fmt: skip
    assert weak['stable'] and abs(weak['moments'][0, 2]) < 1e-8


def test_stoner_integral_of_a_gaussian_shell_in_a_uniform_density():
    r = np.linspace(0.0, 12.0, 6001)
    a, density = 1.3, 0.05
    radial = np.exp(-(r**2) / a**2)
    radial /= np.trapezoid(radial * r**2, r)
    shell = {'species': 'X', 'r': r, 'radial': radial}
    groups = [dict(shell, atom=(0.0, 0.0, 0.0)), dict(shell, atom=(0.5, 0.5, 0.5))]
    matrix = stoner_matrix(groups, {'X': (r, np.full_like(r, density))})
    kernel = lsda_spin_kernel(np.array([density]))[0]
    expected = -2.0 * kernel * np.trapezoid(r**2 * radial**2, r) / (4 * np.pi) * HARTREE_EV
    assert matrix[0, 0] == pytest.approx(expected, rel=1e-10)
    assert matrix[0, 0] > 0 and matrix[0, 1] == 0.0


def test_paoflow_methods_on_a_cubic_model(tmp_path):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import HoppingModel

    hoppings = []
    for axis in range(3):
        step = [0, 0, 0]
        step[axis] = 1
        hoppings += [(0, 0, tuple(step), -1.0), (0, 0, tuple(-s for s in step), -1.0)]
    model = HoppingModel(
        'sc', tuple(hoppings), 1, np.eye(3), 1.0, np.zeros((1, 3)), ELECTRONS, (3, 3, 3)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        out = p.spin_susceptibility(
            field=1.0, nk=(NK,) * 3, smearing='fermi-dirac', degauss=WIDTH, methods=('bare', 'stoner'),
            stoner=0.5 / _density_of_states(), shells=[{'orbitals': np.array([0, 1]), 'l': 0}],
        )  # fmt: skip
    n = _density_of_states()
    assert out['bare'] == pytest.approx(2 * n, rel=1e-4)
    assert out['stoner'] == pytest.approx(4 * n, rel=1e-3)
    assert out['enhancement']['stoner'] == pytest.approx(2.0, rel=1e-3)
    assert (tmp_path / 'spin_susceptibility.dat').exists()


def _write_charge_density(path, mill, rho, b):
    """QE's charge-density.dat (Fortran records): header, b, Miller indices, rho(G)."""

    def record(handle, data: bytes):
        size = np.array([len(data)], dtype='<i4').tobytes()
        handle.write(size + data + size)

    with open(path, 'wb') as handle:
        record(handle, np.array([0, mill.shape[0], 1], dtype='<i4').tobytes())
        record(handle, np.asarray(b, dtype='<f8').tobytes())
        record(handle, np.asarray(mill, dtype='<i4').tobytes())
        record(handle, np.asarray(rho, dtype='<c16').tobytes())


def test_crystal_density_is_the_spherical_average_about_the_atom(tmp_path):
    import os
    import shutil

    from PAOFLOW.hamiltonian.stoner import crystal_densities
    from PAOFLOW.inputs.read_upf import UPF

    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    upf_path = os.path.join(root, 'PSEUDOS', 'nc-sr-05_pbe_standard_upf', 'Fe.upf')
    if not os.path.exists(upf_path):
        pytest.skip('no PSEUDOS folder')
    shutil.copy(upf_path, tmp_path / 'Fe.upf')
    alat = 5.4
    mill = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    rho = np.array([0.03, 0.004, 0.004], dtype=complex)  # n = 0.03 + 0.008 cos(2 pi x / a)
    _write_charge_density(tmp_path / 'charge-density.dat', mill, rho, np.eye(3))
    controller = type('C', (), {})()
    controller.data_arrays = {
        'species': [('Fe', 'Fe.upf')], 'atoms': ['Fe'], 'tau': np.array([[0.25, 0.0, 0.0]]) * alat,
        'b_vectors': np.eye(3),
    }  # fmt: skip
    controller.data_attributes = {'fpath': str(tmp_path), 'alat': alat}
    controller.data_dicts = lambda: (controller.data_arrays, controller.data_attributes)
    r, n = crystal_densities(controller)['Fe']
    upf = UPF(upf_path)
    core = np.asarray(upf.rho_atc)[: r.size]
    g = 2 * np.pi / alat
    # the atom sits on a node of the cosine: the average of cos(g x) about it is 0 at every r
    np.testing.assert_allclose(n - core, 0.03, atol=1e-12)
    # about a crest (shift the atom to x = 0) it is 0.008 j0(g r)
    controller.data_arrays['tau'] = np.zeros((1, 3))
    r, n = crystal_densities(controller)['Fe']
    np.testing.assert_allclose(n - core, 0.03 + 0.008 * np.sinc(g * r / np.pi), atol=1e-12)
