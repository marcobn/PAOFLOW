"""Topology, spin and optical responses against closed forms on models (and JDOS on a DFT save).

- the anomalous Hall conductivity of the Qi-Wu-Zhang model is quantized, :math:`2\\pi\\sigma_{xy} = \\mp 1`
  in units of :math:`e^2/\\hbar` in its two topological phases, 0 in the trivial one;
- the Weyl nodes of the lattice Weyl model sit at :math:`(0, 0, \\pm\\pi/2a)` with energies :math:`\\pm b`
  (and chiralities :math:`\\pm 1` with Z2Pack);
- the spin of a Rashba gas is :math:`\\mp\\hat g/2`, perpendicular to k;
- the Rashba-Edelstein tensor of a Rashba gas has the symmetry of :math:`C_{4v}`: only
  :math:`\\chi_{xy} = -\\chi_{yx}`;
- the interband dielectric function obeys the f-sum rule of the tight-binding model;
- the JDOS of a DFT save is the normalized pair distribution of its eigenvalues.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from _models import bloch, model_run, output, qwz_model, rashba_model
from scipy import constants as C

from PAOFLOW.models.models import parabolic_model, weyl_model

pytestmark = pytest.mark.unit

BOHR = C.physical_constants['Bohr radius'][0]
SILICON = Path(__file__).resolve().parents[2] / 'data' / 'silicon'


@pytest.mark.parametrize('mass, chern', [(1.0, -1), (-1.0, 1), (3.0, 0)])
def test_anomalous_hall_is_quantized(mass, chern):
    p = model_run(qwz_model(m=mass), (60, 60, 1))
    p.gradient_and_momenta()
    p.adaptive_smearing()
    p.anomalous_Hall(emin=-0.05, emax=0.05, ne=3, twoD=True, a_tensor=[[0, 1]])
    sigma = output(p, 'ahcEf_xy.dat')[:, 1]  # e^2/hbar, Fermi level in the gap
    np.testing.assert_allclose(2 * np.pi * sigma, chern, atol=2e-3)


def test_weyl_nodes_of_the_lattice_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from PAOFLOW import PAOFLOW

    p = PAOFLOW.PAOFLOW(model=weyl_model(t=1.0, b=0.3), outputdir='output', verbose=False)
    p.data_controller.data_attributes['abort_on_exception'] = True
    p.write_Hamiltonian()
    p.find_weyl_points(search_grid=[4, 4, 4], symmetrize=False)
    rows = [line.split() for line in open('output/weyl_points.dat').readlines()[3:] if line.strip()]
    nodes = sorted((float(r[3]), r[4], float(r[5])) for r in rows)  # kz (2pi/alat), C, E
    assert len(nodes) == 2  # each node once, from whichever boxes converge to it
    (kz_minus, c_minus, e_minus), (kz_plus, c_plus, e_plus) = nodes
    assert kz_minus == pytest.approx(-0.25, abs=1e-4) and kz_plus == pytest.approx(0.25, abs=1e-4)
    assert e_minus == pytest.approx(-0.3, abs=1e-4) and e_plus == pytest.approx(0.3, abs=1e-4)
    for x, y in (r[1:3] for r in rows):
        assert abs(float(x)) < 1e-4 and abs(float(y)) < 1e-4
    try:
        import tbmodels  # noqa: F401
        import z2pack  # noqa: F401
    except ImportError:
        assert (c_minus, c_plus) == ('?', '?')
    else:
        assert (c_minus, c_plus) == ('-1', '1')


def test_rashba_spin_texture():
    n = 24
    p = model_run(rashba_model(), (n, n, 1))
    p.spin_texture(fermi_up=4.5, fermi_dw=-4.5)
    opath = p.data_controller.data_attributes['opath']
    assert p.data_controller.data_arrays['ind_plot'] == [0, 1]
    k = 2 * np.pi * np.arange(n) / n
    kx, ky = np.meshgrid(k, k, indexing='ij')
    field = np.stack([np.sin(ky), -np.sin(kx)], axis=-1)
    size = np.linalg.norm(field, axis=-1)
    away = size > 1e-6  # the time-reversal-invariant momenta are degenerate
    for band, sign in ((0, -1), (1, 1)):
        spin = np.load(os.path.join(opath, 'spin_text_band_%d.npz' % band))['spinband'][:, :, 0]
        along = np.sum(spin[..., :2] * field, axis=-1)[away] / size[away]
        np.testing.assert_allclose(along, sign * 0.5, atol=1e-12)
        np.testing.assert_allclose(spin[..., 2][away], 0.0, atol=1e-12)
        across = spin[..., 0] * np.sin(kx) + spin[..., 1] * np.sin(ky)
        np.testing.assert_allclose(across[away], 0.0, atol=1e-12)


def test_rashba_edelstein_has_the_c4v_form():
    p = model_run(rashba_model(), (24, 24, 1))
    p.gradient_and_momenta()
    p.adaptive_smearing()
    p.rashba_edelstein(emin=-3.0, emax=-1.0, ne=5)
    opath = p.data_controller.data_attributes['opath']
    chi = {
        os.path.basename(f)[5:7]: np.loadtxt(f)[:, 1]
        for f in glob.glob(os.path.join(opath, 'Ekai_*.dat'))
    }
    scale = np.abs(chi['xy']).max()
    assert scale > 0
    np.testing.assert_allclose(chi['yx'], -chi['xy'], rtol=1e-6)
    for component in ('xx', 'yy', 'zz', 'xz', 'yz', 'zx', 'zy'):
        assert np.abs(chi[component]).max() < 1e-9 * scale


def test_interband_f_sum_rule():
    r""":math:`\int_0^\infty \omega\,\varepsilon_2(\omega)\,d\omega = (\pi/2)\,\omega_p^2`, with the
    tight-binding plasma frequency of an insulator :math:`\omega_p^2 = (e^2/\varepsilon_0\Omega\hbar^2)\,g_s
    \langle\sum_{\rm occ}\langle n|\partial^2_{k_x}H|n\rangle\rangle_k` (the Lorentz oscillators of
    ``dielectric_tensor`` obey it at any broadening)."""
    model = parabolic_model(n_bands=2)
    n = 2000
    p = model_run(model, (n, 1, 1), smearing=None)
    p.gradient_and_momenta()
    p.dielectric_tensor(delta=0.05, emin=0.0, emax=60.0, ne=24001, d_tensor=[[0, 0]],
                        intraband=False)  # fmt: skip
    omega, eps2 = output(p, 'epsi_xx.dat').T
    total = 0.0
    for k in np.arange(n) / n:
        k_cart = (
            np.array([k, 0, 0]) @ np.linalg.inv(model.lattice_vectors).T * 2 * np.pi / model.alat
        )
        _, vectors = np.linalg.eigh(bloch(model, k_cart))
        occupied = vectors[:, :1]
        total += np.real(np.trace(occupied.conj().T @ bloch(model, k_cart, order=2) @ occupied))
    total /= n  # eV bohr^2
    volume = model.volume * BOHR**3
    plasma2 = C.e**2 / (C.epsilon_0 * volume) * 2 * total * C.e * BOHR**2 / C.hbar**2  # 1/s^2
    expected = np.pi / 2 * (C.hbar / C.e) ** 2 * plasma2  # eV^2
    assert np.trapezoid(omega * eps2, omega) == pytest.approx(expected, rel=1e-3)


def test_jdos_of_a_dft_save(monkeypatch, tmp_path):
    r"""``jdos`` reads the eigenvalues of the save (not the PAO bands): the normalized distribution of
    the vertical transitions :math:`\sum_k w_k\sum_{v,c}(f_v - f_c)\,g(\omega - E_c + E_v)`."""
    from PAOFLOW import PAOFLOW

    monkeypatch.chdir(SILICON)
    p = PAOFLOW.PAOFLOW(
        savedir='silicon.save', outputdir=str(tmp_path), smearing=None, verbose=False
    )
    p.data_controller.data_attributes['abort_on_exception'] = True
    p.jdos(delta=0.1, emin=0.0, emax=20.0, ne=401)
    energy, jdos = np.loadtxt(tmp_path / 'jdos.dat').T
    root = ET.parse(SILICON / 'silicon.save' / 'data-file-schema.xml').getroot()
    bands = root.find('output/band_structure')
    hartree = C.physical_constants['Hartree energy in eV'][0]
    top = float(bands.findtext('highestOccupiedLevel')) * hartree
    result = np.zeros_like(energy)
    count = 0.0
    weights = []
    for point in bands.findall('ks_energies'):
        weights.append(float(point.find('k_point').attrib['weight']))
        e = np.array(point.findtext('eigenvalues').split(), dtype=float) * hartree - top
        f = (e <= 1e-9).astype(float)
        pairs = (e[:, None] - e[None, :] > 1e-10) & (f[None, :] > 1e-4) & (f[:, None] < 2)
        upper, lower = np.nonzero(pairs)
        gap = e[upper] - e[lower]
        occupation = f[lower] - f[upper]
        kernel = np.exp(-(((gap[:, None] - energy[None, :]) / 0.1) ** 2)) / (np.sqrt(np.pi) * 0.1)
        result += weights[-1] * np.sum(occupation[:, None] * kernel, axis=0)
        count += occupation.sum()
    result *= len(weights) / count / 2  # spin-degenerate
    # 1e-4: PAOFLOW's hartree-to-eV factor is not scipy's CODATA value
    np.testing.assert_allclose(jdos, result, rtol=1e-3, atol=1e-9)
