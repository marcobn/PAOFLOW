"""ArpesModel on a synthetic arpes/output folder and on a system folder."""

import numpy as np
import pytest

from PAOFLOW.agentic.arpes import ArpesModel
from PAOFLOW.agentic.arpes.model import parse_metadata, read_sigma_table

V_BARE, K_FERMI, LAM, GAMMA, W_D = 3.0, 0.2, 0.5, 0.004, 0.02


def write_outputs(folder, temps=(10.0, 300.0)):
    folder.mkdir(parents=True, exist_ok=True)
    k = np.linspace(0.0, 0.4, 401)
    w = np.linspace(-0.1, 0.02, 241)
    kpoints = np.column_stack([k, np.zeros_like(k), np.zeros_like(k)])
    e_b = V_BARE * (k - K_FERMI)
    metadata = repr(
        {'photon_ev': 21.2, 'energy': (np.float64(-0.1), np.float64(0.02), 241), 'kz': 0.0}
    )
    common = dict(kpoints=kpoints, w=w, energies=e_b[:, None], weights=np.ones((k.size, 1)),
                  bands=np.array([5]), grid=np.array((0, 0)), kx=np.zeros(0), ky=np.zeros(0),
                  metadata=np.array(metadata))  # fmt: skip
    bare = 1e-3 / np.pi / ((w[None, :] - e_b[:, None]) ** 2 + 1e-6)
    np.savez(folder / 'arpes_bare.npz', intensity=bare, raw=bare, **common)
    for t in temps:
        dressed = GAMMA / np.pi / ((w[None, :] - e_b[:, None] + LAM * w[None, :]) ** 2 + GAMMA**2)
        np.savez(
            folder / ('arpes_isotropic_%.0fK.npz' % t), intensity=dressed, raw=dressed, **common
        )
        ww = np.linspace(-0.08, 0.02, 401)
        re = -LAM * ww * np.exp(-0.5 * (ww / W_D) ** 2)
        im = -GAMMA * np.ones_like(ww) * (1 + t / 300.0)
        np.savetxt(folder / ('sigma_isotropic_%.0fK.dat' % t), np.column_stack([ww * 1e3, re * 1e3, im * 1e3]),
                   header='omega (meV)  Re Sigma (meV)  Im Sigma (meV)   T = %.0f K' % t)  # fmt: skip
    np.savetxt(folder / 'sigma_anisotropic_20K.dat', np.column_stack([ww * 1e3] + [re * 1e3, im * 1e3] * 3),
               header='omega (meV)  Re/Im Sigma crossing state  Re/Im Sigma weakest state  Re/Im isotropic (meV), T = 20 K')  # fmt: skip
    # Fan-Migdal: two temperatures, 5 path points, 2 bands, widths a + b|E| (and a T factor)
    path = kpoints[::100]
    energies = np.column_stack([V_BARE * (path[:, 0] - K_FERMI), -1.0 + 0.5 * path[:, 0]])
    sigma = np.zeros((len(temps), 5, 2, ww.size), dtype=complex)
    for it, t in enumerate(temps):
        for ik in range(5):
            for ib in range(2):
                sigma[it, ik, ib] = 0.01 - 1j * (0.002 + 0.01 * abs(energies[ik, ib])) * (
                    1 + t / 300.0
                )
    np.savez(folder / 'self_energy_fan_migdal.npz', k_cart=path, sigma_ev=sigma, energies_ev=energies,
             bands=np.array([5, 4]), w_ev=ww, temps_k=np.array(temps), nq_dense=12, eta_ev=0.01,
             adaptive_smearing=np.array(None, dtype=object), metal=True)  # fmt: skip
    return k, w


def test_parse_metadata_and_sigma_table(tmp_path):
    assert parse_metadata("{'a': np.float64(-0.08), 'b': (1, 2), 'c': None}") == {
        'a': -0.08,
        'b': (1, 2),
        'c': None,
    }
    assert parse_metadata('not a dict') == {'raw': 'not a dict'}
    write_outputs(tmp_path)
    iso = read_sigma_table(tmp_path / 'sigma_isotropic_10K.dat')
    assert iso['labels'] == ['isotropic'] and iso['w_ev'][0] == pytest.approx(-0.08)
    aniso = read_sigma_table(tmp_path / 'sigma_anisotropic_20K.dat')
    assert aniso['labels'] == ['crossing', 'weakest', 'isotropic'] and len(aniso['sigmas']) == 3


def test_arpes_model_dispersion_kink_and_linewidths(tmp_path):
    write_outputs(tmp_path)
    model = ArpesModel(tmp_path)
    assert model.kinds == ['isotropic'] and model.temperatures() == [10.0, 300.0]
    assert model.metadata['photon_ev'] == 21.2
    distance, energies, bands = model.band_energies()
    assert distance[-1] == pytest.approx(0.4) and bands.tolist() == [5]
    assert model.bare_fermi_momentum() == pytest.approx(K_FERMI, abs=1e-6)
    d = model.dispersion('isotropic', 10.0, energy_window=(-0.08, -0.01))
    assert np.allclose(d['k'], K_FERMI + (1 + LAM) * d['energy_ev'] / V_BARE, atol=2e-4)
    assert model.fermi_momentum('isotropic', 10.0) == pytest.approx(K_FERMI, abs=2e-4)
    v = model.velocities('isotropic', 10.0, window_ev=(-0.06, -0.01))
    assert v['v_bare'] == pytest.approx(V_BARE) and v['lambda'] == pytest.approx(LAM, rel=1e-2)
    sigma = model.self_energy_from_map(
        'isotropic', 10.0, energy_window=(-0.08, -0.01), method='lorentzian'
    )
    assert np.allclose(sigma['re_sigma_ev'], -LAM * sigma['w_ev'], atol=1e-3)
    assert np.allclose(sigma['im_sigma_ev'], -GAMMA, rtol=1e-3)
    table = model.self_energy('isotropic', 300.0)
    assert table['label'] == 'isotropic' and table['sigma'].imag[0] == pytest.approx(-2 * GAMMA)
    lam, z = model.mass_enhancement('isotropic', 10.0)
    assert lam == pytest.approx(LAM, rel=1e-3) and z == pytest.approx(1 / (1 + LAM), rel=1e-3)
    assert model.kink_energy('isotropic', 10.0) == pytest.approx(-W_D, abs=1e-3)
    assert model.self_energy('anisotropic', 20.0, column='weakest')['label'] == 'weakest'
    with pytest.raises(KeyError):
        model.self_energy('isotropic', 77.0)
    with pytest.raises(KeyError):
        model.map('isotropic', 77.0)
    k, mdc = model.mdc('isotropic', 10.0, -0.03)
    assert k.size == 401 and np.argmax(mdc) > 0
    w, edc = model.edc('isotropic', 10.0, 200)
    assert w.size == 241
    assert model.fan_migdal_temperatures() == [10.0, 300.0]
    widths = model.linewidths(300.0)
    assert widths['energies_ev'].shape == (5, 2) and widths['bands'].tolist() == [5, 4]
    assert widths['widths_ev'][0, 0] == pytest.approx(
        (0.002 + 0.01 * abs(V_BARE * (0.0 - K_FERMI))) * 2
    )
    temps, series = model.linewidth_vs_temperature(1, 2)
    assert np.allclose(temps, [10.0, 300.0]) and series[1] / series[0] == pytest.approx(
        2 / (1 + 10 / 300)
    )
    state = model.state_self_energy(10.0, 0, 2)
    assert (
        state['energy_ev'] == pytest.approx(V_BARE * (0.2 - K_FERMI))
        and state['sigma'].real[0] == 0.01
    )
    text = model.describe()
    assert '| maps isotropic (K) | 10, 300 |' in text and '| nq_dense | 12 |' in text
    assert '| photon_ev | 21.2 |' in text


def test_arpes_model_without_outputs_and_from_system(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(FileNotFoundError):
        ArpesModel(tmp_path / 'empty')
    (tmp_path / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='pb', root='.', masses_amu=[207.2], nelec=14, qgrid=(3, 3, 3))\n"
        "NAME = 'Pb'\nARPES = dict(photon_ev=21.2, direction=(0.5, 0.0, 0.5))\n"
    )
    write_outputs(tmp_path / 'arpes' / 'output', temps=(10.0,))
    model = ArpesModel.from_system(tmp_path)
    assert model.system['name'] == 'Pb' and model.system['arpes_defaults']['photon_ev'] == 21.2
    assert '| coupling_source | dfpt |' in model.describe()
    only_fan = tmp_path / 'fan'
    only_fan.mkdir()
    (tmp_path / 'arpes' / 'output' / 'self_energy_fan_migdal.npz').rename(
        only_fan / 'self_energy_fan_migdal.npz'
    )
    fan_model = ArpesModel(only_fan)
    assert fan_model.kinds == [] and fan_model.fan_migdal_temperatures() == [10.0]
    with pytest.raises(FileNotFoundError):
        fan_model.map('bare')
