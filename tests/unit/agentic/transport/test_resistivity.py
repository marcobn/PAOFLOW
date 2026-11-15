"""ResistivityModel on a synthetic metal transport/output and on a system folder."""

import numpy as np
import pytest

from PAOFLOW.agentic.transport.resistivity import ResistivityModel, read_resistivity_table


def synthetic_metal(folder, eliashberg=None):
    temps = np.array([20.0, 50.0, 100.0, 200.0, 300.0])
    eph = np.array([0.5, 2.4, 5.2, 10.8, 16.2])
    ee = 1e-5 * temps**2
    rows = np.column_stack([temps, eph + ee, eph, ee])
    np.savetxt(folder / 'resistivity.dat', rows, fmt='%14.6f',
               header='T (K), rho total and per channel (uOhm cm): e-ph (a2F_tr), electron-electron')  # fmt: skip
    omega = np.linspace(0.005, 2.0, 50)
    np.savez(folder / 'emissivity.npz', omega_ev=omega, temps_k=temps,
             eps1=-np.ones((5, 50)) * 100, eps2=np.ones((5, 50)),
             hemispherical_opaque=np.outer(temps / 300.0, omega / 2.0),
             hemispherical_slab=np.outer(temps / 300.0, omega / 2.0),
             directional_slab=np.ones((5, 2, 50)), total_slab=temps / 1e4, total_opaque=temps / 1e4,
             planck_coverage=np.ones(5), thickness_m=np.inf, angles_deg=np.array([0.0, 60.0]),
             plasma_ev=8.3, dc_rate_per_s=np.ones(5), rho_ohm_m=eph * 1e-8, drude='extended')  # fmt: skip
    if eliashberg is not None:
        np.savez(eliashberg, omega=omega, a2F=omega, a2F_tr=omega, **{'lambda': 1.2}, lambda_tr=0.94,
                 omega_plasma_ev=8.34, omega_log=4.5, mu_star=0.1, Tc_allen_dynes=5.2, Tc_mcmillan=4.8)  # fmt: skip
    return temps, eph, ee


def test_read_resistivity_table_names_channels(tmp_path):
    temps, eph, ee = synthetic_metal(tmp_path)
    t, total, channels = read_resistivity_table(tmp_path / 'resistivity.dat')
    assert np.allclose(t, temps) and list(channels) == ['e-ph (a2F_tr)', 'electron-electron']
    assert np.allclose(channels['e-ph (a2F_tr)'], eph) and np.allclose(total, eph + ee)


def test_resistivity_model_reads_interpolates_and_fits(tmp_path):
    temps, eph, ee = synthetic_metal(tmp_path, tmp_path / 'eliashberg.npz')
    model = ResistivityModel(tmp_path, eliashberg=tmp_path / 'eliashberg.npz')
    assert model.channels == ['e-ph (a2F_tr)', 'electron-electron']
    assert model.resistivity(300.0) == pytest.approx(16.2 + 1e-5 * 9e4)
    assert model.resistivity(300.0, 'e-ph (a2F_tr)') == pytest.approx(16.2)
    assert model.resistivity(300.0, rho0_uohm_cm=0.5) == pytest.approx(16.2 + 0.9 + 0.5)
    assert model.resistivity(250.0, 'e-ph (a2F_tr)', interpolate=True) == pytest.approx(13.5)
    # a power-law interpolation reproduces T^n between two points exactly
    assert model.resistivity(
        30.0, 'electron-electron', interpolate=True, law='power'
    ) == pytest.approx(9e-3)
    with pytest.raises(KeyError):
        model.resistivity(250.0)
    with pytest.raises(ValueError):
        model.bracket(400.0)
    assert model.bracket(250.0) == (200.0, 300.0)
    slope = model.slope(window=(100.0, 300.0), channel='e-ph (a2F_tr)')
    assert slope['slope'] == pytest.approx(0.055, rel=1e-3) and slope['n'] == 3
    rrr = model.residual_resistivity_ratio(0.1, 20.0)
    assert rrr == pytest.approx((16.2 + 0.9 + 0.1) / (0.5 + 0.004 + 0.1))
    assert model.emissivity_total(300.0) == pytest.approx(0.03)
    assert model.emissivity_total(250.0, interpolate=True) == pytest.approx(0.025)
    omega, spectrum = model.emissivity_spectrum(300.0)
    assert np.allclose(spectrum, omega / 2.0)
    _, directional = model.emissivity_spectrum(100.0, 'directional_slab', angle_deg=60.0)
    assert np.allclose(directional, 1.0)
    with pytest.raises(KeyError):
        model.emissivity_spectrum(100.0, 'directional_slab', angle_deg=30.0)
    omega, eps1, eps2 = model.dielectric(200.0)
    assert eps1[0] == -100.0 and eps2[0] == 1.0
    assert model.eliashberg['lambda_tr'] == pytest.approx(0.94)
    text = model.describe()
    assert (
        '| lambda_tr | 0.94 |' in text and '| channels | e-ph (a2F_tr), electron-electron |' in text
    )
    assert '300 K: 16.200' in text or '300 K: 17.100' in text


def test_resistivity_model_without_emissivity_raises(tmp_path):
    synthetic_metal(tmp_path)
    (tmp_path / 'emissivity.npz').unlink()
    model = ResistivityModel(tmp_path)
    assert model.emissivity is None and model.eliashberg == {}
    with pytest.raises(FileNotFoundError):
        model.emissivity_total(300.0)


def test_resistivity_model_from_system_folder(tmp_path):
    (tmp_path / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='pb', root='.', masses_amu=[207.2], nelec=14, qgrid=(3, 3, 3))\n"
        "NAME = 'Pb'\nTRANSPORT = dict(rho0_uohm_cm=0.0)\nSUPERCONDUCTIVITY = dict(mu_star=0.1)\n"
    )
    out = tmp_path / 'transport' / 'output'
    out.mkdir(parents=True)
    sc = tmp_path / 'superconductivity' / 'output'
    sc.mkdir(parents=True)
    synthetic_metal(out, sc / 'eliashberg.npz')
    model = ResistivityModel.from_system(tmp_path)
    assert model.system['prefix'] == 'pb' and model.system['transport_defaults'] == {
        'rho0_uohm_cm': 0.0
    }
    assert model.eliashberg_path == sc / 'eliashberg.npz' and model.eliashberg[
        'lambda'
    ] == pytest.approx(1.2)
    assert '| coupling_source | dfpt |' in model.describe()
