"""Eliashberg and gap adapters on synthetic outputs, and the baseline formulas."""

import numpy as np
import pytest

from PAOFLOW.agentic.superconductivity import AnisotropicGapModel, EliashbergModel, GapModel
from PAOFLOW.agentic.superconductivity import baselines as bl
from PAOFLOW.elphon.eph_kq import mcmillan_allen_dynes_tc
from PAOFLOW.agentic.superconductivity.baselines import KB_EV

OMEGA0, LAM = 0.010, 1.2  # an Einstein-like alpha^2F at 10 meV with lambda = 1.2


def einstein_a2f(width=0.0004):
    omega = np.linspace(0.0005, 0.03, 600)
    gauss = np.exp(-0.5 * ((omega - OMEGA0) / width) ** 2) / (width * np.sqrt(2 * np.pi))
    return omega, 0.5 * LAM * OMEGA0 * gauss  # 2 int a2F / omega = lambda


def write_eliashberg(path, transport=True):
    omega, a2f = einstein_a2f()
    moments = bl.a2f_moments(omega, a2f)
    tc = mcmillan_allen_dynes_tc(
        moments['lambda'], moments['omega_log_ev'], moments['omega_2_ev'], 0.1
    )
    extra = (
        {'a2F_tr': 0.8 * a2f, 'lambda_tr': 0.8 * moments['lambda'], 'omega_plasma_ev': 8.3}
        if transport
        else {}
    )
    np.savez(path, omega=omega, a2F=a2f, omega_log=moments['omega_log_ev'], omega_2=moments['omega_2_ev'],
             mu_star=0.1, Tc_mcmillan=tc['Tc_mcmillan_K'], Tc_allen_dynes=tc['Tc_allen_dynes_K'], f1=tc['f1'],
             f2=tc['f2'], lambda_qv=np.ones((4, 3)) * LAM / 12, omega_q=np.full((4, 3), OMEGA0),
             q_weights=np.full(4, 0.25), **{'lambda': moments['lambda']}, **extra)  # fmt: skip
    return moments, tc


def test_baselines_moments_tc_and_bcs():
    omega, a2f = einstein_a2f()
    m = bl.a2f_moments(omega, a2f)
    assert m['lambda'] == pytest.approx(LAM, rel=3e-3)
    assert m['omega_log_ev'] == pytest.approx(OMEGA0, rel=5e-3) and m[
        'omega_2_ev'
    ] == pytest.approx(OMEGA0, rel=2e-3)
    cumulative = bl.cumulative_lambda(omega, a2f)
    assert cumulative[-1] == pytest.approx(m['lambda'], rel=1e-6) and cumulative[0] == 0.0
    tc_ad = bl.allen_dynes_tc(LAM, OMEGA0, OMEGA0, 0.1)
    tc_mcm = bl.mcmillan_tc(LAM, OMEGA0, 0.1)
    ref = mcmillan_allen_dynes_tc(LAM, OMEGA0, OMEGA0, 0.1)
    assert tc_ad == ref['Tc_allen_dynes_K'] and tc_mcm == ref['Tc_mcmillan_K'] and tc_ad > tc_mcm
    assert bl.lambda_from_tc(tc_mcm, OMEGA0, 0.1) == pytest.approx(LAM, rel=1e-6)
    assert bl.lambda_from_tc(tc_ad, OMEGA0, 0.1, omega_2_ev=OMEGA0) == pytest.approx(LAM, rel=1e-6)
    with pytest.raises(ValueError):
        bl.lambda_from_tc(1e4, OMEGA0, 0.1)
    assert bl.bcs_gap_zero_mev(10.0) == pytest.approx(1.764 * KB_EV * 10.0 * 1e3, rel=1e-3)
    curve = bl.bcs_gap_vs_t([0.0, 5.0, 9.0, 9.9, 12.0], 10.0)
    assert curve[0] == pytest.approx(bl.bcs_gap_zero_mev(10.0))
    assert curve[1] / curve[0] == pytest.approx(0.9569, abs=5e-3)  # Muhlschlegel, t = 0.5
    assert curve[2] / curve[0] == pytest.approx(0.5263, abs=5e-3)  # t = 0.9
    assert 0.0 < curve[3] < curve[2] and curve[4] == 0.0
    scaled = bl.bcs_gap_vs_t([5.0], 10.0, gap0_mev=2.0)
    assert scaled[0] == pytest.approx(2.0 * 0.9569, abs=1e-2)
    assert bl.strong_coupling_gap_ratio(7.2, 0.0045) > 3.53  # Pb: about 4.3


def test_eliashberg_model(tmp_path):
    moments, tc = write_eliashberg(tmp_path / 'eliashberg.npz')
    model = EliashbergModel(tmp_path)
    assert model.transport and model.scalars['lambda'] == pytest.approx(moments['lambda'])
    w, f = model.function(np.array([0.0, OMEGA0, 0.1]))
    assert f[0] == 0.0 and f[2] == 0.0 and f[1] > 0.0
    _, cumulative = model.cumulative_lambda()
    assert cumulative[-1] == pytest.approx(moments['lambda'], rel=1e-6)
    assert model.moments()['omega_log_ev'] == pytest.approx(moments['omega_log_ev'])
    assert model.moments(transport=True)['lambda'] == pytest.approx(
        0.8 * moments['lambda'], rel=1e-6
    )
    assert model.lambda_in_window(0.009, 0.011) == pytest.approx(moments['lambda'], rel=2e-2)
    assert model.lambda_in_window(0.02, 0.03) == pytest.approx(0.0, abs=1e-6)
    assert model.tc() == pytest.approx(tc['Tc_allen_dynes_K']) and model.tc(
        formula='mcmillan'
    ) == pytest.approx(tc['Tc_mcmillan_K'])
    assert model.tc(mu_star=0.13) < model.tc(mu_star=0.1)
    with pytest.raises(ValueError):
        model.tc(formula='bcs')
    modes = model.mode_lambda()
    assert modes['lambda_qv'].shape == (4, 3) and modes['q_weights'].sum() == pytest.approx(1.0)
    text = model.describe()
    assert '| lambda_tr | 0.9615 |' in text and '| Tc_allen_dynes (K) |' in text
    write_eliashberg(tmp_path / 'bare.npz', transport=False)
    with pytest.raises(KeyError):
        EliashbergModel(tmp_path / 'bare.npz').function(transport=True)


def write_isotropic(folder, tc=10.0):
    folder.mkdir(parents=True, exist_ok=True)
    temps = np.array([2.0, 5.0, 8.0, 9.5])
    gap0 = 1.0e-3 * bl.bcs_gap_vs_t(temps, tc, 2.0)  # eV
    w_real = np.linspace(1e-4, 0.05, 50)
    extra = {}
    for t in temps:
        tag = '%06.2f' % t
        extra['qdos_%s' % tag] = (w_real > 1.9e-3).astype(float)
        extra['pade_%s' % tag] = np.column_stack(
            [np.full(50, 1.5), np.zeros(50), np.full(50, 2e-3), np.zeros(50)]
        )
    np.savez(folder / 'migdal_eliashberg.npz', temps=temps, gap0_imag=gap0, gap_pade=gap0 * 1.01, gap_acon=gap0 * 1.01,
             Z0=np.full(4, 1.5), Tc_gap=tc, Tc_linear=tc - 0.2, lin_temps=temps, max_eigenvalue=tc / temps,
             mu_star=0.1, w_real=w_real, converged=np.ones(4, dtype=bool), **extra)  # fmt: skip
    return temps, gap0


def test_gap_model(tmp_path):
    temps, gap0 = write_isotropic(tmp_path / 'me')
    model = GapModel(tmp_path / 'me')
    assert np.allclose(model.temperatures, temps) and model.tc == {'gap': 10.0, 'linear': 9.8}
    assert model.gap(5.0) == pytest.approx(1e3 * gap0[1]) and model.gap(
        5.0, 'pade'
    ) == pytest.approx(1.01e3 * gap0[1])
    assert model.gap(6.5, interpolate=True) == pytest.approx(0.5e3 * (gap0[1] + gap0[2]))
    with pytest.raises(KeyError):
        model.gap(6.5)
    assert model.gap_zero() == pytest.approx(1e3 * gap0[0])
    assert model.gap_ratio() == pytest.approx(2e3 * gap0[0] / (KB_EV * 10.0) * 1e-3)
    assert np.allclose(
        model.bcs_gap(), 1e3 * gap0, atol=2e-3
    )  # the synthetic gap is the BCS curve itself
    assert model.mass_renormalisation(8.0) == 1.5
    w, dos = model.quasiparticle_dos(5.0)
    assert w.size == 50 and dos.max() == 1.0
    real = model.real_axis(5.0)
    assert real['z'][0] == 1.5 + 0j and real['gap_ev'][0] == pytest.approx(2e-3)
    with pytest.raises(KeyError):
        model.real_axis(5.0, 'acon')
    t, ev = model.max_eigenvalue()
    assert ev[0] == 5.0
    text = model.describe()
    assert '| Tc gap (K) | 10.000 |' in text and '| converged | 4 of 4 |' in text


def test_anisotropic_gap_model(tmp_path):
    folder = tmp_path / 'me_aniso'
    folder.mkdir()
    temps = np.array([5.0, 20.0])
    bands = np.array([6] * 20 + [7] * 10 + [8] * 30)
    gap0 = np.where(bands == 8, 2.0e-3, 7.0e-3)[None, :] * np.array([1.0, 0.9])[:, None]
    grid = np.linspace(0.0, 9.0, 91)
    dist = np.zeros((2, 91))
    for it in range(2):
        for g, n in ((2.0 * (1.0 - 0.1 * it), 30), (7.0 * (1.0 - 0.1 * it), 30)):
            dist[it] += n * np.exp(-0.5 * ((grid - g) / 0.2) ** 2)
    lam = np.where(bands == 8, 0.33, 0.99)
    np.savez(folder / 'migdal_eliashberg_aniso.npz', temps=temps, gap0=gap0, gap_edge=gap0 * 1.02,
             gap_mean=gap0.mean(axis=1), gap_min=gap0.min(axis=1), gap_max=gap0.max(axis=1), gap_grid_mev=grid,
             gap0_distribution=dist, gap_edge_distribution=dist, lambda_grid=np.linspace(0, 1.2, 10),
             lambda_distribution=np.ones(10), Tc_gap=39.0, mu_star=0.1, lambda_nk=lam, band=bands,
             weight=np.ones(60) / 60, energy_ev=np.zeros(60), k_cart=np.zeros((60, 3)), qdos=np.ones((2, 5)),
             w_real=np.linspace(0, 0.01, 5), converged=np.array([True, True]), **{'lambda': 0.6})  # fmt: skip
    model = AnisotropicGapModel(folder)
    assert model.tc == 39.0 and model.bands.size == 60
    assert model.gaps(5.0).max() == pytest.approx(7.0) and model.gaps(
        20.0, 'edge'
    ).min() == pytest.approx(2.0 * 0.9 * 1.02)
    assert model.gap_range(5.0) == {'min': 2.0, 'mean': pytest.approx(4.5), 'max': 7.0}
    assert model.gap_peaks(5.0) == pytest.approx([2.0, 7.0], abs=0.06)
    assert model.gap_peaks(20.0) == pytest.approx([1.8, 6.3], abs=0.06)
    sheets = model.sheet_gaps(5.0, {'sigma': [6, 7], 'pi': [8]})
    assert sheets['sigma']['gap_mean'] == pytest.approx(7.0) and sheets['pi'][
        'lambda_mean'
    ] == pytest.approx(0.33)
    assert sheets['sigma']['weight'] == pytest.approx(0.5) and sheets['pi']['states'] == 30
    with pytest.raises(KeyError):
        model.sheet_gaps(5.0, {'none': [1]})
    stats = model.lambda_statistics()
    assert stats['min'] == 0.33 and stats['max'] == 0.99 and stats['lambda'] == 0.6
    w, dos = model.quasiparticle_dos(20.0)
    assert dos.size == 5
    text = model.describe()
    assert '| Tc (Delta_max^2 -> 0) (K) | 39.000 |' in text and 'bands 6, 7, 8' in text


def test_from_system_folders(tmp_path):
    (tmp_path / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='pb', root='.', masses_amu=[207.2], nelec=14, qgrid=(3, 3, 3))\n"
        "NAME = 'Pb'\nSUPERCONDUCTIVITY = dict(nk=48, nq=24, mu_star=0.1)\n"
    )
    out = tmp_path / 'superconductivity' / 'output'
    out.mkdir(parents=True)
    write_eliashberg(out / 'eliashberg.npz')
    write_isotropic(out / 'me')
    e = EliashbergModel.from_system(tmp_path)
    assert e.system['name'] == 'Pb' and '| nk (system.py) | 48 |' in e.describe()
    g = GapModel.from_system(tmp_path)
    assert g.system['prefix'] == 'pb' and '| coupling_source | dfpt |' in g.describe()
