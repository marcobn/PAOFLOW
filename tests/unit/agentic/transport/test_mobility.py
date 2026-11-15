"""MobilityModel on a synthetic transport/output and on a system folder."""

import numpy as np
import pytest

from PAOFLOW.agentic.transport.mobility import COLUMNS, MobilityModel


def synthetic_transport(folder):
    temps = np.array([200.0, 300.0, 400.0])
    mu_e = 1300.0 * (temps / 300.0) ** -2.4  # a clean power law
    mu_h = 500.0 * (temps / 300.0) ** -2.2
    n = 1e10 * np.exp(-0.55 / (8.617333262e-5 * temps))
    table = np.column_stack([temps, 0.3 * np.ones(3), n, n, 1.05 * mu_e, 1.05 * mu_h, mu_e, mu_h,
                             1e3 * np.ones(3), 1e3 * np.ones(3), np.array([-900.0, -800.0, -700.0])])  # fmt: skip
    np.savetxt(folder / 'mobility.dat', table, header='T, ...')
    np.savez(folder / 'carrier_scattering.npz', nk_dense=48, nq_dense=48, smearing_ev=0.01,
             doping_cm3=0.0, vbm_ev=-0.3, cbm_ev=0.3, metal=False, scissor_mode='none',
             electrons_dft='PBE', coupling_screening='none', in_window=np.ones(12, dtype=bool))  # fmt: skip
    return temps, mu_e, mu_h, n


def test_mobility_model_reads_interpolates_and_fits(tmp_path):
    temps, mu_e, mu_h, n = synthetic_transport(tmp_path)
    model = MobilityModel(tmp_path)
    assert list(model.table) == list(COLUMNS) and np.allclose(model.temperatures, temps)
    assert model.mobility(300.0, 'e', 'mrta') == pytest.approx(1300.0)
    assert model.mobility(300.0, 'h', 'serta') == pytest.approx(1.05 * 500.0)
    # a power law interpolates exactly in ln T
    assert model.mobility(250.0, 'e', 'mrta', interpolate=True) == pytest.approx(
        1300.0 * (250 / 300) ** -2.4
    )
    assert model.power_law_exponent('e', 'mrta') == pytest.approx(2.4)
    assert model.power_law_exponent('h', 'mrta', window=(250.0, 450.0)) == pytest.approx(2.2)
    # an activated density interpolates exactly in 1/T
    assert model.carrier_density(350.0, 'e', interpolate=True) == pytest.approx(
        1e10 * np.exp(-0.55 / (8.617333262e-5 * 350.0))
    )
    assert model.seebeck(350.0, interpolate=True) == pytest.approx(-750.0)
    assert model.bracket(250.0) == (200.0, 300.0) and model.bracket(300.0) == (300.0, 300.0)
    with pytest.raises(KeyError):
        model.mobility(250.0)
    with pytest.raises(ValueError):
        model.bracket(500.0)
    assert model.metadata['gap_ev'] == pytest.approx(0.6) and model.metadata['window_states'] == 12
    text = model.describe()
    assert '| nk_dense | 48 |' in text and '300 K: 1300 / 500' in text


def test_mobility_model_from_system_folder(tmp_path):
    (tmp_path / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='si', root='.', masses_amu=[28.085], nelec=8, qgrid=(3, 3, 3))\n"
        "NAME = 'Si'\nTRANSPORT = dict(nk=48, nq=48)\n"
    )
    out = tmp_path / 'transport' / 'output'
    out.mkdir(parents=True)
    synthetic_transport(out)
    model = MobilityModel.from_system(tmp_path)
    assert model.system['prefix'] == 'si' and model.system['qgrid'] == '3x3x3'
    assert model.system['transport_defaults'] == {'nk': 48, 'nq': 48}
    assert '| coupling_source | dfpt |' in model.describe()


def test_thermo_optics_shim_reexports_the_adapter():
    from PAOFLOW.agentic.thermo_optics import paoflow_transport

    assert paoflow_transport.MobilityModel is MobilityModel
    assert paoflow_transport.COLUMNS is COLUMNS
