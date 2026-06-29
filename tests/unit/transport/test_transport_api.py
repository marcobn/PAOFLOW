from __future__ import annotations

import pytest

from PAOFLOW.Transport import Transport


def test_initial_state_has_no_conductor_data():
    transport = Transport(data_controller=object())
    assert transport.conductor_data is None
    assert transport.blc_blocks is None
    assert transport.results is None
    assert transport._energy_grid_config is None


def test_require_hamiltonian_blocks_raises_before_build():
    transport = Transport(data_controller=object())
    with pytest.raises(RuntimeError, match='build_hamiltonian_blocks'):
        transport._require_hamiltonian_blocks()


def test_require_step_state_raises_before_build():
    transport = Transport(data_controller=object())
    with pytest.raises(RuntimeError, match='build_hamiltonian_blocks'):
        transport._require_step_state()


def test_require_grid_config_raises_before_configure():
    transport = Transport(data_controller=object())
    with pytest.raises(RuntimeError, match='configure_energy_grid'):
        transport._require_grid_config()


def test_configure_energy_grid_stores_config():
    transport = Transport(data_controller=object())
    transport.configure_energy_grid(emin=-5.0, emax=5.0, ne=100, delta=0.001)
    assert transport._energy_grid_config == {
        'emin': -5.0,
        'emax': 5.0,
        'ne': 100,
        'delta': 0.001,
        'nk': [0, 0],
    }


def test_configure_energy_grid_accepts_custom_nk():
    transport = Transport(data_controller=object())
    transport.configure_energy_grid(emin=-1.0, emax=1.0, ne=50, delta=1e-4, nk=(4, 4))
    assert transport._energy_grid_config['nk'] == [4, 4]


def test_configure_energy_grid_invalidates_results(monkeypatch):
    transport = Transport(data_controller=object())
    transport.results = object()
    transport.configure_energy_grid(emin=-1.0, emax=1.0, ne=10, delta=1e-4)
    assert transport.results is None


def test_configure_outputs_stores_config():
    transport = Transport(data_controller=object())
    transport.configure_outputs(output_dir='./out', postfix='_test', write_kdata=True)
    assert transport._output_config['output_dir'] == './out'
    assert transport._output_config['postfix'] == '_test'
    assert transport._output_config['write_kdata'] is True


def test_configure_outputs_write_flags_stored():
    transport = Transport(data_controller=object())
    transport.configure_outputs(
        output_dir='./out',
        write_green_function=True,
        write_lead_self_energy=True,
    )
    assert transport._output_config['write_green_function'] is True
    assert transport._output_config['write_lead_self_energy'] is True


def test_configure_outputs_invalidates_results(monkeypatch):
    transport = Transport(data_controller=object())
    transport.results = object()
    transport.configure_outputs(output_dir='./out')
    assert transport.results is None


def test_configure_transport_options_stores_formula():
    transport = Transport(data_controller=object())
    transport.configure_transport_options(formula='landauer')
    assert transport._transport_options_config['formula'] == 'landauer'


def test_configure_transport_options_invalidates_results():
    transport = Transport(data_controller=object())
    transport.results = object()
    transport.configure_transport_options(formula='landauer')
    assert transport.results is None


def test_compute_leads_self_energy_requires_hamiltonian_blocks():
    transport = Transport(data_controller=object())
    transport.configure_energy_grid(emin=-1.0, emax=1.0, ne=10, delta=1e-4)
    with pytest.raises(RuntimeError, match='build_hamiltonian_blocks'):
        transport.compute_leads_self_energy()


def test_compute_transmission_requires_energy_grid():
    transport = Transport(data_controller=object())
    transport.conductor_data = object()
    transport.blc_blocks = object()
    with pytest.raises(RuntimeError, match='configure_energy_grid'):
        transport.compute_transmission()


def test_compute_dos_requires_energy_grid():
    transport = Transport(data_controller=object())
    transport.conductor_data = object()
    transport.blc_blocks = object()
    with pytest.raises(RuntimeError, match='configure_energy_grid'):
        transport.compute_dos()
