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
    from PAOFLOW.transport.conductor_orchestration import require_hamiltonian_blocks

    with pytest.raises(RuntimeError, match='build_hamiltonian_blocks'):
        require_hamiltonian_blocks(None, None)


def test_require_grid_config_raises_before_configure():
    from PAOFLOW.transport.conductor_orchestration import require_grid_config

    with pytest.raises(RuntimeError, match='configure_energy_grid'):
        require_grid_config(None)


def test_configure_energy_grid_stores_config():
    transport = Transport(data_controller=object())
    transport.configure_energy_grid(emin=-5.0, emax=5.0, ne=100, delta=0.001)
    assert transport._energy_grid_config == {
        'emin': -5.0,
        'emax': 5.0,
        'ne': 100,
        'delta': 0.001,
        'nk': [0, 0],
        'smearing_type': 'lorentzian',
        'delta_ratio': 5.0e-3,
        'xmax': 25.0,
        'ne_buffer': 1,
        'energy_step': 0.001,
        'nx_smear': 20000,
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


def test_define_blocks_stashes_only_provided_selectors():
    transport = Transport(data_controller=object())
    transport.define_blocks(
        H00_C={'rows': 'ALL', 'cols': 'ALL'}, H_CR={'rows': '1-2', 'cols': '1-2'}
    )
    assert set(transport._block_selectors) == {'H00_C', 'H_CR'}
    assert transport._block_selectors['H_CR'] == {'rows': '1-2', 'cols': '1-2'}


def test_configure_onsite_shifts_applies_to_existing_data():
    from PAOFLOW.transport.data import build_conductor_data

    transport = Transport(data_controller=object())
    transport.conductor_data = build_conductor_data(dimC=1)
    transport.configure_onsite_shifts(shift_C=0.3, shift_corr=0.1)
    assert transport.conductor_data.shift_C == 0.3
    assert transport.conductor_data.shift_corr == 0.1


def test_configure_lead_convergence_stashes_config():
    transport = Transport(data_controller=object())
    transport.configure_lead_convergence(niterx=300, transfer_thr=1e-8, surface=True)
    assert transport._lead_convergence_config['niterx'] == 300
    assert transport._lead_convergence_config['surface'] is True


def test_configure_eigenchannels_applies_without_clobbering_when_unset():
    from PAOFLOW.transport.conductor_orchestration import apply_eigenchannels
    from PAOFLOW.transport.data import build_conductor_data

    # An unset (None) config must not overwrite values already on the model.
    data = build_conductor_data(dimC=1, do_eigenchannels=True)
    apply_eigenchannels(data, None)
    assert data.symmetry.do_eigenchannels is True


def test_configure_eigenchannels_applies_when_set():
    from PAOFLOW.transport.conductor_orchestration import apply_eigenchannels
    from PAOFLOW.transport.data import build_conductor_data

    transport = Transport(data_controller=object())
    transport.configure_eigenchannels(do_eigenchannels=True, neigchnx=4)
    data = build_conductor_data(dimC=1)
    apply_eigenchannels(data, transport._eigenchannel_config)
    assert data.symmetry.do_eigenchannels is True
    assert data.symmetry.neigchnx == 4
