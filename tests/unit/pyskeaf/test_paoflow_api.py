import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import PAOFLOW.pyskeaf.runner as runner
from PAOFLOW.DataController import DataController
from PAOFLOW.PAOFLOW import PAOFLOW
from PAOFLOW.pyskeaf.config import RYDBERG_IN_EV


def _paoflow_stub(output_path):
    return SimpleNamespace(
        data_controller=SimpleNamespace(
            data_attributes={'opath': str(output_path), 'abort_on_exception': True}
        ),
        rank=0,
        report_exception=lambda name: None,
        report_module_time=lambda name: None,
    )


class _SerialComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def bcast(self, value, root=0):
        return value


def _install_serial_mpi(monkeypatch):
    mpi = SimpleNamespace(COMM_WORLD=_SerialComm())
    monkeypatch.setitem(sys.modules, 'mpi4py', SimpleNamespace(MPI=mpi))


def test_paoflow_initializes_from_bxsf_when_savedir_is_unreadable(
    monkeypatch, tmp_path
):
    _install_serial_mpi(monkeypatch)

    output_path = tmp_path / 'output'
    output_path.mkdir()
    (output_path / 'Fermi_surf_band_1.bxsf').touch()

    paoflow = PAOFLOW(
        workpath=str(tmp_path),
        savedir='missing.save',
        outputdir='output',
        header_style='minimal',
    )
    arrays, attributes = paoflow.data_controller.data_dicts()

    assert attributes['bxsf_only'] is True
    assert attributes['opath'] == str(output_path)
    assert 'fpath' not in attributes
    assert set(arrays) >= {'Efield', 'Bfield', 'HubbardU'}


def test_readable_dft_data_take_precedence_over_existing_bxsf(
    monkeypatch, tmp_path
):
    _install_serial_mpi(monkeypatch)
    output_path = tmp_path / 'output'
    output_path.mkdir()
    (output_path / 'Fermi_surf_band_1.bxsf').touch()
    savedir = tmp_path / 'calculation.save'
    savedir.mkdir()
    (savedir / 'data-file-schema.xml').touch()

    monkeypatch.setattr(DataController, 'read_qe_output', lambda self: None)
    monkeypatch.setattr(PAOFLOW, 'memory_check', lambda self: 0.0)
    paoflow = PAOFLOW(
        workpath=str(tmp_path),
        savedir='calculation.save',
        outputdir='output',
        header_style='minimal',
    )
    attributes = paoflow.data_controller.data_attributes

    assert attributes['bxsf_only'] is False
    assert attributes['fpath'] == str(savedir)


def test_initialization_stops_when_dft_and_bxsf_are_both_missing(
    monkeypatch, tmp_path, capsys
):
    _install_serial_mpi(monkeypatch)

    with pytest.raises(SystemExit):
        PAOFLOW(
            workpath=str(tmp_path),
            savedir='missing.save',
            outputdir='empty-output',
            header_style='minimal',
        )

    output = capsys.readouterr().out
    assert 'No readable QE data were found' in output
    assert 'no BXSF files were found' in output


def test_paoflow_pyskeaf_maps_rotation_options_and_custom_stems(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, **kwargs):
        captured['config'] = config
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, 'run_paoflow_bxsf_files', fake_run)
    paoflow = _paoflow_stub(tmp_path)

    PAOFLOW.pyskeaf(
        paoflow,
        fermi_energy=1.5,
        num_interpolation=120,
        azimuthal=(10.0, 20.0),
        polar=(30.0, 60.0),
        num_angles=9,
        bands=('fermi_bot', 'hhh.bxsf'),
        verbose=False,
    )

    cfg = captured['config']
    assert cfg.hvd == 'r'
    assert cfg.numint == 120
    assert cfg.num_rots == 9
    assert math.isclose(math.degrees(cfg.theta_start), 10.0)
    assert math.isclose(math.degrees(cfg.theta_end), 20.0)
    assert math.isclose(math.degrees(cfg.phi_start), 30.0)
    assert math.isclose(math.degrees(cfg.phi_end), 60.0)
    assert captured['filenames'] == ['fermi_bot.bxsf', 'hhh.bxsf']
    assert captured['input_dir'] == Path(tmp_path)
    assert captured['output_dir'] == Path(tmp_path)
    assert captured['write_auxiliary_files'] is False


def test_paoflow_pyskeaf_maps_single_field_and_forces_one_angle(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config, **kwargs):
        captured['config'] = config
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, 'run_paoflow_bxsf_files', fake_run)
    paoflow = _paoflow_stub(tmp_path)

    PAOFLOW.pyskeaf(
        paoflow,
        b_field='b2',
        azimuthal=12.0,
        polar=34.0,
        num_angles=8,
        bands=3,
        verbose=True,
    )

    cfg = captured['config']
    assert cfg.hvd == 'b'
    assert cfg.num_rots == 1
    assert math.isclose(math.degrees(cfg.theta), 12.0)
    assert math.isclose(math.degrees(cfg.phi), 34.0)
    assert captured['filenames'] == ['Fermi_surf_band_3.bxsf']
    assert captured['write_auxiliary_files'] is True


def test_paoflow_pyskeaf_all_selects_only_standard_bands_in_numeric_order(
    monkeypatch, tmp_path
):
    for filename in ('Fermi_surf_band_10.bxsf', 'Fermi_surf_band_2.bxsf', 'manual.bxsf'):
        (tmp_path / filename).touch()
    captured = {}

    def fake_run(config, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(runner, 'run_paoflow_bxsf_files', fake_run)

    PAOFLOW.pyskeaf(_paoflow_stub(tmp_path), bands='All')

    assert captured['filenames'] == [
        'Fermi_surf_band_2.bxsf',
        'Fermi_surf_band_10.bxsf',
    ]


def test_paoflow_pyskeaf_runs_explicit_fermi_energy_tuple(monkeypatch, tmp_path):
    energies_ry = []

    def fake_run(config, **kwargs):
        energies_ry.append(config.fermi_energy)
        return [config.fermi_energy]

    monkeypatch.setattr(runner, 'run_paoflow_bxsf_files', fake_run)

    results = PAOFLOW.pyskeaf(
        _paoflow_stub(tmp_path),
        fermi_energy=(-0.01, 0.0, 0.2, 1.0),
        bands=1,
    )

    assert np.allclose(
        np.asarray(energies_ry) * RYDBERG_IN_EV,
        [-0.01, 0.0, 0.2, 1.0],
    )
    assert len(results) == 4


def test_paoflow_pyskeaf_runs_explicit_fermi_energy_list(monkeypatch, tmp_path):
    energies_ry = []

    def fake_run(config, **kwargs):
        energies_ry.append(config.fermi_energy)
        return []

    monkeypatch.setattr(runner, 'run_paoflow_bxsf_files', fake_run)

    PAOFLOW.pyskeaf(
        _paoflow_stub(tmp_path),
        fermi_energy=[-0.01, 1.0, 6],
        bands=1,
    )

    assert np.allclose(
        np.asarray(energies_ry) * RYDBERG_IN_EV,
        [-0.01, 1.0, 6.0],
    )


def test_paoflow_pyskeaf_rejects_duplicate_energy_filenames(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'run_paoflow_bxsf_files', lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match='distinct output filename'):
        PAOFLOW.pyskeaf(
            _paoflow_stub(tmp_path),
            fermi_energy=(0.0, -0.0),
            bands=1,
        )


@pytest.mark.parametrize(
    ('kwargs', 'exception_type', 'specific_message'),
    [
        ({'num_interpolation': '100'}, TypeError, 'must be an integer'),
        ({'b_field': 'a'}, ValueError, 'must be one of'),
        (
            {'fermi_energy': [-0.01, 'bad', 2.5]},
            TypeError,
            'fermi_energy[1] must be a real number',
        ),
        (
            {'fermi_energy': np.array([-0.01, 0.0, 0.2])},
            TypeError,
            'explicit Python tuple/list',
        ),
        ({'theta': 0.0}, TypeError, 'Unknown pyskeaf option(s): theta'),
    ],
)
def test_paoflow_pyskeaf_input_errors_include_complete_usage(
    tmp_path, kwargs, exception_type, specific_message
):
    with pytest.raises(exception_type) as error:
        PAOFLOW.pyskeaf(_paoflow_stub(tmp_path), bands=1, **kwargs)

    message = str(error.value)
    assert specific_message in message
    assert 'Correct PAOFLOW.pyskeaf() format:' in message
    assert 'fermi_energy=0.0' in message
    assert '[-0.01, 0.0, 0.2, 1.0]' in message
    assert "b_field='non_principal'" in message
    assert 'azimuthal=0.0' in message
    assert "bands='all'" in message
    assert 'verbose=False' in message
