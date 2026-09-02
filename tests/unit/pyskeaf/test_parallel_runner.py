from types import SimpleNamespace

import numpy as np

import PAOFLOW.pyskeaf.runner as runner
from PAOFLOW.pyskeaf.config import SkeafConfig
from PAOFLOW.pyskeaf.io_bxsf import BXSFData


class _RootComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def allgather(self, value):
        return [value, None]

    def gather(self, local_results, root=0):
        return [local_results, [(1, ['angle-1']), (3, ['angle-3'])]]

    def bcast(self, value, root=0):
        return value


class _NonRootComm:
    def __init__(self, ordered_results):
        self._ordered_results = ordered_results
        self._bcast_count = 0

    def Get_rank(self):
        return 1

    def Get_size(self):
        return 2

    def allgather(self, value):
        return [None, value]

    def gather(self, local_results, root=0):
        return None

    def bcast(self, value, root=0):
        self._bcast_count += 1
        if self._bcast_count == 1:
            return None
        return self._ordered_results


def test_mpi_angle_jobs_are_reassembled_in_input_order():
    jobs = [(index, 0.0, float(index)) for index in range(4)]

    def worker(job):
        return job[0], [f'angle-{job[0]}']

    results = runner._run_mpi_angle_jobs(jobs, worker, _RootComm())

    assert results == [
        (0, ['angle-0']),
        (1, ['angle-1']),
        (2, ['angle-2']),
        (3, ['angle-3']),
    ]


def test_nonroot_mpi_rank_does_not_write_output(monkeypatch, tmp_path):
    ordered = [(index, [f'angle-{index}']) for index in range(4)]
    comm = _NonRootComm(ordered)
    monkeypatch.setattr(runner, 'active_mpi_comm', lambda: comm)
    monkeypatch.setattr(
        runner,
        'run_at_angle',
        lambda *args, **kwargs: SimpleNamespace(orbits=['local']),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError('a nonroot MPI rank attempted to write output')

    monkeypatch.setattr(runner, 'write_results_freqvsangle', fail_if_called)
    monkeypatch.setattr(runner, 'write_results_short', fail_if_called)
    monkeypatch.setattr(runner, 'write_results_long', fail_if_called)
    monkeypatch.setattr(runner, 'write_orbit_outlines', fail_if_called)

    cfg = SkeafConfig(
        hvd='r',
        num_rots=4,
        numint=2,
        theta_start=0.0,
        theta_end=0.0,
        phi_start=0.0,
        phi_end=1.0,
    )
    bxsf = SimpleNamespace(filename='mpi-test.bxsf', fermi_energy=0.0)

    result = runner.run_skeaf(cfg, bxsf, write_files=True, output_dir=tmp_path)

    assert result.orbits == ['angle-0', 'angle-1', 'angle-2', 'angle-3']
    assert list(tmp_path.iterdir()) == []


def test_nonverbose_run_writes_only_prefixed_frequency_result(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(runner, 'active_mpi_comm', lambda: None)
    monkeypatch.setattr(
        runner,
        'run_at_angle',
        lambda *args, **kwargs: SimpleNamespace(orbits=[]),
    )
    monkeypatch.setattr(
        runner,
        'write_results_freqvsangle',
        lambda result, path: written.append(path.name),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError('an auxiliary result writer was called')

    monkeypatch.setattr(runner, 'write_results_short', fail_if_called)
    monkeypatch.setattr(runner, 'write_results_long', fail_if_called)
    monkeypatch.setattr(runner, 'write_orbit_outlines', fail_if_called)

    cfg = SkeafConfig(hvd='n', numint=2)
    bxsf = SimpleNamespace(filename='manual_7.bxsf', fermi_energy=0.0, recip_ang=np.eye(3))

    runner.run_skeaf(
        cfg,
        bxsf,
        output_dir=tmp_path,
        output_suffix='manual_7',
        write_auxiliary_files=False,
    )

    assert written == ['qo_results_freqvsangle_manual_7.out']


def test_paoflow_bxsf_band_energies_remain_in_rydberg(monkeypatch, tmp_path):
    path = tmp_path / 'Fermi_surf_band_1.bxsf'
    path.touch()
    energies_ry = np.array([[[-0.01, 0.01]]])
    bxsf = BXSFData(
        filename=str(path),
        fermi_energy=0.0,
        nx=1,
        ny=1,
        nz=2,
        recip_au=np.eye(3),
        recip_ang=np.eye(3),
        energies=energies_ry.copy(),
    )
    passed_energies = []

    monkeypatch.setattr(runner, 'read_bxsf', lambda unused_path: bxsf)

    def fake_run_skeaf(config, data, **kwargs):
        passed_energies.append(data.energies.copy())
        return SimpleNamespace()

    monkeypatch.setattr(runner, 'run_skeaf', fake_run_skeaf)

    runs = runner.run_paoflow_bxsf_files(
        SkeafConfig(fermi_energy=0.0),
        input_dir=tmp_path,
        all_files=True,
    )

    assert len(runs) == 1
    assert runs[0].calculated
    assert np.array_equal(passed_energies[0], energies_ry)
    assert np.isclose(runs[0].minimum_ev, -0.01 * runner._RYDBERG_IN_EV)
    assert np.isclose(runs[0].maximum_ev, 0.01 * runner._RYDBERG_IN_EV)


def test_custom_bxsf_uses_complete_stem_for_output_suffix(monkeypatch, tmp_path):
    path = tmp_path / 'manual_band_7.bxsf'
    path.touch()
    bxsf = BXSFData(
        filename=str(path),
        fermi_energy=0.0,
        nx=1,
        ny=1,
        nz=2,
        recip_au=np.eye(3),
        recip_ang=np.eye(3),
        energies=np.array([[[-0.01, 0.01]]]),
    )
    calls = []
    monkeypatch.setattr(runner, 'read_bxsf', lambda unused_path: bxsf)

    def fake_run_skeaf(config, data, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(runner, 'run_skeaf', fake_run_skeaf)

    runner.run_paoflow_bxsf_files(
        SkeafConfig(fermi_energy=0.0),
        input_dir=tmp_path,
        filenames=[path.name],
        write_auxiliary_files=False,
    )

    assert calls[0]['output_suffix'] == 'manual_band_7'
    assert calls[0]['write_auxiliary_files'] is False


def test_paoflow_bxsf_progress_is_reported_after_each_file(monkeypatch, tmp_path):
    paths = [
        tmp_path / 'Fermi_surf_band_1.bxsf',
        tmp_path / 'Fermi_surf_band_2.bxsf',
    ]
    for path in paths:
        path.touch()

    def fake_read_bxsf(path):
        energies = (
            np.array([[[-0.01, 0.01]]])
            if path.name.endswith('_1.bxsf')
            else np.array([[[0.02, 0.03]]])
        )
        return BXSFData(
            filename=str(path),
            fermi_energy=0.0,
            nx=1,
            ny=1,
            nz=2,
            recip_au=np.eye(3),
            recip_ang=np.eye(3),
            energies=energies,
        )

    monkeypatch.setattr(runner, 'read_bxsf', fake_read_bxsf)
    monkeypatch.setattr(runner, 'run_skeaf', lambda *args, **kwargs: SimpleNamespace())
    progress = []

    runs = runner.run_paoflow_bxsf_files(
        SkeafConfig(fermi_energy=0.0),
        input_dir=tmp_path,
        all_files=True,
        progress_callback=lambda item: progress.append(
            (item.path.name, item.calculated, item.skipped_reason)
        ),
    )

    assert len(runs) == 2
    assert progress[0] == ('Fermi_surf_band_1.bxsf', True, None)
    assert progress[1][0:2] == ('Fermi_surf_band_2.bxsf', False)
    assert 'outside' in progress[1][2]
