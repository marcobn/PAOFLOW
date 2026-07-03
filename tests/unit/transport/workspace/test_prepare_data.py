"""Unit tests for workspace preparation helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.workspace import prepare_data
from PAOFLOW.transport.utils.memusage import MemoryTracker


class DummyData:
    class DummyEnergy:
        smearing_type = 'lorentzian'
        delta = 0.2
        delta_ratio = 0.5
        xmax = 1.0

    class DummySymmetry:
        write_lead_sgm = False
        write_gf = False
        use_sym = True

    energy = DummyEnergy()
    symmetry = DummySymmetry()

    dimL = 1
    dimC = 1
    dimR = 1

    class DummyRuntime:
        nkpts_par = 2
        nrtot_par = 3
        vkpt_par3D = np.zeros((2, 3))
        wk_par = np.array([0.5, 0.5])
        ivr_par3D = np.zeros((3, 2), dtype=int)
        wr_par = np.ones(2)

    _runtime = DummyRuntime()

    def get_runtime_data(self):
        return self._runtime


@pytest.mark.unit
def test_prepare_smearing_registers_memory():
    """Smearing preparation should register memory usage in the tracker."""
    tracker = MemoryTracker()
    data = DummyData()

    smearing = prepare_data.prepare_smearing(data, tracker)

    assert smearing.xgrid is not None
    assert 'smearing' in tracker.sections


@pytest.mark.unit
def test_prepare_kpoints_registers_memory():
    """K-point preparation should register memory usage in the tracker."""
    tracker = MemoryTracker()
    data = DummyData()

    kpoints_data = prepare_data.prepare_kpoints(data, tracker)

    assert kpoints_data.vkpt_par3D is not None
    assert 'kpoints' in tracker.sections


@pytest.mark.unit
def test_prepare_hamiltonian_system_registers_memory():
    """Hamiltonian system preparation should register memory usage sections."""
    tracker = MemoryTracker()
    data = DummyData()

    ham = prepare_data.prepare_hamiltonian_system(data, tracker)

    assert 'hamiltonian data' in tracker.sections
    assert 'correlation data' in tracker.sections
    assert ham.dimC == 1


@pytest.mark.unit
def test_prepare_workspace_allocates():
    """Workspace preparation should allocate and register memory usage."""
    tracker = MemoryTracker()
    data = DummyData()

    workspace = prepare_data.prepare_workspace(data, tracker)

    assert workspace.allocated
    assert 'workspace' in tracker.sections
