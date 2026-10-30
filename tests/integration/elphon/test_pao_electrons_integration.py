"""``pao_hamiltonian_from_save`` on a bundled QE save (Al, projwfc projections)."""

import os

import numpy as np
import pytest

from PAOFLOW.elphon.pao_electrons import check_same_pao_basis, pao_hamiltonian_from_save

SAVE = os.path.join(os.path.dirname(__file__), '..', 'transport', 'example01', 'al5.save')

pytestmark = pytest.mark.integration


def test_pao_hamiltonian_from_save_is_reproducible_and_self_consistent(tmp_path):
    first = pao_hamiltonian_from_save(
        SAVE, projector='projwfc', workpath=str(tmp_path), outputdir='first'
    )
    second = pao_hamiltonian_from_save(
        SAVE,
        projector='projwfc',
        keep_projections=False,
        workpath=str(tmp_path),
        outputdir='second',
    )
    assert first.projections.shape[1] == first.HRs.shape[0] == 20
    assert second.projections is None
    np.testing.assert_array_equal(first.HRs, second.HRs)
    # the save lists the full 1 x 1 x 8 grid: no symmetry expansion
    assert first.expand_wedge is False and second.expand_wedge is False
    assert first.hubbard is None
    assert first.nscf['orbital_positions'].shape == (20, 3)
    summary = check_same_pao_basis(first.data_controller, second.data_controller)
    assert summary['nkpnts'] == (8, 8) and summary['bnd'][0] == summary['bnd'][1]
