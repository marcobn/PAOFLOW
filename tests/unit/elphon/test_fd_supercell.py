"""Displaced runs of the ideal supercell of a crystal (the finite-difference coupling source)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

EXAMPLE = Path(__file__).resolve().parents[3] / 'examples' / 'qe_examples' / 'example01'


def test_supercell_displacements_from_the_primitive_save(tmp_path):
    pytest.importorskip('phonopy')
    from PAOFLOW.elphon.workflows.finite_differences import (
        fd_runs_done,
        reference_save,
        write_supercell_displacements,
    )

    fd = tmp_path / 'qe' / 'fd'
    pseudos = tmp_path / 'pseudo'
    pseudos.mkdir()
    (pseudos / 'Si.pbe-tm-new-gipaw-v2.1.UPF').write_bytes(
        (EXAMPLE / 'Si.pbe-tm-new-gipaw-v2.1.UPF').read_bytes()
    )
    result = write_supercell_displacements(
        str(EXAMPLE / 'silicon.save'), str(fd), (2, 2, 2), prefix='silicon', kgrid=(4, 4, 4),
        pp_dir='../../pseudo',
    )  # fmt: skip
    # every atom of the 16-atom supercell along +-x, +-y, +-z, plus the reference
    assert result['atoms'] == 16 and result['cells'] == 96 and len(result['inputs']) == 97
    manifest = json.loads((fd / 'displacements.json').read_text())
    assert manifest['displacement_mode'] == 'cartesian'
    assert manifest['reference_prefix'] == 'silicon_sc_ref'
    assert {d['sc_atom'] for d in manifest['displacements']} == set(range(16))
    assert manifest['nbnd'] > 16 * 4  # sized to the PAO basis through the relative pseudo_dir
    reference = (fd / 'supercell.in').read_text()
    assert "prefix = 'silicon_sc_ref'" in reference.replace('  ', ' ')
    assert 'nat = 16' in reference.replace('  ', ' ') and '4 4 4 0 0 0' in reference
    assert "'../../pseudo'" in reference
    assert (fd / 'run_fd.sh').is_file() and fd_runs_done(str(fd)) == (0, 97)
    assert reference_save('qe/fd', 'silicon') == 'qe/fd/tmp_silicon_sc_ref/silicon_sc_ref.save'
    # random patterns: 2 nconfig cells, all atoms displaced at once
    result = write_supercell_displacements(
        str(EXAMPLE / 'silicon.save'), str(tmp_path / 'random'), (2, 2, 2), prefix='silicon',
        kgrid=(4, 4, 4), mode='random', nconfig=3, seed=7, pp_dir=str(pseudos),
    )  # fmt: skip
    manifest = json.loads((tmp_path / 'random' / 'displacements.json').read_text())
    assert result['cells'] == 6 and manifest['displacement_mode'] == 'random'
    assert manifest['seed'] == 7 and manifest['nconfig'] == 3
    with pytest.raises(ValueError, match='cartesian'):
        write_supercell_displacements(
            str(EXAMPLE / 'silicon.save'), str(tmp_path / 'x'), (2, 2, 2), prefix='silicon',
            kgrid=(4, 4, 4), mode='symmetry',
        )  # fmt: skip
