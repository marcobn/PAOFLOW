"""The VASP POTCAR reader (:mod:`PAOFLOW.inputs.read_potcar`).

Synthetic checks run everywhere; the checks on real datasets need the ``potpaw_PBE`` library,
located through ``PAOFLOW_POTCAR_DIR`` or ``../potpaw_PBE_64`` next to the repository, and are
skipped without it (the files are licensed and not shipped).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.inputs import read_potcar as module
from PAOFLOW.inputs.read_potcar import (
    AUTOA,
    HARTREE_EV,
    PotcarDataset,
    find_potcar_datasets,
    potcar_title,
    read_potcar,
    split_potcar_blocks,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY = Path(os.environ.get('PAOFLOW_POTCAR_DIR', REPO_ROOT.parent / 'potpaw_PBE_64'))


def _potcar(name: str) -> str:
    path = LIBRARY / name / 'POTCAR'
    if not path.exists():
        pytest.skip('POTCAR library not available (%s)' % path)
    return str(path)


@pytest.fixture(scope='module')
def si():
    return read_potcar(_potcar('Si'))[0]


@pytest.fixture(scope='module')
def ga():
    return read_potcar(_potcar('Ga_d'))[0]


# ---------------------------------------------------------------------------
# synthetic
# ---------------------------------------------------------------------------


def test_split_blocks_and_titles():
    text = '  PAW_PBE A 01Jan2000 \n 1.0\nfoo\n End of Dataset\n\n PAW_PBE B 02Feb2000\n 2.0\n End of Dataset\n'
    blocks = split_potcar_blocks(text)
    assert [potcar_title(b) for b in blocks] == ['PAW_PBE A 01Jan2000', 'PAW_PBE B 02Feb2000']
    assert blocks[0][-1].strip() == 'End of Dataset' and len(blocks[0]) == 4


def test_scan_blocks_reads_keywords_numbers_and_the_psmaxn_line():
    lines = [
        ' local part',
        '  81.25',
        '  1.0 2.0 3.0',
        '   16.25  T',
        ' Non local Part',
        '  2 2 1.2',
    ]
    blocks = module._scan_blocks(lines)
    assert [name for name, _ in blocks] == ['local part', 'PSMAXN', 'Non local Part']
    assert blocks[0][1] == [81.25, 1.0, 2.0, 3.0]
    assert blocks[1][1] == [16.25, 1.0]


def test_find_datasets_by_title_and_library_layout(tmp_path, monkeypatch):
    class _Fake:
        def __init__(self, title):
            self.title = title
            self.name = title.split()[1]
            self.source = 'fake'

    registry = {
        str(tmp_path / 'lib' / 'Ga_d' / 'POTCAR'): [_Fake('PAW_PBE Ga_d 06Jul2010')],
        str(tmp_path / 'lib' / 'As' / 'POTCAR'): [_Fake('PAW_PBE As 22Sep2009')],
        str(tmp_path / 'run' / 'POTCAR'): [_Fake('PAW_PBE Si 05Jan2001')],
    }
    for path in registry:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text('x')
    monkeypatch.setattr(module, 'read_potcar', lambda path, **kw: registry[os.path.abspath(path)])
    found = find_potcar_datasets(
        ['PAW_PBE Ga_d 06Jul2010', 'PAW_PBE As 22Sep2009'],
        str(tmp_path / 'run'),
        str(tmp_path / 'lib'),
    )
    assert [d.title for d in found] == ['PAW_PBE Ga_d 06Jul2010', 'PAW_PBE As 22Sep2009']
    # the run folder first
    assert (
        find_potcar_datasets(['PAW_PBE Si 05Jan2001'], str(tmp_path / 'run'), None)[0].name == 'Si'
    )
    # the environment variable
    monkeypatch.setenv('PAOFLOW_POTCAR_DIR', str(tmp_path / 'lib'))
    assert find_potcar_datasets(['PAW_PBE As 22Sep2009'], None, None)[0].name == 'As'
    # same name, other date: accepted with a warning
    with pytest.warns(RuntimeWarning, match='no POTCAR dataset titled'):
        other = find_potcar_datasets(['PAW_PBE As 01Jan1999'], None, str(tmp_path / 'lib'))
    assert other[0].title == 'PAW_PBE As 22Sep2009'
    # nothing found: the message lists the places searched
    monkeypatch.delenv('PAOFLOW_POTCAR_DIR')
    with pytest.raises(FileNotFoundError, match='PAW_PBE Zr_sv 04Jan2005'):
        find_potcar_datasets(
            ['PAW_PBE Zr_sv 04Jan2005'], str(tmp_path / 'run'), str(tmp_path / 'lib')
        )


# ---------------------------------------------------------------------------
# real datasets
# ---------------------------------------------------------------------------


def test_header_projectors_and_units(si, ga):
    assert si.element == 'Si' and si.name == 'Si' and si.val == 4.0 and si.qexc == 'PBE'
    assert ga.element == 'Ga' and ga.name == 'Ga_d' and ga.val == 13.0
    assert si.ptype == 'PAW' and si.has_augmentation and not si.has_spinorbit and si.nlcc
    assert si.lll == [0, 0, 1, 1] and ga.lll == [2, 2, 0, 0, 1, 1]
    assert all(b['j'] is None for b in ga.beta)
    # the Description rows minus the local channel (Ga_d: the l=3 row) match the blocks
    assert len(ga.description) == 7 and ga.reference_energies.size == 6
    assert ga._energies_ev[0] == pytest.approx(-19.1549037)
    # RMAX of the header (bohr) is the projector cutoff
    assert ga.beta[0]['cutoff_radius'] == pytest.approx(2.341, abs=2e-3)
    assert (
        si.r[si.beta[0]['cutoff_index'] - 1]
        <= si.beta[0]['cutoff_radius']
        < si.r[si.beta[0]['cutoff_index']]
    )
    # the stored grid (angstrom) was converted to bohr and extended to 12 bohr
    assert si.npoints_stored == 351 and si.r[si.npoints_stored - 1] == pytest.approx(
        2.584 / AUTOA, rel=1e-3
    )
    assert si.r[-1] >= 12.0 and si.r.size > si.npoints_stored
    assert np.allclose(si.rab, si.r * np.log(si.r[1] / si.r[0]))


def test_augmentation_charges_match_the_partial_waves(si, ga):
    for data in (si, ga):
        n = data.npoints_stored
        q = np.array(
            [
                [
                    np.sum((data.aewfc[i][:n] * data.aewfc[j][:n] - data.pswfc_partial[i][:n] * data.pswfc_partial[j][:n]) * data.rab[:n])
                    for j in range(data.nproj)
                ]
                for i in range(data.nproj)
            ]
        )  # fmt: skip
        same_l = np.equal.outer(data.lll, data.lll)
        assert np.allclose(data.qqq[same_l], q[same_l], atol=2e-4)
        assert np.all(data.qqq[~same_l] == 0.0)
        assert np.allclose(data.qqq, data.qqq.T, atol=1e-12)
    assert ga.qqq[0, 0] == pytest.approx(0.8811, abs=1e-3)  # the 3d charge of the note


def test_densities_and_local_potential(si, ga):
    for data in (si, ga):
        assert np.sum(data.atrho * data.rab) == pytest.approx(data.val, abs=1e-3)
        # the tables of some datasets (Si) dip slightly below zero where the compensation
        # charge is subtracted: at most 0.1 % of the valence charge
        assert np.sum(np.minimum(data.atrho, 0.0) * data.rab) > -2e-3 * data.val
        assert np.all(data.rho_atc >= -1e-12) and data.rho_atc[-1] == 0.0
        # the local potential is Coulombic at the end of the stored grid and beyond
        last = data.npoints_stored - 1
        assert data.vloc[last] * data.r[last] == pytest.approx(-data.val, abs=1e-2)
        assert data.vloc[-1] * data.r[-1] == pytest.approx(-data.val)
    # the all-electron core density holds Z - Z_v electrons (10 for Si, 18 for Ga)
    assert np.sum(4 * np.pi * si.r**2 * si.rho_core_ae * si.rab) == pytest.approx(10.0, abs=1e-3)
    assert np.sum(4 * np.pi * ga.r**2 * ga.rho_core_ae * ga.rab) == pytest.approx(18.0, abs=1e-3)


def test_pseudo_orbitals_are_the_bound_reference_states(si, ga):
    assert [p['label'] for p in si.pswfc] == ['3S', '3P'] and si.shells == [0, 1]
    assert [p['occ'] for p in si.pswfc] == [2.0, 2.0]
    assert [p['label'] for p in ga.pswfc] == ['3D', '4S', '4P'] and ga.shells == [2, 0, 1]
    assert [p['occ'] for p in ga.pswfc] == [10.0, 2.0, 1.0]
    for data in (si, ga):
        for p in data.pswfc:
            l = data.lll[p['projector']]
            # S-normalised: plain norm + q_ii of the matched projector
            assert data.augmented_norm(p['wfc'], l) == pytest.approx(1.0, abs=2e-3)
            # inside the stored grid the orbital is the pseudo partial wave
            n = data.npoints_stored
            assert np.array_equal(p['wfc'][:n], data.pswfc_partial[p['projector']][:n])
            # the continued tail decays
            assert abs(p['wfc'][-1]) < 0.05 * np.abs(p['wfc']).max()
            assert np.all(np.diff(np.abs(p['wfc'][n:])) <= 1e-12)
    assert si.pswfc[0]['energy'] == pytest.approx(-10.8127223 / HARTREE_EV, rel=1e-6)


def test_scattering_states_of_as_and_iodine_are_not_orbitals():
    arsenic = read_potcar(_potcar('As'))[0]
    assert [p['label'] for p in arsenic.pswfc] == ['4S', '4P']  # the empty 4d is a scattering state
    iron = read_potcar(_potcar('Fe'))[0]
    assert [p['label'] for p in iron.pswfc] == ['3D', '4S']  # the p channel sits at -0.1 Ha


def test_projectors_are_dual_to_the_partial_waves(si, ga):
    for data in (si, ga):
        n = data.nproj
        dual = np.array(
            [
                [np.sum(data.beta[i]['wfc'] * data.pswfc_partial[j] * data.rab) if data.lll[i] == data.lll[j] else 0.0 for j in range(n)]
                for i in range(n)
            ]
        )  # fmt: skip
        assert np.allclose(dual, np.eye(n), atol=1e-10)
        # the raw inverse transforms are dual to a few per cent only
        assert np.allclose(np.diag(data.duality), 1.0, atol=0.2)
        assert np.abs(data.duality - np.eye(n)).max() < 0.4
        for b in data.beta:
            assert np.all(b['wfc'][b['cutoff_index'] :] == 0.0)


def test_reciprocal_tables_are_the_file_values_in_bohr_units(ga):
    q, table = ga.projector_q_tables[0]
    assert q.size == 100 and q[0] == 0.0 and q[1] == pytest.approx(ga.psmaxn / 100)
    assert ga.psmaxn == pytest.approx(16.2516435017787 * AUTOA)
    assert table[7] == pytest.approx(0.26914704 / AUTOA**1.5, rel=1e-6)  # 8th value of the file
    assert ga.lreal_flag is True


def test_screened_strengths(si, ga):
    for data in (si, ga):
        assert data.dion.shape == (data.nproj, data.nproj)
        assert np.allclose(data.dion, data.dion.T)
        same_l = np.equal.outer(data.lll, data.lll)
        assert np.all(data.dion[~same_l] == 0.0) and np.all(data.dion_file[~same_l] == 0.0)
        assert data.dion_screened_asymmetry < 0.1
        assert data.nonlocal_strength == 'screened' and data.dion is data.dion_screened
        alt = data.with_nonlocal_strength('file')
        assert alt.dion is alt.dion_file and alt.qqq is data.qqq
    assert ga.dion_file[0, 0] == pytest.approx(44.5077080701093 / HARTREE_EV)


def test_partial_wave_dipole_integrals_are_well_defined(ga):
    from PAOFLOW.hamiltonian.position_dipoles import partial_wave_dipole_radial

    dipole = partial_wave_dipole_radial(ga)
    assert dipole.shape == (6, 6) and np.all(np.isfinite(dipole))
    assert np.allclose(dipole, dipole.T, atol=1e-10)


def test_summary_and_repr(si):
    text = si.summary()
    assert 'PAW_PBE Si 05Jan2001' in text and '3S' in text and 'scattering state' in text
    assert "pswfc=['3S', '3P']" in repr(si)


def test_ultrasoft_potcar_is_refused():
    lines = [
        '  US A 01Jan2000',
        ' 1.0',
        '   LPAW   =        F    paw PP',
        ' END of PSCTR-controll parameters',
    ]
    with pytest.raises(NotImplementedError):
        PotcarDataset(lines, 'synthetic')
