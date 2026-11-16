"""The basis generator on VASP POTCAR datasets.

The radial solver of :mod:`PAOFLOW.basis_gen` consumes a :class:`PotcarDataset` through the
UPF attributes. With the screened strengths of the dataset the bound reference states of the
PAW generation are eigenstates of the generalized box problem; the pseudo-orbitals are taken
as stored and the augmenting shells come from the box, as for UPF files.
Needs the ``potpaw_PBE`` library (``PAOFLOW_POTCAR_DIR`` or ``../potpaw_PBE_64``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from PAOFLOW.basis_gen import generate_basis_for_pseudo, generate_basis_shells, pseudize_shell
from PAOFLOW.basis_gen.driver import _default_shells
from PAOFLOW.basis_gen.radial import (
    _augmented_overlap_operator,
    lowest_valence_n,
    select_physical_states,
    solve_radial_channel,
)
from PAOFLOW.inputs.read_potcar import HARTREE_EV, read_potcar

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _s_product(dataset, l, r_uni, a, b):
    S = _augmented_overlap_operator(dataset, l, None, r_uni)
    dr = r_uni[1] - r_uni[0]
    return float(a @ S @ b) * dr


@pytest.mark.parametrize('name', ['Si', 'Ga_d', 'Fe'])
def test_bound_reference_states_are_eigenstates_of_the_box_problem(name):
    data = read_potcar(_potcar(name))[0]
    for orbital in data.pswfc:
        i = orbital['projector']
        l = data.lll[i]
        eps, U, r = solve_radial_channel(data, l, n_states=6)
        k = int(np.argmin(np.abs(eps - orbital['energy'])))
        assert abs(eps[k] - orbital['energy']) * HARTREE_EV < 0.3, (name, orbital['label'])
        reference = np.nan_to_num(
            CubicSpline(data.r, orbital['wfc'], extrapolate=False)(r), nan=0.0
        )
        overlap = abs(_s_product(data, l, r, U[k], reference)) / np.sqrt(
            _s_product(data, l, r, U[k], U[k]) * _s_product(data, l, r, reference, reference)
        )
        assert overlap > 0.995, (name, orbital['label'], overlap)


def test_file_strengths_do_not_reproduce_the_reference_states(si):
    """The unscreened D_ij of the file (the QE route's PP_DIJ) misplace the 3s level by eV."""
    alt = si.with_nonlocal_strength('file')
    eps, _, _ = solve_radial_channel(alt, 0, n_states=6)
    closest = eps[np.argmin(np.abs(eps - si.pswfc[0]['energy']))]
    assert abs(closest - si.pswfc[0]['energy']) * HARTREE_EV > 1.0


def test_pseudize_shell_returns_the_stored_orbital_and_box_states(ga):
    r, u, eps = pseudize_shell(ga, 3, 2)
    assert np.isnan(eps)
    assert np.sum(u * u) * (r[1] - r[0]) == pytest.approx(1.0)
    r2, u2, eps2 = pseudize_shell(ga, 4, 2)  # the 4d polarisation shell: a box state
    assert np.isfinite(eps2) and r2.shape == r.shape
    assert np.sum(u * u2) * (r[1] - r[0]) != pytest.approx(1.0)
    with pytest.raises(ValueError, match='below lowest'):
        pseudize_shell(ga, 2, 2)


def test_ghost_filter_keeps_physical_states(si):
    eps, U, r = solve_radial_channel(si, 0, n_states=8)
    ranks = select_physical_states(si, 0, None, eps, U, r)
    # the 3s (reproduces the stored orbital) is excluded; the remaining states are box states
    assert ranks and all(eps[k] > si.pswfc[0]['energy'] for k in ranks)
    assert 0 not in ranks or eps[0] * HARTREE_EV > -5.0


def test_lowest_valence_n_from_the_core_of_the_dataset(si, ga):
    assert lowest_valence_n(si, 2) == 3
    assert lowest_valence_n(ga, 2) == 3  # Ga_d keeps the 3d in the valence
    iron = read_potcar(_potcar('Fe'))[0]
    assert lowest_valence_n(iron, 1) == 4


def test_generate_basis_shells_one_mesh_and_sources(si):
    shells, skipped = generate_basis_shells(si, ['3S', '3P', '3D', '4S', '4P'])
    assert skipped == []
    assert [s.label for s in shells] == ['3S', '3P', '3D', '4S', '4P']
    assert all(np.array_equal(s.r, shells[0].r) for s in shells)
    assert shells[0].from_pseudo_orbital and shells[1].from_pseudo_orbital
    assert all(not s.from_pseudo_orbital for s in shells[2:])
    assert all(s.j is None for s in shells)
    assert shells[0].filename == '3S.dat'


def test_presets_on_a_dataset(si, ga):
    assert _default_shells(si, 'minimal') == ['3S', '3P']
    assert _default_shells(si, 'standard') == ['3S', '3P', '3D', '4S', '4P']
    extended = _default_shells(ga, 'extended')
    assert extended[:3] == ['3D', '4S', '4P'] and '4D' in extended and '5S' in extended


def test_generator_cli_writes_tables_from_a_potcar(tmp_path):
    from PAOFLOW.basis_gen.cli import main

    out = tmp_path / 'BASIS_PS'
    assert main(['--pseudo', _potcar('Si'), '--out', str(out), '--preset', 'minimal']) == 0
    assert sorted(p.name for p in (out / 'Si').iterdir()) == ['3P.dat', '3S.dat']
    data = np.loadtxt(out / 'Si' / '3S.dat')
    assert data.shape[1] == 2 and np.sum(data[:, 1] ** 2) * (
        data[1, 0] - data[0, 0]
    ) == pytest.approx(1.0, abs=1e-6)
    written = generate_basis_for_pseudo(_potcar('Si'), str(tmp_path / 'again'), shells=['3D'])
    assert [Path(p).name for p in written] == ['3D.dat']
