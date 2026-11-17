"""Legacy POTCAR coverage using analytic, freely generated radial data.

No VASP potential tables are distributed here. The optional CrSBr checks use a local
calculation selected through ``PAOFLOW_VASP_LEGACY_RUN``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.integrate import simpson
from scipy.special import erf, spherical_jn

from PAOFLOW.inputs import read_potcar as module
from PAOFLOW.inputs.read_potcar import AUTOA, HARTREE_EV, PotcarDataset, read_potcar

SyntheticPotcar = Callable[..., tuple[str, np.ndarray, np.ndarray]]


def _numbers(values: ArrayLike) -> str:
    return ' '.join('%.15e' % value for value in np.asarray(values).ravel())


def _synthetic_potcar(
    name: str = 'Si', *, legacy: bool = False, omit_configuration: bool = False
) -> tuple[str, np.ndarray, np.ndarray]:
    """Build a toy PAW atom with Gaussian waves and known augmentation norms.

    The Gaussian formulas are independent of any licensed potential. The valence shell
    labels exercise the same channel ordering as the CrSBr calculation; the reference
    energies are arbitrary negative numbers matched explicitly in the header.
    """
    element = name.split('_')[0]
    configurations = {
        'H': [('1S', 1.0)],
        'Si': [('3S', 2.0), ('3P', 2.0)],
        'Cr': [('3P', 6.0), ('3D', 5.0), ('4S', 1.0)],
        'S': [('3S', 2.0), ('3P', 4.0)],
        'Br': [('4S', 2.0), ('4P', 5.0)],
    }
    shells = configurations[element]
    nproj = len(shells)
    zval = sum(occ for _, occ in shells)
    r = np.geomspace(1.0e-4, 12.5, 180)
    rab = r * np.log(r[1] / r[0])
    width = 0.9
    qmax = 16.0
    q = np.arange(module.NPSPTS) * qmax / module.NPSPTS
    qproj = np.arange(module.NPSNL) * qmax / module.NPSNL
    qpotential = np.zeros_like(q)
    qpotential[1:] = -4 * np.pi * zval * np.expm1(-0.25 * width**2 * q[1:] ** 2) / q[1:] ** 2
    # VASP's local table has the singular bare Coulomb term removed.
    qpotential[0] = np.pi * zval * width**2
    vloc = -zval * erf(r / width) / r
    energies = [0.0] if element == 'H' else [-0.6 + 0.1 * i for i in range(nproj)]
    pseudo_norm = 1.0 if element == 'H' else 0.85
    waves, projector_tables = [], []
    for label, _ in shells:
        l = 'SPDF'.index(label[1])
        u = r * np.exp(-r) if element == 'H' else r ** (l + 1) * np.exp(-0.6 * r**2)
        u /= np.sqrt(np.sum(u * u * rab))
        waves.append(np.sqrt(pseudo_norm) * u)
        # The overall projector scale is immaterial: the reader restores duality.
        radial_projector = r**l * np.exp(-0.6 * r**2)
        transform = (
            4
            * np.pi
            * simpson(
                spherical_jn(l, np.multiply.outer(qproj, r)) * (radial_projector * r**2)[None, :],
                x=r,
                axis=1,
            )
        )
        projector_tables.append(transform)

    lines = [
        'PAW_PBE %s analytic-test' % name,
        str(zval),
        ' LPAW = T; LEXCH = PE; POMASS = 1.0',
        ' VRHFIN =%s:%s' % (element, 's1' if element == 'H' else 'synthetic'),
    ]
    if not omit_configuration:
        lines.extend([' Atomic configuration', str(nproj), ' n l j E occ.'])
        for (label, occ), energy in zip(shells, energies):
            lines.append(
                '%s %s 0.0 %.15e %.1f'
                % (label[0], 'SPDF'.index(label[1]), energy * HARTREE_EV, occ)
            )
    lines.extend([' Description', ' l E TYP RCUT'])
    for (label, _), energy in zip(shells, energies):
        lines.append('%d %.15e 23 3.0' % ('SPDF'.index(label[1]), energy * HARTREE_EV))
    lines.extend(
        [
            ' END of PSCTR-controll parameters',
            ' local part',
            _numbers([qmax / AUTOA]),
            _numbers(qpotential * HARTREE_EV * AUTOA**3),
            ' core charge-density (partial)',
            _numbers(np.zeros(module.NPSPTS)),
            ' atomic pseudo charge-density',
            _numbers(zval * np.exp(-0.25 * q**2)),
            ' %.15e T' % (qmax / AUTOA),
        ]
    )
    for (label, _), table in zip(shells, projector_tables):
        lines.extend(
            [
                ' Non local Part',
                '%d 1 %.15e' % ('SPDF'.index(label[1]), 3.0 * AUTOA),
                '1.0',
                ' Reciprocal Space Part',
                _numbers(table * AUTOA**1.5),
                ' Real Space Part',
                _numbers(np.zeros(module.NPSNL)),
            ]
        )
    lines.extend(
        [
            ' PAW radial sets',
            ' augmentation charges (non sperical)',
            _numbers((1.0 - pseudo_norm) * np.eye(nproj)),
            ' uccopancies in atom',
            _numbers(np.diag([occ for _, occ in shells])),
            ' grid',
            _numbers(r * AUTOA),
        ]
    )
    if not legacy:
        lines.extend([' local pseudopotential core', _numbers(vloc * HARTREE_EV)])
    lines.extend(
        [
            ' aepotential',
            _numbers(np.zeros_like(r)),
            ' pspotential' if legacy else ' pspotential valence only',
            _numbers((vloc + 1.0 / r) * HARTREE_EV if element == 'H' else np.zeros_like(r)),
            ' core charge-density',
            _numbers(np.zeros_like(r)),
            ' core charge-density (pseudized)',
            _numbers(np.zeros_like(r)),
        ]
    )
    for u in waves:
        lines.extend(
            [
                ' pseudo wavefunction',
                _numbers(u / np.sqrt(AUTOA)),
                ' ae wavefunction',
                _numbers(u / np.sqrt(pseudo_norm * AUTOA)),
            ]
        )
    lines.append(' End of Dataset')
    return '\n'.join(lines) + '\n', r, vloc


@pytest.fixture
def analytic_tables(monkeypatch: pytest.MonkeyPatch) -> SyntheticPotcar:
    """Small reciprocal tables retain the formats without expensive production meshes."""
    monkeypatch.setattr(module, 'NPSPTS', 128)
    monkeypatch.setattr(module, 'NPSNL', 64)
    monkeypatch.setattr(module, 'vasp_license_notice', lambda: None)
    return _synthetic_potcar


@pytest.mark.parametrize('width', [0.6, 1.3])
def test_reciprocal_local_potential_recovers_soft_coulomb(width: float) -> None:
    """A finite q mesh reproduces an analytic local potential, including its tail."""
    q = np.linspace(0.0, 40.0, 1000, endpoint=False)
    zval = 7.0
    table = np.zeros_like(q)
    table[1:] = -4 * np.pi * zval * np.expm1(-0.25 * width**2 * q[1:] ** 2) / q[1:] ** 2
    table[0] = np.pi * zval * width**2
    r = np.geomspace(1.0e-4, 12.0, 180)
    actual = module._local_potential_from_reciprocal(q, table, zval, r)
    expected = -zval * erf(r / width) / r
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-5)


def test_explicit_configuration_and_modern_radial_data_are_preserved(
    analytic_tables: SyntheticPotcar,
) -> None:
    text, r, vloc = analytic_tables()
    dataset = PotcarDataset(text.splitlines(), source='analytic', r_extend=None)
    np.testing.assert_allclose(dataset.r, r, rtol=1e-12)
    np.testing.assert_allclose(dataset.vloc, vloc, rtol=1e-12)
    assert [orbital['label'] for orbital in dataset.pswfc] == ['3S', '3P']
    assert [orbital['occ'] for orbital in dataset.pswfc] == [2.0, 2.0]
    np.testing.assert_allclose(dataset.reference_energies, [-0.6, -0.5], atol=1e-12)
    for orbital in dataset.pswfc:
        i = orbital['projector']
        np.testing.assert_array_equal(orbital['wfc'], dataset.pswfc_partial[i])
        assert dataset.augmented_norm(orbital['wfc'], dataset.lll[i]) == pytest.approx(
            1.0, abs=1e-10
        )
        assert np.sum(orbital['wfc'] ** 2 * dataset.rab) == pytest.approx(0.85)


def test_old_radial_block_names_and_missing_local_core_are_supported(
    analytic_tables: SyntheticPotcar,
) -> None:
    text, r, vloc = analytic_tables(legacy=True)
    dataset = PotcarDataset(text.splitlines(), source='analytic legacy', r_extend=None)
    assert [orbital['label'] for orbital in dataset.pswfc] == ['3S', '3P']
    np.testing.assert_allclose(dataset.r, r, rtol=1e-12)
    np.testing.assert_allclose(dataset.vloc, vloc, atol=2e-3, rtol=2e-4)
    assert np.all(np.isfinite(dataset.dion_screened))
    assert np.allclose(dataset.qqq, 0.15 * np.eye(2))


def test_missing_configuration_and_placeholder_energy_recover_hydrogen(
    analytic_tables: SyntheticPotcar,
) -> None:
    """The known Coulomb 1s level is recovered without trusting a zero reference energy."""
    from PAOFLOW.basis_gen import generate_basis_shells
    from PAOFLOW.basis_gen.atomic_potential import frozen_effective_potential

    text, _, _ = analytic_tables('H', legacy=True, omit_configuration=True)
    dataset = PotcarDataset(text.splitlines(), source='analytic hydrogen', r_extend=None)
    assert [orbital['label'] for orbital in dataset.pswfc] == ['1S']
    assert dataset.pswfc[0]['occ'] == pytest.approx(1.0)
    assert dataset.pswfc[0]['energy'] == pytest.approx(-0.5, abs=2e-3)
    np.testing.assert_allclose(dataset.reference_energies, [-0.5], atol=2e-3)
    assert dataset.augmented_norm(dataset.pswfc[0]['wfc'], 0) == pytest.approx(1.0, abs=2e-3)
    np.testing.assert_allclose(
        frozen_effective_potential(dataset), -1.0 / dataset.r, atol=2e-3, rtol=2e-4
    )
    shells, skipped = generate_basis_shells(dataset, ['1S'], n_points=240, r_box=10.0)
    assert skipped == [] and [shell.label for shell in shells] == ['1S']
    assert shells[0].from_pseudo_orbital
    assert np.max(np.abs(shells[0].wfc[-10:])) < 0.01 * np.max(np.abs(shells[0].wfc))
    assert np.all(np.isfinite(dataset._effective_potential()))


@pytest.mark.parametrize(
    ('configuration', 'message'),
    [
        ('', 'Atomic configuration.*VRHFIN'),
        ('s3', 'invalid VRHFIN occupation'),
        ('s0.5', 'do not sum to ZVAL'),
        ('s0.5s0.5', 'several occupied s shells'),
        ('p1', 'occupied VRHFIN channels lack PAW projectors'),
    ],
)
def test_missing_atomic_metadata_fails_with_actionable_error(
    analytic_tables: SyntheticPotcar,
    configuration: str,
    message: str,
) -> None:
    """Unsupported headers fail explicitly instead of silently producing an empty basis."""
    text, _, _ = analytic_tables('H', legacy=True, omit_configuration=True)
    text = text.replace('VRHFIN =H:s1', 'VRHFIN =H:' + configuration)
    with pytest.raises(ValueError, match=message):
        PotcarDataset(text.splitlines(), source='invalid analytic header', r_extend=None)


@pytest.mark.parametrize(
    ('element', 'zval', 'configuration', 'expected'),
    [
        ('Ga', 13.0, 's2p1', [(3, 2, 10.0), (4, 0, 2.0), (4, 1, 1.0)]),
        ('Fe', 14.0, 'd7s1', [(3, 1, 6.0), (3, 2, 7.0), (4, 0, 1.0)]),
    ],
)
def test_zval_identifies_unlisted_filled_semicore_channels(
    element: str,
    zval: float,
    configuration: str,
    expected: list[tuple[int, int, float]],
) -> None:
    """A uniquely identified filled semicore shell is retained when VRHFIN omits it."""
    from PAOFLOW.inputs.potcar_atomic import _valence_shells

    dataset = SimpleNamespace(
        element=element,
        val=zval,
        lll=[0, 1, 2],
        valence_configuration=configuration,
        title='analytic ' + element,
    )
    assert _valence_shells(dataset) == expected


def test_concatenated_potcar_cli_generates_every_element(
    tmp_path: Path,
    analytic_tables: SyntheticPotcar,
) -> None:
    from PAOFLOW.basis_gen.cli import main

    potcar = tmp_path / 'POTCAR'
    potcar.write_text(''.join(analytic_tables(name)[0] for name in ('Cr_pv', 'S', 'Br')))
    output = tmp_path / 'basis'
    assert (
        main(
            [
                '--pseudo',
                str(potcar),
                '--out',
                str(output),
                '--preset',
                'minimal',
                '--n-points',
                '240',
            ]
        )
        == 0
    )
    expected = {
        'Cr': {'3P.dat', '3D.dat', '4S.dat'},
        'S': {'3S.dat', '3P.dat'},
        'Br': {'4S.dat', '4P.dat'},
    }
    assert {p.name for p in output.iterdir()} == set(expected)
    for element, files in expected.items():
        assert {p.name for p in (output / element).iterdir()} == files
        for table in (output / element).iterdir():
            radial = np.loadtxt(table)
            assert radial.shape == (239, 2)
            assert np.all(np.isfinite(radial))
            dr = radial[1, 0] - radial[0, 0]
            assert np.sum(radial[:, 1] ** 2) * dr == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize('layout', ['concatenated', 'directory'])
def test_potential_variants_cannot_silently_overwrite_each_other(
    tmp_path: Path,
    analytic_tables: SyntheticPotcar,
    layout: str,
) -> None:
    from PAOFLOW.basis_gen.driver import generate_basis_for_directory, generate_basis_for_pseudo

    output = tmp_path / 'basis'
    if layout == 'concatenated':
        potcar = tmp_path / 'POTCAR'
        potcar.write_text(analytic_tables('Si')[0] + analytic_tables('Si_sv')[0])
        generate = lambda: generate_basis_for_pseudo(
            str(potcar), str(output), preset='minimal', n_points=120
        )
    else:
        library = tmp_path / 'library'
        for name in ('Si', 'Si_sv'):
            dataset_path = library / name
            dataset_path.mkdir(parents=True)
            (dataset_path / 'POTCAR').write_text(analytic_tables(name)[0])
        generate = lambda: generate_basis_for_directory(
            str(library), str(output), preset='minimal', n_points=120
        )
    with pytest.raises(ValueError, match='Si'):
        generate()
    assert not list(output.rglob('*.dat'))


@pytest.fixture(scope='module')
def legacy_run() -> Path:
    location = os.environ.get('PAOFLOW_VASP_LEGACY_RUN')
    if not location:
        pytest.skip('set PAOFLOW_VASP_LEGACY_RUN to a local CrSBr calculation')
    directory = Path(location).expanduser()
    if not (directory / 'POTCAR').is_file():
        pytest.skip('POTCAR unavailable in PAOFLOW_VASP_LEGACY_RUN')
    return directory


def test_local_crsbr_recovers_valence_and_semicore_shells(legacy_run: Path) -> None:
    from PAOFLOW.basis_gen import generate_basis_shells

    datasets = read_potcar(str(legacy_run / 'POTCAR'))
    assert [dataset.name for dataset in datasets] == ['Cr_pv', 'S', 'Br']
    expected = {'Cr_pv': {'3P', '3D', '4S'}, 'S': {'3S', '3P'}, 'Br': {'4S', '4P'}}
    spinor_dimension = 0
    for dataset in datasets:
        labels = [orbital['label'] for orbital in dataset.pswfc]
        assert set(labels) == expected[dataset.name]
        assert np.all(np.isfinite(dataset.reference_energies))
        assert np.any(dataset.reference_energies < 0.0)
        assert np.all(np.isfinite(dataset.vloc))
        assert np.allclose(dataset.qqq, dataset.qqq.T)
        dual = np.array(
            [
                [
                    np.sum(dataset.beta[i]['wfc'] * dataset.pswfc_partial[j] * dataset.rab)
                    if dataset.lll[i] == dataset.lll[j]
                    else 0.0
                    for j in range(dataset.nproj)
                ]
                for i in range(dataset.nproj)
            ]
        )
        np.testing.assert_allclose(dual, np.eye(dataset.nproj), atol=1e-9)
        for orbital in dataset.pswfc:
            assert np.all(np.isfinite(orbital['wfc']))
            assert dataset.augmented_norm(
                orbital['wfc'], dataset.lll[orbital['projector']]
            ) == pytest.approx(1.0, abs=1e-10)
            assert abs(orbital['wfc'][-1]) < 0.05 * np.max(np.abs(orbital['wfc']))
        diagnostics = dataset.legacy_reconstruction
        assert min(diagnostics['orbital_inner_overlap'].values()) > 0.99
        assert max(diagnostics['orbital_energy_error_hartree'].values()) < 0.01
        shells, skipped = generate_basis_shells(dataset, labels, n_points=240, r_box=10.0)
        assert skipped == []
        assert [shell.label for shell in shells] == labels
        for shell in shells:
            dr = shell.r[1] - shell.r[0]
            assert np.sum(shell.wfc**2) * dr == pytest.approx(1.0, abs=1e-10)
            spinor_dimension += 4 * (2 * shell.l + 1)  # two atoms and two spin components
    assert spinor_dimension == 68


def test_local_crsbr_spinor_overlap_and_occupied_projectability(legacy_run: Path) -> None:
    if not (legacy_run / 'WAVECAR').is_file() or not (legacy_run / 'vasprun.xml').is_file():
        pytest.skip('WAVECAR and vasprun.xml required for the PAW overlap check')
    from PAOFLOW.inputs.read_VASP import parse_vasprun_data
    from PAOFLOW.projection.augmentation import build_augmentation, k_plus_g_vectors
    from PAOFLOW.projection.do_atwfc_proj import augmented_proj_k, read_dft_wfc
    from PAOFLOW.projection.radial_shell_basis import build_generated_basis

    class Controller:
        rank = 0

        def __init__(self) -> None:
            self.data_arrays = {}
            self.data_attributes = {'verbose': False, 'dft': 'VASP', 'fpath': str(legacy_run)}

        def data_dicts(self) -> tuple[dict, dict]:
            return self.data_arrays, self.data_attributes

    controller = Controller()
    parse_vasprun_data(controller, str(legacy_run / 'vasprun.xml'))
    augmentation = build_augmentation(controller)
    assert augmentation is not None and augmentation.spin_orbit
    arrays, _ = controller.data_dicts()
    arrays['configuration'] = {
        'Cr': ['3P', '3D', '4S'],
        'S': ['3S', '3P'],
        'Br': ['4S', '4P'],
    }
    basis, _ = build_generated_basis(controller, n_points=800, save_basis=False)
    assert len(basis) == 68
    for ik in (0, 65, 129):
        gkspace, functions = read_dft_wfc(controller, ik, 0, orthonormalize=False)
        assert functions['npol'] == 2
        wfc = functions['wfc']
        beta = augmentation.beta(k_plus_g_vectors(gkspace))
        overlap = augmentation.overlap(wfc, wfc, beta)
        np.testing.assert_allclose(overlap, np.eye(wfc.shape[0]), atol=2e-5, rtol=2e-5)
        projections, _ = augmented_proj_k(controller, augmentation, basis, ik, 0)
        weights = np.sum(np.abs(projections) ** 2, axis=1)
        occupied = arrays['my_eigsmat'][:, ik, 0] < 0.0
        assert np.count_nonzero(occupied) == 50
        assert np.all(weights[occupied] > 0.95)
        assert np.all(weights <= 1.0 + 2.0e-5)
