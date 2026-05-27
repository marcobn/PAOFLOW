"""Unit tests for the auto-augmented internal-basis presets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from PAOFLOW.inputs.basis_presets import (
    SUPPORTED_PRESETS,
    available_ae_shells,
    element_block,
    extended_augmentation,
    minimal_shells_from_upf,
    resolve_configuration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataController:
    """Minimal stand-in matching the ``data_dicts()`` API used by the presets."""

    def __init__(self, arry=None, attr=None):
        self._arry = arry or {}
        self._attr = attr or {}

    def data_dicts(self):
        return self._arry, self._attr


def _make_basis_tree(root: Path, layout: dict[str, list[str]]) -> None:
    """Create a fake ``BASIS/<elem>/<shell>.dat`` directory tree."""
    for elem, shells in layout.items():
        elem_dir = root / elem
        elem_dir.mkdir(parents=True, exist_ok=True)
        for shell in shells:
            (elem_dir / f'{shell}.dat').write_text('0.0 0.0\n')


# ---------------------------------------------------------------------------
# element_block
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    'elem,expected',
    [
        ('H', 's'),
        ('Li', 's'),
        ('Cs', 's'),
        ('C', 'p'),
        ('O', 'p'),
        ('Si', 'p'),
        ('Ga', 'p'),
        ('As', 'p'),
        ('Fe', 'd'),
        ('Au', 'd'),
        ('Pt', 'd'),
        ('Zn', 'd'),
        ('Ce', 'f'),
        ('U', 'f'),
    ],
)
def test_element_block(elem, expected):
    assert element_block(elem) == expected


# ---------------------------------------------------------------------------
# available_ae_shells
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_available_ae_shells_sorted(tmp_path):
    _make_basis_tree(tmp_path, {'Si': ['3P', '1S', '3D', '2P', '2S', '3S', '4S']})
    shells = available_ae_shells(str(tmp_path), 'Si')
    assert shells == ['1S', '2S', '2P', '3S', '3P', '3D', '4S']


@pytest.mark.unit
def test_available_ae_shells_missing_directory(tmp_path):
    assert available_ae_shells(str(tmp_path), 'Xx') == []


@pytest.mark.unit
def test_available_ae_shells_lowercase_normalized(tmp_path):
    _make_basis_tree(tmp_path, {'O': ['2s', '2p']})
    assert available_ae_shells(str(tmp_path), 'O') == ['2S', '2P']


# ---------------------------------------------------------------------------
# extended_augmentation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    'elem,minimal,expected',
    [
        # p-block: 3 rows of S/P starting at nmax + D shells (max(3,nmax-1)..nmax+1)
        ('Si', ['3S', '3P'], ['4S', '4P', '5S', '5P', '3D', '4D']),
        ('O', ['2S', '2P'], ['3S', '3P', '4S', '4P', '3D']),
        ('Ga', ['3D', '4S', '4P'], ['5S', '5P', '6S', '6P', '4D', '5D']),
        # d-block
        ('Fe', ['3D', '4S'], ['4P', '5S', '5P', '6S', '6P', '4D', '5D']),
        ('Au', ['5D', '6S'], ['6P', '7S', '7P', '8S', '8P', '6D', '7D']),
        # s-block
        ('Li', ['2S'], ['2P', '3S', '3P', '4S', '4P', '3D']),
        ('H', ['1S'], ['2S', '2P', '3S', '3P']),  # 1P excluded (n>=2)
        # f-block: also append F shells
        (
            'Ce',
            ['4F', '5D', '6S'],
            ['6P', '7S', '7P', '8S', '8P', '6D', '7D', '5F', '6F'],
        ),
    ],
)
def test_extended_augmentation_rules(elem, minimal, expected):
    assert extended_augmentation(elem, minimal) == expected


@pytest.mark.unit
def test_extended_augmentation_skips_duplicates():
    # Shells already in `minimal` must not appear in the augmentation.
    # Fe with valence including 4P -> 4P is dropped, but the rule still
    # adds 5S/5P/6S/6P/4D/5D as polarization.
    extra = extended_augmentation('Fe', ['3D', '4S', '4P'])
    assert '4P' not in extra
    assert '4S' not in extra
    assert '3D' not in extra
    assert extra == ['5S', '5P', '6S', '6P', '4D', '5D']


@pytest.mark.unit
def test_extended_augmentation_unknown_element():
    assert extended_augmentation('Xx', ['1S']) == []


@pytest.mark.unit
def test_extended_augmentation_empty_minimal():
    assert extended_augmentation('Si', []) == []


# ---------------------------------------------------------------------------
# minimal_shells_from_upf — real UPF file from examples/
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
GAAS_DIR = REPO_ROOT / 'examples' / 'acbn0_examples' / 'GaAs'


@pytest.mark.unit
@pytest.mark.skipif(not GAAS_DIR.exists(), reason='GaAs example pseudopotentials not present')
def test_minimal_shells_from_upf_real_pseudopotential():
    dc = _FakeDataController(
        arry={
            'species': [
                ('As', 'As.pbe-n-kjpaw_psl.1.0.0.UPF'),
                ('Ga', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'),
            ]
        },
        attr={'fpath': str(GAAS_DIR)},
    )
    as_shells = minimal_shells_from_upf(dc, 'As')
    ga_shells = minimal_shells_from_upf(dc, 'Ga')

    # Every label must be a valid 2-character shell (e.g. "4S", "4P")
    for label in as_shells + ga_shells:
        assert len(label) == 2
        assert label[0].isdigit()
        assert label[1] in 'SPDF'

    # Valence configuration for these pseudos: As has 4s4p; Ga has at
    # least 4s4p (PSL pseudos additionally include semicore 3d).
    assert '4S' in as_shells and '4P' in as_shells
    assert '4S' in ga_shells and '4P' in ga_shells


@pytest.mark.unit
def test_minimal_shells_from_upf_raises_when_unconfigured():
    dc = _FakeDataController()
    with pytest.raises(RuntimeError, match='Cannot resolve UPF'):
        minimal_shells_from_upf(dc, 'Si')


# ---------------------------------------------------------------------------
# resolve_configuration — dispatch & validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_configuration_rejects_unknown_preset():
    dc = _FakeDataController(arry={'atoms': ['Si']}, attr={})
    with pytest.raises(ValueError, match='Unknown configuration preset'):
        resolve_configuration(dc, 'huge')


@pytest.mark.unit
def test_resolve_configuration_rejects_non_string():
    dc = _FakeDataController(arry={'atoms': ['Si']}, attr={})
    with pytest.raises(TypeError):
        resolve_configuration(dc, 42)


@pytest.mark.unit
def test_resolve_configuration_extended_requires_basispath():
    dc = _FakeDataController(
        arry={'atoms': ['Si'], 'species': [('Si', 'fake.UPF')]},
        attr={'fpath': '/nowhere'},
    )
    with pytest.raises(ValueError, match='basispath'):
        resolve_configuration(dc, 'extended')


@pytest.mark.unit
def test_resolve_configuration_preset_is_case_insensitive():
    assert all(p.lower() == p for p in SUPPORTED_PRESETS)
    # The dispatcher should accept both 'MINIMAL' and 'minimal' equally
    # (we only check the validation branch — full resolution needs a UPF).
    dc = _FakeDataController(arry={'atoms': ['Si'], 'species': []}, attr={'fpath': '/x'})
    with pytest.raises(RuntimeError):  # no matching species → not a ValueError
        resolve_configuration(dc, 'MINIMAL')


@pytest.mark.unit
@pytest.mark.skipif(not GAAS_DIR.exists(), reason='GaAs example pseudopotentials not present')
def test_resolve_configuration_minimal_end_to_end():
    dc = _FakeDataController(
        arry={
            'atoms': ['Ga', 'As'],
            'species': [
                ('Ga', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'),
                ('As', 'As.pbe-n-kjpaw_psl.1.0.0.UPF'),
            ],
        },
        attr={'fpath': str(GAAS_DIR)},
    )
    cfg = resolve_configuration(dc, 'minimal')
    assert set(cfg.keys()) == {'Ga', 'As'}
    assert '4S' in cfg['Ga'] and '4P' in cfg['Ga']
    assert '4S' in cfg['As'] and '4P' in cfg['As']


@pytest.mark.unit
@pytest.mark.skipif(not GAAS_DIR.exists(), reason='GaAs example pseudopotentials not present')
def test_resolve_configuration_extended_unions_and_filters(tmp_path):
    # Build a fake BASIS/ tree that contains the augmenting shells the
    # rule will request for Ga and As (both p-block, nmax depends on the
    # pseudopotential's valence configuration).
    _make_basis_tree(
        tmp_path,
        {
            # Provide every shell up to 5D so any reasonable augmentation
            # is satisfied regardless of pseudopotential semicore choice.
            'Ga': ['3S', '3P', '3D', '4S', '4P', '4D', '5S', '5P', '5D'],
            'As': ['3S', '3P', '3D', '4S', '4P', '4D', '5S', '5P', '5D'],
        },
    )
    dc = _FakeDataController(
        arry={
            'atoms': ['Ga', 'As'],
            'species': [
                ('Ga', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'),
                ('As', 'As.pbe-n-kjpaw_psl.1.0.0.UPF'),
            ],
        },
        attr={'fpath': str(GAAS_DIR), 'basispath': str(tmp_path) + os.sep},
    )
    pseudo_shells = {elem: minimal_shells_from_upf(dc, elem) for elem in ('Ga', 'As')}
    minimal = resolve_configuration(dc, 'minimal')
    extended = resolve_configuration(dc, 'extended')

    for elem in ('Ga', 'As'):
        # Extended is a superset of minimal …
        assert set(minimal[elem]).issubset(set(extended[elem]))
        # … and strictly larger (the rule added at least one polarization shell).
        assert len(extended[elem]) > len(minimal[elem])
        # Minimal entries come first; augmentations are appended after.
        assert extended[elem][: len(minimal[elem])] == minimal[elem]
        # No duplicates.
        assert len(extended[elem]) == len(set(extended[elem]))
        # Pseudo (UPF) shells must be preserved verbatim.
        assert set(pseudo_shells[elem]).issubset(set(extended[elem]))


@pytest.mark.unit
def test_resolve_configuration_extended_warns_when_basis_missing(tmp_path, recwarn):
    # No <basispath>/As directory exists → expect a warning and a
    # fall-back to the minimal valence list.
    if not GAAS_DIR.exists():
        pytest.skip('GaAs example pseudopotentials not present')
    dc_arry = {'atoms': ['As'], 'species': [('As', 'As.pbe-n-kjpaw_psl.1.0.0.UPF')]}
    dc_attr = {'fpath': str(GAAS_DIR), 'basispath': str(tmp_path) + os.sep}

    dc = _FakeDataController(arry=dc_arry, attr=dc_attr)
    cfg = resolve_configuration(dc, 'extended')
    assert cfg['As']  # fell back to minimal, non-empty
    assert any('No AE basis files' in str(w.message) for w in recwarn.list)
