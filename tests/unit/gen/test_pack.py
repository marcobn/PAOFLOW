"""``paoflow-pack`` and ``paoflow-unpack``: a run of the Si fixture packed, opened and checked."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from PAOFLOW.gen.cases import load_case
from PAOFLOW.gen.pack import CRATE, main_pack, main_unpack, pack, parse_author, unpack

pytestmark = pytest.mark.unit

SAVE = Path(__file__).parents[2] / 'data' / 'silicon' / 'silicon.save'


@pytest.fixture
def calculation(tmp_path, monkeypatch):
    """A folder with a DFT save, a driver script and the outputs of one run of it."""
    from PAOFLOW.PAOFLOW import PAOFLOW

    work = tmp_path / 'si_bands'
    work.mkdir()
    shutil.copytree(SAVE, work / 'silicon.save')
    (work / 'silicon.save' / 'wfc1.dat').write_bytes(b'wavefunctions')
    (work / 'main.py').write_text('# the driver\n')
    (work / 'scf.in').write_text('&control\n/\n')
    (work / 'slurm-1.out').write_text('a log with the name of the cluster')
    monkeypatch.chdir(work)
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    paoflow = PAOFLOW(savedir='silicon.save', outputdir='output', verbose=False)
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=2, nk=20)
    paoflow.finish_execution()
    return work


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_author_with_and_without_orcid():
    assert parse_author('Ada Lovelace 0000-0002-1825-0097') == {
        'name': 'Ada Lovelace',
        'orcid': '0000-0002-1825-0097',
    }
    assert parse_author(' Ada Lovelace ') == {'name': 'Ada Lovelace'}


def test_pack_then_unpack_preserves_every_file(calculation, tmp_path, capsys):
    """The package holds outputs, inputs, script and description, not the wavefunctions or the
    logs; unpacking gives the same bytes, a case Studio opens, and the steps in order."""
    arguments = [
        '--no-questions',
        '--title',
        'Si bands',
        '--author',
        'A. Person 0000-0002-1825-0097',
    ]
    assert main_pack([str(calculation), *arguments]) == 0
    printed = capsys.readouterr().out
    assert '[x] Authors: A. Person' in printed and 'such as Zenodo' in printed
    assert '[x] Not included, listed by checksum' in printed
    archive = calculation / 'si_bands-package.zip'
    names = set(zipfile.ZipFile(archive).namelist())
    assert {'README.md', CRATE, 'case.json', 'main.py', 'scf.in', 'output/bands_0.dat'} <= names
    assert 'silicon.save/data-file-schema.xml' in names
    assert not {'silicon.save/wfc1.dat', 'slurm-1.out'} & names

    assert main_unpack([str(archive), '-o', str(tmp_path / 'opened')]) == 0
    printed = capsys.readouterr().out
    assert 'Read them before you run them' in printed
    assert 'all match' in printed and 'DFT steps first' in printed and 'python main.py' in printed
    opened = tmp_path / 'opened'
    for name in names - {'README.md', CRATE}:
        assert _sha256(opened / name) == _sha256(calculation / name), name
    case = load_case(opened)
    assert (
        case.scripts
        and case.title == 'Si bands'
        and case.steps[0]['command'] == ['python', 'main.py']
    )
    readme = (opened / 'README.md').read_text()
    assert 'Quantum ESPRESSO' in readme and '`bands`' in readme and 'wfc1.dat' in readme
    assert 'PAOFLOW.bands(' in readme and str(calculation) not in readme


def test_crate_describes_the_run(calculation):
    """A Workflow Run Crate: the script is the workflow, the run its action, every file has a
    checksum and its PAOFLOW term, and every reference inside the crate resolves."""
    result = pack(calculation, authors=[{'name': 'A. Person'}], license='CC-BY-4.0')
    with zipfile.ZipFile(result['path']) as archive:
        graph = json.loads(archive.read(CRATE))['@graph']
        names = set(archive.namelist())
    by_id = {entity['@id']: entity for entity in graph}
    root = by_id['./']
    assert {ref['@id'] for ref in root['conformsTo']} >= {
        'https://w3id.org/ro/wfrun/process/0.6',
        'https://w3id.org/ro/wfrun/workflow/0.6',
    }
    assert root['mainEntity'] == {'@id': 'main.py'} and root['license']['@id'].endswith('CC-BY-4.0')
    assert 'ComputationalWorkflow' in by_id['main.py']['@type']
    (action,) = [entity for entity in graph if entity['@type'] == 'CreateAction']
    assert action['instrument'] == {'@id': 'main.py'} and action['endTime'].endswith('Z')
    assert {'@id': 'output/bands_0.dat'} in action['result']
    assert {'@id': 'silicon.save/data-file-schema.xml'} in action['object']
    bands = by_id['output/bands_0.dat']
    assert bands['about'] == {'@id': '#term-bands'} and len(bands['sha256']) == 64
    assert any('optimade' in link for link in by_id['#structure-cell_bohr']['sameAs'])
    assert {ref['@id'] for ref in root['hasPart']} == names - {CRATE}
    local = {
        value['@id']
        for entity in graph
        for value in (v for vs in entity.values() for v in (vs if isinstance(vs, list) else [vs]))
        if isinstance(value, dict) and not value['@id'].startswith('http')
    }
    assert local <= set(by_id)


def test_a_changed_package_is_reported(calculation, tmp_path, capsys):
    result = pack(calculation)
    damaged = tmp_path / 'damaged.zip'
    with zipfile.ZipFile(result['path']) as source, zipfile.ZipFile(damaged, 'w') as target:
        for name in source.namelist():
            data = source.read(name)
            target.writestr(name, b'edited' if name == 'output/bands_0.dat' else data)
    assert unpack(damaged, tmp_path / 'a')['bad'] == ['output/bands_0.dat']
    assert main_unpack([str(damaged), '-o', str(tmp_path / 'b')]) == 1
    assert 'output/bands_0.dat' in capsys.readouterr().out


def test_unpack_refuses_paths_outside_the_folder_and_other_zips(tmp_path):
    escaping = tmp_path / 'escaping.zip'
    with zipfile.ZipFile(escaping, 'w') as archive:
        archive.writestr(CRATE, '{"@graph": []}')
        archive.writestr('../outside.txt', 'x')
    with pytest.raises(ValueError, match='outside the package'):
        unpack(escaping, tmp_path / 'a')
    assert not (tmp_path / 'outside.txt').exists()
    other = tmp_path / 'other.zip'
    with zipfile.ZipFile(other, 'w') as archive:
        archive.writestr('notes.txt', 'x')
    with pytest.raises(ValueError, match='not a PAOFLOW package'):
        unpack(other, tmp_path / 'b')
    # a zip that expands beyond the limit is refused before anything is written
    large = tmp_path / 'large.zip'
    with zipfile.ZipFile(large, 'w') as archive:
        archive.writestr(CRATE, '{"@graph": []}')
    with pytest.raises(ValueError, match='--max-size'):
        unpack(large, tmp_path / 'c', max_gb=1e-9)
    assert not (tmp_path / 'c').exists()
    (tmp_path / 'full').mkdir()
    (tmp_path / 'full' / 'mine.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        unpack(other, tmp_path / 'full')


def test_pack_needs_a_run_record_and_says_what_changed(calculation, tmp_path, capsys):
    (tmp_path / 'empty').mkdir()
    assert main_pack([str(tmp_path / 'empty'), '--no-questions']) == 1
    assert 'folder where you ran' in capsys.readouterr().err
    (calculation / 'output' / 'bands_0.dat').write_text('edited after the run')
    lines = dict((text, fine) for fine, text in pack(calculation)['checklist'])
    assert lines['Changed since the run that recorded them: output/bands_0.dat'] is False
    assert lines['Authors missing: answer the question, or pass --author "Name ORCID"'] is False


def test_crate_passes_the_validator(calculation, tmp_path):
    """The crate against the run-crate profiles, when the validator is installed (dev extra)."""
    validator = shutil.which('rocrate-validator')
    if validator is None:
        pytest.skip('rocrate-validator is not installed (pip install roc-validator)')
    folder = unpack(pack(calculation, license='CC-BY-4.0')['path'], tmp_path / 'opened')['folder']
    done = subprocess.run(
        [validator, 'validate', str(folder)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
