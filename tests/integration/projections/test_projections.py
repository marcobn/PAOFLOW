"""PAO projections of small QE saves, checked against the DFT eigenvalues of the saves themselves.

The model Hamiltonians of ``tests/unit/closed_forms`` test the properties; these saves test what a
model cannot: the QE readers and the two projection routes (``projwfc.x``, and the internal
projections on the ``minimal`` and ``standard`` bases with the augmented metric of ultrasoft and PAW
pseudopotentials and the spinor basis of fully relativistic ones), the PAO Hamiltonian and the
symmetry unfolding of the wedge. No reference output is stored:

- the PAO bands equal the DFT eigenvalues at the k-points the two grids share (projectable bands
  below E_F + 1 eV);
- the internal ``minimal`` basis (the pseudo-atomic orbitals of the UPF, which projwfc.x reads too)
  gives the projwfc.x Hamiltonian;
- with spin-orbit coupling and inversion every band is a Kramers pair; a collinear magnet matches in
  both spin channels.

The saves are the release asset ``projection_test_assets.tar.gz``
(``.github/assets_generation/projections/build_assets.sh``): set ``PAOFLOW_PROJECTION_ASSET_URL``
and ``PAOFLOW_PROJECTION_ASSET_SHA256``, or ``PAOFLOW_PROJECTION_ASSET_ARCHIVE`` for a local tarball.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from tests.integration.qe.assets import ensure_assets_extracted, resolve_asset_source

pytestmark = pytest.mark.integration

HARTREE_EV = 27.211386245988
#: system: (prefix, largest |PAO - DFT| (eV) of the projectable bands below E_F + 1 eV, Kramers pairs)
SYSTEMS = {
    'si': ('silicon', 1e-3, False),
    'gaas_paw': ('GaAs', 2e-3, False),
    'pt_fr': ('pt', 2e-3, True),
    'pt_lsda': ('pt', 2e-3, False),
    'bi_bilayer': ('Bi', 1e-2, True),
}


@pytest.fixture(scope='session')
def assets_root() -> Path:
    source = resolve_asset_source(
        env_archive_var='PAOFLOW_PROJECTION_ASSET_ARCHIVE',
        env_url_var='PAOFLOW_PROJECTION_ASSET_URL',
        env_sha256_var='PAOFLOW_PROJECTION_ASSET_SHA256',
        env_version_var='PAOFLOW_PROJECTION_ASSET_VERSION',
        cache_key='projections',
    )
    if source.archive_path is None and source.url is None:
        pytest.skip(
            'set PAOFLOW_PROJECTION_ASSET_URL (and _SHA256) or PAOFLOW_PROJECTION_ASSET_ARCHIVE'
        )
    return ensure_assets_extracted(source)


def _dft(save: Path) -> tuple[np.ndarray, list[np.ndarray], float, np.ndarray]:
    """k-points (cartesian, 2pi/alat), eigenvalues per spin channel (eV - E_F), E_F, b vectors."""
    root = ET.parse(save / 'data-file-schema.xml').getroot()
    bands = root.find('output/band_structure')
    reference = bands.findtext('fermi_energy') or bands.findtext('highestOccupiedLevel')
    fermi = float(reference) * HARTREE_EV
    points, values = [], []
    for point in bands.findall('ks_energies'):
        points.append([float(x) for x in point.find('k_point').text.split()])
        values.append(
            np.array(point.findtext('eigenvalues').split(), dtype=float) * HARTREE_EV - fermi
        )
    values = np.array(values)
    lsda = (bands.findtext('lsda') or '').strip().lower() == 'true'
    channels = (
        [values[:, : int(bands.findtext('nbnd_up'))], values[:, int(bands.findtext('nbnd_up')) :]]
        if lsda
        else [values]
    )
    b = np.array([[float(x) for x in root.find('output/basis_set/reciprocal_lattice/' + v).text.split()]
                  for v in ('b1', 'b2', 'b3')])  # fmt: skip
    return np.array(points), channels, fermi, b


@pytest.fixture(scope='module', params=sorted(SYSTEMS))
def system(request, assets_root):
    """A writable copy of the save, with its internal bases generated from its UPFs."""
    from PAOFLOW.basis_gen import generate_basis_for_pseudo
    from PAOFLOW.basis_gen.driver import _default_shells
    from PAOFLOW.inputs.read_upf import UPF

    name = request.param
    prefix = SYSTEMS[name][0]
    work = Path(tempfile.mkdtemp(prefix='projections_'))
    shutil.copytree(assets_root / name / (prefix + '.save'), work / (prefix + '.save'))
    basis = work / 'BASIS_PS'
    for upf in sorted((work / (prefix + '.save')).glob('*.[uU][pP][fF]')):
        generate_basis_for_pseudo(
            str(upf), str(basis), shells=_default_shells(UPF(str(upf)), preset='standard')
        )
    cwd = os.getcwd()
    os.chdir(work)
    try:
        yield name, work, prefix, {}
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)


def _pao(system, route: str):
    """E_k (full grid, every spin channel), bnd and the grid of a projection route (cached)."""
    from PAOFLOW import PAOFLOW

    name, work, prefix, cache = system
    if route not in cache:
        p = PAOFLOW.PAOFLOW(
            savedir=prefix + '.save', outputdir=str(work / ('out_' + route)), verbose=False
        )
        p.data_controller.data_attributes['abort_on_exception'] = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if route == 'projwfc':
                p.read_atomic_proj_QE()
            else:
                p.projections(basispath=str(work / 'BASIS_PS') + os.sep, configuration=route)
            p.projectability(pthr=0.95)
            p.pao_hamiltonian()
            p.pao_eigh()
        arrays, attr = p.data_controller.data_dicts()
        cache[route] = (
            np.array(arrays['E_k']),
            int(attr['bnd']),
            (attr['nk1'], attr['nk2'], attr['nk3']),
        )
    return cache[route]


@pytest.mark.parametrize('route', ['projwfc', 'minimal', 'standard'])
def test_pao_bands_are_the_dft_bands(system, route):
    name, work, prefix, _ = system
    energies, bnd, grid = _pao(system, route)
    points, channels, _fermi, b = _dft(work / (prefix + '.save'))
    n1, n2, n3 = grid
    crystal = np.array(
        [[i / n1, j / n2, k / n3] for i in range(n1) for j in range(n2) for k in range(n3)]
    )
    cartesian = crystal @ b
    worst, shared = 0.0, 0
    for index, point in enumerate(points):
        hit = np.flatnonzero(np.all(np.abs(cartesian - point) < 1e-6, axis=1))
        if not hit.size:
            continue
        shared += 1
        for spin, dft in enumerate(channels):
            pao = np.sort(energies[hit[0], :, spin])
            for n in range(min(bnd, len(pao), dft.shape[1])):
                if dft[index, n] < 1.0:
                    worst = max(worst, abs(pao[n] - dft[index, n]))
    assert shared >= 1
    assert worst < SYSTEMS[name][1], '%s %s: %.2e eV' % (name, route, worst)


def test_internal_minimal_is_projwfc(system):
    """The two routes project on the same orbitals: the same bands below E_F + 1 eV."""
    projwfc, bnd_p, _ = _pao(system, 'projwfc')
    minimal, bnd_m, _ = _pao(system, 'minimal')
    assert bnd_p == bnd_m
    bands = min(bnd_p, projwfc.shape[1])
    first, second = np.sort(projwfc, axis=1)[:, :bands], np.sort(minimal, axis=1)[:, :bands]
    low = first < 1.0
    np.testing.assert_allclose(second[low], first[low], atol=1e-4)


@pytest.mark.parametrize('route', ['projwfc', 'minimal', 'standard'])
def test_kramers_pairs(system, route):
    """Spin-orbit coupling with inversion and time reversal: every band below E_F + 1 eV is a pair
    (to the degeneracy of the DFT eigenvalues of the save). Until 17 Nov 2026 the standard basis of
    ultrasoft / PAW runs split them by 1-4 meV (radial functions misassigned in the j blocks)."""
    name = system[0]
    if not SYSTEMS[name][2]:
        pytest.skip('no spin-orbit coupling with inversion')
    energies, bnd, _ = _pao(system, route)
    pairs = np.sort(energies[:, :, 0], axis=1)[:, : bnd - bnd % 2].reshape(len(energies), -1, 2)
    low = pairs[..., 0] < 1.0
    np.testing.assert_allclose(pairs[..., 0][low], pairs[..., 1][low], atol=1e-4)
