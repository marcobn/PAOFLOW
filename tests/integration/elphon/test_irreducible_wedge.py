"""nscf on the irreducible wedge (``K_POINTS automatic``) vs the explicit full-grid nscf of the same scf.

The asset holds ``si_ibz.save`` (16 irreducible points of the 6^3 grid, 12 bands) next to
``si.save`` (the 216-point list).  The rotated Bloch functions agree with the stored ones up to
a unitary mixing inside degenerate multiplets and the nscf convergence; the Hamiltonian and the
PAO-basis vertex agree to the same level.
"""

import os

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_vertex import CoarseVertexSetup
from PAOFLOW.elphon.elph_bloch import read_nscf
from PAOFLOW.elphon.pao_electrons import pao_hamiltonian_from_save
from PAOFLOW.elphon.plane_waves import read_plane_wave_block
from PAOFLOW.elphon.wavefunction_symmetry import provider_for

pytestmark = pytest.mark.integration

NBANDS = 8  # the lowest bands are converged tightly and never cut a multiplet (12 stored)


def _complete_groups(e, nb, tol=1e-5):
    groups, start = [], 0
    for b in range(1, nb + 1):
        if b == nb or e[b] - e[b - 1] > tol:
            groups.append((start, b))
            start = b
    if len(e) > nb and e[nb] - e[nb - 1] < tol:
        groups = groups[:-1]
    return groups


@pytest.fixture(scope='module')
def saves(elphon_assets):
    ibz = os.path.join(elphon_assets, 'si_ibz.save')
    if not os.path.isdir(ibz):
        pytest.skip('asset has no si_ibz.save')
    return os.path.join(elphon_assets, 'si.save'), ibz


def test_rotated_blocks_match_the_full_grid_nscf(saves):
    full_dir, ibz_dir = saves
    full = read_nscf(full_dir)
    provider = provider_for(ibz_dir)
    assert provider.kpts_irr_cryst.shape[0] == 16 and provider.nk == 216
    assert np.abs(provider.kpts_cryst - full['kpts_cryst']).max() < 1e-6  # same order as the list
    unfolded = read_nscf(ibz_dir, unfold=True)
    assert np.abs(unfolded['eigs_ry'][:, :NBANDS] - full['eigs_ry'][:, :NBANDS]).max() < 1e-6
    worst = 0.0
    for ik in range(provider.nk):
        rot = provider.block(ik)
        ref = read_plane_wave_block(full_dir, ik, full['alat'])
        assert np.abs(rot.xk_cart - ref.xk_cart).max() < 1e-6
        index = {tuple(m): i for i, m in enumerate(ref.miller.T)}
        idx = np.array([index.get(tuple(m), -1) for m in rot.miller.T])
        common = idx >= 0
        assert common.sum() == ref.npw  # same sphere
        overlap = ref.coefficients[:, idx[common]].conj() @ rot.coefficients[:, common].T
        for a, b in _complete_groups(full['eigs_ry'][ik], NBANDS):
            block = overlap[a:b, a:b]
            worst = max(worst, np.abs(block @ block.conj().T - np.eye(b - a)).max())
            others = np.r_[0:a, b:NBANDS]
            if others.size:
                worst = max(worst, np.abs(overlap[a:b][:, others]).max())
    assert worst < 1e-4, f'rotated blocks deviate by {worst:.2e} inside degenerate groups'


def test_hamiltonian_and_pao_vertex_from_the_irreducible_wedge(saves, tmp_path):
    full_dir, ibz_dir = saves
    for name in ('full', 'ibz', 'sym'):
        (tmp_path / name).mkdir()
    kw = {'configuration': 'minimal', 'pthr': 0.9, 'outputdir': 'out'}
    el_full = pao_hamiltonian_from_save(full_dir, workpath=str(tmp_path / 'full'), **kw)
    el_ibz = pao_hamiltonian_from_save(
        ibz_dir, workpath=str(tmp_path / 'ibz'), unfold_kpoints=True, **kw
    )
    el_sym = pao_hamiltonian_from_save(ibz_dir, workpath=str(tmp_path / 'sym'), **kw)
    assert el_full.projections.shape == el_ibz.projections.shape
    assert not el_ibz.expand_wedge and el_sym.expand_wedge
    scale = np.abs(el_full.HRs).max()
    assert np.abs(el_ibz.HRs - el_full.HRs).max() / scale < 1e-4
    assert np.abs(el_sym.HRs - el_full.HRs).max() / scale < 1e-4
    dfpt = os.path.join(os.path.dirname(full_dir), 'save')
    setup_full = CoarseVertexSetup(full_dir, dfpt, (3, 3, 3))
    setup_ibz = CoarseVertexSetup(ibz_dir, dfpt, (3, 3, 3))
    assert setup_full.provider is None and setup_ibz.provider is not None
    assert setup_ibz.nk == 216 and setup_ibz.nscf_grid() == (6, 6, 6)
    with pytest.raises(ValueError, match='irreducible wedge'):
        CoarseVertexSetup(ibz_dir, dfpt, (3, 3, 3), unfold_kpoints=False)
    bands = np.arange(NBANDS)
    member = next(m for m in setup_full.members if m.isym != 0 and np.abs(m.umklapp).max() > 0)
    twin = next(m for m in setup_ibz.members if np.allclose(m.q_cryst, member.q_cryst))
    g_full = setup_full.pao_vertex_k(member, el_full.projections, (6, 6, 6), 'pao', bands)
    g_ibz = setup_ibz.pao_vertex_k(twin, el_ibz.projections, (6, 6, 6), 'pao', bands)
    assert np.abs(g_full - g_ibz).max() / np.abs(g_full).max() < 1e-4
