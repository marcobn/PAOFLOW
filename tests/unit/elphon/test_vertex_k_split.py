"""k-split of the coarse DFPT vertex: k-point subsets of ``pao_vertex_k`` and partial sums of ``g_q(R_e)``."""

from types import SimpleNamespace

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_vertex import CoarseVertexSetup, k_blocks
from PAOFLOW.elphon.do_pao_eph_dense_q import _broadcast_cells, _sum_on_node_root
from PAOFLOW.elphon.elph_bloch import pao_vertex_to_cells

pytestmark = pytest.mark.unit


def _grid_kpoints(ng):
    return np.array(
        [
            [i / ng[0], j / ng[1], k / ng[2]]
            for i in range(ng[0])
            for j in range(ng[1])
            for k in range(ng[2])
        ]
    )


def _fake_setup(ng, nbnd, nat):
    """A stand-in for ``CoarseVertexSetup`` whose matrix elements encode (k, k + q)."""
    kpts = _grid_kpoints(ng)
    nk = kpts.shape[0]

    def matrix_elements(member, potential, ik, ikq, g0, left=None, right=None):
        nawf = right.shape[1]
        base = (ik + 1) + 1j * (ikq + 1) + 0.01 * float(np.sum(g0))
        return base * np.arange(1, 3 * nat * nawf * nawf + 1).reshape(3 * nat, nawf, nawf)

    return SimpleNamespace(
        nk=nk,
        nbnd=nbnd,
        nat=nat,
        kpts_cryst=kpts,
        provider=None,
        potential_at=lambda member: None,
        matrix_elements=matrix_elements,
    )


def test_pao_vertex_k_subset_matches_rows_of_the_full_vertex():
    ng, nbnd, nawf, nat = (2, 2, 3), 6, 4, 2
    setup = _fake_setup(ng, nbnd, nat)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nbnd, nawf, setup.nk)) + 0j
    member = SimpleNamespace(q_cryst=np.array([0.5, 0.0, 1.0 / 3.0]))
    full = CoarseVertexSetup.pao_vertex_k(setup, member, A, ng, 'pao')
    assert full.shape == (setup.nk, nawf, nawf, 3 * nat)
    rows = [7, 2, 11]
    part = CoarseVertexSetup.pao_vertex_k(setup, member, A, ng, 'pao', kpoints=rows)
    assert part.shape == (3, nawf, nawf, 3 * nat)
    assert np.array_equal(part, full[rows])
    empty = CoarseVertexSetup.pao_vertex_k(setup, member, A, ng, 'pao', kpoints=[])
    assert empty.shape == (0, nawf, nawf, 3 * nat)


def test_k_blocks_tile_the_grid_in_order():
    blocks = k_blocks(10, 4)
    assert [b.tolist() for b in blocks] == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    blocks = k_blocks(2, 3)
    assert [b.size for b in blocks] == [1, 1, 0]
    assert np.concatenate(k_blocks(7, 3)).tolist() == list(range(7))


def test_partial_cell_transforms_add_up_to_the_full_one():
    """g_q(R_e) of the full grid = sum over k-blocks of the transform of each block."""
    ng = (3, 2, 2)
    kpts = _grid_kpoints(ng)
    nk = kpts.shape[0]
    rng = np.random.default_rng(1)
    gk = rng.standard_normal((nk, 3, 3, 6)) + 1j * rng.standard_normal((nk, 3, 3, 6))
    kidx = np.round(kpts * np.asarray(ng)).astype(int) % np.asarray(ng)
    full = pao_vertex_to_cells(gk, kidx, ng)
    total = np.zeros_like(full)
    for block in k_blocks(nk, 5):
        total += pao_vertex_to_cells(gk[block], kidx[block], ng)
    assert np.allclose(total, full, atol=1e-13, rtol=0)


def test_node_collectives_are_identities_on_one_rank():
    from mpi4py import MPI

    comm = MPI.COMM_SELF
    rng = np.random.default_rng(2)
    array = rng.standard_normal((3, 2, 4)) + 0j
    expected = array.copy()
    _sum_on_node_root(comm, array)
    assert np.array_equal(array, expected)
    g = rng.standard_normal((2, 2, 3, 2, 1, 1, 2, 1, 1)) + 0j
    expected = g.copy()
    labels = np.array([[0, 0, 0], [1, 0, 0]])
    _broadcast_cells(comm, g, labels, {(0, 0, 0): 0, (1, 0, 0): 0})
    assert np.array_equal(g, expected)
    with pytest.raises(ValueError):
        _sum_on_node_root(SimpleNamespace(Get_size=lambda: 2, Get_rank=lambda: 0), array[:, ::2])
