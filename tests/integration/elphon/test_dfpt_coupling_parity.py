"""Native DFPT vertex vs EPW's ``epmatq`` on the Si tutorial data (6^3 k, 3^3 q, 12 bands)."""

import os

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_vertex import CoarseVertexSetup
from PAOFLOW.elphon.elph_bloch import kq_index_map

pytestmark = pytest.mark.integration


def test_band_vertex_matches_epw_reference(elphon_assets):
    ref = np.load(os.path.join(elphon_assets, 'reference', 'epmatq_subset.npz'))
    setup = CoarseVertexSetup(
        os.path.join(elphon_assets, 'si.save'),
        os.path.join(elphon_assets, 'save'),
        tuple(int(n) for n in ref['qgrid']),
    )
    assert setup.nk == 216 and setup.nbnd == 12 and len(setup.members) == 27
    labels = {tuple(np.rint(m.q_cryst * 3).astype(int) % 3): m for m in setup.members}
    for q_cryst, epmatq, tol in zip(ref['q_cryst'], ref['epmatq'], ref['tolerance']):
        member = labels[tuple(np.rint(q_cryst * 3).astype(int) % 3)]
        d = setup.band_vertex_at_q(member)  # (nk, nb, nb, 3nat)
        reference = np.moveaxis(epmatq, 2, 0)
        rel = np.abs(d - reference).max() / np.abs(reference).max()
        assert rel < tol, 'q = %s: relative deviation %.2e > %.0e' % (q_cryst, rel, tol)
    # the stored q-points cover a Gamma and a rotated star member with an umklapp vector
    _, g0 = kq_index_map(setup.kpts_cryst, ref['q_cryst'][-1], setup.nscf_grid())
    assert (g0 != 0).any()


def test_hermiticity_between_q_and_minus_q(elphon_assets):
    """g_mn(k, q)^* = g_nm(k+q, -q) to the DFPT convergence of the reference run."""
    ref = np.load(os.path.join(elphon_assets, 'reference', 'epmatq_subset.npz'))
    setup = CoarseVertexSetup(
        os.path.join(elphon_assets, 'si.save'), os.path.join(elphon_assets, 'save'), (3, 3, 3)
    )
    members = {tuple(np.rint(m.q_cryst * 3).astype(int) % 3): m for m in setup.members}
    q = np.array([0.0, 0.0, 1.0 / 3.0])
    plus = setup.band_vertex_at_q(members[tuple(np.rint(q * 3).astype(int) % 3)])
    minus = setup.band_vertex_at_q(members[tuple(np.rint(-q * 3).astype(int) % 3)])
    ikq, _ = kq_index_map(setup.kpts_cryst, q, setup.nscf_grid())
    dev = np.abs(np.conj(plus) - minus[ikq].transpose(0, 2, 1, 3)).max() / np.abs(plus).max()
    assert dev < float(ref['hermiticity_tolerance'])
