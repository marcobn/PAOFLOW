import numpy as np
import pytest
from mpi4py import MPI
from scipy import linalg as spl

from PAOFLOW.DataController import DataController
from PAOFLOW.hamiltonian.do_build_pao_hamiltonian import build_Hks

NAWF, NBNDS, NKPNTS, NSPIN = 8, 14, 11, 2
ETA = 5.0


class _StubController:
    """Minimal controller exercising the real projection accessor."""

    local_projections = DataController.local_projections

    def __init__(self, arrays, attributes):
        self.data_arrays = arrays
        self.data_attributes = attributes
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


def _inputs(seed):
    rng = np.random.default_rng(seed)
    shape = (NBNDS, NAWF, NKPNTS, NSPIN)
    U = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    eigs = np.sort(rng.normal(size=(NBNDS, NKPNTS, NSPIN)), axis=0)
    return U, eigs


def _reference(U, eigs, shift_type):
    """Serial, per-k transcription of the documented construction."""
    Hks = np.zeros((NAWF, NAWF, NKPNTS, NSPIN), dtype=complex)
    for ik in range(NKPNTS):
        for ispin in range(NSPIN):
            my_eigs = eigs[:, ik, ispin]
            UU = np.transpose(U[:, :, ik, ispin])
            proj_k = np.real(np.sum(np.conj(UU) * UU, axis=0))
            UU[:, :NAWF] = UU[:, :NAWF] / np.sqrt(np.where(proj_k > 0.0, proj_k, 1.0))[:NAWF]
            sel = [n for n in range(NBNDS) if my_eigs[n] <= ETA]
            ac = UU[:, sel]
            ee1 = np.diag(my_eigs[sel])
            core = ac.dot(ee1).dot(np.conj(ac).T)
            if shift_type == 0:
                h = core + ETA * (np.identity(NAWF) - ac.dot(np.conj(ac).T))
            elif shift_type == 1:
                aux_p = spl.inv(np.dot(np.conj(ac).T, ac))
                h = core + ETA * (np.identity(NAWF) - ac.dot(aux_p).dot(np.conj(ac).T))
            else:
                h = core
            Hks[:, :, ik, ispin] = 0.5 * (h + np.conj(h.T))
    return Hks


@pytest.mark.parametrize('shift_type', [0, 1, 2])
def test_build_hks_matches_reference_construction(shift_type):
    U, eigs = _inputs(shift_type)
    attributes = {
        'bnd': NBNDS,
        'nawf': NAWF,
        'nspin': NSPIN,
        'nkpnts': NKPNTS,
        'npool': 1,
        'shift': ETA,
        'shift_type': shift_type,
        'pthr_local': 0.0,
    }
    Hks = build_Hks(_StubController({'U': U.copy(), 'my_eigsmat': eigs}, attributes))

    assert Hks.shape == (NAWF, NAWF, NKPNTS, NSPIN)
    assert np.array_equal(Hks, _reference(U.copy(), eigs, shift_type))


def test_build_hks_returns_hermitian_blocks():
    U, eigs = _inputs(99)
    attributes = {
        'bnd': NBNDS,
        'nawf': NAWF,
        'nspin': NSPIN,
        'nkpnts': NKPNTS,
        'npool': 1,
        'shift': ETA,
        'shift_type': 0,
        'pthr_local': 0.0,
    }
    Hks = build_Hks(_StubController({'U': U, 'my_eigsmat': eigs}, attributes))

    for ik in range(NKPNTS):
        for ispin in range(NSPIN):
            block = Hks[:, :, ik, ispin]
            assert np.allclose(block, np.conj(block.T), atol=1e-14)
