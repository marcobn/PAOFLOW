"""Deterministic spin/orbital texture and REE calculation under MPI."""

import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from PAOFLOW.response.do_rashba_edelstein import do_rashba_edelstein
from PAOFLOW.topology.do_orbital_texture import do_orbital_texture
from PAOFLOW.topology.do_spin_texture import do_spin_texture
from PAOFLOW.utils.communication import scatter_full

comm = MPI.COMM_WORLD
rank = comm.rank
output = Path(sys.argv[1])
if rank == 0:
    output.mkdir(parents=True, exist_ok=True)
comm.Barrier()


class Controller:
    def __init__(self) -> None:
        self.comm, self.rank = comm, rank
        self.data_arrays = {}
        self.data_attributes = dict(
            npool=1,
            nawf=2,
            nk1=5,
            nk2=1,
            nk3=1,
            nkpnts=5,
            fermi_dw=-1.0,
            fermi_up=1.0,
            opath=str(output),
            smearing='gauss',
        )

    def data_dicts(self) -> tuple[dict, dict]:
        return self.data_arrays, self.data_attributes


controller = Controller()
arrays = controller.data_arrays
pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
arrays['Sj'], arrays['Lj'] = 0.5 * pauli, 1.5 * pauli
full_v = np.empty((5, 2, 2, 1), dtype=complex)
full_p = np.empty((5, 3, 2, 2, 1), dtype=complex)
full_e = np.empty((5, 2, 1), dtype=float)
for k in range(5):
    angle, phase = 0.12 * (k + 1), np.exp(0.21j * (k + 1))
    c, s = np.cos(angle), np.sin(angle)
    full_v[k, :, :, 0] = [[c, -s * phase.conjugate()], [s * phase, c]]
    full_e[k, :, 0] = [-0.25 + 0.015 * k, 0.21 + 0.012 * k]
    for component in range(3):
        full_p[k, component, :, :, 0] = [
            [0.7 + 0.1 * k + 0.2 * component, 0.03j],
            [-0.03j, -0.4 + 0.02 * k + 0.1 * component],
        ]
for name, value in [
    ('v_k', full_v),
    ('pksp', full_p),
    ('E_k', full_e),
    ('deltakp', np.full((5, 2, 1), 0.08)),
]:
    arrays[name] = scatter_full(value if rank == 0 else None, 1)

local_counts = comm.allgather(arrays['v_k'].shape[0])
for name, producer, prefix in [
    ('sktxt', do_spin_texture, 'spin_'),
    ('oktxt', do_orbital_texture, 'orbital_'),
]:
    producer(controller)
    assert arrays[name].shape == (local_counts[rank], 3, 2), (rank, name, arrays[name].shape)
    do_rashba_edelstein(
        controller,
        np.linspace(-0.4, 0.4, 9),
        0.0,
        1e-30,
        False,
        1.0,
        1.0,
        True,
        arrays[name],
        prefix,
    )

if rank == 0:
    (output / 'counts.json').write_text(json.dumps(local_counts))
    print('MPI texture -> REE check passed; local k-point counts:', local_counts)
