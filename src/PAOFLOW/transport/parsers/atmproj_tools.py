import numpy as np

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.DataController import DataController
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.grid.rgrid import get_rgrid
from PAOFLOW.transport.io.write_data import populate_real_space_hamiltonian
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.transport.utils.timing import timed_function


@timed_function('atmproj_to_internal')
@headered_function('Conductor Initialization')
def parse_atomic_proj(
    data: ConductorData, data_controller: DataController
) -> dict[str, np.ndarray]:
    opts = data.atomic_proj

    arry, attr = data_controller.data_dicts()

    alat = float(attr['alat'])
    arry['wk'] = arry['kpnts_wght'] / np.sum(arry['kpnts_wght'])
    arry['vkpts_cartesian'] = arry['kpnts'].T * (2.0 * np.pi / alat)  # bohr^-1

    log.log_proj_summary(data_controller, data)

    hk_data = reshape_pao_hamiltonian(data_controller)

    nk = np.array([1, 1, 4], dtype=int)  # TODO: confirm hardcoded grid
    nr = nk
    ivr, wr = get_rgrid(nr)
    hk_data.update({'ivr': ivr, 'wr': wr, 'nk': nk, 'nr': nr})
    arry.update(hk_data)

    populate_real_space_hamiltonian(data_controller, hk_data, opts.do_overlap_transformation)

    return hk_data


def reshape_pao_hamiltonian(data_controller: DataController) -> dict[str, np.ndarray]:
    arry, attr = data_controller.data_dicts()
    Hks_raw = data_controller.full_hamiltonian_k()  # (nawf, nawf, nk1, nk2, nk3, nspin)
    HRs_raw = arry['HRs']  # shape: (nawf, nawf, nk1, nk2, nk3, nspin)
    nspin = attr['nspin']
    nkpnts = attr['nkpnts']
    nawf = attr['nawf']

    # reshape to (nawf, nawf, nkpnts, nspin)
    Hks_reshaped = Hks_raw.reshape((nawf, nawf, nkpnts, nspin))
    # transpose to (nspin, nkpnts, nawf, nawf)
    Hk = np.transpose(Hks_reshaped, (3, 2, 1, 0))

    HRs_reshaped = HRs_raw.reshape((nawf, nawf, nkpnts, nspin))
    HR = np.transpose(HRs_reshaped, (3, 2, 1, 0))

    Sks_raw = (
        arry['Sks'] if 'Sks' in arry else None
    )  # TODO Paoflow uses the acbn0 flag to transform Sk in case of non-orthogonality. Check if we need Sks computed with or without the acbn0 flag to get the right answer. In the current implementation, Sks is without the acbn0 flag.
    Sk = Sks_raw[:, :nawf, :] if Sks_raw is not None else None
    Sk = np.transpose(Sk, (2, 1, 0)) if Sk is not None else None

    SRs_raw = arry['SRs'] if 'SRs' in arry else None
    SR = SRs_raw[:, :nawf, :] if SRs_raw is not None else None
    SR = np.transpose(SR, (2, 1, 0)) if SR is not None else None
    return {'Hk': Hk, 'Sk': Sk, 'HR': HR, 'SR': SR}
