def add_ext_field(data_controller):
    import numpy as np

    from .constants import ANGSTROM_AU

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    nawf, _, nk1, nk2, nk3, nspin = arrays['HRs'].shape
    arrays['HRs'] = np.reshape(arrays['HRs'], (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')

    l = 0
    natoms = attributes['natoms']
    nwf = nawf // natoms
    tau_wf = np.zeros((nawf, 3), dtype=float)
    for n in range(attributes['natoms']):
        for i in range(nwf):
            tau_wf[l, :] = arrays['tau'][n, :]
            l += 1

    tau_wf /= ANGSTROM_AU

    if arrays['Efield'].any() != 0.0:
        for n in range(nawf):
            arrays['HRs'][n, n, 0, :] -= arrays['Efield'].dot(tau_wf[n, :])

    if arrays['Bfield'].any() != 0.0:
        field = (
            arrays['Bfield'][0] * arrays['Sj'][0]
            + arrays['Bfield'][1] * arrays['Sj'][1]
            + arrays['Bfield'][2] * arrays['Sj'][2]
        )
        for n in range(nawf):
            for m in range(nawf):
                arrays['HRs'][n, m, 0, :] -= field[n, m]

    if arrays['HubbardU'].any() != 0:
        for n in range(nawf):
            arrays['HRs'][n, n, 0, :] -= arrays['HubbardU'][n] / 2.0

    arrays['HRs'] = np.reshape(arrays['HRs'], (nawf, nawf, nk1, nk2, nk3, nspin), order='C')
