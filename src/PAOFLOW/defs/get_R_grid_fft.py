def get_R_grid_fft(data_controller, nr1, nr2, nr3):
    import numpy as np

    arrays = data_controller.data_arrays

    nrtot = nr1 * nr2 * nr3

    a_vectors = arrays['a_vectors']

    arrays['R'] = np.zeros((nrtot, 3), dtype=float)
    arrays['idx'] = np.zeros((nr1, nr2, nr3), dtype=int)
    arrays['Rfft'] = np.zeros((nr1, nr2, nr3, 3), dtype=float)
    arrays['R_wght'] = np.ones((nrtot), dtype=float)

    for i in range(nr1):
        for j in range(nr2):
            for k in range(nr3):
                n = k + j * nr3 + i * nr2 * nr3
                Rx = float(i) / float(nr1)
                Ry = float(j) / float(nr2)
                Rz = float(k) / float(nr3)
                if Rx >= 0.5:
                    Rx = Rx - 1.0
                if Ry >= 0.5:
                    Ry = Ry - 1.0
                if Rz >= 0.5:
                    Rz = Rz - 1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)

                # evec = np.array([[1,0,0],[0,1,0],[0,0,1]])*attributes['alat']
                # arrays['R'][n,:] = Rx*nr1*evec[0,:] + Ry*nr2*evec[1,:] + Rz*nr3*evec[2,:]
                arrays['R'][n, :] = (
                    Rx * nr1 * a_vectors[0, :]
                    + Ry * nr2 * a_vectors[1, :]
                    + Rz * nr3 * a_vectors[2, :]
                )
                arrays['Rfft'][i, j, k, :] = arrays['R'][n, :]
                arrays['idx'][i, j, k] = n
