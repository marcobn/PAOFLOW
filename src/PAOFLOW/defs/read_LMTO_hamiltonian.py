def read_RS_LMTO_hamiltonian(inputpath, filename, nawf, nspin, twoD):
    import os
    import numpy as np

    fpath = os.path.join(inputpath + filename)

    f = open(fpath, 'r')
    nlines = 0
    # Read first non-empty, non-comment line for nlines
    for line in f:
        nlines += 1

    f.seek(0)
    nk1 = 3  # Hardcoded
    nk2 = 3  # Hardcoded
    if twoD == True:
        nk3 = 1  # Hardcoded
    else:
        nk3 = 3  # Hardcoded

    HRs_hr = np.zeros((nawf, nawf, nk1, nk2, nk3, nspin), dtype=complex)
    HRs = np.zeros((nawf, nawf, nk1, nk2, nk3, nspin), dtype=complex)
    
    for i in range(nlines):
        nx, ny, nz, l, m, HRs_real, HRs_imag = f.readline().split()
        nx = int(nx)
        ny = int(ny)
        nz = int(nz)
        l = int(l)
        m = int(m)
        HRs_hr[l - 1, m - 1, nx, ny, nz, 0] = complex(
            float(HRs_real), float(HRs_imag)
        )  # the minus sing is because of python index

    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                Rx = float(i) / float(nk1)
                Ry = float(j) / float(nk2)
                Rz = float(k) / float(nk3)
                if Rx >= 0.5:
                    Rx = Rx - 1.0
                if Ry >= 0.5:
                    Ry = Ry - 1.0
                if Rz >= 0.5:
                    Rz = Rz - 1.0
                Rx -= int(Rx)
                Ry -= int(Ry)
                Rz -= int(Rz)
                # the minus sign in Rx*nk1 is due to the Fourier transformation (Ri-Rj)
                ix = -round(Rx * nk1, 0)
                iy = -round(Ry * nk2, 0)
                iz = -round(Rz * nk3, 0)
                HRs[:, :, i, j, k, :] = HRs_hr[:, :, int(ix), int(iy), int(iz), :]

    return HRs
