def do_non_ortho(Hks, Sks):
    import numpy as np
    from scipy import linalg as spl

    # Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(1/2)*Hks*Sks^(1/2)

    nawf, _, nkpnts, nspin = Hks.shape
    S2k = np.zeros((nawf, nawf, nkpnts), dtype=complex)
    for ik in range(nkpnts):
        S2k[:, :, ik] = spl.sqrtm(Sks[:nawf, :nawf, ik])

    Hks_no = np.zeros((nawf, nawf, nkpnts, nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            Hks_no[:, :, ik, ispin] = np.dot(S2k[:, :, ik], Hks[:, :, ik, ispin]).dot(S2k[:, :, ik])

    return Hks_no
