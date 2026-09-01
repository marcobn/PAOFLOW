import cmath

import numpy as np

# ---------------------------------------------------------------------------
# Generic on-site SOC builder
# ---------------------------------------------------------------------------
#
# The hardcoded ``soc_<l>_<token>`` matrices below were written one per
# pseudopotential layout (``sp``, ``spd``, ``sspd``, ``spds`` ...).  Adding a
# new layout (e.g. extended AE bases produced by ``configuration='extended'``)
# previously required a new hand-derived matrix.  The functions in this
# section build the same L·S blocks from the canonical ``L`` operators in
# the QE tesseral basis, and place them at the correct offset for any
# ``shells`` list.  ``do_spin_orbit_H`` dispatches to them when
# ``orb_pseudo[n] == 'generic'`` (set by :meth:`DataController.build_arrays_adhoc_soc`
# for any layout not covered by the hardcoded set).
#
# The convention follows the existing kernels:
#   * Tesseral order per shell: m=0, then (m=+kc, m=+ks) for k=1..l
#     (i.e. for l=1: pz, px, py; for l=2: dz2, dxz, dyz, dx2-y2, dxy).
#   * Quantisation axis n̂ = (sinθ cosφ, sinθ sinφ, cosθ);
#     θ̂ = (cosθ cosφ, cosθ sinφ, -sinθ); φ̂ = (-sinφ, cosφ, 0).
#   * H_UU = +(L·n̂)/2,  H_DD = -(L·n̂)/2,
#     H_UD = (L·θ̂ - i L·φ̂)/2,  H_DU = H_UD†.


def _angular_momentum_matrices(l):
    """Return ``(Lx, Ly, Lz)`` as Hermitian ``(2l+1, 2l+1)`` complex matrices
    in the QE real-tesseral basis.

    Hardcoded for ``l = 0, 1, 2, 3`` (the only channels exposed by the rest
    of PAOFLOW's adhoc SOC).  Values are read off the existing
    ``soc_p_*`` / ``soc_d_*``/``soc_f_*`` kernels so the generic builder is
    bit-for-bit compatible with them.
    """
    if l == 0:
        z = np.zeros((1, 1), dtype=complex)
        return z, z, z
    if l == 1:
        Lx = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]], dtype=complex)
        Ly = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
        Lz = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
        return Lx, Ly, Lz
    if l == 2:
        s3 = np.sqrt(3.0)
        Lx = np.zeros((5, 5), dtype=complex)
        Lx[0, 2] = 1j * s3
        Lx[2, 0] = -1j * s3
        Lx[1, 4] = 1j
        Lx[4, 1] = -1j
        Lx[2, 3] = -1j
        Lx[3, 2] = 1j
        Ly = np.zeros((5, 5), dtype=complex)
        Ly[0, 1] = -1j * s3
        Ly[1, 0] = 1j * s3
        Ly[1, 3] = -1j
        Ly[3, 1] = 1j
        Ly[2, 4] = -1j
        Ly[4, 2] = 1j
        Lz = np.zeros((5, 5), dtype=complex)
        Lz[1, 2] = -1j
        Lz[2, 1] = 1j
        Lz[3, 4] = -2j
        Lz[4, 3] = 2j
        return Lx, Ly, Lz
    ########################## FROM wikipedia 
    # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics 
    # l=3
    # in the order Y_3,m m= [0,1,-1,2,-2,3,-3]
    if l==3:
        s6,s10=np.sqrt(6.0),np.sqrt(10.0)

        Lz=np.zeros((7,7), dtype=complex)
        Lz[1,2],Lz[2,1],Lz[3,4],Lz[4,3],Lz[5,6],Lz[6,5]= -1j,1j,-2j,2j,-3j,3j
        
        Lx = np.array([
    [0.+0.j,          0.+0.j,          0.+s6*1j, 0.+0.j,          0.+0.j,          0.+0.j,          0.+0.j],
    [0.+0.j,          0.+0.j,          0.+0.j,         0.+0.j,          0.+s10/2j, 0.+0.j,          0.+0.j],
    [0.-s6*1j,  0.+0.j,          0.+0.j,         0.-s10/2j, 0.+0.j,          0.+0.j,          0.+0.j],
    [0.+0.j,          0.+0.j,          0.+s10/2j, 0.+0.j,          0.+0.j,          0.+0.j,          0.+s6/2j],
    [0.+0.j,          0.-s10/2j, 0.+0.j,          0.+0.j,          0.+0.j,          0.-s6/2j, 0.+0.j],
    [0.+0.j,          0.+0.j,          0.+0.j,         0.+0.j,          0.+s6/2j, 0.+0.j,          0.+0.j],
    [0.+0.j,          0.+0.j,          0.+0.j,         0.-s6/2j, 0.+0.j,          0.+0.j,          0.+0.j]
], dtype=complex)

        Ly = np.array([
    [0.+0.j,          0.-s6*1j, 0.+0.j,          0.+0.j,          0.+0.j,          0.+0.j,          0.+0.j],
    [0.+s6*1j,  0.+0.j,          0.+0.j,         0.-s10/2j, 0.+0.j,          0.+0.j,          0.+0.j],
    [0.+0.j,          0.+0.j,          0.+0.j,         0.+0.j,          0.-s10/2j, 0.+0.j,          0.+0.j],
    [0.+0.j,          0.+s10/2j, 0.+0.j,          0.+0.j,          0.+0.j,          0.-s6/2j, 0.+0.j],
    [0.+0.j,          0.+0.j,          0.+s10/2j, 0.+0.j,          0.+0.j,          0.+0.j,          0.-s6/2j],
    [0.+0.j,          0.+0.j,          0.+0.j,         0.+s6/2j, 0.+0.j,          0.+0.j,          0.+0.j],
    [0.+0.j,          0.+0.j,          0.+0.j,         0.+0.j,          0.+s6/2j, 0.+0.j,          0.+0.j]
], dtype=complex)
  
        return Lx,Ly,Lz




def _shell_soc_block(theta, phi, l):
    """Build the ``(2(2l+1)) x (2(2l+1))`` L·S block for a single shell.

    Layout: rows/cols ``0..2l`` are spin-up, ``2l+1..4l+1`` are spin-down,
    matching the convention of the hardcoded ``soc_<l>_<token>`` kernels.
    """
    Lx, Ly, Lz = _angular_momentum_matrices(l)
    n = 2 * l + 1
    sT, cT = np.sin(theta), np.cos(theta)
    sP, cP = np.sin(phi), np.cos(phi)
    L3 = (sT * cP) * Lx + (sT * sP) * Ly + cT * Lz  # L · n̂
    L1 = (cT * cP) * Lx + (cT * sP) * Ly + (-sT) * Lz  # L · θ̂
    L2 = (-sP) * Lx + cP * Ly  # L · φ̂
    H = np.zeros((2 * n, 2 * n), dtype=complex)
    H[:n, :n] = 0.5 * L3
    H[n:, n:] = -0.5 * L3
    H[:n, n:] = 0.5 * (L1 - 1j * L2)
    H[n:, :n] = H[:n, n:].conj().T
    return H


def build_generic_soc(theta, phi, shells_list, norb, active_shells=None):
    """Build per-l SOC contributions for an arbitrary shells layout.

    Parameters
    ----------
    theta, phi : float
        Quantisation-axis angles (radians).
    shells_list : sequence of int
        Per-shell angular momenta in the order they appear in the atom's
        orbital basis (e.g. ``[0, 1, 2]`` for ``spd``, ``[0, 1, 2, 0]``
        for ``spds``, ``[0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2]`` for an
        extended Pt basis).
    norb : int
        Total number of orbitals per atom (must equal ``sum(2l+1)``).
    active_shells : sequence of bool or float, optional
        Per-shell weight applied to the SOC kernel of each shell.
        Booleans are interpreted as 0/1.  ``None`` (default) uses a
        hydrogenic-style ``1/k**2`` falloff over the occurrence index of
        each l > 0 channel: the valence shell (k=1) gets weight 1, the
        first augmentation (k=2) gets 0.25, the second (k=3) gets
        ~0.111, etc.  l=0 shells always get 0.

    Returns
    -------
    (HR_soc_p, HR_soc_d) : tuple of two ``(2*norb, 2*norb)`` complex arrays
        SOC blocks restricted to l=1 and l=2 shells respectively.  Caller
        weights them by ``lambda_p`` and ``lambda_d``.

    Notes
    -----
    The per-element ``lambda_p`` / ``lambda_d`` are fit (or read from
    tables) for the valence p/d shell.  Applying the same ``lambda`` to
    every shell of the same l in an extended/augmented basis triple-counts
    SOC on unoccupied polarisation channels and corrupts response
    functions (e.g. spin Hall conductivity).  Conversely, zeroing the
    augmentation shells entirely under-broadens the conduction-side
    response.  The default ``1/k**2`` heuristic is a compromise that
    preserves the valence kernel exactly (matching the hardcoded
    minimal-basis kernels ``soc_p_spd``, ``soc_p_spds``, ...) while
    leaving a small residual weight on the augmentation channels.  Pass
    an explicit ``active_shells`` (bool or float per shell) to override.
    """
    HR_p = np.zeros((2 * norb, 2 * norb), dtype=complex)
    HR_d = np.zeros((2 * norb, 2 * norb), dtype=complex)
    HR_f = np.zeros((2 * norb, 2 * norb), dtype=complex)
    if active_shells is None:
        counts = {}
        weights = []
        for l in shells_list:
            if l == 0:
                weights.append(0.0)
                continue
            counts[l] = counts.get(l, 0) + 1
            weights.append(1.0 / counts[l] ** 2)
    else:
        if len(active_shells) != len(shells_list):
            raise ValueError(
                'active_shells length (%d) does not match shells_list length (%d).'
                % (len(active_shells), len(shells_list))
            )
        weights = [float(w) for w in active_shells]
    offset = 0
    for l, w in zip(shells_list, weights):
        n = 2 * l + 1
        if l == 0 or w == 0.0:
            offset += n
            continue
        H = _shell_soc_block(theta, phi, l)
        if l == 1:
            target = HR_p
        elif l == 2:
            target = HR_d
        elif l == 3:
            target = HR_f
        else:
            offset += n
            continue
        sl_up = slice(offset, offset + n)
        sl_dn = slice(offset + norb, offset + norb + n)
        target[sl_up, sl_up] += w * H[:n, :n]
        target[sl_dn, sl_dn] += w * H[n:, n:]
        target[sl_up, sl_dn] += w * H[:n, n:]
        target[sl_dn, sl_up] += w * H[n:, :n]
        offset += n
    if offset != norb:
        raise ValueError(
            'shells_list orbital count (%d) does not match norb (%d).' % (offset, norb)
        )
    return HR_p, HR_d,HR_f


# ---------------------------------------------------------------------------


def do_spin_orbit_H(data_controller):
    """Construct the tight-binding spin-orbit Hamiltonian in real space.

    Follows the formalism of Abate and Asdente, *Phys. Rev.* **140**, A1303
    (1965).

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``naw``, ``orb_pseudo``, ``lambda_p``, ``lambda_d``.,``lambda_f``
        Required attributes: ``natoms``, ``theta``, ``phi``.

    Returns
    -------
    None
        Modifies ``data_arrays`` in place:

        - ``HRs`` : replaced by the spin-orbit-coupled Hamiltonian of shape
          ``(2*nawf, 2*nawf, nk1, nk2, nk3, 1)``.
        - ``naw`` : extended by appending a copy of itself.

        Updates ``data_attributes``:

        - ``nawf`` : set to ``2 * nawf``.

    Notes
    -----
    For a non-magnetic system the spin-degenerate block is
    :math:`H_{\\uparrow\\uparrow} = H_{\\downarrow\\downarrow} = H_0`.  For
    a magnetic system the two spin channels are taken from the two spin
    components of ``HRs``.  The SOC term is added only at the
    :math:`\\mathbf{R}=(0,0,0)` site; p-/d-/f-channel couplings are weighted
    by ``lambda_p`` and ``lambda_d``,``lambda_f``, respectively.
    """
    # construct TB spin orbit Hamiltonian (following Abate and Asdente, Phys. Rev. 140, A1303 (1965))

    arry, attr = data_controller.data_dicts()

    natoms = attr['natoms']
    theta, phi = attr['theta'], attr['phi']
    norb_array, orb_pseudo = arry['naw'], arry['orb_pseudo']

    nawf, _, nk1, nk2, nk3, nspin = arry['HRs'].shape

    socStrengh = np.zeros((natoms, 3), dtype=float)
    socStrengh[:, 0] = arry['lambda_p'][0:natoms]
    socStrengh[:, 1] = arry['lambda_d'][0:natoms]
    socStrengh[:, 2] = arry['lambda_f'][0:natoms]

    HR_double = np.zeros((2 * nawf, 2 * nawf, nk1, nk2, nk3, 1), dtype=complex)

    # nonmagnetic :  copy H at the upper (lower) left (right) of the double matrix HR_double
    if nspin == 1:
        HR_double[:nawf, :nawf, :, :, :, 0] = HR_double[
            nawf : 2 * nawf, nawf : 2 * nawf, :, :, :, 0
        ] = arry['HRs'][:nawf, :nawf, :, :, :, 0]
    # magnetic :  copy H_up (H_down) at the upper (lower) left (right) of the double matrix
    else:
        HR_double[:nawf, :nawf, :, :, :, 0] = arry['HRs'][:nawf, :nawf, :, :, :, 0]
        HR_double[nawf : 2 * nawf, nawf : 2 * nawf, :, :, :, 0] = arry['HRs'][
            :nawf, :nawf, :, :, :, 1
        ]

    ##### Need clarification
    offset = np.zeros(natoms, dtype=int)
    for i in range(1, natoms):
        offset[i] = offset[i - 1] + norb_array[i - 1]

    for n, norb in enumerate(norb_array):
        HR_soc_p = np.zeros((2 * norb, 2 * norb), dtype=complex)
        HR_soc_d = np.zeros((2 * norb, 2 * norb), dtype=complex)
        HR_soc_f = np.zeros((2 * norb, 2 * norb), dtype=complex)
        if orb_pseudo[n] == 's':
            pass
        if orb_pseudo[n] == 'sp':
            HR_soc_p[:, :] = soc_p_sp(theta, phi, norb)
        if orb_pseudo[n] == 'spd':
            HR_soc_p[:, :] = soc_p_spd(theta, phi, norb)
            HR_soc_d[:, :] = soc_d_spd(theta, phi, norb)
        if orb_pseudo[n] == 'ps':
            HR_soc_p[:, :] = soc_p_ps(theta, phi, norb)
        if orb_pseudo[n] == 'sspd':
            HR_soc_p[:, :] = soc_p_sspd(theta, phi, norb)
            HR_soc_d[:, :] = soc_d_sspd(theta, phi, norb)
        if orb_pseudo[n] == 'spds':
            HR_soc_p[:, :] = soc_p_spds(theta, phi, norb)
            HR_soc_d[:, :] = soc_d_spds(theta, phi, norb)
        if orb_pseudo[n] == 'ssp':
            pass
        if orb_pseudo[n] == 'ssppd':
            HR_soc_p[:, :] = soc_p_ssppd(theta, phi, norb)
            HR_soc_d[:, :] = soc_d_ssppd(theta, phi, norb)
        if orb_pseudo[n] == 'generic':
            shells_for_atom = arry['shells'][arry['atoms'][n]]
            sw_map = arry.get('soc_shell_weights')
            weights = sw_map.get(arry['atoms'][n]) if sw_map else None
            HR_soc_p[:, :], HR_soc_d[:, :],HR_soc_f[:, :] = build_generic_soc(
                theta, phi, shells_for_atom, norb, active_shells=weights
            )

        uui = offset[n]
        uuj = uui + norb
        udi = ddi = uui + nawf
        udj = ddj = uuj + nawf

        # Up-Up
        HR_double[uui:uuj, uui:uuj, 0, 0, 0, 0] += (
            socStrengh[n, 0] * HR_soc_p[0:norb, 0:norb]
            + socStrengh[n, 1] * HR_soc_d[0:norb, 0:norb]
            + socStrengh[n, 2] * HR_soc_f[0:norb, 0:norb]
        )
        # Down-Down
        HR_double[ddi:ddj, ddi:ddj, 0, 0, 0, 0] += (
            socStrengh[n, 0] * HR_soc_p[norb : 2 * norb, norb : 2 * norb]
            + socStrengh[n, 1] * HR_soc_d[norb : 2 * norb, norb : 2 * norb]
            +socStrengh[n, 2] * HR_soc_f[norb : 2 * norb, norb : 2 * norb]
        )
        # Up-Down
        HR_double[uui:uuj, udi:udj, 0, 0, 0, 0] += (
            socStrengh[n, 0] * HR_soc_p[0:norb, norb : 2 * norb]
            + socStrengh[n, 1] * HR_soc_d[0:norb, norb : 2 * norb]
            + socStrengh[n, 2] * HR_soc_f[0:norb, norb : 2 * norb]
        )
        # Down-Up
        HR_double[ddi:ddj, uui:uuj, 0, 0, 0, 0] += (
            socStrengh[n, 0] * HR_soc_p[norb : 2 * norb, 0:norb]
            + socStrengh[n, 1] * HR_soc_d[norb : 2 * norb, 0:norb]
            + socStrengh[n, 2] * HR_soc_f[norb : 2 * norb, 0:norb]
        )

        del arry['HRs']
        arry['HRs'] = HR_double
        attr['nawf'] = arry['HRs'].shape[0]

    arry['naw'] = np.append(arry['naw'], arry['naw'])

################### PSEUDOPOTENTIAL PS ##############################33
def soc_p_ps(theta, phi, norb):
    """Build the p-orbital spin-orbit coupling matrix for the **PS** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        Spin-orbit coupling matrix in the
        :math:`|\\uparrow\\rangle \\oplus |\\downarrow\\rangle` basis.  The
        p-state block occupies indices 0–2 (up) and ``norb``–``norb+2``
        (down).
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    # Spin Up - Spin Up  part of the p-satets Hamiltonian
    HR_soc[0, 1] = -0.5j * sTheta * sPhi
    HR_soc[0, 2] = 0.5j * sTheta * cPhi
    HR_soc[1, 2] = -0.5j * cTheta
    HR_soc[1, 0] = np.conj(HR_soc[0, 1])
    HR_soc[2, 0] = np.conj(HR_soc[0, 2])
    HR_soc[2, 1] = np.conj(HR_soc[1, 2])
    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[5:8, 5:8] = -HR_soc[1:4, 1:4]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[0, 5] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[0, 6] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[1, 6] = 0.5j * sTheta
    HR_soc[1, 4] = -HR_soc[0, 5]
    HR_soc[2, 4] = -HR_soc[0, 6]
    HR_soc[2, 5] = -HR_soc[1, 6]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[5, 0] = np.conj(HR_soc[0, 5])
    HR_soc[6, 0] = np.conj(HR_soc[0, 6])
    HR_soc[4, 1] = np.conj(HR_soc[1, 4])
    HR_soc[6, 1] = np.conj(HR_soc[1, 6])
    HR_soc[4, 2] = np.conj(HR_soc[2, 4])
    HR_soc[5, 2] = np.conj(HR_soc[2, 5])

    return HR_soc


################## END PSEUDOPOTENTIAL PS ##############################


################## PSEUDOPOTENTIAL SP ##############################
def soc_p_sp(theta, phi, norb):
    """Build the p-orbital spin-orbit coupling matrix for the **SP** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        Spin-orbit coupling matrix; the p-state block occupies indices 1–3
        (s precedes p in the SP basis).
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    # Spin Up - Spin Up  part of the p-satets Hamiltonian
    HR_soc[1, 2] = -0.5j * sTheta * sPhi
    HR_soc[1, 3] = 0.5j * sTheta * cPhi
    HR_soc[2, 3] = -0.5j * cTheta
    HR_soc[2, 1] = np.conj(HR_soc[1, 2])
    HR_soc[3, 1] = np.conj(HR_soc[1, 3])
    HR_soc[3, 2] = np.conj(HR_soc[2, 3])
    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[5:8, 5:8] = -HR_soc[1:4, 1:4]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[1, 6] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[1, 7] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[2, 7] = 0.5j * sTheta
    HR_soc[2, 5] = -HR_soc[1, 6]
    HR_soc[3, 5] = -HR_soc[1, 7]
    HR_soc[3, 6] = -HR_soc[2, 7]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[6, 1] = np.conj(HR_soc[1, 6])
    HR_soc[7, 1] = np.conj(HR_soc[1, 7])
    HR_soc[5, 2] = np.conj(HR_soc[2, 5])
    HR_soc[7, 2] = np.conj(HR_soc[2, 7])
    HR_soc[5, 3] = np.conj(HR_soc[3, 5])
    HR_soc[6, 3] = np.conj(HR_soc[3, 6])

    return HR_soc


################## END PSEUDOPOTENTIAL SP ##############################


################## PSEUDOPOTENTIAL SPD ##############################
def soc_p_spd(theta, phi, norb):
    """Build the p-orbital spin-orbit coupling matrix for the **SPD** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        p-channel spin-orbit coupling matrix in the SPD orbital ordering.
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    # Spin Up - Spin Up  part of the p-satets Hamiltonian
    HR_soc[1, 2] = -0.5j * sTheta * sPhi
    HR_soc[1, 3] = 0.5j * sTheta * cPhi
    HR_soc[2, 3] = -0.5j * cTheta
    HR_soc[2, 1] = np.conj(HR_soc[1, 2])
    HR_soc[3, 1] = np.conj(HR_soc[1, 3])
    HR_soc[3, 2] = np.conj(HR_soc[2, 3])
    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[10:13, 10:13] = -HR_soc[1:4, 1:4]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[1, 11] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[1, 12] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[2, 12] = 0.5j * sTheta
    HR_soc[2, 10] = -HR_soc[1, 11]
    HR_soc[3, 10] = -HR_soc[1, 12]
    HR_soc[3, 11] = -HR_soc[2, 12]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[11, 1] = np.conj(HR_soc[1, 11])
    HR_soc[12, 1] = np.conj(HR_soc[1, 12])
    HR_soc[10, 2] = np.conj(HR_soc[2, 10])
    HR_soc[12, 2] = np.conj(HR_soc[2, 12])
    HR_soc[10, 3] = np.conj(HR_soc[3, 10])
    HR_soc[11, 3] = np.conj(HR_soc[3, 11])

    return HR_soc


def soc_d_spd(theta, phi, norb):
    """Build the d-orbital spin-orbit coupling matrix for the **SPD** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        d-channel spin-orbit coupling matrix in the SPD orbital ordering
        (d-states occupy indices 4–8).
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    s3 = np.sqrt(3.0)

    # Spin Up - Spin Up  part of the d-satets Hamiltonian
    HR_soc[4, 5] = -s3 * 0.5j * sTheta * sPhi
    HR_soc[4, 6] = s3 * 0.5j * sTheta * cPhi
    HR_soc[5, 6] = -0.5j * cTheta
    HR_soc[5, 7] = -0.5j * sTheta * sPhi
    HR_soc[5, 8] = 0.5j * sTheta * cPhi
    HR_soc[6, 7] = -0.5j * sTheta * cPhi
    HR_soc[6, 8] = -0.5j * sTheta * sPhi
    HR_soc[7, 8] = -1.0j * cTheta
    HR_soc[5, 4] = np.conj(HR_soc[4, 5])
    HR_soc[6, 4] = np.conj(HR_soc[4, 6])
    HR_soc[6, 5] = np.conj(HR_soc[5, 6])
    HR_soc[7, 5] = np.conj(HR_soc[5, 7])
    HR_soc[8, 5] = np.conj(HR_soc[5, 8])
    HR_soc[7, 6] = np.conj(HR_soc[6, 7])
    HR_soc[8, 6] = np.conj(HR_soc[6, 8])

    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[13:18, 13:18] = -HR_soc[4:9, 4:9]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[4, 14] = -s3 * 0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[4, 15] = -s3 * 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[5, 15] = 0.5j * sTheta
    HR_soc[5, 16] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[5, 17] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[6, 16] = 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[6, 17] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[7, 17] = 1j * sTheta
    HR_soc[5, 13] = -HR_soc[4, 14]
    HR_soc[6, 13] = -HR_soc[4, 15]
    HR_soc[6, 14] = -HR_soc[5, 15]
    HR_soc[7, 14] = -HR_soc[5, 16]
    HR_soc[8, 14] = -HR_soc[5, 17]
    HR_soc[7, 15] = -HR_soc[6, 16]
    HR_soc[8, 15] = -HR_soc[6, 17]
    HR_soc[8, 16] = -HR_soc[7, 17]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[14, 4] = np.conj(HR_soc[4, 14])
    HR_soc[15, 4] = np.conj(HR_soc[4, 15])
    HR_soc[15, 5] = np.conj(HR_soc[5, 15])
    HR_soc[16, 5] = np.conj(HR_soc[5, 16])
    HR_soc[17, 5] = np.conj(HR_soc[5, 17])
    HR_soc[16, 6] = np.conj(HR_soc[6, 16])
    HR_soc[17, 6] = np.conj(HR_soc[6, 17])
    HR_soc[17, 7] = np.conj(HR_soc[7, 17])
    HR_soc[13, 5] = np.conj(HR_soc[5, 13])
    HR_soc[13, 6] = np.conj(HR_soc[6, 13])
    HR_soc[14, 6] = np.conj(HR_soc[6, 14])
    HR_soc[14, 7] = np.conj(HR_soc[7, 14])
    HR_soc[14, 8] = np.conj(HR_soc[8, 14])
    HR_soc[15, 7] = np.conj(HR_soc[7, 15])
    HR_soc[15, 8] = np.conj(HR_soc[8, 15])

    return HR_soc


################## END PSEUDOPOTENTIAL SPD ##############################


################## PSEUDOPOTENTIAL SSPD ##############################
def soc_p_sspd(theta, phi, norb):
    """Build the p-orbital spin-orbit coupling matrix for the **SSPD** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        p-channel spin-orbit coupling matrix in the SSPD orbital ordering
        (p-states occupy indices 2–4).

    Notes
    -----
    Orbital layout is hardcoded to the s, s*, p, d basis.
    """
    # Hardcoded to s,p,d. This must change latter.
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    # Spin Up - Spin Up  part of the p-satets Hamiltonian
    HR_soc[2, 3] = -0.5j * sTheta * sPhi
    HR_soc[2, 4] = 0.5j * sTheta * cPhi
    HR_soc[3, 4] = -0.5j * cTheta
    HR_soc[3, 2] = np.conj(HR_soc[2, 3])
    HR_soc[4, 2] = np.conj(HR_soc[2, 4])
    HR_soc[4, 3] = np.conj(HR_soc[3, 4])
    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[12:15, 12:15] = -HR_soc[2:5, 2:5]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[2, 13] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[2, 14] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[3, 14] = 0.5j * sTheta
    HR_soc[3, 12] = -HR_soc[2, 13]
    HR_soc[4, 12] = -HR_soc[2, 14]
    HR_soc[4, 13] = -HR_soc[3, 14]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[13, 2] = np.conj(HR_soc[2, 13])
    HR_soc[14, 2] = np.conj(HR_soc[2, 14])
    HR_soc[12, 3] = np.conj(HR_soc[3, 12])
    HR_soc[14, 3] = np.conj(HR_soc[3, 14])
    HR_soc[12, 4] = np.conj(HR_soc[4, 12])
    HR_soc[13, 4] = np.conj(HR_soc[4, 13])

    return HR_soc


def soc_d_sspd(theta, phi, norb):
    """Build the d-orbital spin-orbit coupling matrix for the **SSPD** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        d-channel spin-orbit coupling matrix in the SSPD orbital ordering
        (d-states occupy indices 5–9).

    Notes
    -----
    Orbital layout is hardcoded to the s, s*, p, d basis.
    """
    # Hardcoded to s,p,d. This must change latter.
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    s3 = cmath.sqrt(3.0)

    # Spin Up - Spin Up  part of the d-satets Hamiltonian
    HR_soc[5, 6] = -s3 * 0.5j * sTheta * sPhi
    HR_soc[5, 7] = s3 * 0.5j * sTheta * cPhi
    HR_soc[6, 7] = -0.5j * cTheta
    HR_soc[6, 8] = -0.5j * sTheta * sPhi
    HR_soc[6, 9] = 0.5j * sTheta * cPhi
    HR_soc[7, 8] = -0.5j * sTheta * cPhi
    HR_soc[7, 9] = -0.5j * sTheta * sPhi
    HR_soc[8, 9] = -1j * cTheta
    HR_soc[6, 5] = np.conj(HR_soc[5, 6])
    HR_soc[7, 5] = np.conj(HR_soc[5, 7])
    HR_soc[7, 6] = np.conj(HR_soc[6, 7])
    HR_soc[8, 6] = np.conj(HR_soc[6, 8])
    HR_soc[9, 6] = np.conj(HR_soc[6, 9])
    HR_soc[8, 7] = np.conj(HR_soc[7, 8])
    HR_soc[9, 7] = np.conj(HR_soc[7, 9])
    HR_soc[9, 8] = np.conj(HR_soc[8, 9])

    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[15:20, 15:20] = -HR_soc[5:10, 5:10]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[5, 16] = -s3 * 0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[5, 17] = -s3 * 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[6, 17] = 0.5j * sTheta
    HR_soc[6, 18] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[6, 19] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[7, 18] = 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[7, 19] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[8, 19] = 1j * sTheta
    HR_soc[6, 15] = -HR_soc[5, 16]
    HR_soc[7, 15] = -HR_soc[5, 17]
    HR_soc[7, 16] = -HR_soc[6, 17]
    HR_soc[8, 16] = -HR_soc[6, 18]
    HR_soc[9, 16] = -HR_soc[6, 19]
    HR_soc[8, 17] = -HR_soc[7, 18]
    HR_soc[9, 17] = -HR_soc[7, 19]
    HR_soc[9, 18] = -HR_soc[8, 19]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[16, 5] = np.conj(HR_soc[5, 16])
    HR_soc[17, 5] = np.conj(HR_soc[5, 17])
    HR_soc[17, 6] = np.conj(HR_soc[6, 17])
    HR_soc[18, 6] = np.conj(HR_soc[6, 18])
    HR_soc[19, 6] = np.conj(HR_soc[6, 19])
    HR_soc[18, 7] = np.conj(HR_soc[7, 18])
    HR_soc[19, 7] = np.conj(HR_soc[7, 19])
    HR_soc[19, 8] = np.conj(HR_soc[8, 19])
    HR_soc[15, 6] = np.conj(HR_soc[6, 15])
    HR_soc[15, 7] = np.conj(HR_soc[7, 15])
    HR_soc[16, 7] = np.conj(HR_soc[7, 16])
    HR_soc[16, 8] = np.conj(HR_soc[8, 16])
    HR_soc[16, 9] = np.conj(HR_soc[9, 16])
    HR_soc[17, 8] = np.conj(HR_soc[8, 17])
    HR_soc[17, 9] = np.conj(HR_soc[9, 17])

    return HR_soc


################## END PSEUDOPOTENTIAL SSPD ##############################


################## PSEUDOPOTENTIAL SPDS ##############################
def soc_p_spds(theta, phi, norb):
    """p-orbital SOC matrix for the **SPDS** layout (s, p, d, s_semicore).

    Identical to :func:`soc_p_spd` except the down-spin block offset is
    parametrised on ``norb`` (=10 here) instead of the hard-coded ``9``.
    The trailing semicore-s orbital at index 9 is left untouched (the
    SOC operator vanishes on l=0).
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    # Up-Up p block (indices 1..3)
    HR_soc[1, 2] = -0.5j * sTheta * sPhi
    HR_soc[1, 3] = 0.5j * sTheta * cPhi
    HR_soc[2, 3] = -0.5j * cTheta
    HR_soc[2, 1] = np.conj(HR_soc[1, 2])
    HR_soc[3, 1] = np.conj(HR_soc[1, 3])
    HR_soc[3, 2] = np.conj(HR_soc[2, 3])
    # Down-Down p block (indices 1+norb .. 3+norb)
    d1, d2, d3 = 1 + norb, 2 + norb, 3 + norb
    HR_soc[d1 : d3 + 1, d1 : d3 + 1] = -HR_soc[1:4, 1:4]
    # Up-Down p block
    HR_soc[1, d2] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[1, d3] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[2, d3] = 0.5j * sTheta
    HR_soc[2, d1] = -HR_soc[1, d2]
    HR_soc[3, d1] = -HR_soc[1, d3]
    HR_soc[3, d2] = -HR_soc[2, d3]
    # Down-Up p block (Hermitian conjugate)
    HR_soc[d2, 1] = np.conj(HR_soc[1, d2])
    HR_soc[d3, 1] = np.conj(HR_soc[1, d3])
    HR_soc[d1, 2] = np.conj(HR_soc[2, d1])
    HR_soc[d3, 2] = np.conj(HR_soc[2, d3])
    HR_soc[d1, 3] = np.conj(HR_soc[3, d1])
    HR_soc[d2, 3] = np.conj(HR_soc[3, d2])

    return HR_soc


def soc_d_spds(theta, phi, norb):
    """d-orbital SOC matrix for the **SPDS** layout (d at indices 4..8)."""
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)
    s3 = np.sqrt(3.0)

    # Up-Up d block (indices 4..8)
    HR_soc[4, 5] = -s3 * 0.5j * sTheta * sPhi
    HR_soc[4, 6] = s3 * 0.5j * sTheta * cPhi
    HR_soc[5, 6] = -0.5j * cTheta
    HR_soc[5, 7] = -0.5j * sTheta * sPhi
    HR_soc[5, 8] = 0.5j * sTheta * cPhi
    HR_soc[6, 7] = -0.5j * sTheta * cPhi
    HR_soc[6, 8] = -0.5j * sTheta * sPhi
    HR_soc[7, 8] = -1.0j * cTheta
    HR_soc[5, 4] = np.conj(HR_soc[4, 5])
    HR_soc[6, 4] = np.conj(HR_soc[4, 6])
    HR_soc[6, 5] = np.conj(HR_soc[5, 6])
    HR_soc[7, 5] = np.conj(HR_soc[5, 7])
    HR_soc[8, 5] = np.conj(HR_soc[5, 8])
    HR_soc[7, 6] = np.conj(HR_soc[6, 7])
    HR_soc[8, 6] = np.conj(HR_soc[6, 8])
    HR_soc[8, 7] = np.conj(HR_soc[7, 8])
    # Down-Down d block (indices 4+norb .. 8+norb)
    e4, e5, e6, e7, e8 = 4 + norb, 5 + norb, 6 + norb, 7 + norb, 8 + norb
    HR_soc[e4 : e8 + 1, e4 : e8 + 1] = -HR_soc[4:9, 4:9]
    # Up-Down d block
    HR_soc[4, e5] = -s3 * 0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[4, e6] = -s3 * 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[5, e6] = 0.5j * sTheta
    HR_soc[5, e7] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[5, e8] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[6, e7] = 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[6, e8] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[7, e8] = 1j * sTheta
    HR_soc[5, e4] = -HR_soc[4, e5]
    HR_soc[6, e4] = -HR_soc[4, e6]
    HR_soc[6, e5] = -HR_soc[5, e6]
    HR_soc[7, e5] = -HR_soc[5, e7]
    HR_soc[8, e5] = -HR_soc[5, e8]
    HR_soc[7, e6] = -HR_soc[6, e7]
    HR_soc[8, e6] = -HR_soc[6, e8]
    HR_soc[8, e7] = -HR_soc[7, e8]
    # Down-Up d block (Hermitian conjugate)
    HR_soc[e5, 4] = np.conj(HR_soc[4, e5])
    HR_soc[e6, 4] = np.conj(HR_soc[4, e6])
    HR_soc[e6, 5] = np.conj(HR_soc[5, e6])
    HR_soc[e7, 5] = np.conj(HR_soc[5, e7])
    HR_soc[e8, 5] = np.conj(HR_soc[5, e8])
    HR_soc[e7, 6] = np.conj(HR_soc[6, e7])
    HR_soc[e8, 6] = np.conj(HR_soc[6, e8])
    HR_soc[e8, 7] = np.conj(HR_soc[7, e8])
    HR_soc[e4, 5] = np.conj(HR_soc[5, e4])
    HR_soc[e4, 6] = np.conj(HR_soc[6, e4])
    HR_soc[e5, 6] = np.conj(HR_soc[6, e5])
    HR_soc[e5, 7] = np.conj(HR_soc[7, e5])
    HR_soc[e5, 8] = np.conj(HR_soc[8, e5])
    HR_soc[e6, 7] = np.conj(HR_soc[7, e6])
    HR_soc[e6, 8] = np.conj(HR_soc[8, e6])
    HR_soc[e7, 8] = np.conj(HR_soc[8, e7])

    return HR_soc


################## END PSEUDOPOTENTIAL SPDS ##############################


################## PSEUDOPOTENTIAL SSPPD ##############################
def soc_p_ssppd(theta, phi, norb):
    """Build the p-orbital spin-orbit coupling matrix for the **SSPPD** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        p-channel spin-orbit coupling matrix in the SSPPD orbital ordering
        (p-states occupy indices 5–7).
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    # Spin Up - Spin Up  part of the p-satets Hamiltonian
    HR_soc[5, 6] = -0.5j * sTheta * sPhi
    HR_soc[5, 7] = 0.5j * sTheta * cPhi
    HR_soc[6, 7] = -0.5j * cTheta
    HR_soc[6, 5] = np.conj(HR_soc[5, 6])
    HR_soc[7, 5] = np.conj(HR_soc[5, 7])
    HR_soc[7, 6] = np.conj(HR_soc[6, 7])
    # Spin Down - Spin Down  part of the p-satets Hamiltonian
    HR_soc[18:21, 18:21] = -HR_soc[5:8, 5:8]
    # Spin Up - Spin Down  part of the p-satets Hamiltonian
    HR_soc[5, 19] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[5, 20] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[6, 20] = 0.5j * sTheta
    HR_soc[6, 18] = -HR_soc[5, 19]
    HR_soc[7, 18] = -HR_soc[5, 20]
    HR_soc[7, 19] = -HR_soc[6, 20]
    # Spin Down - Spin Up  part of the p-satets Hamiltonian
    HR_soc[19, 5] = np.conj(HR_soc[5, 19])
    HR_soc[20, 5] = np.conj(HR_soc[5, 20])
    HR_soc[18, 6] = np.conj(HR_soc[6, 18])
    HR_soc[20, 6] = np.conj(HR_soc[6, 20])
    HR_soc[18, 7] = np.conj(HR_soc[7, 18])
    HR_soc[19, 7] = np.conj(HR_soc[7, 19])

    return HR_soc


def soc_d_ssppd(theta, phi, norb):
    """Build the d-orbital spin-orbit coupling matrix for the **SSPPD** pseudopotential type.

    Parameters
    ----------
    theta : float
        Polar angle (radians) of the quantisation axis.
    phi : float
        Azimuthal angle (radians) of the quantisation axis.
    norb : int
        Total number of orbitals per atom (spin-up block size).

    Returns
    -------
    np.ndarray, shape ``(2*norb, 2*norb)``, complex
        d-channel spin-orbit coupling matrix in the SSPPD orbital ordering.
    """
    HR_soc = np.zeros((2 * norb, 2 * norb), dtype=complex)

    sTheta, sPhi = np.sin(theta), np.sin(phi)
    cTheta, cPhi = np.cos(theta), np.cos(phi)

    s3 = cmath.sqrt(3.0)

    # Spin Up - Spin Up  part of the d-satets Hamiltonian
    HR_soc[8, 9] = -s3 * 0.5j * sTheta * sPhi
    HR_soc[8, 10] = s3 * 0.5j * sTheta * cPhi
    HR_soc[9, 10] = -0.5j * cTheta
    HR_soc[9, 11] = -0.5j * sTheta * sPhi
    HR_soc[9, 12] = 0.5j * sTheta * cPhi
    HR_soc[10, 11] = -0.5j * sTheta * cPhi
    HR_soc[10, 12] = -0.5j * sTheta * sPhi
    HR_soc[11, 12] = -1j * cTheta
    HR_soc[9, 8] = np.conj(HR_soc[8, 9])
    HR_soc[10, 8] = np.conj(HR_soc[8, 10])
    HR_soc[10, 9] = np.conj(HR_soc[9, 10])
    HR_soc[11, 9] = np.conj(HR_soc[9, 11])
    HR_soc[12, 9] = np.conj(HR_soc[9, 12])
    HR_soc[11, 10] = np.conj(HR_soc[10, 11])
    HR_soc[12, 10] = np.conj(HR_soc[10, 12])
    # Spin Down - Spin Down  part of the d-satets Hamiltonian
    HR_soc[21:26, 21:26] = -HR_soc[8:13, 8:13]
    # Spin Up - Spin Down  part of the d-satets Hamiltonian
    HR_soc[8, 22] = -s3 * 0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[8, 23] = -s3 * 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[9, 23] = 0.5j * sTheta
    HR_soc[9, 24] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[9, 25] = -0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[10, 24] = 0.5 * complex(sPhi, -cTheta * cPhi)
    HR_soc[10, 25] = -0.5 * complex(cPhi, cTheta * sPhi)
    HR_soc[11, 25] = 1j * sTheta
    HR_soc[9, 21] = -HR_soc[8, 22]
    HR_soc[10, 21] = -HR_soc[8, 23]
    HR_soc[10, 22] = -HR_soc[9, 23]
    HR_soc[11, 22] = -HR_soc[9, 24]
    HR_soc[12, 22] = -HR_soc[9, 25]
    HR_soc[11, 23] = -HR_soc[10, 24]
    HR_soc[12, 23] = -HR_soc[10, 25]
    HR_soc[12, 24] = -HR_soc[11, 25]
    # Spin Down - Spin Up  part of the d-satets Hamiltonian
    HR_soc[22, 8] = np.conj(HR_soc[8, 22])
    HR_soc[23, 8] = np.conj(HR_soc[8, 23])
    HR_soc[23, 9] = np.conj(HR_soc[9, 23])
    HR_soc[24, 9] = np.conj(HR_soc[9, 24])
    HR_soc[25, 9] = np.conj(HR_soc[9, 25])
    HR_soc[24, 10] = np.conj(HR_soc[10, 24])
    HR_soc[25, 10] = np.conj(HR_soc[10, 25])
    HR_soc[25, 11] = np.conj(HR_soc[11, 25])
    HR_soc[21, 9] = np.conj(HR_soc[9, 21])
    HR_soc[21, 10] = np.conj(HR_soc[10, 21])
    HR_soc[22, 10] = np.conj(HR_soc[10, 22])
    HR_soc[22, 11] = np.conj(HR_soc[11, 22])
    HR_soc[22, 12] = np.conj(HR_soc[12, 22])
    HR_soc[23, 11] = np.conj(HR_soc[11, 23])
    HR_soc[23, 12] = np.conj(HR_soc[12, 23])

    return HR_soc


################## END PSEUDOPOTENTIAL SSPPD ##############################