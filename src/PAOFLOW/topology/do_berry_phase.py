import numpy as np
import scipy.linalg as la

from ..utils.constants import ANGSTROM_AU


def do_berry_phase(self):
    """Compute the Berry or Zak phase for a set of occupied (or selected) bands.

    Dispatches to one of four k-space sampling strategies based on
    ``attr['berry_kspace_method']``:

    ``'path'``
        Evaluate the phase along a 1-D high-symmetry path.  The result is a
        single scalar phase (or a vector of per-band phases when
        ``berry_eigvals`` is True) written to ``<fname>.dat``.

    ``'track'``
        Sweep a 1-D path as a function of a transverse k-coordinate supplied
        by ``berry_kpath_funct``.  Returns a 1-D array of phases (one per
        transverse point) and writes ``<fname>.dat``.

    ``'circle'``
        Integrate along a circular contour of radius ``berry_kradius``
        centered at ``berry_kcenter`` in the kx–ky plane.  Writes a single
        scalar to ``<fname>.dat``.

    ``'square'``
        Tile the k-plane with elementary plaquettes and compute a phase on
        each.  The resulting 2-D phase map (Berry flux) is written to
        ``<fname>.dat`` together with a corner k-grid file.

    Parameters
    ----------
    self : PAOFLOW
        The calling :class:`PAOFLOW.PAOFLOW` instance; provides access to
        ``data_controller`` and indirectly to all DataController arrays and
        attributes.

    Reads from DataController
    -------------------------
    ``berry_occupied``, ``berry_sub``, ``berry_contin``, ``berry_fname``,
    ``berry_kspace_method``, ``berry_nk1``, ``berry_nk2``, ``berry_kpath_funct``,
    ``berry_kradius``, ``berry_kcenter``, ``berry_kxlim``, ``berry_kylim``,
    ``berry_eigvals``, ``berry_closed``, ``berry_method``, ``nelec``, ``HRs``.

    Writes to DataController
    ------------------------
    ``berry_phase`` (array or scalar), ``berry_flux`` (square mode only).
    """
    import os

    arry, attr = self.data_controller.data_dicts()

    occupied = attr['berry_occupied']
    sub = arry['berry_sub']
    contin = attr['berry_contin']
    fname = attr['berry_fname']

    kxlim = arry['berry_kxlim']
    kylim = arry['berry_kylim']

    if occupied:
        ovr_ndim = int(attr['nelec'])
    elif sub != None:
        ovr_ndim = len(sub)
    else:
        ovr_ndim = arry['HRs'].shape[0]

    if attr['berry_kspace_method'] == 'path':
        # Calculate the bands
        do_berry_bands(self.data_controller)
        # Calculate the Berry/Zak phase
        phase = do_phase(self.data_controller)

        if not attr['berry_eigvals']:
            attr['berry_phase'] = phase
        else:
            arry['berry_phase'] = phase

        if not attr['berry_eigvals']:
            # Write 'berry_phase.dat'
            with open(os.path.join(attr['opath'], fname + '.dat'), 'w') as f:
                f.write(f'1D Berry phase (mod pi): phi = {phase / np.pi: 2.6f} \n')
        else:
            self.data_controller.write_bands(fname, phase)

    elif attr['berry_kspace_method'] == 'track':
        nk1, nk2 = attr['berry_nk1'], attr['berry_nk2']
        kpath_funct = attr['berry_kpath_funct']
        kpts = np.linspace(0.0, 1.0, nk2)

        if not attr['berry_eigvals']:
            phase = np.zeros(kpts.shape[0])
        else:
            phase = np.zeros((kpts.shape[0], ovr_ndim))

        for ik, k in enumerate(kpts):
            attr['berry_path'], arry['berry_high_sym_points'] = kpath_funct(k)

            do_berry_bands(self.data_controller)
            phase[ik] = do_phase(self.data_controller)

        if not attr['berry_eigvals']:
            if contin:
                phase = berry_phase_cont(phase, phase[0])
            phase -= phase[0]

            with open(os.path.join(attr['opath'], fname + '.dat'), 'w') as f:
                f.write('# k\tphi\n')
                for ik, k in enumerate(kpts):
                    f.write(f'{k: 2.6f}\t{phase[ik]: 2.6f}\n')
        else:
            if contin:
                phase = berry_eigvals_cont(phase, phase[0, :])

            self.data_controller.write_berry_bands(fname, phase)

        arry['berry_phase'] = phase

    elif attr['berry_kspace_method'] == 'circle':
        nk = attr['berry_nk']
        kradius = attr['berry_kradius']
        kcenter = arry['berry_kcenter']
        kz = kcenter[2]

        ang = np.linspace(0, 2 * np.pi, nk)

        path = []

        for ik, phi in enumerate(ang):
            kx, ky = np.cos(phi), np.sin(phi)
            kx *= kradius
            ky *= kradius

            kx += kcenter[0]
            ky += kcenter[1]

            path.append([kx, ky, kz])

        arry['berry_kq'] = np.array(path).T
        arry['berry_contour'] = np.copy(arry['berry_kq'])

        do_berry_bands(self.data_controller)
        phase = do_phase(self.data_controller)

        attr['berry_phase'] = phase

        # Write 'berry_phase.dat'
        with open(os.path.join(attr['opath'], fname + '.dat'), 'w') as f:
            f.write(f'k-space circle of radius {kradius} /Ang^-1\ncentered at k-point {kcenter}\n')
            f.write(f'Berry phase: {phase: 2.6f} \n')

    elif attr['berry_kspace_method'] == 'square':
        nk1 = attr['berry_nk1']
        nk2 = attr['berry_nk2']

        if nk1 % 2 != 0:
            nk1 += 1
        if nk2 % 2 != 0:
            nk2 += 1

        kx_points = np.linspace(kxlim[0], kxlim[1], nk1)
        ky_points = np.linspace(kylim[0], kylim[1], nk2)

        kpts = np.zeros((nk1, nk2, 3))
        kgrid_centers = np.zeros((nk1 - 1, nk2 - 1, 3))
        phases = np.zeros((nk1 - 1, nk2 - 1))

        for jk in range(nk2):
            for ik in range(nk1):
                kpts[ik, jk] = [kx_points[ik], ky_points[jk], 0]

        for jk in range(nk2 - 1):
            for ik in range(nk1 - 1):
                k00 = kpts[ik, jk]
                k10 = kpts[ik + 1, jk]
                k11 = kpts[ik + 1, jk + 1]
                k01 = kpts[ik, jk + 1]

                arry['berry_kq'] = np.array([k00, k10, k11, k01, k00]).T
                arry['berry_contour'] = np.copy(arry['berry_kq'])
                kgrid_centers[ik, jk] = np.array([k00, k10, k11, k01, k00]).mean(axis=0)

                do_berry_bands(self.data_controller)
                phases[ik, jk] = do_phase(self.data_controller)

        arry['berry_kgrid'] = kpts
        arry['berry_kgrid_centers'] = kgrid_centers

        for i in range(phases.shape[1]):
            if i == 0:
                clos = phases[0, 0]
            else:
                clos = phases[0, i - 1]
            phases[:, i] = berry_phase_cont(phases[:, i], clos)

        arry['berry_phase'] = phases
        attr['berry_flux'] = arry['berry_phase'].sum()

        with open(os.path.join(attr['opath'], fname + '.dat'), 'w') as f:
            f.write(
                f'# xlim: ({kpts[0, 0, 0]},{kpts[-1, 0, 0]}); ylim: ({kpts[0, 0, 1]},{kpts[0, -1, 1]})\n'
            )
            f.write(f'# phases with shape: ({nk1 - 1},{nk2 - 1})\n')
            f.write('# kx,ky are mesh centers\n')
            f.write('# kx\tky\tphi\n')
            for jk in range(nk2 - 1):
                for ik in range(nk1 - 1):
                    f.write(
                        f'{kgrid_centers[ik, jk][0]: 2.12f}\t{kgrid_centers[ik, jk][1]: 2.12f}\t{phases[ik, jk]: 2.12f}\n'
                    )

        with open(os.path.join(attr['opath'], fname + '_kgrid_corners.dat'), 'w') as f:
            f.write('# kx\tky\n')
            for jk in range(nk2):
                for ik in range(nk1):
                    f.write(f'{kpts[ik, jk][0]: 2.12f}\t{kpts[ik, jk][1]: 2.12f}\n')


def do_phase(data_controller):
    """Compute the Berry (or Zak) phase from pre-computed Bloch eigenvectors.

    Constructs the gauge-invariant phase as the argument of the determinant of
    the product of overlap matrices between consecutive k-points along the
    contour stored in ``arry['berry_kq']``:

    .. math::

        \\phi = -\\operatorname{Im} \\ln \\det
               \\prod_{k} \\langle u_k | u_{k+1} \\rangle

    Each overlap matrix is SVD-projected to the nearest unitary matrix before
    multiplication, which makes the result independent of the individual gauge
    choices at each k-point.

    For the Zak variant (``method='zak'``) the boundary link includes the
    reciprocal-lattice phase factor ``exp(-i G·r)`` that accounts for the
    non-periodicity of the Bloch factor across the full Brillouin zone.

    When ``berry_eigvals`` is True, the eigenvalues of the product matrix are
    returned instead of its determinant, yielding per-band (Wannier-centre)
    phases.

    Parameters
    ----------
    data_controller : DataController
        Provides ``berry_v_k``, ``berry_contour``, ``berry_closed``,
        ``berry_method``, ``berry_eigvals``, ``berry_occupied``, ``berry_sub``,
        ``nelec``, ``alat``, ``b_vectors``, ``tau``, ``naw``, ``HRs``.

    Returns
    -------
    float or ndarray
        If ``berry_eigvals`` is False: single scalar phase in radians.
        If ``berry_eigvals`` is True: 1-D array of per-band phases, sorted.
    """
    arry, attr = data_controller.data_dicts()

    v_kp = arry['berry_v_k']

    nkpts = v_kp.shape[0]

    berry_eigvals = attr['berry_eigvals']

    closed = attr['berry_closed']
    method = attr['berry_method']
    sub = arry['berry_sub']
    occupied = attr['berry_occupied']

    alat = attr['alat'] / ANGSTROM_AU
    b_vectors = arry['b_vectors'] * (1 / alat)

    tau = arry['tau']
    naw = arry['naw']

    contour = arry['berry_contour']

    if np.allclose(contour[:, 0], contour[:, -1]):
        closed = False

    method = method.lower()
    if method == 'berry':
        pass
    elif method == 'zak':
        closed = True

    # assumes that occupancy does not change throughout the choosen path

    if occupied:
        sub = None
        occ_idx = np.arange(0, attr['nelec'], 1, dtype=int)
        num_occupied = len(occ_idx)
        dim = num_occupied
    elif sub is not None:
        occ_idx = np.copy(sub)
        num_occupied = len(sub)
        dim = len(sub)
    else:
        occ_idx = np.arange(0, arry['HRs'].shape[0], 1, dtype=int)
        num_occupied = len(occ_idx)
        dim = num_occupied

    prd = np.eye(dim, dtype=complex)
    ovr = np.zeros([dim, dim], dtype=complex)

    for ik in range(nkpts - 1):
        jk = ik + 1

        left_states = v_kp[ik, :, :, 0]
        right_states = v_kp[jk, :, :, 0]

        if occupied or sub is not None:
            left_states_idx = occ_idx
            right_states_idx = occ_idx

            left_occ = np.zeros((left_states.shape[0], len(left_states_idx)), dtype=complex)
            right_occ = np.zeros((right_states.shape[0], len(right_states_idx)), dtype=complex)

            for idx, occ in enumerate(left_states_idx):
                left_occ[:, idx] = left_states[:, occ]
            for idx, occ in enumerate(right_states_idx):
                right_occ[:, idx] = right_states[:, occ]
            left_states = left_occ
            right_states = right_occ

        else:
            pass

        ovr = np.dot(left_states.conj().T, right_states)
        Z_ovr, sig_ovr, W_dag_ovr = la.svd(ovr)
        ovr = np.matmul(Z_ovr, W_dag_ovr)

        prd = np.dot(prd, ovr)

    if closed:
        ik = nkpts - 1
        jk = 0

        left_states = v_kp[ik, :, :, 0]
        right_states = v_kp[jk, :, :, 0]

        if method == 'zak':
            axis = contour[:, 1] - contour[:, 0]
            axis /= np.dot(axis.T, axis) ** 0.5

            G = np.dot(axis, b_vectors) * 2 * np.pi

            orb_sites = np.repeat(tau, naw, axis=0)

            phase = np.dot(orb_sites, G).reshape(-1, 1)

            left_states = right_states * np.exp(-1j * phase)

        if occupied or sub is not None:
            left_states_idx = occ_idx
            right_states_idx = occ_idx

            left_occ = np.zeros((left_states.shape[0], len(left_states_idx)), dtype=complex)
            right_occ = np.zeros((right_states.shape[0], len(right_states_idx)), dtype=complex)

            for idx, occ in enumerate(left_states_idx):
                left_occ[:, idx] = left_states[:, occ]
            for idx, occ in enumerate(right_states_idx):
                right_occ[:, idx] = right_states[:, occ]
            left_states = left_occ
            right_states = right_occ

        else:
            pass

        ovr = np.dot(left_states.conj().T, right_states)

        Z_ovr, _, W_dag_ovr = la.svd(ovr)
        ovr = np.matmul(Z_ovr, W_dag_ovr)

        prd = np.dot(prd, ovr)

    if not berry_eigvals:
        det = la.det(prd)
        ret = -1.0 * np.angle(det)

        return ret

    else:
        evals, _ = la.eig(prd)
        ret = -1.0 * np.angle(evals)
        ret = np.sort(ret)

        return ret


def bands_calc(data_controller):
    """Diagonalise H(k) on the Berry-phase k-path and return eigenvalues and eigenvectors.

    Scatters ``arry['berry_kq']`` across k-point pools, evaluates the
    Hamiltonian at each k via :func:`band_loop_H`, then solves the
    generalised eigenvalue problem with ``scipy.linalg.eigh``.

    Parameters
    ----------
    data_controller : DataController
        Must contain ``HRs`` (real-space Hamiltonian), ``berry_kq``
        (k-point array, shape ``(3, nkpts)``), ``R``, ``npool``, ``nspin``.

    Returns
    -------
    E_kp_aux : ndarray, shape (nkpts_local, nawf, nspin)
        Band eigenvalues along the local k-point slice.
    v_kp_aux : ndarray, shape (nkpts_local, nawf, nawf, nspin)
        Corresponding eigenvectors (columns are eigenstates).

    Side effects
    ------------
    Stores the local k-slice Hamiltonian in ``arry['berry_Hks']``.
    """
    from ..utils.communication import scatter_full

    arry, attr = data_controller.data_dicts()

    npool = attr['npool']
    nawf, _, _, _, _, nspin = arry['HRs'].shape

    kq_aux = scatter_full(arry['berry_kq'].T, npool).T

    Hks_aux = band_loop_H(data_controller, kq_aux)

    E_kp_aux = np.zeros((kq_aux.shape[1], nawf, nspin), dtype=float, order='C')
    v_kp_aux = np.zeros((kq_aux.shape[1], nawf, nawf, nspin), dtype=complex, order='C')

    for ispin in range(nspin):
        for ik in range(kq_aux.shape[1]):
            E_kp_aux[ik, :, ispin], v_kp_aux[ik, :, :, ispin] = la.eigh(
                Hks_aux[:, :, ik, ispin],
                b=(None),
                lower=False,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=True,
            )

    arry['berry_Hks'] = Hks_aux

    Hks_aux = None
    return E_kp_aux, v_kp_aux


def band_loop_H(data_controller, kq_aux):
    """Evaluate the PAO Hamiltonian H(k) at an arbitrary set of k-points.

    Performs the Fourier sum

    .. math::

        H(\\mathbf{k}) = \\sum_{\\mathbf{R}} H(\\mathbf{R})
                         e^{2\\pi i \\mathbf{k} \\cdot \\mathbf{R}}

    using a single ``numpy.tensordot`` contraction over real-space lattice
    vectors ``R``.

    Parameters
    ----------
    data_controller : DataController
        Must contain ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``) and
        ``R`` (shape ``(nR, 3)``, lattice vectors in crystal coordinates).
    kq_aux : ndarray, shape (3, nkpts_local)
        k-points for the local MPI slice, in crystal coordinates.

    Returns
    -------
    Haux : ndarray, shape (nawf, nawf, nkpts_local, nspin), complex
        H(k) at each requested k-point for each spin channel.
    """
    arry, _ = data_controller.data_dicts()

    nksize = kq_aux.shape[1]
    nawf, _, nk1, nk2, nk3, nspin = arry['HRs'].shape

    HRs = np.reshape(arry['HRs'], (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')
    kdot = np.tensordot(arry['R'], 2.0j * np.pi * kq_aux, axes=([1], [0]))
    np.exp(kdot, kdot)
    Haux = np.zeros((nawf, nawf, nksize, nspin), dtype=complex, order='C')

    for ispin in range(nspin):
        Haux[:, :, :, ispin] = np.tensordot(HRs[:, :, :, ispin], kdot, axes=([2], [0]))

    kdot = None
    return Haux


def do_berry_bands(data_controller):
    """Set up the real-space lattice grid and interpolate bands on the Berry-phase contour.

    Orchestrates the full band interpolation required before phase evaluation:

    1. Converts ``alat`` from Bohr to Ångström for the duration of the call.
    2. Calls :func:`get_R_grid_fft` to populate the real-space lattice-vector
       array ``R`` and associated weights.
    3. For ``'path'`` and ``'track'`` methods, builds the k-point mesh via
       :func:`berry_kpnts_interpolation_mesh`.
    4. Rotates k-points from crystal to Cartesian (reciprocal-space) coordinates
       using ``b_vectors``.
    5. Calls :func:`bands_calc` and stores eigenvalues and eigenvectors in
       ``arry['berry_E_k']`` and ``arry['berry_v_k']``.
    6. Restores ``alat`` to Bohr before returning.

    Parameters
    ----------
    data_controller : DataController
        Must contain ``HRs``, ``b_vectors``, ``alat``, ``berry_kspace_method``,
        and (for path/track modes) ``berry_path``, ``berry_high_sym_points``,
        ``berry_nk``, ``ibrav``, ``a_vectors``.

    Side effects
    ------------
    Populates ``arry['berry_E_k']``, ``arry['berry_v_k']``, ``arry['R']``,
    ``arry['R_wght']``, and (path/track) ``arry['berry_kq']``,
    ``arry['berry_contour']``.
    """
    from ..utils.constants import ANGSTROM_AU
    from ..utils.get_R_grid_fft import get_R_grid_fft

    arry, attr = data_controller.data_dicts()

    # Bohr to Angstrom
    attr['alat'] /= ANGSTROM_AU

    # --------------------------------------------
    # Compute bands on a selected path in the BZ
    # --------------------------------------------

    _, _, nk1, nk2, nk3, _ = arry['HRs'].shape

    # Define real space lattice vectors
    get_R_grid_fft(data_controller, nk1, nk2, nk3)

    # Define k-point mesh for bands interpolation
    if attr['berry_kspace_method'] == 'path' or attr['berry_kspace_method'] == 'track':
        berry_kpnts_interpolation_mesh(data_controller)

    nkpi = arry['berry_kq'].shape[1]
    for n in range(nkpi):
        arry['berry_kq'][:, n] = np.dot(arry['berry_kq'][:, n], arry['b_vectors'])

    # Compute the bands along the path in the IBZ
    arry['berry_E_k'], arry['berry_v_k'] = bands_calc(data_controller)

    # Angstrom to Bohr
    attr['alat'] *= ANGSTROM_AU


def berry_kpnts_interpolation_mesh(data_controller):
    """
    Get path between HSP
    Arguments:
        nk (int): total number of points in path

    Returns:
        kpoints : array of arrays kx,ky,kz
        numK    : Total no. of k-points
    """

    from ..spectrum.kpnts_interpolation_mesh import get_path

    arry, attr = data_controller.data_dicts()

    dk = 0.00001
    nk, alat, ibrav = attr['berry_nk'], attr['alat'], attr['ibrav']
    a_vectors, b_vectors = arry['a_vectors'], arry['b_vectors']
    band_path, high_sym_points = attr['berry_path'], arry['berry_high_sym_points']

    bp, hsp = (band_path, high_sym_points) if len(high_sym_points) != 0 else (None, None)

    points, _ = get_path(ibrav, alat, a_vectors, dk, b_vectors, bp, hsp)

    scaled_dk = dk * (points.shape[1] / nk)

    points, path_file = get_path(ibrav, alat, a_vectors, scaled_dk, b_vectors, bp, hsp)

    data_controller.write_kpnts_path('berry_phase_kpath_points.txt', path_file, points, b_vectors)

    arry['berry_kq'] = points
    arry['berry_contour'] = np.copy(arry['berry_kq'])


def no_2pi(x, ref):
    """Shift ``x`` by multiples of 2π until it is as close as possible to ``ref``.

    Parameters
    ----------
    x : float
        Phase value to adjust.
    ref : float
        Reference phase to match.

    Returns
    -------
    float
        ``x`` shifted by an integer multiple of 2π such that
        ``|ref - x| ≤ π``.
    """

    while abs(ref - x) > np.pi:
        if ref - x > np.pi:
            x += 2.0 * np.pi
        elif ref - x < -1.0 * np.pi:
            x -= 2.0 * np.pi

    return x


def berry_phase_cont(pha, clos):
    """
    Reads in 1d array of numbers *pha* and makes sure that they are
    continuous, i.e., that there are no jumps of 2pi. First number is
    made as close to *clos* as possible.
    """
    ret = np.copy(pha)

    # go through entire list and "iron out" 2pi jumps
    for i in range(len(ret)):
        # which number to compare to
        if i == 0:
            cmpr = clos
        else:
            cmpr = ret[i - 1]
        # make sure there are no 2pi jumps
        ret[i] = no_2pi(ret[i], cmpr)

    return ret


def berry_eigvals_cont(arr_pha, clos):
    """Reads in 2d array of phases *arr_pha* and makes sure that they
    are continuous along first index, i.e., that there are no jumps of
    2pi. First array of phasese is made as close to *clos* as
    possible."""

    ret = np.zeros_like(arr_pha)

    # go over all points
    for i in range(arr_pha.shape[0]):
        # which phases to compare to
        if i == 0:
            cmpr = clos
        else:
            cmpr = ret[i - 1, :]
        # remember which indices are still available to be matched
        avail = list(range(arr_pha.shape[1]))
        # go over all phases in cmpr[:]
        for j in range(cmpr.shape[0]):
            # minimal distance between pairs
            min_dist = 1.0e10
            # closest index
            best_k = None
            # go over each phase in arr_pha[i,:]
            for k in avail:
                cur_dist = np.abs(np.exp(1.0j * cmpr[j]) - np.exp(1.0j * arr_pha[i, k]))
                if cur_dist <= min_dist:
                    min_dist = cur_dist
                    best_k = k
            # remove this index from being possible pair later
            avail.pop(avail.index(best_k))
            # store phase in correct place
            ret[i, j] = arr_pha[i, best_k]
            # make sure there are no 2pi jumps
            ret[i, j] = no_2pi(ret[i, j], cmpr[j])

    return ret
