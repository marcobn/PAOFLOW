from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..DataController import DataController


def occupied_band_count(arry: dict, attr: dict) -> int:
    """Number of occupied bands of the spin channel used by the Wilson loops.

    Parameters
    ----------
    arry : dict
        DataController arrays; ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``) is read
        for collinear spin-polarized runs.
    attr : dict
        DataController attributes; ``nelec`` (electrons per cell) and ``dftSO`` (spinor bands:
        fully relativistic runs, ``adhoc_spin_orbit``, spinful models).

    Returns
    -------
    int
        ``nelec`` for spinor bands, ``nelec / 2`` for spin-degenerate bands (``nspin = 1``
        without spin-orbit coupling); for ``nspin = 2`` the number of bands of channel 0 below
        the Fermi level, which must be the same at every k-point of the coarse grid.

    Raises
    ------
    ValueError
        If the count is not an integer (a metal or a fractional filling), or if it is not between
        1 and the number of bands; pass the band indices with ``sub`` instead.

    Notes
    -----
    ``nelec`` counts electrons, as in the DFT readers and the model loaders. Before 16 Nov 2026
    the Wilson loops took the lowest ``nelec`` bands, twice the occupied subspace of a
    spin-degenerate system.
    """
    nawf, nspin = arry['HRs'].shape[0], arry['HRs'].shape[-1]
    if nspin == 2:
        from ..response.nonlinear_optics import channel_electron_counts

        count = channel_electron_counts(arry['HRs'])[0]
    else:
        spin_factor = 1.0 if attr.get('dftSO', False) else 2.0
        count = float(attr['nelec']) / spin_factor
    if abs(count - round(count)) > 1.0e-6:
        raise ValueError(
            f'nelec = {float(attr["nelec"]):g} gives {count:g} occupied bands, not an integer '
            '(a metal?): pass the band indices with sub=[...]'
        )
    nocc = round(count)
    if not 0 < nocc <= nawf:
        raise ValueError(f'{nocc} occupied bands of {nawf}: check nelec')
    return nocc


def orbital_fractional_positions(arry: dict, attr: dict) -> NDArray[np.float64]:
    """Centre of every orbital of ``HRs`` in crystal (fractional) coordinates.

    Parameters
    ----------
    arry : dict
        DataController arrays: ``HRs``, ``a_vectors`` (alat units), and ``basis`` (projections,
        centres in bohr) or ``tau`` with ``naw``.
    attr : dict
        DataController attributes: ``alat`` (bohr) and ``tau_units`` (``'alat'`` for the model
        loaders; absent, i.e. bohr, for DFT runs).

    Returns
    -------
    ndarray, shape ``(nawf, 3)``
        Fractional coordinates :math:`\\mathbf s_n` of the orbital centres,
        :math:`\\boldsymbol\\tau_n = \\sum_i s_{n,i}\\,\\mathbf a_i`.

    Raises
    ------
    ValueError
        If no centre can be assigned to every orbital.

    Notes
    -----
    The centres are those of :func:`PAOFLOW.hamiltonian.orbital_centres.orbital_centres_bohr`.
    """
    from ..hamiltonian.orbital_centres import orbital_centres_bohr

    centres_alat = orbital_centres_bohr(arry, attr) / float(attr['alat'])
    return centres_alat @ np.linalg.inv(np.asarray(arry['a_vectors'], dtype=float))


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

    Notes
    -----
    Reads ``berry_occupied``, ``berry_sub``, ``berry_contin``, ``berry_fname``,
    ``berry_kspace_method``, ``berry_nk1``, ``berry_nk2``, ``berry_kpath_funct``,
    ``berry_kradius``, ``berry_kcenter``, ``berry_kxlim``, ``berry_kylim``,
    ``berry_eigvals``, ``berry_closed``, ``berry_method``, ``nelec``, ``HRs``
    from DataController.

    Writes ``berry_phase`` (array or scalar) and ``berry_flux`` (square mode only)
    to DataController.
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
        ovr_ndim = occupied_band_count(arry, attr)
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
            # one loop: its eigenphases, i.e. the hybrid Wannier centres along the loop
            self.data_controller.write_berry_bands(fname, phase[None, :])

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

            self.data_controller.write_berry_bands(fname, phase, kpts)

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


def do_phase(data_controller: DataController) -> float | NDArray[np.float64]:
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

    ``method='berry'`` uses the eigenvectors of PAOFLOW's :math:`H(\\mathbf k)`, which is in the
    lattice gauge (periodic in :math:`\\mathbf k`). ``method='zak'`` uses the cell-centred
    coefficients :math:`\\tilde c_n(\\mathbf k) = e^{-2\\pi i\\mathbf k\\cdot\\mathbf s_n}c_n(\\mathbf k)`
    (:math:`\\mathbf s_n` the crystal coordinates of orbital :math:`n`, :math:`\\mathbf k` crystal) at
    every point, and closes the loop with
    :math:`\\tilde c(\\mathbf k_0 + \\mathbf G) = e^{-2\\pi i(\\mathbf k_0 + \\mathbf G)\\cdot\\mathbf s}c(\\mathbf k_0)`,
    :math:`\\mathbf G` the reciprocal vector spanned by the contour. On a loop across the zone
    its phase is :math:`2\\pi` times the Wannier centre along the loop (in units of the lattice
    vector), the lattice-gauge phase plus :math:`2\\pi\\sum_n\\overline{|c_n|^2}\\,\\mathbf G\\cdot
    \\mathbf s_n`.

    When ``berry_eigvals`` is True, the eigenvalues of the product matrix are
    returned instead of its determinant, yielding per-band (Wannier-centre)
    phases.

    Parameters
    ----------
    data_controller : DataController
        Provides ``berry_v_k``, ``berry_contour`` (crystal coordinates), ``berry_closed``,
        ``berry_method``, ``berry_eigvals``, ``berry_occupied``, ``berry_sub``, ``nelec``,
        ``dftSO``, ``HRs``; for ``method='zak'`` also ``alat``, ``a_vectors`` and the orbital
        centres (``basis`` or ``tau``/``naw``, see :func:`orbital_fractional_positions`).

    Returns
    -------
    float or ndarray
        If ``berry_eigvals`` is False: single scalar phase in radians.
        If ``berry_eigvals`` is True: 1-D array of per-band phases, sorted.

    Notes
    -----
    The occupied subspace is the lowest :func:`occupied_band_count` bands (``nelec`` divided by
    the spin factor). Until 16 Nov 2026 it was the lowest ``nelec`` bands, and ``method='zak'``
    replaced the closing link by the overlap of the first state with itself times
    :math:`e^{-i\\mathbf G\\cdot\\boldsymbol\\tau}`, with mixed units.
    """
    arry, attr = data_controller.data_dicts()

    v_kp = arry['berry_v_k']

    nkpts = v_kp.shape[0]

    berry_eigvals = attr['berry_eigvals']

    closed = attr['berry_closed']
    method = attr['berry_method'].lower()
    sub = arry['berry_sub']
    occupied = attr['berry_occupied']

    contour = arry['berry_contour']

    if np.allclose(contour[:, 0], contour[:, -1]):
        closed = False

    if method == 'zak':
        closed = True

    # assumes that occupancy does not change throughout the choosen path
    if occupied:
        occ_idx = np.arange(occupied_band_count(arry, attr))
    elif sub is not None:
        occ_idx = np.asarray(sub, dtype=int)
    else:
        occ_idx = np.arange(arry['HRs'].shape[0])
    dim = len(occ_idx)

    states = v_kp[:, :, :, 0][:, :, occ_idx]
    closing = states[0]
    if method == 'zak':
        if contour.shape[1] != nkpts:
            raise ValueError(f'the contour has {contour.shape[1]} points, the eigenvectors {nkpts}')
        frac = orbital_fractional_positions(arry, attr)
        kcrys = np.asarray(contour, dtype=float).T
        states = np.exp(-2.0j * np.pi * kcrys @ frac.T)[:, :, None] * states
        G = np.rint(kcrys[-1] - kcrys[0])
        closing = (
            np.exp(-2.0j * np.pi * (kcrys[0] + G) @ frac.T)[:, None] * v_kp[0, :, :, 0][:, occ_idx]
        )

    prd = np.eye(dim, dtype=complex)

    for ik in range(nkpts - 1):
        ovr = np.dot(states[ik].conj().T, states[ik + 1])
        Z_ovr, _, W_dag_ovr = la.svd(ovr)
        ovr = np.matmul(Z_ovr, W_dag_ovr)

        prd = np.dot(prd, ovr)

    if closed:
        ovr = np.dot(states[nkpts - 1].conj().T, closing)

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

    Notes
    -----
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
                         e^{-2\\pi i \\mathbf{k} \\cdot \\mathbf{R}}

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
    kdot = np.tensordot(
        arry['R'], -2.0j * np.pi * kq_aux, axes=([1], [0])
    )  # H(k), PAOFLOW's convention
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

    Notes
    -----
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
    nk, alat, ibrav = attr['berry_nk'], attr['alat'], attr.get('ibrav', 0)
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
