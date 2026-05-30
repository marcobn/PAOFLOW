import numpy as np
from mpi4py import MPI
from scipy import linalg as spl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def bands_calc(data_controller):
    """Compute band eigenvalues and eigenvectors scattered over MPI pools.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``kq`` (shape ``(3, nkpi)``), ``R`` (shape ``(nrtot, 3)``).
        Required attribute: ``npool``.

    Returns
    -------
    E_kp_aux : np.ndarray, shape ``(nkpi_local, nawf, nspin)``
        Eigenvalues on the local k-point subset.  Complex if ``kq`` is
        complex (e.g. for complex band structures), float otherwise.
    v_kp_aux : np.ndarray, shape ``(nkpi_local, nawf, nawf, nspin)``, complex
        Eigenvectors on the local k-point subset.

    Notes
    -----
    Each MPI rank operates on a local slice of the k-path obtained via
    :func:`scatter_full`.  Eigenvalues are computed with
    ``scipy.linalg.eigh`` (real k) or ``scipy.linalg.eig`` (complex k).
    """
    from ..utils.communication import scatter_full

    arrays, attributes = data_controller.data_dicts()

    npool = attributes['npool']

    kq_aux = scatter_full(arrays['kq'].T, npool).T

    nawf, _, _, _, _, nspin = arrays['HRs'].shape
    Hks_aux = band_loop_H(data_controller, kq_aux)

    if np.iscomplex(arrays['kq']).any():
        E_kp_aux = np.zeros((kq_aux.shape[1], nawf, nspin), dtype=complex, order='C')
    else:
        E_kp_aux = np.zeros((kq_aux.shape[1], nawf, nspin), dtype=float, order='C')
    v_kp_aux = np.zeros((kq_aux.shape[1], nawf, nawf, nspin), dtype=complex, order='C')

    for ispin in range(nspin):
        for ik in range(kq_aux.shape[1]):
            if np.iscomplex(arrays['kq']).any():
                E_kp_aux[ik, :, ispin], v_kp_aux[ik, :, :, ispin] = spl.eig(
                    Hks_aux[:, :, ik, ispin],
                    b=(None),
                    overwrite_a=True,
                    overwrite_b=True,
                    check_finite=True,
                )
            else:
                E_kp_aux[ik, :, ispin], v_kp_aux[ik, :, :, ispin] = spl.eigh(
                    Hks_aux[:, :, ik, ispin],
                    b=(None),
                    lower=False,
                    overwrite_a=True,
                    overwrite_b=True,
                    check_finite=True,
                )

    Hks_aux = None
    return E_kp_aux, v_kp_aux


def band_loop_H(data_controller, kq_aux):
    """Evaluate the Bloch Hamiltonian :math:`H(\\mathbf{k})` on a set of k-points.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``R`` (shape ``(nrtot, 3)``).
    kq_aux : np.ndarray, shape ``(3, nksize)``
        Local k-points (Cartesian, units of :math:`2\\pi/a`) for which
        :math:`H(\\mathbf{k})` is evaluated.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nksize, nspin)``, complex
        :math:`H(\\mathbf{k})` evaluated at each k-point in ``kq_aux``.

    Notes
    -----
    Uses the Fourier sum

    .. math::

        H(\\mathbf{k}) = \\sum_{\\mathbf{R}} H(\\mathbf{R})\\,
            e^{2\\pi i \\mathbf{k} \\cdot \\mathbf{R}}

    implemented as a tensor contraction between the flattened ``HRs`` and
    the phase factors :math:`e^{2\\pi i \\mathbf{k}\\cdot\\mathbf{R}}`.
    """
    arrays, _ = data_controller.data_dicts()

    nksize = kq_aux.shape[1]
    nawf, _, nk1, nk2, nk3, nspin = arrays['HRs'].shape

    HRs = np.reshape(arrays['HRs'], (nawf, nawf, nk1 * nk2 * nk3, nspin), order='C')
    kdot = np.tensordot(arrays['R'], 2.0j * np.pi * kq_aux, axes=([1], [0]))
    np.exp(kdot, kdot)
    Haux = np.zeros((nawf, nawf, nksize, nspin), dtype=complex, order='C')

    for ispin in range(nspin):
        Haux[..., ispin] = np.tensordot(HRs[..., ispin], kdot, axes=([2], [0]))

    return Haux


def do_bands(data_controller):
    """Compute the electronic band structure along a k-path via Fourier interpolation.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``b_vectors`` (shape ``(3, 3)``).
        Required attributes: ``alat``, ``npool``.  Either ``ibrav`` or a
        pre-loaded ``kq`` array must be present.

    Returns
    -------
    None
        Adds or updates the following entries in ``data_controller.data_arrays``:

        - ``E_k`` : np.ndarray, shape ``(nkpi, nawf, nspin)`` — band eigenvalues
          (in eV) along the requested k-path.
        - ``v_k`` : np.ndarray, shape ``(nkpi, nawf, nawf, nspin)``, complex —
          corresponding eigenvectors.

    Notes
    -----
    The real-space grid ``R`` is built by :func:`get_R_grid_fft` if not
    already present, and the k-path is generated by
    :func:`kpnts_interpolation_mesh` when ``ibrav`` is provided.  k-vectors
    supplied in reduced crystal coordinates are rotated to Cartesian by
    multiplication with ``b_vectors`` before the Fourier sum.  If ``kq``
    is complex (e.g. for complex band structures), eigenvalues are sorted
    in ascending order of the real part.
    """
    from mpi4py import MPI

    from ..utils.constants import ANGSTROM_AU
    from ..utils.get_R_grid_fft import get_R_grid_fft
    from .kpnts_interpolation_mesh import kpnts_interpolation_mesh

    rank = MPI.COMM_WORLD.Get_rank()

    arrays, attributes = data_controller.data_dicts()

    # Bohr to Angstrom
    attributes['alat'] /= ANGSTROM_AU

    #### What about ONEDIM?
    attributes['onedim'] = False
    if not attributes['onedim']:
        # --------------------------------------------
        # Compute bands on a selected path in the BZ
        # --------------------------------------------

        _, _, nk1, nk2, nk3, nspin = arrays['HRs'].shape

        # Define real space lattice vectors
        get_R_grid_fft(data_controller, nk1, nk2, nk3)

        # Define k-point mesh for bands interpolation
        if 'ibrav' in attributes:
            kpnts_interpolation_mesh(data_controller)
        if 'kq' not in arrays:
            print('need external kq for bands')

        nkpi = arrays['kq'].shape[1]
        for n in range(nkpi):
            arrays['kq'][:, n] = np.dot(arrays['kq'][:, n], arrays['b_vectors'])

        # Compute the bands along the path in the IBZ
        arrays['E_k'], arrays['v_k'] = bands_calc(data_controller)
        if np.iscomplex(arrays['kq']).any():
            for ispin in range(nspin):
                sorted_E_k = []
                for k in range(arrays['kq'].shape[1]):
                    E = np.array([item for sublist in arrays['E_k'][k] for item in sublist])
                    idx = np.argsort(E)
                    sorted_E_k.append(E[idx])
                arrays['E_k'][:, :, ispin] = np.array(sorted_E_k)
            arrays['v_k'] = arrays['v_k'][:, idx]

    else:
        if rank == 0:
            print('OneDim bands deprecated. Use band_path and high_sym_points')

    # Angstrom to Bohr
    attributes['alat'] *= ANGSTROM_AU
