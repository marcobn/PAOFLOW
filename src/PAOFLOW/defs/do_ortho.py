import numpy as np


def do_ortho(Hks, Sks):
    """Apply the orthogonalising similarity transformation to the Hamiltonian.

    Parameters
    ----------
    Hks : np.ndarray, shape ``(nawf, nawf, nkpnts, nspin)``, complex
        Non-orthogonal PAO Hamiltonian in k-space.
    Sks : np.ndarray, shape ``(nawf, nawf, nkpnts)``, complex
        Overlap matrix :math:`S(\\mathbf{k})` in k-space.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nkpnts, nspin)``, complex
        Orthogonalised Hamiltonian
        :math:`\\tilde{H} = S^{-1/2} H S^{-1/2}`.

    Notes
    -----
    The transformation is applied k-point by k-point:

    .. math::

        \\tilde{H}(\\mathbf{k}) =
            S^{-1/2}(\\mathbf{k})\\, H(\\mathbf{k})\\, S^{-1/2}(\\mathbf{k})

    where :math:`S^{-1/2}` is the matrix inverse of the matrix square root of
    :math:`S`, computed via ``scipy.linalg.sqrtm``.
    """
    from scipy import linalg as spl

    # If orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(-1/2)*Hks*Sks^(-1/2)

    nawf, _, nkpnts, nspin = Hks.shape
    S2k = np.zeros((nawf, nawf, nkpnts), dtype=complex)
    for ik in range(nkpnts):
        S2k[:, :, ik] = npl.inv(spl.sqrtm(Sks[:, :, ik]))

    Hks_o = np.zeros((nawf, nawf, nkpnts, nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            Hks_o[:, :, ik, ispin] = np.dot(S2k[:, :, ik], Hks[:, :, ik, ispin]).dot(S2k[:, :, ik])

    return Hks_o


def do_orthogonalize(data_controller):
    """Orthogonalise the PAO Hamiltonian and transform back to real space.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``HRs`` (shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``),
        ``SRs`` (shape ``(nawf, nawf, nk1, nk2, nk3)``).
        Required attributes: ``nkpnts``, ``nawf``, ``nk1``, ``nk2``, ``nk3``,
        ``nspin``, ``use_cuda``.

    Returns
    -------
    None
        Modifies ``data_controller.data_arrays`` and
        ``data_controller.data_attributes`` in place:

        - ``HRs`` : np.ndarray — replaced with the orthogonalised
          real-space Hamiltonian, broadcast to all MPI ranks.
        - ``Hks`` : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3, nspin)``
          — orthogonalised k-space Hamiltonian (intermediate; may be removed
          by subsequent steps).
        - ``Sks`` : np.ndarray, shape ``(nawf, nawf, nk1, nk2, nk3)``
          — overlap matrix in k-space (intermediate).

        Deletes ``SRs``.  Sets attribute ``acbn0 = False``.

    Notes
    -----
    The workflow is: ``HRs``, ``SRs`` → FFT → ``Hks``, ``Sks``
    → :func:`do_ortho` → orthogonal ``Hks`` → inverse FFT → ``HRs``.
    When CUDA is available the FFTs are delegated to :func:`cuda_fftn` and
    :func:`cuda_ifftn`.
    """
    from scipy import fftpack as FFT

    arrays, attributes = data_controller.data_dicts()

    nktot = attributes['nkpnts']
    nawf, _, nk1, nk2, nk3, nspin = arrays['HRs'].shape

    if attributes['use_cuda']:
        from .cuda_fft import cuda_fftn

        arrays['Hks'] = cuda_fftn(np.moveaxis(arrays['HRs'], [0, 1], [3, 4]), axes=[0, 1, 2])
        arrays['Sks'] = cuda_fftn(np.moveaxis(arrays['SRs'], [0, 1], [3, 4]), axes=[0, 1, 2])
        arrays['Hks'] = np.reshape(
            np.moveaxis(arrays['Hks'], [3, 4], [0, 1]), (nawf, nawf, nktot, nspin), order='C'
        )
        arrays['Sks'] = np.reshape(
            np.moveaxis(arrays['Sks'], [3, 4], [0, 1]), (nawf, nawf, nktot), order='C'
        )
    else:
        arrays['Hks'] = FFT.fftn(arrays['HRs'], axes=[2, 3, 4])
        arrays['Sks'] = FFT.fftn(arrays['SRs'], axes=[2, 3, 4])
        arrays['Hks'] = np.reshape(arrays['Hks'], (nawf, nawf, nktot, nspin), order='C')
        arrays['Sks'] = np.reshape(arrays['Sks'], (nawf, nawf, nktot), order='C')

    arrays['Hks'] = do_ortho(arrays['Hks'], arrays['Sks'])
    arrays['Hks'] = np.reshape(arrays['Hks'], (nawf, nawf, nk1, nk2, nk3, nspin), order='C')
    arrays['Sks'] = np.reshape(arrays['Sks'], (nawf, nawf, nk1, nk2, nk3), order='C')
    if attributes['use_cuda']:
        from .cuda_fft import cuda_ifftn

        arrays['HRs'] = np.moveaxis(
            cuda_ifftn(np.moveaxis(arrays['Hks'], [0, 1], [3, 4]), axes=[0, 1, 2]), [3, 4], [0, 1]
        )
    else:
        arrays['HRs'] = FFT.ifftn(arrays['Hks'], axes=[2, 3, 4])

    data_controller.broadcast_single_array('HRs')

    attributes['acbn0'] = False
    del arrays['SRs']
