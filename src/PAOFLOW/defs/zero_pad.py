#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo, R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np


def zero_pad(aux, nk1, nk2, nk3, nfft1, nfft2, nfft3):
    """Zero-pad a 3-D frequency-domain array while preserving Hermitian symmetry.

    Parameters
    ----------
    aux : np.ndarray, shape ``(nk1, nk2, nk3)``, complex
        Input frequency-domain data (e.g. a real-space Hamiltonian slice
        after an inverse FFT).
    nk1 : int
        Current size of ``aux`` along axis 0.
    nk2 : int
        Current size of ``aux`` along axis 1.
    nk3 : int
        Current size of ``aux`` along axis 2.
    nfft1 : int
        Number of zeros to insert along axis 0 (``nfft1 = nk1p - nk1``).
    nfft2 : int
        Number of zeros to insert along axis 1.
    nfft3 : int
        Number of zeros to insert along axis 2.

    Returns
    -------
    np.ndarray, shape ``(nk1+nfft1, nk2+nfft2, nk3+nfft3)``, complex
        Zero-padded frequency-domain array.  A forward FFT of this array
        yields an interpolated real-space representation on a finer grid.

    Notes
    -----
    Zeros are inserted at the centre of the spectrum (around the Nyquist
    frequency) rather than at the end, so that the relationship
    :math:`A(-k) = A^*(k)` required for real-valued data is preserved.
    For an even-sized dimension of length :math:`N`, the Nyquist component
    at index :math:`N/2` is halved and its copy placed at both
    :math:`sk` and :math:`-sk` to prevent double-counting.

    This is used by :func:`do_double_grid` to perform Fourier interpolation
    of the PAO Hamiltonian onto a denser k-grid without introducing
    Gibbs artefacts.
    """
    # post-padding dimensions
    nk1p = nfft1 + nk1
    nk2p = nfft2 + nk2
    nk3p = nfft3 + nk3
    # halfway points
    sk1 = int((nk1 + 1) / 2)
    sk2 = int((nk2 + 1) / 2)
    sk3 = int((nk3 + 1) / 2)
    # parities (even <-> p==1)
    p1 = (nk1 & 1) ^ 1
    p2 = (nk2 & 1) ^ 1
    p3 = (nk3 & 1) ^ 1

    # accomodate nfft==0
    if nfft1 == 0:
        p1 = 0
    if nfft2 == 0:
        p2 = 0
    if nfft3 == 0:
        p3 = 0

    # first dimension
    auxp1 = np.zeros((nk1, nk2, nk3p), dtype=complex)
    auxp1[:, :, : sk3 + p3] = aux[:, :, : sk3 + p3]
    auxp1[:, :, nfft3 + sk3 :] = aux[:, :, sk3:]
    # second dimension
    auxp2 = np.zeros((nk1, nk2p, nk3p), dtype=complex)
    auxp2[:, : sk2 + p2, :] = auxp1[:, : sk2 + p2, :]
    auxp2[:, nfft2 + sk2 :, :] = auxp1[:, sk2:, :]
    # third dimension
    auxp3 = np.zeros((nk1p, nk2p, nk3p), dtype=complex)
    auxp3[: sk1 + p1, :, :] = auxp2[: sk1 + p1, :, :]
    auxp3[nfft1 + sk1 :, :, :] = auxp2[sk1:, :, :]

    # halve Nyquist axes
    if p1:
        auxp3[sk1, :, :] /= 2
        auxp3[-sk1, :, :] /= 2
    if p2:
        auxp3[:, sk2, :] /= 2
        auxp3[:, -sk2, :] /= 2
    if p3:
        auxp3[:, :, sk3] /= 2
        auxp3[:, :, -sk3] /= 2

    return auxp3


def zero_pad_float(aux, nk1, nk2, nk3, nfft1, nfft2, nfft3):
    """Zero-pad a 3-D real-valued frequency-domain array (deprecated).

    .. deprecated::
        Use :func:`zero_pad` instead.  This function uses a legacy padding
        algorithm that does not preserve Hermitian symmetry for even-sized
        grids and inserts zeros at incorrect positions for odd-sized grids.
        It also accepts only real-valued input.

    Parameters
    ----------
    aux : np.ndarray, shape ``(nk1, nk2, nk3)``, float
        Input frequency-domain data.
    nk1 : int
        Current size of ``aux`` along axis 0.
    nk2 : int
        Current size of ``aux`` along axis 1.
    nk3 : int
        Current size of ``aux`` along axis 2.
    nfft1 : int
        Number of zeros to pad along axis 0.
    nfft2 : int
        Number of zeros to pad along axis 1.
    nfft3 : int
        Number of zeros to pad along axis 2.

    Returns
    -------
    np.ndarray, shape ``(nk1+nfft1, nk2+nfft2, nk3+nfft3)``, float
        Zero-padded array.
    """
    # zero padding for FFT interpolation in 3D
    nk1p = nfft1 + nk1
    nk2p = nfft2 + nk2
    nk3p = nfft3 + nk3
    # first dimension
    auxp1 = np.zeros((nk1, nk2, nk3p), dtype=float)
    auxp1[:, :, : int(nk3 / 2)] = aux[:, :, : int(nk3 / 2)]
    auxp1[:, :, int(nfft3 + nk3 / 2) :] = aux[:, :, int(nk3 / 2) :]
    # second dimension
    auxp2 = np.zeros((nk1, nk2p, nk3p), dtype=float)
    auxp2[:, : int(nk2 / 2), :] = auxp1[:, : int(nk2 / 2), :]
    auxp2[:, int(nfft2 + nk2 / 2) :, :] = auxp1[:, int(nk2 / 2) :, :]
    # third dimension
    auxp3 = np.zeros((nk1p, nk2p, nk3p), dtype=float)
    auxp3[: int(nk1 / 2), :, :] = auxp2[: int(nk1 / 2), :, :]
    auxp3[int(nfft1 + nk1 / 2) :, :, :] = auxp2[int(nk1 / 2) :, :, :]

    return auxp3
