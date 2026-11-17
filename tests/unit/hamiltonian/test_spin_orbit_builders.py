"""The hand-written L.S kernels of ``adhoc_spin_orbit`` are Hermitian and have the L.S spectrum.

Until 16 Nov 2026 ``soc_d_spd``, ``soc_d_sspd`` and ``soc_d_ssppd`` set only the upper triangle of part
of the d block: PAOFLOW's upper-triangle eigensolvers saw the right matrix, every routine that uses the
whole matrix (numpy's default eigensolvers, Green's functions, products with H) a non-Hermitian one.
"""

from __future__ import annotations

import numpy as np
import pytest

import PAOFLOW.hamiltonian.do_spin_orbit as soc

pytestmark = pytest.mark.unit

ANGLES = [(0.0, 0.0), (0.4, 1.3), (1.2, 4.0), (np.pi / 2, np.pi / 3), (2.9, 5.9)]
D_LAYOUTS = {'spd': (9, 4), 'sspd': (10, 5), 'ssppd': (13, 8), 'spds': (10, 4)}
P_LAYOUTS = {'ps': 4, 'sp': 4, 'spd': 9, 'sspd': 10, 'spds': 10, 'ssppd': 13}


def _d_block(matrix: np.ndarray, start: int, norb: int) -> np.ndarray:
    index = np.r_[start : start + 5, norb + start : norb + start + 5]
    return matrix[np.ix_(index, index)]


@pytest.mark.parametrize('layout', sorted(D_LAYOUTS))
@pytest.mark.parametrize('theta, phi', ANGLES)
def test_d_kernels_are_hermitian_with_the_l_dot_s_spectrum(layout, theta, phi):
    norb, start = D_LAYOUTS[layout]
    kernel = getattr(soc, f'soc_d_{layout}')(theta, phi, norb)
    np.testing.assert_allclose(kernel, kernel.conj().T, atol=1e-15)
    eigenvalues = np.linalg.eigvalsh(kernel)
    assert np.sum(np.abs(eigenvalues - 1.0) < 1e-12) == 6  # j = 5/2: L.S = l / 2 = 1
    assert np.sum(np.abs(eigenvalues + 1.5) < 1e-12) == 4  # j = 3/2: L.S = -(l + 1) / 2
    assert np.sum(np.abs(eigenvalues) < 1e-12) == 2 * norb - 10  # nothing outside the d shell
    # the same d block as soc_d_spds, written independently, and as the generic builder
    reference = _d_block(soc.soc_d_spds(theta, phi, 10), 4, 10)
    np.testing.assert_allclose(_d_block(kernel, start, norb), reference, atol=1e-15)
    np.testing.assert_allclose(reference, soc._shell_soc_block(theta, phi, 2), atol=1e-15)


@pytest.mark.parametrize('layout', sorted(P_LAYOUTS))
@pytest.mark.parametrize('theta, phi', ANGLES)
def test_p_kernels_are_hermitian_with_the_l_dot_s_spectrum(layout, theta, phi):
    norb = P_LAYOUTS[layout]
    kernel = getattr(soc, f'soc_p_{layout}')(theta, phi, norb)
    np.testing.assert_allclose(kernel, kernel.conj().T, atol=1e-15)
    eigenvalues = np.linalg.eigvalsh(kernel)
    assert np.sum(np.abs(eigenvalues - 0.5) < 1e-12) == 4
    assert np.sum(np.abs(eigenvalues + 1.0) < 1e-12) == 2
