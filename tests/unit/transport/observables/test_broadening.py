import numpy as np
import pytest

from PAOFLOW.transport.calculators.broadening import compute_broadening_matrix


@pytest.mark.unit
def test_compute_broadening_matrix_matches_existing_definition():
    sigma = np.array([[1.0 + 0.2j, 2.0 - 0.5j], [-1.0 + 0.4j, 3.0 - 0.7j]])

    gamma = compute_broadening_matrix(sigma)

    np.testing.assert_allclose(gamma, 1j * (sigma - sigma.conj().T))
