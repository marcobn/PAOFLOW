"""Unit tests for smearing function evaluation."""

import numpy as np
import pytest

from PAOFLOW.transport.smearing.smearing_base import smearing_func


@pytest.mark.unit
def test_smearing_func_lorentzian_value():
    """Lorentzian smearing should match the analytical expression."""
    value = smearing_func(0.0, 'lorentzian')

    assert value == pytest.approx(1.0 / np.pi)


@pytest.mark.unit
def test_smearing_func_gaussian_value():
    """Gaussian smearing should match the analytical expression at x=0."""
    value = smearing_func(0.0, 'gaussian')

    assert value == pytest.approx(1.0 / np.sqrt(np.pi))


@pytest.mark.unit
def test_smearing_func_aliases():
    """Aliases for smearing types should map to the same expression."""
    assert smearing_func(0.1, 'fd') == pytest.approx(smearing_func(0.1, 'fermi-dirac'))
    assert smearing_func(0.2, 'mp') == pytest.approx(smearing_func(0.2, 'methfessel-paxton'))
    assert smearing_func(0.3, 'mv') == pytest.approx(smearing_func(0.3, 'marzari-vanderbilt'))


@pytest.mark.unit
def test_smearing_func_invalid_type():
    """Unknown smearing types should raise a ValueError."""
    with pytest.raises(ValueError):
        smearing_func(0.0, 'unknown')
