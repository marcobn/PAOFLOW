import numpy as np
import pytest

from PAOFLOW.transport.calculators.current import locate


@pytest.mark.unit
def test_locate_ascending():
    xx = np.array([0.0, 1.0, 2.0, 3.0])

    assert locate(xx, 1.5) == 1


@pytest.mark.unit
def test_locate_descending():
    xx = np.array([3.0, 2.0, 1.0, 0.0])

    assert locate(xx, 1.5) == 1


@pytest.mark.unit
def test_locate_out_of_bounds():
    xx = np.array([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        locate(xx, -0.5)

    with pytest.raises(ValueError):
        locate(xx, 3.0)
