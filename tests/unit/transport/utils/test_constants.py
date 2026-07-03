import pytest

from PAOFLOW.transport.utils import constants


@pytest.mark.unit
def test_constants_inverse_pairs():
    assert constants.kb_au == pytest.approx(1.0 / constants.au_kb)
    assert constants.evtory == pytest.approx(1.0 / constants.rydtoev)


@pytest.mark.unit
def test_constants_aliases():
    assert constants.uma_au == constants.scmass
    assert constants.terahertz == constants.au_terahertz
