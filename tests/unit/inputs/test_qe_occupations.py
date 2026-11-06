"""Occupations of a pw.x run read from data-file-schema.xml (type and width in Ry)."""

import xml.etree.ElementTree as ET

import pytest

from PAOFLOW.inputs.read_QE_xml import qe_occupations


@pytest.mark.unit
@pytest.mark.parametrize(
    'xml, expected',
    [
        (
            '<output><band_structure><smearing degauss="1.0E-02">mp</smearing></band_structure></output>',
            ('mp', 0.02),
        ),
        (
            '<output><band_structure><smearing degauss="8.867E-03">fd</smearing></band_structure></output>',
            ('fd', 0.017734),
        ),
        ('<output><band_structure><nbnd>8</nbnd></band_structure></output>', (None, None)),
    ],
)
def test_qe_occupations(xml, expected):
    smearing, degauss = qe_occupations(ET.fromstring(xml))
    assert smearing == expected[0]
    if expected[1] is None:
        assert degauss is None
    else:
        assert degauss == pytest.approx(expected[1])
