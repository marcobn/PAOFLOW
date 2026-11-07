"""The output readers on the shared parser return what the original readers returned."""

from __future__ import annotations

import glob
import os
import re

import legacy_read_pao_output as legacy
import numpy as np
import pytest

from PAOFLOW.inputs import read_pao_output as reader

EXAMPLES = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples')


def _files(pattern):
    return sorted(glob.glob(os.path.join(EXAMPLES, '**', 'Reference', pattern), recursive=True))


@pytest.mark.parametrize(
    'path', _files('*dos*_0.dat') + _files('*Ef_*.dat') + _files('eps*_xx.dat')
)
def test_two_column_files(path):
    old = legacy.read_dos_PAO(path)
    new = reader.read_dos_PAO(path)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('path', _files('bands_0.dat'))
def test_band_files(path):
    np.testing.assert_array_equal(legacy.read_bands_PAO(path), reader.read_bands_PAO(path))


TRANSPORT = re.compile(r'^(sigma|Seebeck|kappa|PF)(gauss|m-p|mv|fd)?_\d+\.dat$')


@pytest.mark.parametrize(
    'path', [p for p in _files('*_0.dat') if TRANSPORT.match(os.path.basename(p))]
)
def test_transport_files(path):
    for a, b in zip(legacy.read_transport_PAO(path), reader.read_transport_PAO(path)):
        np.testing.assert_array_equal(a, b)


def test_transport_with_several_temperatures(tmp_path):
    rows = []
    for temperature in (300.0, 600.0):
        for energy in (-1.0, 0.0, 1.0):
            rows.append(
                '%8.2f % .5f %s\n'
                % (
                    temperature,
                    energy,
                    ' '.join('%9.5e' % (temperature * energy + c) for c in range(6)),
                )
            )
    path = tmp_path / 'sigma_0.dat'
    path.write_text(''.join(rows))
    for a, b in zip(legacy.read_transport_PAO(str(path)), reader.read_transport_PAO(str(path))):
        np.testing.assert_array_equal(a, b)


def test_header_lines_and_clear_errors(tmp_path):
    with_header = tmp_path / 'AHC_xy.dat'
    with_header.write_text('# E (eV)   AHC (S/cm)\n0.0 1.5\n0.1 2.5\n')
    energies, values = reader.read_dos_PAO(str(with_header))
    assert energies.tolist() == [0.0, 0.1] and values.tolist() == [1.5, 2.5]
    hall = tmp_path / 'hall_trace_gauss_0.dat'
    hall.write_text('  300.00 -1.00000  1.0e-09 \n')
    with pytest.raises(ValueError, match='hall_trace'):
        reader.read_transport_PAO(str(hall))
    sparse = tmp_path / 'kpath_points.txt'
    sparse.write_text('0.0  G\n0.5  X\n')
    with pytest.raises(ValueError, match='sparse'):
        reader.read_band_path_PAO(str(sparse))
