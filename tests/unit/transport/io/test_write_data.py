"""Unit tests for transport IO writers."""

import numpy as np
import pytest

from PAOFLOW.transport.io.write_data import (
    write_data,
    write_eigenchannels,
    write_operator_xml,
)


@pytest.mark.unit
def test_write_data_writes_1d_series(tmp_path):
    """write_data should create a text file with energy/value columns."""
    egrid = np.array([0.0, 1.0])
    data = np.array([2.0, 3.0])

    write_data(egrid, data, 'conductance', tmp_path, verbose=False)

    out_file = tmp_path / 'conductance.dat'
    assert out_file.exists()
    lines = out_file.read_text().splitlines()
    assert lines[0].startswith('# E (eV)')
    assert len(lines) == 3


@pytest.mark.unit
def test_write_eigenchannels_writes_npz(tmp_path):
    """Eigenchannel writer should store data and metadata in npz."""
    data = np.eye(2, dtype=complex)

    filepath = write_eigenchannels(
        data=data,
        ie=1,
        ik=2,
        vkpt=np.array([0.0, 0.0, 0.0]),
        transport_direction='z',
        output_dir=tmp_path,
        verbose=False,
    )

    assert filepath.exists()
    loaded = np.load(filepath)
    assert 'eigenchannels' in loaded
    assert loaded['ie'] == 1


@pytest.mark.unit
def test_write_operator_xml_validations(tmp_path):
    """write_operator_xml should validate required parameters."""
    with pytest.raises(ValueError):
        write_operator_xml(
            output_dir=tmp_path,
            filename='op.xml',
            operator_matrix=None,
            dimwann=1,
            dynamical=True,
            grid=None,
        )

    with pytest.raises(ValueError):
        write_operator_xml(
            output_dir=tmp_path,
            filename='op.xml',
            operator_matrix=None,
            dimwann=1,
            dynamical=False,
            ivr=None,
            vr=None,
        )


@pytest.mark.unit
def test_write_operator_xml_writes_basic_file(tmp_path):
    """Minimal operator XML should be written with required tags."""
    operator_matrix = np.zeros((1, 1, 1, 1), dtype=complex)
    ivr = np.array([[0, 0, 0]])

    write_operator_xml(
        output_dir=tmp_path,
        filename='op.xml',
        operator_matrix=operator_matrix,
        ivr=ivr,
        dimwann=1,
        dynamical=False,
        nomega=1,
        nrtot=1,
    )

    out_file = tmp_path / 'op.xml'
    assert out_file.exists()
    assert '<OPERATOR>' in out_file.read_text()
