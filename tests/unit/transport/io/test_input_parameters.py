"""Unit tests for input parameter validation models."""

import pytest

from PAOFLOW.transport.io.input_parameters import (
    ConductorData,
    FileNamesData,
    KPointGridSettings,
)


@pytest.mark.unit
def test_file_names_requires_datafile_c():
    """datafile_C is required and should not be empty."""
    with pytest.raises(ValueError):
        FileNamesData(datafile_C='')


@pytest.mark.unit
def test_kpoint_grid_settings_rejects_invalid_shifts():
    """Shift values must be 0 or 1 only."""
    with pytest.raises(ValueError):
        KPointGridSettings(s=[0, 2])


@pytest.mark.unit
def test_conductor_data_hamiltonian_tags_mapping():
    """hamiltonian_tags should map YAML keys into block tags."""
    data = ConductorData(
        filename='dummy.yaml',
        validate=False,
        datafile_C='C',
        datafile_L='L',
        datafile_R='R',
        dimC=1,
        dimL=1,
        dimR=1,
        transport_direction=1,
        H00_C={'rows': [0], 'cols': [1]},
    )

    tags = data.hamiltonian_tags

    assert tags['block_00C']['rows'] == [0]
    assert tags['block_00C']['cols'] == [1]
    assert tags['block_00C']['rows_sgm'] == [0]
    assert tags['block_00C']['cols_sgm'] == [1]
