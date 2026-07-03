"""Unit tests for YAML input loading helpers."""

import textwrap

import pytest

from PAOFLOW.transport.io.get_input_params import (
    get_input_from_yaml,
    load_conductor_data_from_yaml,
    load_current_data_from_yaml,
)


@pytest.mark.unit
def test_get_input_from_yaml_reads_dict(tmp_path):
    """YAML file contents should be returned as a dictionary."""
    yaml_text = """
    input_conductor:
      dimC: 1
    """
    yaml_path = tmp_path / 'input.yaml'
    yaml_path.write_text(textwrap.dedent(yaml_text))

    data = get_input_from_yaml(str(yaml_path))

    assert data['input_conductor']['dimC'] == 1


@pytest.mark.unit
def test_load_conductor_data_from_yaml_minimal(tmp_path):
    """Conductor input should validate and be returned as a ConductorData object."""
    yaml_text = """
    input_conductor:
      datafile_C: C
      datafile_L: L
      datafile_R: R
      dimC: 2
      dimL: 1
      dimR: 1
      transport_direction: 3
    """
    yaml_path = tmp_path / 'conductor.yaml'
    yaml_path.write_text(textwrap.dedent(yaml_text))

    data = load_conductor_data_from_yaml(str(yaml_path))

    assert data.dimC == 2
    assert data.file_names.datafile_C == 'C'


@pytest.mark.unit
def test_load_current_data_from_yaml_minimal(tmp_path):
    """Current inputs should be validated and returned as a dictionary."""
    yaml_text = """
    input:
      filein: trans.dat
      fileout: out.dat
      Vmin: 0.0
      Vmax: 1.0
      nV: 5
      sigma: 0.1
      mu_L: 0.5
      mu_R: -0.5
    """
    yaml_path = tmp_path / 'current.yaml'
    yaml_path.write_text(textwrap.dedent(yaml_text))

    data = load_current_data_from_yaml(str(yaml_path))

    assert data['filein'] == 'trans.dat'
    assert data['nV'] == 5
