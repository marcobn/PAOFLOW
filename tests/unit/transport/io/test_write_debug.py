"""Unit tests for debug-only transport IO writers."""

import numpy as np
import pytest

from PAOFLOW.transport.io.write_debug import (
    write_overlap_files,
    write_projectability_files,
)


class DummyDataController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


@pytest.mark.unit
def test_write_projectability_files(tmp_path):
    """Projectability writer should emit a text file for each spin."""
    arry = {
        'Hk': np.zeros((1, 1, 1, 1), dtype=complex),
        'U': np.ones((1, 1, 1, 1), dtype=complex),
        'my_eigsmat': np.zeros((1, 1, 1)),
    }
    attr = {'nbnds': 1}

    write_projectability_files(str(tmp_path), DummyDataController(arry, attr))

    out_file = tmp_path / 'projectability.txt'
    assert out_file.exists()


@pytest.mark.unit
def test_write_overlap_files(tmp_path):
    """Overlap writer should emit kovp.txt when overlap transformation is enabled."""
    arry = {'Sk': np.zeros((1, 1, 1), dtype=complex)}

    write_overlap_files(
        str(tmp_path), DummyDataController(arry, {}), do_overlap_transformation=True
    )

    out_file = tmp_path / 'kovp.txt'
    assert out_file.exists()
