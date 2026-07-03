"""Unit tests for parser_base helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.parsers.parser_base import parse_index_array


@pytest.mark.unit
def test_parse_index_array_ranges_and_repeats():
    """Ranges and repeats should expand into 0-based indices with xval placeholders."""
    indices = parse_index_array('1-3,5,2x', max_value=6, xval=-1)

    np.testing.assert_allclose(indices, [0, 1, 2, 4, -1, -1])


@pytest.mark.unit
def test_parse_index_array_all():
    """The 'all' keyword should return a full range."""
    indices = parse_index_array('all', max_value=4)

    np.testing.assert_allclose(indices, [0, 1, 2, 3])


@pytest.mark.unit
def test_parse_index_array_rejects_out_of_range():
    """Indices above max_value should raise a ValueError."""
    with pytest.raises(ValueError):
        parse_index_array('1-5', max_value=4)
