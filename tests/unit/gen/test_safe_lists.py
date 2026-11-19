"""The eval-free list expressions of the generator (laser wavelengths)."""

from __future__ import annotations

import pytest

from PAOFLOW.gen.safe_lists import evaluate_number_list


@pytest.mark.parametrize(
    'expression, expected',
    [
        ('[n for n in range(450, 470, 5)]', [450.0, 455.0, 460.0, 465.0]),
        ('range(3)', [0.0, 1.0, 2.0]),
        ('[488, 514.5, 2 * 266]', [488.0, 514.5, 532.0]),
        ('(1, -2)', [1.0, -2.0]),
        ('[1240 / e for e in [2.0, 2.5]]', [620.0, 496.0]),
        ('[n for n in range(10) if n % 3 == 0]', [0.0, 3.0, 6.0, 9.0]),
        ('532', [532.0]),
    ],
)
def test_allowed_expressions(expression, expected):
    assert evaluate_number_list(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    'expression',
    [
        '__import__("os").system("echo hi")',
        '().__class__.__base__.__subclasses__()',
        'open("x")',
        '[x for x in range(3) for y in range(2)]',
        'range(10**9)',
        'lambda: 1',
        '[n.real for n in range(3)]',
        'True',
        'range(1.5)',
        '[1, 2',
    ],
)
def test_rejected_expressions(expression):
    with pytest.raises(ValueError):
        evaluate_number_list(expression)
