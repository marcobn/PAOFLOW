"""Real spherical harmonics of ``upf_gaussfit.build_basis_dict``: right count and right shapes for every l <= 3."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.projection.upf_gaussfit import build_basis_dict

EXPONENTS = [1.1, 0.35]
RADIAL = [1.0, 0.4]


def _real_harmonics(l: int, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[np.ndarray]:
    """Real solid harmonics r^l Y_lm without the common prefactor sqrt((2l+1)!!/4pi), in the builder's order."""
    if l == 0:
        return [np.ones_like(x)]
    if l == 1:
        return [z, y, x]
    if l == 2:
        return [(2 * z**2 - x**2 - y**2) / (2 * np.sqrt(3)), x * z, y * z, (x**2 - y**2) / 2, x * y]
    return [
        z * (2 * z**2 - 3 * x**2 - 3 * y**2) / (2 * np.sqrt(15)),
        x * (4 * z**2 - x**2 - y**2) / (2 * np.sqrt(10)),
        y * (4 * z**2 - x**2 - y**2) / (2 * np.sqrt(10)),
        z * (x**2 - y**2) / 2,
        x * y * z,
        x * (x**2 - 3 * y**2) / (2 * np.sqrt(6)),
        y * (3 * x**2 - y**2) / (2 * np.sqrt(6)),
    ]


@pytest.mark.parametrize('l', [0, 1, 2, 3])
def test_shell_components_are_the_real_harmonics(l):
    shell = build_basis_dict(['X'], [l], [np.array(RADIAL)], [EXPONENTS])[0]
    assert len(shell) == 2 * l + 1
    rng = np.random.default_rng(l)
    x, y, z = rng.normal(size=(3, 500))
    r2 = x * x + y * y + z * z
    radial = sum(c * np.exp(-e * r2) for c, e in zip(RADIAL, EXPONENTS))
    for component, harmonic in zip(shell, _real_harmonics(l, x, y, z)):
        value = sum(c * x**lx * y**ly * z**lz * np.exp(-e * r2) for lx, ly, lz, c, e in component)
        assert np.allclose(value, harmonic * radial, atol=1e-12)
