"""Crystal directions and SKEAF magnetic-field angles.

SKEAF/pyskeaf define the field by its polar angle (from +z) and azimuthal angle
(in the xy-plane from +x) in the Cartesian frame of the BXSF cell. A measured
direction must be translated with an explicit convention:

* ``'reciprocal'``: field normal to the (hkl) planes (sample facets, Laue indices);
* ``'real'``: field along the [hkl] atomic row.

The two coincide only for single-index directions of orthogonal axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_primitive_cell(scf_file: str | Path) -> NDArray[np.float64]:
    """Primitive vectors (rows) from the ``CELL_PARAMETERS`` card of a Quantum ESPRESSO input."""
    lines = Path(scf_file).read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip().upper().startswith('CELL_PARAMETERS'):
            return np.array(
                [[float(value) for value in lines[index + row].split()[:3]] for row in (1, 2, 3)]
            )
    raise ValueError(f'CELL_PARAMETERS not found in {scf_file}')


def conventional_cell(
    primitive_cell: NDArray[np.float64],
    conventional_from_primitive: list[list[float]] | None = None,
) -> NDArray[np.float64]:
    """Conventional vectors as integer combinations (rows) of the primitive vectors."""
    primitive_cell = np.asarray(primitive_cell, dtype=float)
    if not conventional_from_primitive:
        return primitive_cell
    return np.asarray(conventional_from_primitive, dtype=float) @ primitive_cell


def reciprocal_vectors(lattice_vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows ``b_i`` with ``b_i . a_j = 2 pi delta_ij``."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice_vectors, dtype=float)).T


def direction_to_field_angles(
    conventional: NDArray[np.float64],
    miller_indices: list[float],
    convention: str = 'reciprocal',
) -> tuple[float, float]:
    """Convert conventional-cell indices to SKEAF ``(azimuthal, polar)`` angles in degrees.

    Parameters
    ----------
    conventional : ndarray, shape ``(3, 3)``
        Conventional lattice vectors (rows).
    miller_indices : list[float]
        Direction indices; may be non-integer (e.g. a rotation angle between axes).
    convention : {'reciprocal', 'real'}
        Field normal to (hkl) or along [hkl].

    Returns
    -------
    tuple[float, float]
        Azimuthal angle in ``[0, 360)`` and polar angle in ``[0, 180]``.
    """
    basis = (
        np.asarray(conventional, float)
        if convention == 'real'
        else reciprocal_vectors(conventional)
    )
    direction = np.asarray(miller_indices, float) @ basis
    azimuthal = np.degrees(np.arctan2(direction[1], direction[0])) % 360.0
    polar = np.degrees(np.arccos(np.clip(direction[2] / np.linalg.norm(direction), -1.0, 1.0)))
    return float(azimuthal), float(polar)


def field_angles_to_direction(
    conventional: NDArray[np.float64],
    azimuthal: float,
    polar: float,
    convention: str = 'reciprocal',
) -> NDArray[np.float64]:
    """Inverse of :func:`direction_to_field_angles`, normalised to the smallest non-zero index = 1.

    Notes
    -----
    In the reciprocal convention the field direction is
    :math:`\\hat{n} \\propto h\\,\\mathbf{a}^* + k\\,\\mathbf{b}^* + l\\,\\mathbf{c}^*`, so the
    indices follow from :math:`h_i = \\hat{n}\\cdot\\mathbf{a}_i`. Use it to check index
    labels quoted by experimentalists against the measured rotation angle.
    """
    azimuthal_rad, polar_rad = np.radians(azimuthal), np.radians(polar)
    unit = np.array(
        [
            np.sin(polar_rad) * np.cos(azimuthal_rad),
            np.sin(polar_rad) * np.sin(azimuthal_rad),
            np.cos(polar_rad),
        ]
    )
    conventional = np.asarray(conventional, float)
    if convention == 'reciprocal':
        indices = conventional @ unit
    else:
        indices = np.linalg.solve(conventional.T, unit)
    nonzero = indices[np.abs(indices) > 1e-9]
    return indices / (np.min(np.abs(nonzero)) if len(nonzero) else 1.0)


def direction_report(
    conventional: NDArray[np.float64],
    directions: list[dict],
    convention: str = 'reciprocal',
    tolerance_deg: float = 0.05,
) -> str:
    """Text check of every measured direction in both conventions against its target angles.

    Parameters
    ----------
    conventional : ndarray, shape ``(3, 3)``
        Conventional lattice vectors.
    directions : list[dict]
        Entries with ``label``, ``hkl`` and optional target ``azimuthal`` / ``polar``.
    convention : {'reciprocal', 'real'}
        Convention selected for the analysis.
    tolerance_deg : float, optional
        Allowed mismatch before a direction is flagged.

    Returns
    -------
    str
        Report text; mismatches are marked ``MISMATCH``.
    """
    lines = [
        'conventional vectors (rows):',
        np.array2string(np.asarray(conventional), precision=6),
        f'lengths: {np.round(np.linalg.norm(conventional, axis=1), 6)}',
        '',
    ]
    for direction in directions:
        if direction.get('hkl') is None:
            continue
        azimuthal, polar = direction_to_field_angles(conventional, direction['hkl'], convention)
        target_az, target_pol = direction.get('azimuthal'), direction.get('polar')
        polar_ok = target_pol is None or abs(polar - target_pol) < tolerance_deg
        azimuthal_ok = (
            target_az is None
            or polar < 1e-6
            or abs(((azimuthal - target_az + 180) % 360) - 180) < tolerance_deg
        )
        verdict = (
            'OK' if polar_ok and azimuthal_ok else 'MISMATCH (convention / centring / indices?)'
        )
        lines.append(
            f'{direction["label"]} hkl={direction["hkl"]}: target az={target_az}, pol={target_pol}'
            f' -> {verdict}'
        )
        for name in ('reciprocal', 'real'):
            az, pol = direction_to_field_angles(conventional, direction['hkl'], name)
            tag = '  <== selected' if name == convention else ''
            lines.append(f'   {name:10s}: azimuthal {az:8.3f}, polar {pol:8.3f}{tag}')
    return '\n'.join(lines)
