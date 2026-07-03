"""Geometry helpers — reciprocal-lattice volumes and field-direction setup.

Mirrors the Fortran utility code in ``skeaf_v1p3p0_r149.F90``:

* ``unit_cell_volume`` reproduces the ``unitcellvolume = |det(plr)|``
  computation at line 1056.
* ``set_field_angle`` reproduces the subroutine ``psetangle`` (lines
  3071–3129), which converts an ``hvd`` selector ('a', 'b', 'c', 'n', 'r')
  into spherical angles (theta, phi) of the magnetic field axis.

Conventions
-----------
* All angles are in **radians** (the Fortran stores degrees in user input
  and converts to radians internally; we standardise on radians).
* ``recip`` is a (3, 3) array with ``recip[i]`` the i-th reciprocal lattice
  vector in Cartesian Å^-1 *with* the 2π factor (i.e. the ``plr*`` form).
* The H-vector spherical convention matches Fortran ``psetangle``:

      theta = atan2(v_y, v_x)             # azimuth in xy-plane
      phi   = acos(v_z / |v|)             # polar angle from +z
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


HvdLiteral = Literal['a', 'b', 'c', 'n', 'r']


def unit_cell_volume(recip: np.ndarray) -> float:
    """Return the BZ (reciprocal unit cell) volume ``|det(recip)|``.

    Parameters
    ----------
    recip : ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows (Å^-1, with 2π factor).
    """
    if recip.shape != (3, 3):
        raise ValueError(f'recip must be (3, 3), got {recip.shape}')
    return float(abs(np.linalg.det(recip)))


def set_field_angle(
    recip: np.ndarray,
    hvd: HvdLiteral,
    theta: float = 0.0,
    phi: float = 0.0,
) -> tuple[float, float]:
    """Compute the (theta, phi) of the magnetic-field axis from an ``hvd`` flag.

    Reproduces ``psetangle`` (skeaf_v1p3p0_r149.F90 lines 3071–3129).

    Parameters
    ----------
    recip : ndarray, shape (3, 3)
        Reciprocal lattice vectors as rows (Å^-1, with 2π factor).  Only used
        for ``hvd in {'a', 'b', 'c'}``.
    hvd : {'a', 'b', 'c', 'n', 'r'}
        Field-direction selector:

        * ``'a'``, ``'b'``, ``'c'`` — align field with reciprocal vector 0, 1
          or 2 respectively.
        * ``'n'`` — return the user-supplied ``theta``, ``phi`` unchanged
          (caller has already obtained them from input).
        * ``'r'`` — auto-rotation; this routine returns ``(theta, phi)``
          unchanged for the caller to set in a sweep.

    theta, phi : float, optional
        Used for ``hvd in {'n', 'r'}``; angles in **radians**.

    Returns
    -------
    (theta_rad, phi_rad) : tuple of float
    """
    if hvd not in ('a', 'b', 'c', 'n', 'r'):
        raise ValueError(f"hvd must be one of 'a','b','c','n','r' — got {hvd!r}")

    if hvd in ('n', 'r'):
        return float(theta), float(phi)

    idx = {'a': 0, 'b': 1, 'c': 2}[hvd]
    v = recip[idx]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

    if vx == 0.0 and vy == 0.0:
        return 0.0, 0.0

    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    return math.atan2(vy, vx), math.acos(vz / norm)


def k_axis_lengths(recip: np.ndarray) -> np.ndarray:
    """Return the three reciprocal-lattice vector lengths ``|b_i|``."""
    return np.linalg.norm(recip, axis=1)


def k_axis_angles(recip: np.ndarray) -> tuple[float, float, float]:
    """Return the inter-axis angles ``(angle_xy, angle_xz, angle_yz)`` in radians.

    Mirrors ``anglelatxy``, ``anglelatxz``, ``anglelatyz`` from preadbxsf
    (lines ~3060).
    """
    lens = k_axis_lengths(recip)

    def _angle(i: int, j: int) -> float:
        cos = float(np.dot(recip[i], recip[j]) / (lens[i] * lens[j]))
        return math.acos(max(-1.0, min(1.0, cos)))

    return _angle(0, 1), _angle(0, 2), _angle(1, 2)
