"""The orbit outlines of pyskeaf lie on the Fermi surface in the BZ frame (origin of the slicing frame)."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.pyskeaf.constants import CONV_AU_TO_ANG, CONV_FSAREA_TO_KT
from PAOFLOW.pyskeaf.io_bxsf import BXSFData
from PAOFLOW.pyskeaf.runner import run_at_angle


def _sphere_bxsf(n=33, k_f=0.3, centre=(0.5, 0.5, 0.5)):
    """Free-electron sphere of radius k_f (A^-1) in a cubic reciprocal cell of side 1 A^-1."""
    recip_ang = np.eye(3)  # A^-1, 2 pi included
    recip_au = (
        recip_ang * CONV_AU_TO_ANG / (2.0 * np.pi)
    )  # a.u.^-1 without 2 pi (reader convention)
    frac = np.arange(n) / (n - 1)  # the BXSF general grid includes the periodic end point
    f1, f2, f3 = np.meshgrid(frac, frac, frac, indexing='ij')
    k = np.stack([f1, f2, f3], axis=-1) @ recip_ang - np.asarray(centre)
    energies = np.sum(k**2, axis=-1) - k_f**2  # Ryd, E_F = 0 on the sphere
    return BXSFData(
        filename='sphere.bxsf', fermi_energy=0.0, nx=n, ny=n, nz=n, recip_au=recip_au,
        recip_ang=recip_ang, energies=energies,
    )  # fmt: skip


@pytest.mark.unit
@pytest.mark.parametrize(
    'theta_phi', [(0.0, 0.0), (np.radians(45.0), np.radians(90.0)), (0.3, 0.9)]
)
def test_orbit_outline_lies_on_the_fermi_sphere(theta_phi):
    k_f, centre = 0.3, np.array([0.5, 0.5, 0.5])
    bxsf = _sphere_bxsf(k_f=k_f, centre=centre)
    result = run_at_angle(bxsf, theta_phi[0], theta_phi[1], numint=25)
    orbits = [o for o in result.orbits if o.outline_ang is not None]
    assert orbits, 'no orbit with an outline'
    # the extremal orbit of a sphere is its great circle: F = pi k_f^2 x conversion
    orbit = max(orbits, key=lambda o: o.num_copies)
    assert orbit.frequency_kT == pytest.approx(np.pi * k_f**2 * CONV_FSAREA_TO_KT, rel=0.05)
    pts = orbit.outline_ang
    # bring every point into the cell of the sphere centre and measure its radius
    rel = pts - centre
    rel -= np.rint(rel @ np.linalg.inv(bxsf.recip_ang)) @ bxsf.recip_ang  # subtract lattice vectors
    radius = np.linalg.norm(rel, axis=1)
    assert np.median(radius) == pytest.approx(k_f, rel=0.05)
    assert np.percentile(np.abs(radius - k_f), 90) < 0.05 * k_f
    assert np.allclose(orbit.outline_au, orbit.outline_ang * CONV_AU_TO_ANG)
