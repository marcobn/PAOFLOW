"""Synthetic Fermi surfaces, SKEAF tables and images with known answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.pyskeaf.io_bxsf import BXSFData, write_bxsf

BOHR_ANG = 0.529177209
# reciprocal vectors of 1 A^-1 (with 2 pi) along x, y, z
RECIP_AU = np.eye(3) * BOHR_ANG / (2.0 * np.pi)


def write_band(path: Path, energies: np.ndarray, recip_au: np.ndarray = RECIP_AU) -> Path:
    """Write one band (general grid, periodic end points included) with PAOFLOW's writer."""
    nx, ny, nz = energies.shape
    data = BXSFData(filename=str(path), fermi_energy=0.0, nx=nx, ny=ny, nz=nz, recip_au=recip_au,
                    recip_ang=2.0 * np.pi * recip_au / BOHR_ANG, energies=energies)  # fmt: skip
    write_bxsf(path, data)
    return path


def fractional_grid(points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, points)
    return np.meshgrid(axis, axis, axis, indexing='ij')


def tight_binding(points: int, offset: float) -> np.ndarray:
    """-(cos 2pi f1 + cos 2pi f2 + cos 2pi f3) + offset (periodic, smooth)."""
    f1, f2, f3 = fractional_grid(points)
    return -(np.cos(2 * np.pi * f1) + np.cos(2 * np.pi * f2) + np.cos(2 * np.pi * f3)) + offset


def sphere(points: int, centre: tuple[float, float, float], radius: float) -> np.ndarray:
    """|k - k0|^2 - r^2 with k in A^-1 (reciprocal vectors of 1 A^-1)."""
    f1, f2, f3 = fractional_grid(points)
    return (f1 - centre[0]) ** 2 + (f2 - centre[1]) ** 2 + (f3 - centre[2]) ** 2 - radius**2


@pytest.fixture
def pocket_bands(tmp_path: Path) -> Path:
    """Bands 1, 2 (Kramers pair: electron pocket at Gamma), 3 (hole pocket at the corner), 4 (empty)."""
    folder = tmp_path / 'bands'
    folder.mkdir()
    electron = tight_binding(17, 2.0)
    write_band(folder / 'Fermi_surf_band_1.bxsf', electron)
    write_band(folder / 'Fermi_surf_band_2.bxsf', electron + 1e-6)
    write_band(folder / 'Fermi_surf_band_3.bxsf', tight_binding(17, -2.0))
    write_band(folder / 'Fermi_surf_band_4.bxsf', tight_binding(17, 5.0))
    return folder


@pytest.fixture
def two_spheres(tmp_path: Path) -> tuple[Path, float]:
    """Two spherical pockets (bands 1 and 2) separated by a known k-space gap (A^-1)."""
    folder = tmp_path / 'spheres'
    folder.mkdir()
    radius, gap = 0.135, 0.03
    shift = radius + gap / 2
    write_band(folder / 'Fermi_surf_band_1.bxsf', sphere(25, (0.5 - shift, 0.5, 0.5), radius))
    write_band(folder / 'Fermi_surf_band_2.bxsf', sphere(25, (0.5 + shift, 0.5, 0.5), radius))
    return folder, gap


def write_frequency_file(path: Path, rows: list[tuple]) -> Path:
    """Write a qo_EF_*_freqvsangle_*.out file; rows = (az, pol, F_kT, m*, curv, type, copies)."""
    header = (' Azimuthal(deg) ,    Polar(deg)  ,      Freq(kT)    ,     mstar(me)    ,     Curv(kTA2)   ,'
              '   Type(+e-h) ,   NumOrbCopy \n')  # fmt: skip
    body = ''.join(
        f'{a:16.6f},{p:16.6f},{f:18.6E},{m:18.6E},{c:18.6E},{t:14.3f},{n:14d}\n'
        for a, p, f, m, c, t, n in rows
    )
    path.write_text(header + body)
    return path


@pytest.fixture
def frequency_run(tmp_path: Path) -> Path:
    """Two bands over a 3-angle polar sweep at azimuthal 30, with a wall artefact (2 copies)."""
    folder = tmp_path / 'run'
    folder.mkdir()
    rows_band1, rows_band2 = [], []
    for polar in (0.0, 45.0, 90.0):
        rows_band1.append((30.0, polar, 0.5 + polar / 1000, 0.5, -100.0, -1.0, 150))
        rows_band1.append((30.0, polar, 0.5 + polar / 1000 + 1e-5, 0.5, -100.0, -1.0, 140))
        rows_band1.append((30.0, polar, 0.123, 0.2, -50.0, -1.0, 2))
        rows_band2.append((30.0, polar, 1.2, 1.1, 300.0, 1.0, 200))
    write_frequency_file(folder / 'qo_EF_0_freqvsangle_1.out', rows_band1)
    write_frequency_file(folder / 'qo_EF_0_freqvsangle_2.out', rows_band2)
    (folder / 'qo_EF_0_freqvsangle_3.out').write_text('')
    return folder
