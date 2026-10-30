"""Orbit-resolved mass enhancement: symmetry unfolding, orbit averaging, file parsing."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.agentic.qo_analysis.mass_enhancement import (
    FermiSurfaceLambda,
    cartesian_to_crystal,
    enhanced_masses,
    lattice_point_group,
    orbit_lambda,
    read_orbit_outlines,
)

BOHR = 0.529177209
FCC = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])


def _coupling(nk, at, states):
    """fs_coupling dict with explicit states: list of (k_cryst, band, lambda, energy)."""
    k = np.array([s[0] for s in states], float)
    return {
        'nk_dense': nk,
        'at': at,
        'fsthick_ev': 0.2,
        'k_cryst': k,
        'band': np.array([s[1] for s in states], int),
        'lambda_nk': np.array([s[2] for s in states], float),
        'energy_ev': np.array([s[3] for s in states], float),
    }


@pytest.mark.unit
def test_lattice_point_group_sizes():
    assert lattice_point_group(np.eye(3)).shape[0] == 48
    assert lattice_point_group(FCC).shape[0] == 48
    assert lattice_point_group(np.diag([1.0, 1.0, 1.7])).shape[0] == 16


@pytest.mark.unit
def test_unfolding_reaches_the_star_of_a_cubic_point():
    nk = 8
    fsc = _coupling(nk, np.eye(3), [([1 / 8, 2 / 8, 0.0], 0, 0.7, 0.0)])
    fsl = FermiSurfaceLambda(fsc)
    # the star of (1,2,0)/8 under O_h has 24 members; every one carries the same lambda
    assert fsl.nstates_full == 24
    for k in ([2 / 8, 1 / 8, 0.0], [0.0, -1 / 8, 2 / 8], [-2 / 8, 0.0, 1 / 8]):
        assert fsl.lookup(np.array([k]), band=0)[0] == pytest.approx(0.7)
    assert np.isnan(fsl.lookup(np.array([[3 / 8, 3 / 8, 3 / 8]]), band=0)[0])


@pytest.mark.unit
def test_orbit_average_of_an_angular_lambda():
    """lambda = 0.5 + 0.5 cos^2(phi) on a circle: arclength average 0.75."""
    nk = 40
    alat = 2.0 * np.pi / BOHR  # a = 2 pi Angstrom -> reciprocal vectors of 1 A^-1 (2 pi included)
    grid = np.arange(nk) / nk
    kx, ky = np.meshgrid(grid, grid, indexing='ij')
    phi = np.arctan2(np.where(ky > 0.5, ky - 1, ky), np.where(kx > 0.5, kx - 1, kx))
    states = [
        ([x, y, 0.0], 0, 0.5 + 0.5 * np.cos(p) ** 2, 0.0)
        for x, y, p in zip(kx.ravel(), ky.ravel(), phi.ravel())
    ]
    fsl = FermiSurfaceLambda(_coupling(nk, np.eye(3), states), sym_rots=np.eye(3)[None])
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    radius = 0.3  # A^-1: crystal coordinate 0.3 / (2 pi) * 2 pi = 0.3 of the cell
    circle = np.column_stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])
    assert np.allclose(cartesian_to_crystal(circle, np.eye(3), alat)[:, :2], circle[:, :2])
    stats = orbit_lambda(circle, fsl, alat, band=0)
    assert stats['matched_fraction'] == pytest.approx(1.0)
    assert stats['lambda_orbit'] == pytest.approx(0.75, abs=0.02)
    assert stats['lambda_min'] == pytest.approx(0.5, abs=0.03) and stats[
        'lambda_max'
    ] == pytest.approx(1.0, abs=0.03)


def _write_skeaf_files(tmp_path, band, orbits):
    """Minimal freqvsangle + orbitoutlines files of pyskeaf for one band."""
    freq = ['Azimuthal(deg),Polar(deg),Freq(kT),mstar(me),Curv(kTA2),Type(+e-h),NumOrbCopy']
    outline = []
    by_angle = {}
    for o in orbits:
        by_angle.setdefault((o['theta'], o['phi']), []).append(o)
    for (theta, phi), group in by_angle.items():
        outline.append(
            f'Azimuthal(deg) = {theta:10.6f} , Polar(deg) = {phi:10.6f} , Number of orbits = {len(group):5d} , (Angstrom^-1) units'
        )
        for o in group:
            freq.append(f'{theta:.6f},{phi:.6f},{o["F"]:.6f},{o["m"]:.6f},-1.0,1,{o["copies"]}')
            outline.append(
                f'Slice   = {1:6d} , Freq(kT, average of all copies)              = {o["F"]:8.4f}'
            )
            outline.append(
                f'Orbit # = {0:6d} , Freq(kT, this largest copy used for outline) = {o["F"]:8.4f}'
            )
            outline.append(f'Points  = {len(o["points"]):6d}')
            outline.append('  kx            ky            kz')
            outline += [f' {x:13.6E} {y:13.6E} {z:13.6E}' for x, y, z in o['points']]
    (tmp_path / f'qo_EF_0_freqvsangle_{band}.out').write_text('\n'.join(freq) + '\n')
    (tmp_path / f'qo_EF_0_orbitoutlines_invAng_{band}.out').write_text('\n'.join(outline) + '\n')


@pytest.mark.unit
def test_read_orbit_outlines_and_enhanced_masses(tmp_path):
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    circle = np.column_stack([0.2 * np.cos(t), 0.2 * np.sin(t), np.zeros_like(t)])
    _write_skeaf_files(
        tmp_path,
        3,
        [
            {'theta': 0.0, 'phi': 0.0, 'F': 1.5, 'm': 0.8, 'copies': 120, 'points': circle},
            {'theta': 0.0, 'phi': 0.0, 'F': 4.0, 'm': -1.2, 'copies': 30, 'points': circle * 2},
        ],
    )
    parsed = read_orbit_outlines(tmp_path / 'qo_EF_0_orbitoutlines_invAng_3.out')
    assert len(parsed) == 2 and parsed[1]['frequency_kT'] == pytest.approx(4.0)
    assert parsed[0]['points'].shape == (50, 3)
    # uniform lambda = 0.6 on band 2 (BXSF band 3) everywhere on a 20^3 grid
    nk, alat = 20, 2.0 * np.pi / BOHR
    g = np.arange(nk) / nk
    kx, ky, kz = np.meshgrid(g, g, g, indexing='ij')
    states = [([x, y, z], 2, 0.6, 0.0) for x, y, z in zip(kx.ravel(), ky.ravel(), kz.ravel())]
    table = enhanced_masses(
        tmp_path, _coupling(nk, np.eye(3), states), alat, sym_rots=np.eye(3)[None]
    )
    assert len(table) == 2
    assert table.lambda_orbit.to_numpy() == pytest.approx([0.6, 0.6])
    assert table.mstar_enhanced.to_numpy() == pytest.approx([0.8 * 1.6, 1.2 * 1.6])
    assert list(table.carrier) == ['e', 'e'] and table.matched_fraction.min() == pytest.approx(1.0)


@pytest.mark.unit
def test_lambda_from_pair_coupling_file_layout():
    """A coupling dict in the write_fs_coupling layout (no lambda_nk) is accepted."""
    from PAOFLOW.elphon.anisotropic_eliashberg import coupling_strength

    rng = np.random.default_rng(0)
    n, nfreq = 6, 20
    freq = np.linspace(0.001, 0.02, nfreq)
    lam_pair = rng.uniform(0.1, 1.0, (n, n, nfreq))
    lam_pair = 0.5 * (lam_pair + lam_pair.transpose(1, 0, 2))
    weight = np.full(n, 1.0 / n)
    base = {
        'coupling': lam_pair, 'coulomb': np.zeros((n, n)), 'freq_ev': freq, 'weight': weight,
        'band': np.zeros(n, int), 'energy_ev': np.zeros(n), 'nk_dense': 4, 'at': np.eye(3), 'fsthick_ev': 0.2,
        'k_cryst': np.array([[i / 4, 0.0, 0.0] for i in range(4)] + [[0.0, 1 / 4, 0.0], [0.0, 2 / 4, 0.0]]),
    }  # fmt: skip
    expected = coupling_strength(base)['lambda_nk']
    fsl = FermiSurfaceLambda(base, sym_rots=np.eye(3)[None])
    got = fsl.lookup(base['k_cryst'], band=0)
    assert np.allclose(got, expected)


@pytest.mark.unit
def test_missing_orbit_outlines_raise(tmp_path):
    (tmp_path / 'qo_EF_0_freqvsangle_1.out').write_text(
        'Azimuthal(deg),Polar(deg),Freq(kT),mstar(me),Curv(kTA2),Type(+e-h),NumOrbCopy\n0,0,1.0,0.5,-1.0,1,10\n'
    )
    nk, alat = 4, 2.0 * np.pi / BOHR
    g = np.arange(nk) / nk
    kx, ky, kz = np.meshgrid(g, g, g, indexing='ij')
    states = [([x, y, z], 0, 0.6, 0.0) for x, y, z in zip(kx.ravel(), ky.ravel(), kz.ravel())]
    with pytest.raises(FileNotFoundError, match='write_auxiliary_files'):
        enhanced_masses(tmp_path, _coupling(nk, np.eye(3), states), alat, sym_rots=np.eye(3)[None])


@pytest.mark.unit
def test_sheet_band_indices_map_bxsf_sheets_to_pao_bands(tmp_path):
    """The FermiSurf_<spin>.bxsf header lists the 1-based PAO band of every sheet."""
    from PAOFLOW.agentic.qo_analysis.mass_enhancement import sheet_band_indices

    assert sheet_band_indices(tmp_path) is None
    (tmp_path / 'FermiSurf_0.bxsf').write_text(
        'BEGIN_INFO\n  Fermi Energy: 0.0\nEND_INFO\nBEGIN_BLOCK_BANDGRID_3D\nband_energies\n'
        'BEGIN_BANDGRID_3D_BANDS\n  2\n  3 3 3\n  BAND:    13\n 1 2 3\n  BAND:    14\n 1 2 3\n'
        'END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3D\n'
    )
    assert sheet_band_indices(tmp_path) == {1: 12, 2: 13}
    # enhanced_masses uses the map: states on PAO band 13 (BXSF band 14 = sheet 2)
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    circle = np.column_stack([0.2 * np.cos(t), 0.2 * np.sin(t), np.zeros_like(t)])
    _write_skeaf_files(
        tmp_path,
        2,
        [{'theta': 0.0, 'phi': 0.0, 'F': 1.5, 'm': 0.8, 'copies': 10, 'points': circle}],
    )
    nk, alat = 20, 2.0 * np.pi / BOHR
    g = np.arange(nk) / nk
    kx, ky, kz = np.meshgrid(g, g, g, indexing='ij')
    states = [([x, y, z], 13, 0.4, 0.0) for x, y, z in zip(kx.ravel(), ky.ravel(), kz.ravel())]
    table = enhanced_masses(
        tmp_path, _coupling(nk, np.eye(3), states), alat, sym_rots=np.eye(3)[None]
    )
    assert list(table.pao_band) == [13] and table.matched_fraction.iloc[0] == pytest.approx(1.0)
    assert table.lambda_orbit.iloc[0] == pytest.approx(0.4)
    # without the header the legacy sheet - 1 convention applies and finds nothing on band 1
    (tmp_path / 'FermiSurf_0.bxsf').unlink()
    table = enhanced_masses(
        tmp_path, _coupling(nk, np.eye(3), states), alat, sym_rots=np.eye(3)[None]
    )
    assert list(table.pao_band) == [1] and table.matched_fraction.iloc[0] == 0.0
    # an explicit map overrides both
    table = enhanced_masses(
        tmp_path, _coupling(nk, np.eye(3), states), alat, sym_rots=np.eye(3)[None], band_map={2: 13}
    )
    assert table.lambda_orbit.iloc[0] == pytest.approx(0.4)


@pytest.mark.unit
def test_lookup_takes_the_closest_state_among_the_neighbouring_grid_points():
    """States only on even grid planes: the odd planes are served by their neighbours."""
    nk = 20
    g = np.arange(nk) / nk
    kx, ky, kz = np.meshgrid(g, g, g, indexing='ij')
    even = (np.rint(kx.ravel() * nk).astype(int) % 2) == 0
    states = [
        ([x, y, z], 0, 0.3 + x, 0.0)
        for x, y, z, keep in zip(kx.ravel(), ky.ravel(), kz.ravel(), even)
        if keep
    ]
    fsl = FermiSurfaceLambda(_coupling(nk, np.eye(3), states), sym_rots=np.eye(3)[None])
    odd_points = np.array([[1 / nk, 0.25, 0.5], [3 / nk, 0.1, 0.9], [5 / nk, 0.0, 0.0]])
    assert np.all(np.isnan(fsl.lookup(odd_points, band=0, neighbours=0)))
    lam = fsl.lookup(odd_points, band=0, neighbours=1)
    assert np.all(np.isfinite(lam))
    # both neighbours along x carry the same energy (0): either is acceptable, lambda = 0.3 + x
    assert np.all(
        (np.abs(lam - (0.3 + odd_points[:, 0] - 1 / nk)) < 1e-9)
        | (np.abs(lam - (0.3 + odd_points[:, 0] + 1 / nk)) < 1e-9)
    )
    # band=None takes the state closest to the Fermi level among bands and neighbours
    assert np.all(np.isfinite(fsl.lookup(odd_points, band=None)))
