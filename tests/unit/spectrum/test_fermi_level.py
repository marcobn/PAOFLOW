"""Charge-neutral Fermi level of the PAO bands by electron counting (``spectrum.fermi_level``).

Closed forms for the solver (an unequal-degeneracy two-level system, the intrinsic level of a wide
gap at 1 K, doping), a one-orbital cubic model through ``HRs`` (particle-hole symmetry and a direct
count), the guard against a valence count the basis does not hold, and the PAOFLOW method on the
silicon fixture, dense and sparse (base cell).
"""

import os

import numpy as np
import pytest

from PAOFLOW.spectrum.fermi_level import (
    BOLTZMANN_EV_PER_K,
    BandCount,
    charge_neutral_levels,
    coarse_grid_energies,
    neutral_level_differs,
    occupied_states,
    solve_charge_neutral_level,
    usable_symmetry_operations,
)

SILICON = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'silicon'))


def _count(valence, conduction, window=10.0):
    valence, conduction = np.asarray(valence, float), np.asarray(conduction, float)
    return BandCount(
        valence_window=valence[np.abs(valence) <= window],
        conduction_window=conduction[np.abs(conduction) <= window],
        holes_outside=float(np.sum(valence > window)),
        electrons_outside=float(np.sum(conduction < -window)),
        valence_states=float(valence.size),
        valence_top=float(valence.max()),
        conduction_bottom=float(conduction.min()),
    )


@pytest.mark.parametrize('temperature', [30.0, 300.0, 1000.0])
def test_two_level_system_with_unequal_degeneracy(temperature):
    """f(e0) + 2 f(e1) = 1: with z = exp(-mu/kT), a b z^2 - a z - 2 = 0 (a, b = exp(e/kT))."""
    nk, e0, e1 = 500, -0.04, 0.03
    kT = BOLTZMANN_EV_PER_K * temperature
    mu = solve_charge_neutral_level(
        _count(np.full(nk, e0), np.full(2 * nk, e1)), nk, temperature, 10.0
    )
    a, b = np.exp(e0 / kT), np.exp(e1 / kT)
    z = (a + np.sqrt(a * a + 8.0 * a * b)) / (2.0 * a * b)
    assert mu == pytest.approx(-kT * np.log(z), abs=1e-12)


@pytest.mark.parametrize('temperature', [1.0, 10.0, 300.0])
def test_intrinsic_level_of_a_wide_gap(temperature):
    """Gap 2 eV, conduction level twice degenerate: mu = -(kT/2) ln 2 (non-degenerate limit).

    At 1 K the occupations are exp(-11600): exact only because the balance is solved in logs.
    """
    nk = 400
    count = _count(np.full(nk, -1.0), np.full(2 * nk, 1.0))
    mu = solve_charge_neutral_level(count, nk, temperature, 10.0)
    assert mu == pytest.approx(-0.5 * BOLTZMANN_EV_PER_K * temperature * np.log(2.0), rel=1e-9)
    assert occupied_states(count, mu, temperature) == pytest.approx(nk, abs=1e-9)


def test_doping_and_the_direct_count():
    rng = np.random.default_rng(7)
    nk, temperature = 300, 300.0
    kT = BOLTZMANN_EV_PER_K * temperature
    valence, conduction = rng.uniform(-1.0, 0.05, 3 * nk), rng.uniform(-0.05, 1.0, 3 * nk)
    count = _count(valence, conduction)
    for target in (3 * nk, 3 * nk + 1.0, 3 * nk - 2.5):
        mu = solve_charge_neutral_level(count, target, temperature, 10.0)
        occupied = np.sum(1.0 / (1.0 + np.exp((np.concatenate([valence, conduction]) - mu) / kT)))
        assert occupied == pytest.approx(target, abs=1e-9)


def _cubic_point_group():
    """The 48 signed permutation matrices (O_h on fractional k of a simple cubic lattice)."""
    from itertools import permutations, product

    return np.array([
        np.eye(3)[list(order)] * np.array(signs)[:, None]
        for order in permutations(range(3)) for signs in product((1, -1), repeat=3)
    ])  # fmt: skip


def _cubic_s_band(onsite, hopping, second, grid=6, anisotropy=0.0):
    """One orbital on a simple cubic lattice: HRs of E(k) by inverse FFT on ``grid``^3."""
    k = np.arange(grid) / grid
    c1, c2, c3 = np.meshgrid(*(np.cos(2 * np.pi * k),) * 3, indexing='ij')
    energies = (
        onsite
        - 2.0 * hopping * (c1 + c2 + c3)
        - 4.0 * second * (c1 * c2 + c2 * c3 + c3 * c1)
        - 2.0 * anisotropy * c1
    )
    HRs = np.fft.ifftn(energies).reshape(1, 1, grid, grid, grid, 1)
    return HRs, (lambda kk: onsite - 2.0 * hopping * np.cos(2 * np.pi * kk).sum(-1)
                 - 4.0 * second * sum(np.cos(2 * np.pi * kk[:, i]) * np.cos(2 * np.pi * kk[:, j])
                                      for i, j in ((0, 1), (1, 2), (2, 0))))  # fmt: skip


def test_half_filled_cubic_band_is_particle_hole_symmetric():
    """E(k + (1/2,1/2,1/2)) = 2 eps0 - E(k) on an even grid: mu = eps0 at any temperature."""
    HRs, _ = _cubic_s_band(onsite=0.1, hopping=1.0, second=0.0)
    for record in charge_neutral_levels(
        HRs, np.eye(3), 1.0, None, (12, 12, 12), [10.0, 300.0, 3000.0], 1.0, 2.0
    ):
        assert record['fermi_level_ev'] == pytest.approx(0.1, abs=1e-9)
        assert not record['gapped'] and record['electrons'] == 1.0
        assert record['electrons_at_dft_level'] < 1.0  # the band is shifted up by 0.1 eV


def test_cubic_band_with_second_neighbours_matches_the_direct_count():
    HRs, band = _cubic_s_band(onsite=0.0, hopping=0.5, second=0.12)
    nk, temperature, nelec = (10, 10, 10), 500.0, 0.8
    kT = BOLTZMANN_EV_PER_K * temperature
    grid = np.stack(np.meshgrid(*(np.arange(10) / 10,) * 3, indexing='ij'), -1).reshape(-1, 3)
    energies = band(grid)
    for doping in (0.0, 0.05):
        (record,) = charge_neutral_levels(
            HRs, np.eye(3), 1.0, None, nk, [temperature], nelec, 2.0, doping=doping,
            check_nelec=False, kblock=37,
        )  # fmt: skip
        occupied = 2.0 * np.mean(1.0 / (1.0 + np.exp((energies - record['fermi_level_ev']) / kT)))
        assert occupied == pytest.approx(nelec + doping, abs=1e-10)


def test_irreducible_wedge_gives_the_full_grid_level():
    HRs, _ = _cubic_s_band(onsite=0.0, hopping=0.5, second=0.12)
    options = dict(nelec=0.8, spin_factor=2.0, check_nelec=False)
    (full,) = charge_neutral_levels(HRs, np.eye(3), 1.0, None, (12, 12, 12), [300.0], **options)
    (wedge,) = charge_neutral_levels(
        HRs, np.eye(3), 1.0, None, (12, 12, 12), [300.0],
        symmetry_operations_cryst=_cubic_point_group(), **options,
    )  # fmt: skip
    assert full['symmetry_operations'] == 0 and full['k_points'] == 12**3
    assert wedge['symmetry_operations'] == 48 and wedge['k_points'] < 12**3 // 20
    assert wedge['fermi_level_ev'] == pytest.approx(full['fermi_level_ev'], abs=1e-12)


def test_operations_the_bands_or_the_grid_do_not_have_are_dropped():
    operations = _cubic_point_group()
    HRs, _ = _cubic_s_band(onsite=0.0, hopping=0.5, second=0.12, anisotropy=0.1)
    coarse = coarse_grid_energies(HRs, np.eye(3), 1.0)
    assert usable_symmetry_operations(operations, (12, 12, 12), (6, 6, 6), coarse) is None
    keeps_x = operations[np.abs(operations[:, 0, 0]) == 1]  # x stays on x: a symmetry left
    assert len(usable_symmetry_operations(keeps_x, (12, 12, 12), (6, 6, 6), coarse)) == 16
    isotropic = coarse_grid_energies(_cubic_s_band(0.0, 0.5, 0.12)[0], np.eye(3), 1.0)
    assert usable_symmetry_operations(operations, (12, 12, 10), (6, 6, 6), isotropic) is None
    assert usable_symmetry_operations(None, (12, 12, 12), (6, 6, 6), isotropic) is None


def test_a_valence_count_the_basis_does_not_hold_raises():
    HRs, _ = _cubic_s_band(onsite=0.0, hopping=1.0, second=0.0)
    with pytest.raises(ValueError, match='outside the PAO basis'):
        charge_neutral_levels(HRs, np.eye(3), 1.0, None, (8, 8, 8), [300.0], 1.8, 2.0)


def test_warning_threshold():
    assert neutral_level_differs(0.006, 10.0) and not neutral_level_differs(0.004, 10.0)
    assert not neutral_level_differs(0.012, 300.0)  # kT/2 = 12.9 meV at 300 K
    assert neutral_level_differs(-0.0737, 300.0)


def _silicon(outdir, sparse):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(SILICON)
    try:
        pao = PAOFLOW(
            savedir='silicon.save', outputdir=outdir, smearing='gauss', verbose=False,
            sparse=sparse, sparse_config={'hopping_threshold': 0.0} if sparse else None,
        )  # fmt: skip
        pao.read_atomic_proj_QE()
        pao.projectability()
        pao.pao_hamiltonian()
    finally:
        os.chdir(cwd)
    return pao


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(SILICON, 'silicon.save')), reason='silicon fixture not available'
)
def test_silicon_intrinsic_level_dense_and_sparse(tmp_path):
    dense = _silicon(str(tmp_path / 'dense'), sparse=False)
    levels = dense.charge_neutral_fermi_level(temperature=[300.0, 600.0], nk=(12, 12, 12))
    attr = dense.data_controller.data_dicts()[1]
    records = attr['charge_neutral_fermi_level']
    assert len(records) == 2 and all(r['gapped'] for r in records)
    assert records[0]['symmetry_operations'] > 1 and records[0]['k_points'] < 12**3
    full = dense.charge_neutral_fermi_level(temperature=300.0, nk=(12, 12, 12), symmetry=False)
    assert full == pytest.approx(levels[0], abs=1e-9)
    del records[2:]
    assert all(r['valence_top_ev'] < level < r['conduction_bottom_ev']
               for r, level in zip(records, levels))  # fmt: skip
    assert records[0]['electrons'] == pytest.approx(float(attr['nelec']))

    sparse = _silicon(str(tmp_path / 'sparse'), sparse=True)
    level = sparse.charge_neutral_fermi_level(temperature=300.0, nk=(12, 12, 12))
    assert isinstance(level, float)
    assert level == pytest.approx(levels[0], abs=1e-6)
    assert 'HRs' not in sparse.data_controller.data_arrays  # the bond list is kept

    from PAOFLOW.writers.output_manifest import run_record

    facts = run_record(attr, 0.0, attr['opath'])
    assert [r['temperature_k'] for r in facts['charge_neutral_fermi_level']] == [300.0, 600.0]
