"""Hopping-list models of PAOFLOW.models.models: spectra, the DataController round trip, the dict labels."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW import PAOFLOW
from PAOFLOW.hamiltonian.bloch_derivatives import (
    bloch_terms_from_HRs,
    hamiltonian_block,
    orbital_positions_from_data,
)
from PAOFLOW.models import models
from PAOFLOW.response import nonlinear_optics as nlo

pytestmark = pytest.mark.unit

VALLEYS = np.array(
    [[2 / 3, 1 / 3, 0.0], [1 / 3, 2 / 3, 0.0]]
)  # K, K' of a_1 = x, a_2 at 60 degrees


def bands(model: models.HoppingModel, k_cryst: np.ndarray) -> np.ndarray:
    terms = bloch_terms_from_HRs(
        model.real_space_hamiltonian(), model.lattice_vectors, model.alat, model.orbital_positions
    )
    return np.linalg.eigvalsh(hamiltonian_block(terms, np.atleast_2d(k_cryst)))


def test_honeycomb_gaps_at_the_valleys():
    """Gapped graphene: |Delta| at K and K'; Haldane: 2 |Delta/2 -+ 3 sqrt(3) t2 sin(phi)|."""
    valleys = np.array([[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.0]])
    gaps = np.diff(bands(models.honeycomb_model(delta=0.6, t=-1.0), valleys), axis=1)[:, 0]
    assert np.allclose(gaps, 0.6, atol=1e-12)
    haldane = models.honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=np.pi / 2)
    gaps = np.sort(np.diff(bands(haldane, valleys), axis=1)[:, 0])
    expected = np.sort(2.0 * np.abs([0.1 - 3 * np.sqrt(3) * 0.1, 0.1 + 3 * np.sqrt(3) * 0.1]))
    assert np.allclose(gaps, expected, atol=1e-12)


def test_weyl_nodes_and_rice_mele_gap():
    """Weyl: degenerate at k = (0, 0, +-1/4) with energies +-b; Rice-Mele: gap 2 sqrt(Delta^2 + 4 dt^2) at the zone edge."""
    weyl = models.weyl_model(t=1.0, b=0.3)
    nodes = bands(weyl, np.array([[0, 0, 0.25], [0, 0, -0.25]]))
    assert np.allclose(nodes, [[0.3, 0.3], [-0.3, -0.3]], atol=1e-12)
    chain = models.rice_mele_model(t=1.0, dt=0.3, delta=0.4)
    gap = np.diff(bands(chain, np.array([0.5, 0.0, 0.0])), axis=1)[0, 0]
    assert gap == pytest.approx(2.0 * np.sqrt(0.4**2 + (2 * 0.3) ** 2), abs=1e-12)


def test_single_band_dispersion():
    """The band is sum_R h(R) e^{2 pi i k.R} (partners added), the second orbital stays at 12 eV."""
    band = {(1, 0, 0): -1.0, (0, 1, 0): -0.8, (2, 0, 0): 0.25j}
    model = models.single_band_model(band, np.diag([1.0, 1.0, 10.0]), 5.0, 0.8, (5, 5, 1))
    k = np.array([[0.13, 0.27, 0.0], [-0.31, 0.05, 0.0]])
    exact = sum(2.0 * (h * np.exp(2j * np.pi * (k @ np.array(R)))).real for R, h in band.items())
    assert np.allclose(bands(model, k), np.stack([exact, np.full(2, 12.0)], axis=1), atol=1e-12)


def test_grid_too_small_is_refused():
    with pytest.raises(ValueError, match='cannot hold'):
        models.hoppings_to_HRs([(0, 0, (2, 0, 0), 1.0)], 1, (4, 1, 1))


@pytest.mark.parametrize(
    'params, reference',
    [
        (
            {'label': 'honeycomb', 'delta': 0.6, 't': -1.0, 't2': 0.15, 'phi': 1.0, 'strain': 0.2},
            models.honeycomb_model(0.6, -1.0, 0.15, 1.0, strain=0.2),
        ),
        ({'label': 'weyl', 't': 1.0, 'b': 0.3}, models.weyl_model(1.0, 0.3)),
        ({'label': 'Rice_Mele', 'delta': -0.4}, models.rice_mele_model(delta=-0.4)),
        ({'label': 'tmd', 'material': 'WSe2'}, models.tmd_model('WSe2')),
        ({'label': 'TMD', 'material': 'MoS2', 'soc': False}, models.tmd_model('MoS2', soc=False)),
        (
            {'label': 'Kane_Mele', 't': 1.0, 'soc_par': 0.1, 'alat': 1.0},
            models.kane_mele_model(t=1.0, soc=0.1, rashba=0.0, delta=0.0, alat=1.889727),
        ),
        (models.kane_mele_model(), models.kane_mele_model()),
    ],
)
def test_paoflow_model_round_trip(tmp_path, params, reference):
    """PAOFLOW(model=...) stores the HRs, cell and orbital centres the optical responses read."""
    pao = PAOFLOW.PAOFLOW(model=params, outputdir=str(tmp_path / 'out'), verbose=False)
    arrays, attributes = pao.data_controller.data_dicts()
    assert np.array_equal(arrays['HRs'], reference.real_space_hamiltonian())
    assert np.allclose(arrays['a_vectors'], reference.lattice_vectors)
    assert attributes['alat'] == reference.alat
    assert attributes['omega'] == pytest.approx(reference.volume)
    assert attributes['nelec'] == reference.nelec
    assert attributes['dftSO'] is reference.spinful
    positions = orbital_positions_from_data(pao.data_controller)
    assert np.allclose(positions, reference.orbital_positions, atol=1e-12)
    if reference.spinful:
        half = reference.number_of_orbitals // 2
        assert np.allclose(arrays['Sj'][2], 0.5 * np.diag([1.0] * half + [-1.0] * half))


def test_unknown_parameter_is_refused(tmp_path):
    with pytest.raises(TypeError):
        PAOFLOW.PAOFLOW(
            model={'label': 'weyl', 'mass': 1.0}, outputdir=str(tmp_path / 'out'), verbose=False
        )


def test_unknown_kane_mele_parameter_is_refused(tmp_path):
    with pytest.raises(TypeError):
        PAOFLOW.PAOFLOW(
            model={'label': 'Kane_Mele', 't': 1.0, 'soc_par': 0.1, 'soc': 0.1},
            outputdir=str(tmp_path / 'out'),
            verbose=False,
        )


def test_kane_mele_spectrum_at_the_valleys():
    """Without Rashba the energies at K and K' are +-(lambda_v +- 3 sqrt(3) lambda_SO)."""
    for delta in (0.1, 0.5):
        model = models.kane_mele_model(t=1.0, soc=0.06, rashba=0.0, delta=delta)
        masses = np.array([delta + 3 * np.sqrt(3) * 0.06, delta - 3 * np.sqrt(3) * 0.06])
        expected = np.sort(np.concatenate([masses, -masses]))
        assert np.allclose(bands(model, VALLEYS), [expected, expected], atol=1e-12)


def chern_number(model: models.HoppingModel, orbitals: list[int], band: int, n: int = 24) -> float:
    """Fukui-Hatsugai-Suzuki Chern number of ``band`` of the block ``orbitals`` on an n x n grid."""
    terms = bloch_terms_from_HRs(
        model.real_space_hamiltonian(), model.lattice_vectors, model.alat, model.orbital_positions
    )
    grid = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, [0.0], indexing='ij'), -1)
    H = hamiltonian_block(terms, grid.reshape(-1, 3))[:, orbitals][:, :, orbitals]
    states = np.linalg.eigh(H)[1][:, :, band].reshape(n, n, -1)
    # the lattice-gauge Hamiltonian is periodic, so the states are too: links wrap around
    link_1 = np.einsum('ijk,ijk->ij', states.conj(), np.roll(states, -1, axis=0))
    link_2 = np.einsum('ijk,ijk->ij', states.conj(), np.roll(states, -1, axis=1))
    flux = np.angle(link_1 * np.roll(link_2, -1, axis=0) / (np.roll(link_1, -1, axis=1) * link_2))
    return flux.sum() / (2 * np.pi)


def test_kane_mele_spin_chern_transition():
    """S_z conserved without Rashba: the spin-up valence band has Chern number +-1 below the
    transition lambda_v = 3 sqrt(3) lambda_SO (quantum spin Hall) and 0 above (trivial)."""
    soc = 0.06
    up = [0, 1]  # block order: (A up, B up | A down, B down)
    topological = models.kane_mele_model(
        t=1.0, soc=soc, rashba=0.0, delta=0.5 * 3 * np.sqrt(3) * soc
    )
    trivial = models.kane_mele_model(t=1.0, soc=soc, rashba=0.0, delta=1.5 * 3 * np.sqrt(3) * soc)
    assert abs(chern_number(topological, up, 0)) == pytest.approx(1.0, abs=1e-8)
    assert chern_number(trivial, up, 0) == pytest.approx(0.0, abs=1e-8)
    down = [2, 3]
    assert chern_number(topological, down, 0) == pytest.approx(
        -chern_number(topological, up, 0), abs=1e-8
    )


def liu_closed_form(parameters: tuple, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """H^NN(k) of Liu et al., PRB 88, 085433, Eqs. (4)-(11) (k in units of 1/a)."""
    _, e1, e2, t0, t1, t2, t11, t12, t22 = parameters
    al, be = kx / 2, np.sqrt(3) * ky / 2
    h0 = 2 * t0 * (np.cos(2 * al) + 2 * np.cos(al) * np.cos(be)) + e1
    h1 = -2 * np.sqrt(3) * t2 * np.sin(al) * np.sin(be) + 2j * t1 * (
        np.sin(2 * al) + np.sin(al) * np.cos(be)
    )
    h2 = 2 * t2 * (np.cos(2 * al) - np.cos(al) * np.cos(be)) + 2j * np.sqrt(3) * t1 * np.cos(
        al
    ) * np.sin(be)
    h11 = 2 * t11 * np.cos(2 * al) + (t11 + 3 * t22) * np.cos(al) * np.cos(be) + e2
    h22 = 2 * t22 * np.cos(2 * al) + (3 * t11 + t22) * np.cos(al) * np.cos(be) + e2
    h12 = np.sqrt(3) * (t22 - t11) * np.sin(al) * np.sin(be) + 4j * t12 * np.sin(al) * (
        np.cos(al) - np.cos(be)
    )
    return np.stack(
        [
            np.stack([h0, h1, h2], -1),
            np.stack([np.conj(h1), h11, h12], -1),
            np.stack([np.conj(h2), np.conj(h12), h22], -1),
        ],
        -2,
    )


@pytest.mark.parametrize('material, functional', [('MoS2', 'GGA'), ('WTe2', 'LDA'), ('WS2', 'GGA')])
def test_tmd_hamiltonian_is_the_closed_form(material, functional):
    """The hopping list (E(R1) and its C3 images) reproduces the paper's H(k) matrix."""
    model = models.tmd_model(material, functional, soc=False)
    k = np.random.default_rng(7).random((12, 3)) * [1, 1, 0]
    terms = bloch_terms_from_HRs(
        model.real_space_hamiltonian(), model.lattice_vectors, model.alat, model.orbital_positions
    )
    cartesian = 2 * np.pi * k @ np.linalg.inv(model.lattice_vectors).T
    parameters = models.TMD_NEAREST_NEIGHBOUR[functional][material]
    expected = liu_closed_form(parameters, cartesian[:, 0], cartesian[:, 1])
    assert np.allclose(hamiltonian_block(terms, k), expected, atol=1e-12)


def test_tmd_gap_and_spin_valley_locking():
    """MoS2 (GGA): K gap 1.663 eV without SOC; with SOC the valence top splits by exactly 2 lambda,
    spin up above at K and below at K' (time reversal), the d_z2 conduction band does not split."""
    parameters = models.TMD_NEAREST_NEIGHBOUR['GGA']['MoS2']
    _, e1, e2, t0, _, _, t11, t12, t22 = parameters
    gap = e1 - 3 * t0 - (e2 - 1.5 * (t11 + t22) - 3 * np.sqrt(3) * abs(t12))
    energies = bands(models.tmd_model('MoS2', soc=False), VALLEYS)
    assert energies[0, 1] - energies[0, 0] == pytest.approx(gap, abs=1e-12)
    assert gap == pytest.approx(1.663, abs=1e-3)

    model = models.tmd_model('MoS2')
    terms = bloch_terms_from_HRs(
        model.real_space_hamiltonian(), model.lattice_vectors, model.alat, model.orbital_positions
    )
    H = hamiltonian_block(terms, VALLEYS)
    assert np.allclose(H[:, :3, 3:], 0.0)  # S_z conserved
    up = np.linalg.eigvalsh(H[:, :3, :3])
    down = np.linalg.eigvalsh(H[:, 3:, 3:])
    splitting = 2 * models.TMD_SPIN_ORBIT['MoS2']
    assert up[0, 0] - down[0, 0] == pytest.approx(splitting, abs=1e-12)  # K
    assert up[1, 0] - down[1, 0] == pytest.approx(-splitting, abs=1e-12)  # K'
    assert np.allclose(up[:, 1], down[:, 1], atol=1e-12)  # conduction band minimum
    assert np.allclose(up[0], down[1], atol=1e-12)  # time reversal: up at K = down at K'


def test_tmd_bad_arguments():
    with pytest.raises(ValueError, match='material'):
        models.tmd_model('MoO2')
    with pytest.raises(ValueError, match='GGA'):
        models.tmd_model('MoS2', 'LDA', soc=True)
    assert models.tmd_model('MoS2', 'LDA', soc=0.07).spinful


@pytest.mark.parametrize(
    'model',
    [
        models.kane_mele_model(t=-1.0, soc=0.06, rashba=0.05, delta=0.4, alat=1.0),
        models.tmd_model('MoS2'),
        models.honeycomb_model(delta=0.6, t=-1.0, alat=1.0),
    ],
    ids=['kane_mele', 'tmd', 'honeycomb'],
)
def test_point_group_selection_rules_of_the_shift_current(model):
    """D3h / C3v with the mirror x -> -x: sigma^yyy = -sigma^yxx, sigma^xxx = 0 (as built)."""
    energies = np.array([1.2, 2.0]) if model.name != 'tmd_MoS2' else np.array([1.8, 2.4])
    out = nlo.nonlinear_optical_spectra(
        model.real_space_hamiltonian(),
        model.lattice_vectors,
        model.alat,
        model.volume,
        model.orbital_positions,
        model.nelec,
        (30, 30, 1),
        energies,
        components=['yyy', 'yxx', 'xxx'],
        eta=0.1,
        spin_factor=1.0 if model.spinful else 2.0,
        want=('shift',),
    )
    yyy, yxx, xxx = out['shift']
    scale = np.abs(yyy).max()
    assert scale > 0
    assert np.allclose(yxx, -yyy, atol=1e-10 * scale)
    assert np.allclose(xxx, 0.0, atol=1e-10 * scale)


@pytest.mark.parametrize('n_bands', [2, 3, 4])
def test_parabolic_gamma_levels_and_kane_masses(n_bands):
    """Gamma: the polar pair E_g/2 +- sqrt(E_g^2/4 + Q^2) (2, 3 bands); Kane masses along x for A = 0."""
    gap, t_c, t_sigma, t_pi, kane_p, q = 1.5, 0.5, 0.4, 0.1, 1.0, 0.3
    model = models.parabolic_model(n_bands=n_bands, asymmetry=q)
    levels = bands(model, np.zeros(3))[0]
    pair = gap / 2 + np.array([-1.0, 1.0]) * np.sqrt(gap**2 / 4 + q**2)
    if n_bands < 4:
        expected = np.sort(np.concatenate([pair, np.zeros(n_bands - 2)]))
    else:
        expected = np.array([0.0, 0.0, 0.0, gap])  # Kane's B term vanishes at Gamma
    assert np.allclose(levels, expected, atol=1e-12)

    symmetric = models.parabolic_model(n_bands=n_bands, asymmetry=0.0)
    step = 1e-3
    energies = bands(symmetric, np.array([[-step, 0, 0], [0, 0, 0], [step, 0, 0]]))
    curvature = (energies[0] + energies[2] - 2 * energies[1]) / (
        2 * np.pi * step / symmetric.alat
    ) ** 2
    a2 = symmetric.alat**2
    light, heavy = -2 * a2 * (t_sigma + kane_p**2 / gap), -2 * a2 * t_pi
    expected = np.sort([2 * a2 * (t_c + kane_p**2 / gap), light] + [heavy] * (n_bands - 2))
    assert np.allclose(curvature, expected, rtol=1e-4)  # O((k a)^2) finite-difference error


def _photocurrents(model, nk, components):
    return nlo.nonlinear_optical_spectra(
        model.real_space_hamiltonian(),
        model.lattice_vectors,
        model.alat,
        model.volume,
        model.orbital_positions,
        model.nelec,
        nk,
        np.array([1.9, 2.4]),
        components=components,
        eta=0.1,
        degeneracy_tol=1e-4,
        want=('shift', 'injection'),
    )


def test_parabolic_symmetry_classes():
    """2D, mirror x -> -x: A = 0 kills everything; phase 0 (T) gives shift and circular injection
    only, phase pi/2 (PT) the magnetic injection only (Watanabe and Yanase 2021)."""
    comps = ['yyy', 'yxx', 'xxy', 'xxx']
    responses = ('shift', 'shiftm', 'injectc', 'injectm')

    def scales(**kwargs):
        out = _photocurrents(models.parabolic_model(n_bands=3, **kwargs), (40, 40, 1), comps)
        return {key: np.abs(out[key]).max() for key in responses}, out

    centrosymmetric, _ = scales(asymmetry=0.0)
    polar, out_t = scales(phase=0.0)
    magnetic, _ = scales(phase=np.pi / 2)
    assert max(centrosymmetric['shift'], centrosymmetric['shiftm']) < 1e-12 * polar['shift']
    assert max(centrosymmetric['injectc'], centrosymmetric['injectm']) < 1e-12 * polar['injectc']
    assert magnetic['injectc'] < 1e-12 * polar['injectc']
    assert polar['shiftm'] < 1e-12 * polar['shift'] and polar['injectm'] < 1e-12 * polar['injectc']
    assert (
        magnetic['shift'] < 1e-12 * polar['shift'] and magnetic['injectm'] > 1e-3 * polar['injectc']
    )
    assert np.allclose(out_t['shift'][3], 0.0, atol=1e-12 * polar['shift'])  # xxx: mirror x -> -x
    flipped = _photocurrents(models.parabolic_model(n_bands=3, asymmetry=-0.3), (40, 40, 1), comps)
    assert np.allclose(flipped['shift'], -out_t['shift'], atol=1e-12 * polar['shift'])


def test_parabolic_td_components():
    """Four bands with Kane's B term (T_d): only the xyz-type components survive, all equal."""
    model = models.parabolic_model(n_bands=4)
    out = _photocurrents(model, (10, 10, 10), ['xyz', 'yzx', 'zxy', 'xxx', 'xyy', 'xxy'])
    shift = out['shift']
    scale = np.abs(shift[0]).max()
    assert scale > 0
    assert np.allclose(shift[1], shift[0], atol=1e-10 * scale)
    assert np.allclose(shift[2], shift[0], atol=1e-10 * scale)
    assert np.allclose(shift[3:], 0.0, atol=1e-10 * scale)


def test_parabolic_label_and_bad_band_count(tmp_path):
    pao = PAOFLOW.PAOFLOW(
        model={'label': 'parabolic', 'n_bands': 3, 'phase': 0.5},
        outputdir=str(tmp_path / 'out'),
        verbose=False,
    )
    arrays, attributes = pao.data_controller.data_dicts()
    reference = models.parabolic_model(n_bands=3, phase=0.5)
    assert np.array_equal(arrays['HRs'], reference.real_space_hamiltonian())
    assert attributes['nelec'] == 4.0
    assert arrays['Dnm'].shape == (3, 3, 3)
    with pytest.raises(ValueError):
        models.parabolic_model(n_bands=5)
