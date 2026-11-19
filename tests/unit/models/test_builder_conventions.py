"""Storage convention and completeness of the model builders (15 November 2026).

Every builder must store the hopping to the cell ``R`` at the ``HRs`` index ``-R`` (PAOFLOW's convention,
the one of the QE route), so that quantities that pair ``HRs`` with the orbital centres, band unfolding
of a model against a DFT run, and the band path ``bands()`` agree:

- ``Slater_Koster``, ``SK_EDTB`` and the dual-parameter builder used to store ``+R``;
- the legacy ``graphene``, ``cubium`` and ``cubium2`` lacked attributes (``bnd``, ``Dnm``) that
  ``pao_eigh`` and ``gradient_and_momenta`` need;
- the path routines (``do_bands`` and the topology loops) evaluated :math:`H(-\\mathbf k)`;
- ``surface_project`` passed a stray positional argument to ``_extract_hamiltonian``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW import PAOFLOW
from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
from PAOFLOW.models import models as M
from PAOFLOW.models.band_unfold import _detect_fourier_sign, paoflow_to_model_dict, unfold_bands
from PAOFLOW.utils.get_R_grid_fft import get_R_grid_fft

pytestmark = pytest.mark.unit

EXAMPLES = Path(__file__).resolve().parents[2] / 'data' / 'siliconSC'
ONSITE = {'s': -5.19278, 'px': 1.05825, 'py': 1.05825, 'pz': 1.05825}
HOPPINGS = {'sss': -2.36233, 'sps': 1.86401, 'pps': 2.85882, 'ppp': -0.94687}
FCC = [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]


def _silicon(label='Slater_Koster', screening=None, cubic=False):
    """Si (diamond), sp3 nearest neighbours (Papaconstantopoulos): 2-atom fcc cell or 8-atom cube."""
    if cubic:
        lattice = np.eye(3).tolist()
        base = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        taus = [b for b in base] + [list(np.add(b, 0.25)) for b in base]
    else:
        lattice, taus = FCC, [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    atoms = {str(i): {'name': 'Si', 'tau': list(map(float, t)), 'orbitals': ['s', 'px', 'py', 'pz'], **ONSITE}
             for i, t in enumerate(taus)}  # fmt: skip
    model = {'a_vectors': lattice, 'atoms': atoms, 'hoppings': dict(HOPPINGS)}
    if screening is not None:
        model['screening'] = screening
    return {'label': label, 'alat': 10.26, 'model': model}


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _skip_without_pythtb2(model) -> None:
    """The *_pythtb labels build PythTB 2 models (pythtb.Lattice; PythTB 2 needs Python >= 3.12)."""
    if isinstance(model, dict) and str(model.get('label', '')).endswith('_pythtb'):
        pythtb = pytest.importorskip('pythtb')
        if not hasattr(pythtb, 'Lattice'):
            pytest.skip('the *_pythtb models need PythTB 2 (Python >= 3.12)')


def _load(model, tmp_path, name='run'):
    _skip_without_pythtb2(model)
    with _quiet():
        return PAOFLOW.PAOFLOW(
            model=model, outputdir=str(tmp_path / name), smearing='gauss', verbose=False
        )


def _stored_sign(p):
    """+1 if HRs holds the hopping to tau_B + R at the index R, -1 for PAOFLOW's -R (bond-length test)."""
    arrays, attr = p.data_controller.data_dicts()
    hrs = arrays['HRs']
    nawf, _, n1, n2, n3, nspin = hrs.shape
    get_R_grid_fft(p.data_controller, n1, n2, n3)
    norb = np.asarray(arrays.get('norbitals', arrays.get('naw')), dtype=int)
    tau = np.atleast_2d(arrays['tau'])
    return _detect_fourier_sign(hrs.reshape(nawf, nawf, -1, nspin), arrays['R'], tau, norb)


@pytest.mark.parametrize('label', ['Slater_Koster', 'SK_EDTB'])
def test_slater_koster_builders_store_hoppings_at_minus_R(label, tmp_path):
    screening = {'r_cut': 9.0, 'gamma': 0.05} if label == 'SK_EDTB' else None
    assert _stored_sign(_load(_silicon(label, screening), tmp_path)) == -1
    # the same as the hopping-list models and the QE route
    assert _stored_sign(_load(M.honeycomb_model(), tmp_path, 'hc')) == -1


def test_slater_koster_path_hamiltonian_with_orbital_centres_matches_the_grid(tmp_path):
    """The pair-resolved Wigner-Seitz layer (orbital centres + cells) reproduces pao_eigh only if the
    storage convention is PAOFLOW's: it missed by 5.8 eV with the hoppings at +R."""
    p = _load(_silicon(), tmp_path)
    arrays, attr = p.data_controller.data_dicts()
    avec = np.asarray(arrays['a_vectors'], float)
    positions = np.repeat(np.asarray(arrays['tau']) @ np.linalg.inv(avec), 4, axis=0)
    terms = bloch_terms_from_HRs(np.array(arrays['HRs']), avec, attr['alat'], positions)
    with _quiet():
        p.interpolated_hamiltonian(nfft1=6, nfft2=6, nfft3=6)
        p.pao_eigh()
    k_cryst = (avec @ np.asarray(arrays['kgrid'])).T
    energies = np.linalg.eigvalsh(hamiltonian_block(terms, k_cryst))
    assert np.abs(energies - arrays['E_k'][:, :, 0]).max() < 1e-10


def test_sk_edtb_without_screening_is_slater_koster(tmp_path):
    plain = _load(_silicon(), tmp_path, 'a').data_controller.data_arrays['HRs']
    edtb = _load(
        _silicon('SK_EDTB', {'r_cut': 9.0, 'gamma': 0.0}), tmp_path, 'b'
    ).data_controller.data_arrays['HRs']
    assert np.allclose(plain, edtb, atol=1e-12)


def test_slater_koster_pdos_covers_every_band(tmp_path):
    """shift = inf: the PDOS is no longer cut at 0 eV."""
    p = _load(_silicon(), tmp_path)
    assert np.isinf(p.data_controller.data_attributes['shift'])
    with _quiet():
        p.interpolated_hamiltonian(nfft1=8, nfft2=8, nfft3=8)
        p.pao_eigh()
        p.gradient_and_momenta()
        p.adaptive_smearing(min_width=0.05)
        p.dos(emin=-16.0, emax=8.0, ne=481, do_pdos=True)
    pdos = np.loadtxt(tmp_path / 'run' / 'pdosdk_sum_0.dat')
    assert pdos[-1, 0] == pytest.approx(8.0)
    assert np.trapezoid(pdos[:, 1], pdos[:, 0]) == pytest.approx(8.0, abs=1e-3)


def test_unfolding_of_a_slater_koster_supercell_is_route_independent(tmp_path):
    """The supercell through the model runner and through paoflow_to_model_dict (the DFT-route helper,
    fourier_sign = -1) gives the same weights: they differed by up to 1 while the builder stored +R."""
    pc, sc = _silicon(), _silicon(cubic=True)
    sym_points = {'G': [0.0, 0.0, 0.0], 'X': [0.0, 0.5, 0.5], 'L': [0.5, 0.5, 0.5]}
    with _quiet():
        runner = unfold_bands(
            pc, sc, sym_points=sym_points, path_str='L-G-X', nk_per_seg=4, verbose=False
        )
        dft_like = paoflow_to_model_dict(_load(sc, tmp_path, 'sc').data_controller)
        dft_like['model'] = copy.deepcopy(sc['model'])
        helper = unfold_bands(
            pc, dft_like, sym_points=sym_points, path_str='L-G-X', nk_per_seg=4, verbose=False
        )
    assert np.allclose(runner.E_sc, helper.E_sc, atol=1e-10)
    assert np.allclose(runner.W, helper.W, atol=1e-10)
    # the unfolded supercell states reproduce the primitive bands
    for ik in range(runner.W.shape[0]):
        bright = runner.E_sc[ik][runner.W[ik] > 0.5]
        assert np.abs(bright[:, None] - runner.E_pc[ik][None, :]).min(axis=1).max() < 1e-8


def test_surface_projection_runs_on_a_slater_koster_model():
    from PAOFLOW.models.surface_project import project_bulk_bands

    with _quiet():
        result = project_bulk_bands(_silicon(), [0, 0, 1], nk_par=6, nk_perp=6, verbose=False)
    values = [
        np.asarray(getattr(result, name))
        for name in vars(result)
        if isinstance(getattr(result, name), np.ndarray)
    ]
    assert values and all(np.isfinite(v).all() for v in values if v.dtype.kind in 'fc')


def test_dual_parameter_hamiltonian_follows_the_paoflow_convention(tmp_path):
    from PAOFLOW.models.dual_params import DualParamModel

    params = json.loads((EXAMPLES / 'Si_SK_params.json').read_text())
    geometry = json.loads((EXAMPLES / 'Si_geometry.json').read_text())
    model = DualParamModel(params, params, geometry, labels=['bulk', 'bulk'])
    with _quiet():
        hrs, _ = model.build_hamiltonian(verbose=False)
        p = _load(model.to_model_dict(), tmp_path)
    p.data_controller.data_arrays['HRs'] = hrs  # the documented use: replace the SK_EDTB HRs
    assert _stored_sign(p) == -1


@pytest.mark.parametrize(
    'model, ngrid',
    [({'label': 'graphene', 'delta': 0.5, 't': -2.7}, (24, 24, 1)),
     ({'label': 'cubium', 't': 1.0}, (8, 8, 8)),
     ({'label': 'cubium2', 't': 1.0, 'Eg': 1.0}, (8, 8, 8))],
)  # fmt: skip
def test_legacy_builders_run_the_whole_pipeline(model, ngrid, tmp_path):
    p = _load(model, tmp_path)
    with _quiet():
        p.interpolated_hamiltonian(nfft1=ngrid[0], nfft2=ngrid[1], nfft3=ngrid[2])
        p.pao_eigh()
        p.gradient_and_momenta()
        p.adaptive_smearing()
        p.dos(emin=-15.0, emax=15.0, ne=1501, do_pdos=False)
    dos = np.loadtxt(tmp_path / 'run' / 'dosdk_0.dat')
    nawf = p.data_controller.data_attributes['nawf']
    assert np.isfinite(dos[:, 1]).all()
    assert np.trapezoid(dos[:, 1], dos[:, 0]) == pytest.approx(nawf, abs=5e-3)


def test_legacy_graphene_has_the_bands_of_honeycomb_model(tmp_path):
    legacy = _load({'label': 'graphene', 'delta': 0.4, 't': -2.7}, tmp_path, 'a')
    modern = _load(M.honeycomb_model(delta=0.4, t=-2.7), tmp_path, 'b')
    for p in (legacy, modern):
        with _quiet():
            p.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=1)
            p.pao_eigh()
    assert np.allclose(
        legacy.data_controller.data_arrays['E_k'],
        modern.data_controller.data_arrays['E_k'],
        atol=1e-12,
    )


def test_pythtb_labels_take_a_filling(tmp_path):
    p = _load({'label': 'cubium_pythtb', 't': 1.0, 'nelec': 1.0}, tmp_path)
    assert p.data_controller.data_attributes['nelec'] == 1.0


def test_bands_evaluates_H_at_plus_k(tmp_path):
    """Haldane model with Delta and phi (P and T broken): bands() along a path is H(+k), not H(-k)."""
    hm = M.honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=np.pi / 2)
    p = _load(hm, tmp_path)
    arrays, attr = p.data_controller.data_dicts()
    avec = np.asarray(arrays['a_vectors'], float)
    terms = bloch_terms_from_HRs(np.array(arrays['HRs']), avec, attr['alat'], hm.orbital_positions)
    with _quiet():
        p.bands(
            ibrav=4,
            nk=60,
            band_path='G-K-M-G',
            high_sym_points={'G': [0, 0, 0], 'K': [2 / 3, 1 / 3, 0], 'M': [0.5, 0, 0]},
        )
    k_cryst = (avec @ np.asarray(arrays['kq'])).T
    e_plus = np.linalg.eigvalsh(hamiltonian_block(terms, k_cryst))
    e_minus = np.linalg.eigvalsh(hamiltonian_block(terms, -k_cryst))
    assert np.abs(arrays['E_k'][:, :, 0] - e_plus).max() < 1e-10
    assert np.abs(arrays['E_k'][:, :, 0] - e_minus).max() > 0.1
