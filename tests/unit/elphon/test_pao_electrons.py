"""The electronic side of the dense pipeline: basis identity, Hubbard provenance, decoupled HRs."""

import warnings

import numpy as np
import pytest

from PAOFLOW.elphon.pao_electrons import (
    check_orbital_dimensions,
    check_same_pao_basis,
    format_hubbard_info,
    hubbard_info_from_save,
)

NAWF = 4


def _stub(**over):
    """``(arrays, attributes)`` of a two-atom Si-like run in a 4-orbital basis."""
    tau = np.array([[0.0, 0.0, 0.0], [2.583, 2.583, 2.583]])
    basis = [
        {'atom': 'Si', 'tau': tau[0], 'l': 0, 'm': 1, 'label': '3S'},
        {'atom': 'Si', 'tau': tau[0], 'l': 1, 'm': 1, 'label': '3P'},
        {'atom': 'Si', 'tau': tau[1], 'l': 0, 'm': 1, 'label': '3S'},
        {'atom': 'Si', 'tau': tau[1], 'l': 1, 'm': 1, 'label': '3P'},
    ]
    arrays = {
        'species': [('Si', '/some/where/Si.upf')],
        'atoms': ['Si', 'Si'],
        'tau': tau,
        'a_vectors': np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]),
        'basis': basis,
        'configuration': {'Si': ['3S', '3P']},
    }
    attributes = {
        'nawf': NAWF, 'nspin': 1, 'dftSO': False, 'natoms': 2, 'nelec': 8.0, 'alat': 10.33,
        'nbnds': 21, 'nkpnts': 1728, 'nk1': 12, 'nk2': 12, 'nk3': 12, 'bnd': 11, 'shift': 10.59,
        'Efermi': 6.2, 'pthr': 0.95,
    }  # fmt: skip
    for key, value in over.items():
        (attributes if key in attributes else arrays)[key] = value
    return arrays, attributes


def test_check_same_pao_basis_passes_and_reports():
    reference = _stub()
    other = _stub(nbnds=30, nkpnts=195, nk1=18, nk2=18, nk3=18, bnd=13, shift=9.1, Efermi=6.9)
    summary = check_same_pao_basis(reference, other)
    assert summary['nbnds'] == (21, 30) and summary['nkpnts'] == (1728, 195)
    assert summary['kgrid'] == ((12, 12, 12), (18, 18, 18))
    assert summary['bnd'] == (11, 13) and summary['Efermi'] == (6.2, 6.9)
    # a different pseudopotential directory with the same file name is the same pseudopotential
    same = _stub(species=[('Si', '/else/Si.upf')])
    check_same_pao_basis(reference, same)


@pytest.mark.parametrize(
    'change, field',
    [
        ({'nawf': 5}, 'nawf'),
        ({'species': [('Si', 'Si.pbe-tm.UPF')]}, 'pseudopotential'),
        ({'tau': np.array([[0.0, 0.0, 0.0], [2.6, 2.583, 2.583]])}, 'tau'),
        (
            {'a_vectors': np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.01]])},
            'a_vectors',
        ),
        ({'configuration': {'Si': ['3S', '3P', '3D']}}, 'basis configuration'),
        ({'nelec': 10.0}, 'nelec'),
    ],
)
def test_check_same_pao_basis_mismatches(change, field):
    with pytest.raises(ValueError, match=field):
        check_same_pao_basis(_stub(), _stub(**change))


def test_check_same_pao_basis_orbital_order_and_several_mismatches():
    arrays, attributes = _stub()
    arrays['basis'] = [arrays['basis'][1], arrays['basis'][0]] + arrays['basis'][2:]
    with pytest.raises(ValueError, match=r'orbital basis \(orbital 0\)'):
        check_same_pao_basis(_stub(), (arrays, attributes))
    with pytest.raises(ValueError) as info:
        check_same_pao_basis(_stub(), _stub(nawf=6, nspin=2))
    assert 'nawf' in str(info.value) and 'nspin' in str(info.value)


def test_check_orbital_dimensions():
    A = np.zeros((5, NAWF, 8), complex)
    HRs = np.zeros((NAWF, NAWF, 2, 2, 2, 1), complex)
    check_orbital_dimensions(A, HRs, np.zeros((NAWF, 3)))
    with pytest.raises(ValueError, match='same PAO basis'):
        check_orbital_dimensions(A, HRs[:3, :3])
    with pytest.raises(ValueError, match='orbital_positions'):
        check_orbital_dimensions(A, HRs, np.zeros((NAWF + 1, 3)))


_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <dft>
      <functional>PBE</functional>
      %s
    </dft>
  </output>
</qes:espresso>
"""
_DFTU = """<dftU new_format="true">
        <lda_plus_u_kind>2</lda_plus_u_kind>
        <Hubbard_U specie="Si" label="3s">0.000000</Hubbard_U>
        <Hubbard_U specie="Si" label="3p">0.110297</Hubbard_U>
        <Hubbard_V specie1="Si" label1="3p" specie2="Si" label2="3p" index1="1" index2="2">0.061082</Hubbard_V>
        <U_projection_type>ortho-atomic</U_projection_type>
        <Hubbard_ns specie="Si" label="3p" spin="1" index="1" rank="2" dims="3 3">
          0.5 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.5
        </Hubbard_ns>
      </dftU>"""


def test_hubbard_info_from_save(tmp_path):
    (tmp_path / 'data-file-schema.xml').write_text(_XML % _DFTU)
    info = hubbard_info_from_save(tmp_path)
    assert info['kind'] == 'DFT+U+V' and info['projectors'] == 'ortho-atomic'
    tags = [t for t, _, _ in info['terms']]
    assert tags == ['Hubbard_U', 'Hubbard_U', 'Hubbard_V']  # the occupation matrix is skipped
    assert info['terms'][1][2] == pytest.approx(0.110297 * 27.211386245988)
    text = format_hubbard_info(info)
    assert text.startswith(
        'DFT+U+V (ortho-atomic): Hubbard_U Si 3p 3.001 eV; Hubbard_V Si 3p Si 3p 1 2'
    )
    assert 'Hubbard_U Si 3s' not in text  # zero U omitted
    (tmp_path / 'data-file-schema.xml').write_text(_XML % '')
    assert hubbard_info_from_save(tmp_path) is None
    assert format_hubbard_info(None) == 'PBE'


def test_check_direct_dielectric_gap_warns_only_on_mismatch():
    from PAOFLOW.elphon.phonon_assisted_absorption import (
        check_direct_dielectric_gap,
        dielectric_onset_ev,
    )

    omega = np.arange(0.01, 10.0 + 1e-9, 0.01)
    eps2 = np.where(omega >= 2.5, 20.0, 0.0)
    assert dielectric_onset_ev(omega, eps2) == pytest.approx(2.5)
    assert np.isnan(dielectric_onset_ev(omega, np.zeros_like(omega)))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = check_direct_dielectric_gap(
            {'direct_gap_dft_ev': 2.55}, (omega, np.ones_like(omega), eps2)
        )
    assert out['consistent'] and out['onset_ev'] == pytest.approx(2.5)
    with pytest.warns(UserWarning, match='onset 3.20 eV vs the direct gap 2.55 eV'):
        out = check_direct_dielectric_gap(
            {'direct_gap_dft_ev': 2.55},
            (omega, np.ones_like(omega), np.where(omega >= 3.2, 20.0, 0.0)),
        )
    assert not out['consistent']
    # the components are averaged; ``direct_gap_ev`` is the fallback key
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        check_direct_dielectric_gap(
            {'direct_gap_ev': 2.5}, (omega, np.ones((3, omega.size)), np.tile(eps2, (3, 1)))
        )


# --------------------------------------------------------------------------- #
# End to end: the dense pipeline with a Hamiltonian decoupled from the coupling
# --------------------------------------------------------------------------- #
def _grid(n):
    axes = [np.arange(n) / n] * 3
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def _gapped_HRs(ng=2, onsite=(-3.0, -2.5, 3.0, 3.5), hopping=0.2):
    """Simple-cubic nearest-neighbour tight binding with a gap of several eV; PAOFLOW layout."""
    HRs = np.zeros((NAWF, NAWF, ng, ng, ng, 1), dtype=complex)
    HRs[:, :, 0, 0, 0, 0] = np.diag(onsite)
    rng = np.random.default_rng(3)
    mixing = rng.standard_normal((NAWF, NAWF)) * 0.02
    mixing = mixing + mixing.T
    for axis in range(3):
        for sign in (1, -1):
            n = tuple((-sign * np.eye(3, dtype=int)[axis]) % ng)
            HRs[:, :, n[0], n[1], n[2], 0] += hopping * np.eye(NAWF) + mixing
    return HRs


def _synthetic_epw(path, ng, nat=1, seed=5):
    from .synthetic_epw import write_epw_dir

    rng = np.random.default_rng(seed)
    nk = ng**3
    q_cryst = _grid(ng)
    nq = len(q_cryst)
    nm = 3 * nat
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)  # noqa: E731
    epmatq = 0.05 * cplx(NAWF, NAWF, nk, nm, nq)  # Ry/bohr, random but fixed
    spring, floor = 0.3, 0.2
    dynq = np.stack(
        [(2 * spring * np.sum(1 - np.cos(2 * np.pi * q)) + floor) * np.eye(nm) for q in q_cryst],
        axis=-1,
    ).astype(complex)
    et = rng.standard_normal((NAWF, nk))
    write_epw_dir(path, 'tb', q_cryst, np.eye(3), et, dynq, epmatq, NAWF, nk, nat)
    return q_cryst


def _run(epw_dir, HRs, ng=2, **extra):
    from PAOFLOW.elphon.phonon_assisted_absorption import phonon_assisted_absorption_dense_q

    A = np.repeat(np.eye(NAWF, dtype=complex)[:, :, None], ng**3, axis=2)  # PAO basis = Bloch basis
    return phonon_assisted_absorption_dense_q(
        A, HRs, _grid(ng), np.eye(3), np.eye(3), 10.0, 1000.0, str(epw_dir), (ng, ng, ng),
        (ng, ng, ng), masses_amu=[28.0], nelec=4, nk_dense=4, nq_dense=2,
        omega_ev=(0.5, 8.0, 0.5), temps_k=(300.0,), degauss_ev=0.1, fsthick_ev=10.0,
        etas_ev=(0.05,), tau_cryst=np.zeros((1, 3)), species=['X'], **extra,
    )  # fmt: skip


def test_dense_absorption_with_a_decoupled_hamiltonian(tmp_path):
    pytest.importorskip('mpi4py')
    _synthetic_epw(tmp_path, 2)
    HRs = _gapped_HRs()
    out = _run(tmp_path, HRs)
    assert out['indirect_gap_dft_ev'] > 1.5 and np.asarray(out['eps2_indirect']).max() > 0.0
    assert out['scissor_0k_ev'] == 0.0

    # the 0 K scissor is recorded: target gap minus the Hamiltonian gap (Varshni: E0 at 0 K)
    from PAOFLOW.elphon.phonon_assisted_absorption import varshni_gap

    target = _run(tmp_path, HRs, target_gap_ev=varshni_gap(1.2, 4.73e-4, 636.0))
    assert target['scissor_0k_ev'] == pytest.approx(1.2 - out['indirect_gap_dft_ev'])
    assert target['scissor_ev'][0] == pytest.approx(
        varshni_gap(1.2, 4.73e-4, 636.0)(300.0) - out['indirect_gap_dft_ev']
    )

    # a rigid shift of every band (same eigenvectors) changes nothing
    shifted = HRs.copy()
    shifted[:, :, 0, 0, 0, 0] += 0.7 * np.eye(NAWF)
    out_shifted = _run(tmp_path, shifted)
    for key in (
        'eps2_indirect',
        'eps2_direct',
        'alpha_cm',
        'indirect_gap_dft_ev',
        'direct_gap_dft_ev',
    ):
        np.testing.assert_allclose(out_shifted[key], out[key], rtol=1e-8, atol=1e-12)

    # the Hamiltonian passed is the one used: scaled bands scale the gaps and change the spectrum
    out_scaled = _run(tmp_path, 1.1 * HRs)
    assert out_scaled['indirect_gap_dft_ev'] == pytest.approx(1.1 * out['indirect_gap_dft_ev'])
    assert out_scaled['direct_gap_dft_ev'] == pytest.approx(1.1 * out['direct_gap_dft_ev'])
    assert not np.allclose(out_scaled['eps2_direct'], out['eps2_direct'])

    # n(omega) of another run enters alpha only, interpolated on the photon grid
    omega = np.linspace(0.0, 10.0, 101)
    out_n = _run(tmp_path, HRs, refractive_index=(omega, 3.0 + 0.1 * omega))
    n_grid = np.interp(out_n['omega_ev'], omega, 3.0 + 0.1 * omega)
    np.testing.assert_allclose(out_n['refractive_index'], n_grid)
    np.testing.assert_allclose(out_n['alpha_cm'] * n_grid, out['alpha_cm'] * 3.4, rtol=1e-10)
    np.testing.assert_array_equal(out_n['eps2_indirect'], out['eps2_indirect'])

    # None = the EPW constant 3.4
    out_none = _run(tmp_path, HRs, refractive_index=None)
    np.testing.assert_array_equal(out_none['alpha_cm'], out['alpha_cm'])
    assert float(out_none['refractive_index']) == 3.4

    # a Hamiltonian in another basis is refused before any heavy work
    with pytest.raises(ValueError, match='same PAO basis'):
        _run(tmp_path, HRs[:3, :3])
