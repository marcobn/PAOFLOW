"""Unit tests for the pseudo-atom radial solver and basis_gen driver.

These tests pin down the behaviour of
:mod:`PAOFLOW.basis_gen.radial` (the ``solve_radial_channel`` /
``pseudize_shell`` API) and :mod:`PAOFLOW.basis_gen.driver` (the
``generate_basis_for_pseudo`` / ``generate_basis_for_directory``
wrappers) against the two reference UPFs already shipped with
``examples/qe_examples/example16``: Si (scalar-rel ONCV) and Pt
(fully-relativistic ONCV).

The single guardrail underpinning these tests is the PSWFC
reproduction check: for every occupied valence shell stored in
``upf.pswfc``, re-solving the radial equation with frozen-density
V_eff = V_loc + V_H + V_xc must match the stored radial function to
within a few percent L2.  Eigenvalues that come out positive (unbound
in the confining box, expected for ``6P``-like augmentation channels)
are not pinned numerically.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.basis_gen import (
    generate_basis_for_directory,
    generate_basis_for_pseudo,
    pseudize_shell,
    solve_radial_channel,
)
from PAOFLOW.inputs.read_upf import UPF

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SI_UPF = _REPO_ROOT / 'examples/qe_examples/example16/Si_ONCV/Si_ONCV_PBE_sr.UPF'
_PT_UPF = _REPO_ROOT / 'examples/qe_examples/example16/Pt_REL/Pt_ONCV_PBE_fr.upf'


@pytest.fixture(scope='module')
def si_upf():
    if not _SI_UPF.exists():
        pytest.skip(f'reference UPF not available: {_SI_UPF}')
    return UPF(str(_SI_UPF))


@pytest.fixture(scope='module')
def pt_upf():
    if not _PT_UPF.exists():
        pytest.skip(f'reference UPF not available: {_PT_UPF}')
    return UPF(str(_PT_UPF))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l_from_label(lab):
    return 'SPDF'.index(lab[1].upper())


def _interp_to_uniform(r_src, f_src, r_uni):
    """Linear interpolation onto the uniform mesh; zero outside r_src."""
    out = np.zeros_like(r_uni)
    mask = (r_uni >= r_src[0]) & (r_uni <= r_src[-1])
    out[mask] = np.interp(r_uni[mask], r_src, f_src)
    return out


def _l2_relative(u_ref, u_solved, dr):
    """Sign-aligned L2 difference normalised by ||u_ref||."""
    # both are u(r) = r R(r); use the inner product to fix the sign
    s = np.sign(np.sum(u_ref * u_solved))
    if s == 0:
        s = 1.0
    diff = u_ref - s * u_solved
    return float(np.sqrt(np.sum(diff * diff) * dr) /
                 np.sqrt(np.sum(u_ref * u_ref) * dr))


def _pswfc_u_on_uniform(upf, label, r_uni, j=None):
    """Pull the stored PSWFC for ``label`` (matched by (n, L) and j) onto r_uni.

    Returns u(r) = r * R(r), L2-normalised on the uniform mesh.
    """
    target_label = label.upper()
    candidates = []
    for i, c in enumerate(upf.pswfc):
        if c['label'].upper() != target_label:
            continue
        if j is not None and i < len(upf.jchia):
            if abs(float(upf.jchia[i]) - float(j)) > 1e-6:
                continue
        candidates.append(i)
    if not candidates:
        pytest.skip(f'PSWFC entry not found for label={label!r} j={j}')
    i = candidates[0]
    R_log = upf.pswfc[i]['wfc']  # already u = r R in UPF convention
    u = _interp_to_uniform(upf.r, R_log, r_uni)
    dr = r_uni[1] - r_uni[0]
    nrm = np.sqrt(np.sum(u * u) * dr)
    if nrm > 0:
        u = u / nrm
    return u


# ---------------------------------------------------------------------------
# Solver: API contract
# ---------------------------------------------------------------------------


def test_solve_radial_channel_returns_normalised_states(si_upf):
    eps, U, r = solve_radial_channel(si_upf, l=0, n_points=1500, n_states=4)
    dr = r[1] - r[0]
    # Sorted ascending.
    assert np.all(np.diff(eps) >= -1e-10)
    # u normalised on the uniform mesh.
    for n in range(U.shape[0]):
        assert np.isclose(np.sum(U[n] * U[n]) * dr, 1.0, atol=1e-8)
    # Uniform mesh, interior only.
    assert r[0] > 0.0
    assert np.allclose(np.diff(r), dr, atol=1e-12)


def test_pseudize_shell_returns_eigenstate_with_correct_eigenvalue(si_upf):
    r, u, eps = pseudize_shell(si_upf, n=3, l=0, n_points=1500)
    eps_all, U, r_all = solve_radial_channel(si_upf, l=0, n_points=1500, n_states=4)
    # pseudize_shell selects the lowest-rank S state since Si PSWFC start at n=3.
    assert np.allclose(r, r_all)
    assert np.isclose(eps, eps_all[0], atol=1e-10)
    # u is u_0 up to sign.
    s = np.sign(np.sum(u * U[0]))
    assert np.allclose(u, s * U[0])


def test_pseudize_shell_rejects_below_lowest_n(si_upf):
    # Si PSWFC start at n=3 for L=0; n=2 must raise.
    with pytest.raises(ValueError, match='below lowest'):
        pseudize_shell(si_upf, n=2, l=0)


# ---------------------------------------------------------------------------
# PSWFC reproduction (validation guardrail)
# ---------------------------------------------------------------------------


# Tolerance taken from validated standalone runs (L2 errors 3-7%).
_PSWFC_L2_TOL = 0.10


@pytest.mark.unit
@pytest.mark.parametrize('label', ['3S', '3P'])
def test_si_pswfc_reproduction(si_upf, label):
    n = int(label[0])
    l = _l_from_label(label)
    r, u_solved, eps = pseudize_shell(si_upf, n=n, l=l, n_points=2000)
    u_ref = _pswfc_u_on_uniform(si_upf, label, r)
    err = _l2_relative(u_ref, u_solved, r[1] - r[0])
    assert err < _PSWFC_L2_TOL, f'Si {label} L2 error {err:.3f} > {_PSWFC_L2_TOL}'
    # Bound state in the box.
    assert eps < 0.0, f'Si {label} eigenvalue should be bound (got {eps:+.3f} Ha)'


@pytest.mark.unit
@pytest.mark.parametrize(
    'label,j',
    [
        ('5S', 0.5),
        ('5P', 0.5),
        ('5P', 1.5),
        ('5D', 1.5),
        ('5D', 2.5),
        ('6S', 0.5),
    ],
)
def test_pt_pswfc_reproduction(pt_upf, label, j):
    n = int(label[0])
    l = _l_from_label(label)
    r, u_solved, eps = pseudize_shell(pt_upf, n=n, l=l, j=j, n_points=2000)
    u_ref = _pswfc_u_on_uniform(pt_upf, label, r, j=j)
    err = _l2_relative(u_ref, u_solved, r[1] - r[0])
    assert err < _PSWFC_L2_TOL, f'Pt {label} j={j} L2 error {err:.3f} > {_PSWFC_L2_TOL}'
    assert eps < 0.0, f'Pt {label} j={j} eigenvalue should be bound (got {eps:+.3f} Ha)'


def test_pt_spin_orbit_splitting_signs(pt_upf):
    """j = l - 1/2 lies deeper than j = l + 1/2 for both 5P and 5D."""
    _, _, e_p_minus = pseudize_shell(pt_upf, n=5, l=1, j=0.5)
    _, _, e_p_plus = pseudize_shell(pt_upf, n=5, l=1, j=1.5)
    _, _, e_d_minus = pseudize_shell(pt_upf, n=5, l=2, j=1.5)
    _, _, e_d_plus = pseudize_shell(pt_upf, n=5, l=2, j=2.5)
    assert e_p_minus < e_p_plus, '5P_1/2 must be below 5P_3/2'
    assert e_d_minus < e_d_plus, '5D_3/2 must be below 5D_5/2'


# ---------------------------------------------------------------------------
# Driver: file outputs
# ---------------------------------------------------------------------------


def _read_two_col(path):
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def test_driver_writes_minimal_set_for_si(tmp_path, si_upf):
    written = generate_basis_for_pseudo(
        str(_SI_UPF), str(tmp_path), preset='minimal',
    )
    files = sorted(os.path.basename(p) for p in written)
    assert files == ['3P.dat', '3S.dat']
    # Round-trip file format.
    r, wfc = _read_two_col(tmp_path / 'Si' / '3S.dat')
    dr = r[1] - r[0]
    assert np.allclose(np.diff(r), dr, atol=1e-10)
    assert np.isclose(np.sum(wfc * wfc) * dr, 1.0, atol=1e-6)


def test_driver_extended_si_includes_augmentation(tmp_path):
    written = generate_basis_for_pseudo(
        str(_SI_UPF), str(tmp_path), preset='extended', verbose=False,
    )
    names = {os.path.basename(p) for p in written}
    # Minimal (3S, 3P) plus the extended augmentation rule (rows of nS/nP, plus D).
    for required in ('3S.dat', '3P.dat', '3D.dat', '4S.dat', '4P.dat'):
        assert required in names, f'missing {required} from extended Si basis'


def test_driver_pt_so_emits_j_resolved_and_average(tmp_path):
    written = generate_basis_for_pseudo(
        str(_PT_UPF), str(tmp_path), preset='standard',
    )
    names = {os.path.basename(p) for p in written}
    # SO shells: j-resolved + j-averaged (l > 0).
    for required in (
        '5S.dat',
        '5P.dat', '5P_j1.dat', '5P_j3.dat',
        '5D.dat', '5D_j3.dat', '5D_j5.dat',
        '6S.dat',
    ):
        assert required in names, f'missing {required} from Pt SO standard basis'

    # j-averaged file is the degeneracy-weighted mean of the two j components.
    r_m, u_m = _read_two_col(tmp_path / 'Pt' / '5P_j1.dat')
    _, u_p = _read_two_col(tmp_path / 'Pt' / '5P_j3.dat')
    _, u_avg = _read_two_col(tmp_path / 'Pt' / '5P.dat')
    # weights: Jm = 2(1 - 1/2)+1 = 2, Jp = 2(1 + 1/2)+1 = 4 => (2 u_m + 4 u_p)/6
    expected = (2.0 * u_m + 4.0 * u_p) / 6.0
    assert np.allclose(u_avg, expected, atol=1e-12)


def test_driver_overwrite_false_skips_existing(tmp_path):
    out = str(tmp_path)
    first = generate_basis_for_pseudo(str(_SI_UPF), out, preset='minimal')
    assert first  # something was written
    second = generate_basis_for_pseudo(
        str(_SI_UPF), out, preset='minimal', overwrite=False,
    )
    assert second == [], 'overwrite=False must skip all existing files'


def test_driver_unknown_preset_raises(tmp_path):
    with pytest.raises(ValueError, match='unknown preset'):
        generate_basis_for_pseudo(str(_SI_UPF), str(tmp_path), preset='bogus')


def test_directory_driver_collects_all_upfs(tmp_path):
    """``generate_basis_for_directory`` walks every UPF in a folder."""
    # Use the Pt_REL directory (single UPF) for a stable smoke test.
    pseudo_dir = _PT_UPF.parent
    out = generate_basis_for_directory(str(pseudo_dir), str(tmp_path), preset='minimal')
    assert 'Pt' in out
    assert all(os.path.exists(p) for p in out['Pt'])


def test_directory_driver_errors_when_no_upf(tmp_path):
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        generate_basis_for_directory(str(empty), str(tmp_path / 'out'))
