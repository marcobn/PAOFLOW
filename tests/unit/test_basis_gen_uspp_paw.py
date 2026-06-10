"""Unit tests for ultrasoft / PAW support in PAOFLOW.basis_gen.

Two reference UPFs already shipped with the repository are used:

* USPP scalar-rel : ``examples/qe_examples/example03/Pt.pz-n-rrkjus_psl.0.1.UPF``
* PAW scalar-rel  : ``examples/acbn0_examples/GaAs/As.pbe-n-kjpaw_psl.1.0.0.UPF``

These tests pin down:

1. The augmentation parser (``upf.qqq``, ``upf.has_augmentation``) and
   the NLCC parser (``upf.rho_atc``) populate the expected fields.
2. The solver builds a positive-definite overlap operator ``S`` and
   the resulting u(r) satisfies ``<u|S|u> dr = 1``.
3. For occupied PSWFC channels the lowest-rank generalized eigenstate
   reproduces the stored PSWFC overlap to within a relaxed tolerance
   (USPP/PAW solver is best-effort -- see ``pseudize_shell`` docstring).
4. The end-to-end driver writes a ``minimal`` basis on disk for the
   PAW pseudo without raising.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.basis_gen import (
    generate_basis_for_pseudo,
    pseudize_shell,
    solve_radial_channel,
)
from PAOFLOW.basis_gen.radial import _default_j, _interp_to_uniform, _select_projectors
from PAOFLOW.inputs.read_upf import UPF

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PT_USPP = _REPO_ROOT / 'examples/qe_examples/example03/Pt.pz-n-rrkjus_psl.0.1.UPF'
_AS_PAW = _REPO_ROOT / 'examples/acbn0_examples/GaAs/As.pbe-n-kjpaw_psl.1.0.0.UPF'


@pytest.fixture(scope='module')
def pt_uspp():
    if not _PT_USPP.exists():
        pytest.skip(f'reference UPF not available: {_PT_USPP}')
    return UPF(str(_PT_USPP))


@pytest.fixture(scope='module')
def as_paw():
    if not _AS_PAW.exists():
        pytest.skip(f'reference UPF not available: {_AS_PAW}')
    return UPF(str(_AS_PAW))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_pt_uspp_parser_populates_augmentation_and_nlcc(pt_uspp):
    assert pt_uspp.ptype == 'USPP'
    assert pt_uspp.has_augmentation is True
    assert pt_uspp.qqq is not None
    assert pt_uspp.qqq.shape == (pt_uspp.nproj, pt_uspp.nproj)
    assert np.max(np.abs(pt_uspp.qqq - pt_uspp.qqq.T)) < 1e-12
    # PP_NLCC table is present in this UPF even though core_correction is false.
    assert pt_uspp.rho_atc is not None
    assert pt_uspp.rho_atc.shape == pt_uspp.r.shape


def test_as_paw_parser_populates_augmentation_and_nlcc(as_paw):
    assert as_paw.ptype == 'PAW'
    assert as_paw.has_augmentation is True
    assert as_paw.qqq is not None
    assert as_paw.qqq.shape == (as_paw.nproj, as_paw.nproj)
    assert as_paw.nlcc is True
    assert as_paw.rho_atc is not None
    assert as_paw.rho_atc.shape == as_paw.r.shape
    # Core density should be non-negative and decay at large r.
    assert np.all(as_paw.rho_atc >= -1e-10)
    assert as_paw.rho_atc[-1] < as_paw.rho_atc[0]


# ---------------------------------------------------------------------------
# Solver: augmentation overlap operator
# ---------------------------------------------------------------------------


def _build_S(upf, l, n_points=1600):
    """Replicate solve_radial_channel's S construction for inspection."""
    r_box = float(min(upf.r[-1], 10.0))
    dr = r_box / n_points
    r = np.arange(1, n_points) * dr
    pidx = _select_projectors(upf, l, _default_j(upf, l, None))
    if not pidx:
        return r, dr, None
    A = np.zeros((r.size, len(pidx)))
    for c, ip in enumerate(pidx):
        b = upf.beta[ip]
        A[:, c] = _interp_to_uniform(upf.r, b['wfc'], r, cutoff_index=b.get('cutoff_index'))
    Q = upf.qqq[np.ix_(pidx, pidx)]
    S = np.eye(r.size) + dr * (A @ Q @ A.T)
    return r, dr, S


@pytest.mark.parametrize('l', [0, 1, 2])
def test_pt_uspp_overlap_is_positive_definite(pt_uspp, l):
    _, _, S = _build_S(pt_uspp, l)
    assert S is not None, f'no projectors for l={l}'
    ev = np.linalg.eigvalsh(S)
    assert ev.min() > 0.0, f'l={l}: S not positive definite (min eig {ev.min():.3e})'


@pytest.mark.parametrize('l', [0, 1])
def test_as_paw_overlap_is_positive_definite(as_paw, l):
    _, _, S = _build_S(as_paw, l)
    assert S is not None, f'no projectors for l={l}'
    ev = np.linalg.eigvalsh(S)
    assert ev.min() > 0.0, f'l={l}: S not positive definite (min eig {ev.min():.3e})'


def test_solver_uses_s_normalisation_for_uspp(pt_uspp):
    """For USPP the returned u(r) must satisfy <u|S|u> dr = 1, not <u|u> dr."""
    eps, U, r = solve_radial_channel(pt_uspp, l=0, n_points=1600, n_states=3)
    dr = r[1] - r[0]
    _, _, S = _build_S(pt_uspp, l=0, n_points=1600)
    for n in range(U.shape[0]):
        s_norm = float(U[n] @ S @ U[n]) * dr
        assert np.isclose(s_norm, 1.0, atol=1e-8), f'state {n} S-norm = {s_norm:.6f}'


# ---------------------------------------------------------------------------
# PSWFC reproduction (relaxed tolerance for USPP/PAW)
# ---------------------------------------------------------------------------


def _pswfc_u(upf, label, r_uni):
    target = label.upper()
    for c in upf.pswfc:
        if c['label'].upper() == target:
            u = np.interp(r_uni, upf.r, c['wfc'], left=0.0, right=0.0)
            return u
    raise LookupError(label)


def _s_norm_inner(u_a, u_b, S, dr):
    return float(u_a @ S @ u_b) * dr


@pytest.mark.parametrize('label,n,l', [('6S', 6, 0), ('6P', 6, 1)])
def test_pt_uspp_lowest_state_overlaps_pswfc(pt_uspp, label, n, l):
    """The L2 overlap with the stored PSWFC (renormalised on the box mesh) > 0.95."""
    r, u, _ = pseudize_shell(pt_uspp, n=n, l=l, n_points=1800)
    dr = r[1] - r[0]
    u_ref = _pswfc_u(pt_uspp, label, r)
    u_ref = u_ref / np.sqrt(np.sum(u_ref * u_ref) * dr)
    ov = abs(np.sum(u_ref * u) * dr)
    assert ov > 0.95, f'Pt {label}: overlap = {ov:.3f}'


@pytest.mark.parametrize('label,n,l', [('4S', 4, 0), ('4P', 4, 1)])
def test_as_paw_lowest_state_overlaps_pswfc(as_paw, label, n, l):
    """As PAW (no semi-core in occupied channels) yields a clean lowest state."""
    r, u, _ = pseudize_shell(as_paw, n=n, l=l, n_points=1800)
    dr = r[1] - r[0]
    u_ref = _pswfc_u(as_paw, label, r)
    u_ref = u_ref / np.sqrt(np.sum(u_ref * u_ref) * dr)
    ov = abs(np.sum(u_ref * u) * dr)
    # Relaxed tolerance: PAW box-confined frozen-rho solver is best-effort.
    assert ov > 0.80, f'As {label}: overlap = {ov:.3f}'


# ---------------------------------------------------------------------------
# Driver end-to-end
# ---------------------------------------------------------------------------


def test_driver_writes_paw_minimal_basis(tmp_path):
    if not _AS_PAW.exists():
        pytest.skip(f'reference UPF not available: {_AS_PAW}')
    written = generate_basis_for_pseudo(str(_AS_PAW), str(tmp_path), preset='minimal')
    names = sorted(os.path.basename(p) for p in written)
    assert names == ['4P.dat', '4S.dat']
    arr = np.loadtxt(tmp_path / 'As' / '4S.dat')
    assert arr.shape[1] == 2
    dr = arr[1, 0] - arr[0, 0]
    assert np.allclose(np.diff(arr[:, 0]), dr, atol=1e-10)


def test_driver_runs_for_uspp(tmp_path):
    if not _PT_USPP.exists():
        pytest.skip(f'reference UPF not available: {_PT_USPP}')
    written = generate_basis_for_pseudo(str(_PT_USPP), str(tmp_path), preset='minimal')
    names = {os.path.basename(p) for p in written}
    # Pt USPP PSWFC includes 5D, 6S (scalar-rel here).
    for required in ('5D.dat', '6S.dat'):
        assert required in names, f'missing {required} from USPP Pt basis'
