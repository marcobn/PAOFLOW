"""Hellmann-Feynman & Hermiticity invariants for the PAOFLOW velocity operator.

These checks pin down properties of ``pksp`` (the band-resolved momentum
matrix elements built by ``do_momentum``) that **must** survive once the
non-local pseudopotential velocity correction is later inserted between
``gradient()`` and ``momenta()`` (see ``TODOs/nonlocal_velocity_correction.md``):

* ``pksp`` is Hermitian in the band indices at every k-point;
* the diagonal ``pksp[k, l, n, n]`` equals the band group velocity
  :math:`\\partial E_n / \\partial k_l` (Hellmann-Feynman) — the upcoming
  correction is purely off-diagonal in the band index and must not touch
  this;
* time-reversal symmetry: for a spinless real Hamiltonian (cubium),
  :math:`p_{nn}(-\\mathbf{k}) = -p_{nn}(\\mathbf{k})`.

The test uses the built-in ``cubium`` model (single-band simple-cubic
nearest-neighbor TB), which has an analytic dispersion

.. math::
    E(\\mathbf{k}) = -E_F + 2 t \\sum_l \\cos(k_l a)

and therefore an analytic group velocity
:math:`-2 t a \\sin(k_l a)` to compare against (in eV·bohr).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mpi4py')


@pytest.fixture(scope='module')
def cubium_velocity(tmp_path_factory):
    """Build the cubium TB model and run gradient + momenta on an 8^3 FFT grid.

    Returns a dict with the slices needed by every assertion below.  Scoped
    to module so the (relatively expensive) PAOFLOW pipeline runs once per
    test session.
    """
    from PAOFLOW import PAOFLOW

    workdir = tmp_path_factory.mktemp('cubium_hf')
    cwd = Path.cwd()
    os.chdir(workdir)
    try:
        t = 1.0
        model = {'label': 'cubium', 't': t}
        paoflow = PAOFLOW.PAOFLOW(
            model=model,
            workpath=str(workdir),
            outputdir='output',
            verbose=False,
        )
        paoflow.interpolated_hamiltonian(nfft1=8, nfft2=8, nfft3=8)
        paoflow.pao_eigh()

        # cubium is a 1-orbital basis with the orbital at tau=(0,0,0): the
        # orbital-position correction matrix Dnm vanishes.  The cubium model
        # builder does not populate it, so we inject zeros to satisfy the
        # do_gradient API (no physics change).
        arrays_pre, attr_pre = paoflow.data_controller.data_dicts()
        if 'Dnm' not in arrays_pre:
            arrays_pre['Dnm'] = np.zeros((attr_pre['nawf'], attr_pre['nawf'], 3))

        paoflow.gradient_and_momenta()

        arrays, attr = paoflow.data_controller.data_dicts()
        # Single-rank / single-pool assumption: local k-block == global grid.
        nktot_local = arrays['pksp'].shape[0]
        assert nktot_local == arrays['kgrid'].shape[1], (
            'cubium_velocity fixture assumes 1 MPI rank / 1 pool '
            f'(got nktot_local={nktot_local}, kgrid={arrays["kgrid"].shape[1]}).'
        )

        return {
            'pksp': arrays['pksp'].copy(),
            'E_k': arrays['E_k'].copy(),
            'kgrid': arrays['kgrid'].copy(),
            'alat': attr['alat'],
            't': t,
            'Efermi': attr['Efermi'],
            'nk1': attr['nk1'],
            'nk2': attr['nk2'],
            'nk3': attr['nk3'],
        }
    finally:
        os.chdir(cwd)


def test_pksp_hermitian_in_band_indices(cubium_velocity):
    """``pksp[k, l, :, :]`` is Hermitian for every (k, l, spin)."""
    pksp = cubium_velocity['pksp']  # (nktot, 3, nawf, nawf, nspin)
    nktot, three, nawf, _, nspin = pksp.shape

    for l in range(three):
        for s in range(nspin):
            block = pksp[:, l, :, :, s]
            herm_defect = np.abs(block - np.conj(np.swapaxes(block, 1, 2))).max()
            assert herm_defect < 1e-12, (
                f'pksp not Hermitian for direction l={l}, spin={s}: '
                f'max |p - p^dagger| = {herm_defect:.3e}'
            )


def test_diag_pksp_matches_analytical_group_velocity(cubium_velocity):
    """Hellmann-Feynman: diag(pksp) == -dE/dk for the cubium dispersion.

    Cubium has only nearest-neighbor hopping, so on any FFT grid the
    discrete FFT gradient is exact (only ±1 modes contribute).  We can
    therefore compare against the analytic form to machine precision.

    Sign convention
    ---------------
    PAOFLOW stores ``pksp`` as :math:`\\langle n|dH/d(-k)|m\\rangle`
    (i.e. the opposite sign of the conventional momentum operator).  This
    can be checked in ``writers/write4bt2.py`` where the BoltzTraP2 export
    flips the sign back: ``mommat = -np.real(pksp)``.  Downstream uses in
    ``do_epsilon`` and ``do_Hall`` are quadratic in ``pksp``, so the sign
    is invisible to derived observables; but the invariant is part of
    PAOFLOW's internal contract and must be preserved when the non-local
    velocity correction is added.

    For cubium :math:`E(\\mathbf{k}) = -E_F + 2 t \\sum_l \\cos(k_l a)` so
    :math:`-dE/dk_l = +2 t a \\sin(k_l a)`.
    """
    d = cubium_velocity
    pksp = d['pksp']
    kgrid = d['kgrid']  # (3, nktot), in units of 2π/alat
    alat = d['alat']  # bohr
    t = d['t']

    # k_l · a (the lattice vector length equals alat for cubium)
    k_dot_a = 2.0 * np.pi * kgrid  # (3, nktot)
    minus_dE_dk = +2.0 * t * alat * np.sin(k_dot_a)  # (3, nktot) eV·bohr

    diag = np.real(pksp[:, :, 0, 0, 0]).T  # (3, nktot)

    np.testing.assert_allclose(diag, minus_dE_dk, atol=1e-10, rtol=1e-10)


def test_pksp_diagonal_is_real(cubium_velocity):
    """Group velocity is observable → diag(pksp) must be real."""
    pksp = cubium_velocity['pksp']
    nawf = pksp.shape[2]
    for n in range(nawf):
        imag_max = np.abs(np.imag(pksp[:, :, n, n, :])).max()
        assert imag_max < 1e-12, f'diag(pksp)[n={n}] has imaginary part {imag_max:.3e}'


def test_pksp_diagonal_obeys_time_reversal(cubium_velocity):
    """For a real, spinless TB Hamiltonian, v_n(-k) = -v_n(k)."""
    d = cubium_velocity
    pksp = d['pksp']
    kgrid = d['kgrid']  # (3, nktot) in 2π/alat units

    # Map each k to its -k partner on the periodic grid.  k is folded to
    # [-0.5, 0.5); the partner is found by (-k) mod 1 → same folding.
    nk_axis = np.array([d['nk1'], d['nk2'], d['nk3']], dtype=int)
    idx = np.rint(kgrid.T * nk_axis).astype(int) % nk_axis  # (nktot, 3)
    flat = idx[:, 2] + idx[:, 1] * nk_axis[2] + idx[:, 0] * nk_axis[1] * nk_axis[2]
    # Sanity: flat should be the identity permutation for the +k side.
    assert np.array_equal(flat, np.arange(flat.size))

    neg_idx = (-np.rint(kgrid.T * nk_axis).astype(int)) % nk_axis
    neg_flat = neg_idx[:, 2] + neg_idx[:, 1] * nk_axis[2] + neg_idx[:, 0] * nk_axis[1] * nk_axis[2]

    diag = np.real(pksp[:, :, 0, 0, 0])  # (nktot, 3)
    diag_at_minus_k = diag[neg_flat]  # (nktot, 3)

    np.testing.assert_allclose(diag_at_minus_k, -diag, atol=1e-10, rtol=1e-10)
