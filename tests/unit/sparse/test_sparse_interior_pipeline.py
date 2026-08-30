"""Interior window wired through the real pipeline on example01.

The load-bearing test is the DoS parity one: a DoS computed from an interior
window must agree, *inside the window*, with the same DoS computed the normal
from-the-bottom way.  That is the whole claim of the feature -- the states
below ``elo`` were never computed, and the assertion is that they were not
needed for what was plotted.  It only holds with a margin of several smearing
widths between ``elo`` and the plotted range, which is exactly why
``interior_window`` clamps rather than trusting the caller.

Requires the example01 QE data; skipped when absent.
"""

import os

import numpy as np
import pytest

EXAMPLE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
    )
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(EXAMPLE, 'silicon.save')),
    reason='example01 QE data not available',
)

MESH = 4  # 4^3 = 64 k-points: enough for a DoS, cheap enough for a test


def _driver(outdir):
    from PAOFLOW.SparsePAOFLOW import SparsePAOFLOW

    p = SparsePAOFLOW(
        savedir='silicon.save',
        outputdir=outdir,
        smearing='gauss',
        npool=1,
        verbose=False,
        threshold=1.0e-4,
        hk_solver='auto',
    )
    p.read_atomic_proj_QE()
    p.projectability()
    p.pao_hamiltonian()
    return p


@pytest.fixture(scope='module')
def in_example(tmp_path_factory):
    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    yield tmp_path_factory.mktemp('interior')
    os.chdir(cwd)


def test_interior_dos_matches_the_full_solve_inside_the_window(in_example):
    """States below elo were never computed; the DoS in the window is unaffected.

    The reference must compute the FULL spectrum (nev = nawf), not the default
    ``bnd`` projectable subset -- otherwise the reference is itself truncated
    at the top and the two runs differ for a second, unrelated reason.

    The window has to clear the plotted range by more than the adaptive
    smearing reach, so the test measures that reach from the reference run
    rather than assuming it: Yates widths scale as nkpnts^(-1/3) and on a
    coarse mesh they reach several eV, wide enough that no useful window
    exists at all.  That is a real property of the method, which is why
    interior_window carries smear_margin_eV and warns when it is too small.
    """
    plot_lo, plot_hi = -1.0, 1.0
    fine = 12  # the native QE grid; coarser meshes smear too wide to test on
    nawf = 18

    ref = _driver(str(in_example / 'full'))
    ref.energy_window(emin=-12.0, emax=5.0, margin=0.0, nev=nawf)  # every band
    ref.interpolated_hamiltonian(nfft1=fine, nfft2=fine, nfft3=fine)
    ref.dos(emin=plot_lo, emax=plot_hi, ne=200, do_pdos=False)
    dos_full = np.array(ref.data_controller.data_arrays['dosdk'])
    ref_E = np.array(ref.data_controller.data_arrays['E_k'])
    dmax = float(np.max(ref.data_controller.data_arrays['deltakp']))

    elo, ehi = plot_lo - 5.0 * dmax, plot_hi + 5.0 * dmax
    skipped = int((ref_E < elo).sum())
    assert skipped > 0, (
        'test premise: the window must exclude real states (elo=%.2f, spectrum bottom '
        '%.2f) or this proves nothing' % (elo, float(ref_E.min()))
    )

    itr = _driver(str(in_example / 'interior'))
    itr.interior_window(elo, ehi)
    itr.interpolated_hamiltonian(nfft1=fine, nfft2=fine, nfft3=fine)
    itr.dos(emin=plot_lo, emax=plot_hi, ne=200, do_pdos=False)
    dos_int = np.array(itr.data_controller.data_arrays['dosdk'])
    itr_E = np.array(itr.data_controller.data_arrays['E_k'])

    assert np.all(itr_E[np.isfinite(itr_E)] >= elo - 1e-9), 'interior returned states below elo'
    assert dos_int.shape == dos_full.shape
    scale = max(float(np.abs(dos_full).max()), 1e-30)
    dev = float(np.abs(dos_int - dos_full).max())
    assert dev < 1e-5 * scale, (
        'interior DoS deviates by %.3e (peak %.3e) after skipping %d states below %.2f eV'
        % (dev, scale, skipped, elo)
    )


def test_mesh_pads_to_a_rectangular_block_and_sets_bnd(in_example):
    p = _driver(str(in_example / 'pad'))
    p.interior_window(-3.0, 3.0)
    p.interpolated_hamiltonian(nfft1=MESH, nfft2=MESH, nfft3=MESH)
    p.dos(emin=-1.0, emax=1.0, ne=50, do_pdos=False)

    arrays, attr = p.data_controller.data_dicts()
    E_k, velkp, deltakp = arrays['E_k'], arrays['velkp'], arrays['deltakp']
    m = E_k.shape[1]
    assert m > 0 and attr['bnd'] == m
    assert velkp.shape == (E_k.shape[0], 3, m, E_k.shape[2])
    assert deltakp.shape == E_k.shape
    # padding must be inert: far outside the window, zero velocity, nonzero width
    pad = E_k > 3.0 + 1.0
    assert np.all(deltakp > 0.0), 'a zero smearing width would divide by zero'
    if pad.any():
        assert np.allclose(velkp[:, 0, :, :][pad], 0.0)


def test_dos_outside_the_window_is_clamped_not_silently_zero(in_example, capsys):
    p = _driver(str(in_example / 'clamp'))
    p.interior_window(-2.0, 2.0)
    p.interpolated_hamiltonian(nfft1=MESH, nfft2=MESH, nfft3=MESH)
    p.dos(emin=-12.0, emax=2.2, ne=50, do_pdos=False)  # far outside on both sides
    assert 'clamped' in capsys.readouterr().out


def test_disjoint_range_skips_the_property_and_the_run_continues(in_example, capsys):
    p = _driver(str(in_example / 'skip'))
    p.interior_window(-2.0, 2.0)
    p.interpolated_hamiltonian(nfft1=MESH, nfft2=MESH, nfft3=MESH)
    p.dos(emin=5.0, emax=9.0, ne=50, do_pdos=False)  # no overlap at all

    out = capsys.readouterr().out
    assert 'SKIPPED' in out
    assert [prop for prop, _ in p._skipped] == ['dos']
    # the run continues: a supportable range still works afterwards
    p.dos(emin=-1.0, emax=1.0, ne=50, do_pdos=False)
    assert 'dosdk' in p.data_controller.data_arrays


def test_skips_are_restated_at_the_end(in_example, capsys):
    p = _driver(str(in_example / 'report'))
    p.interior_window(-2.0, 2.0)
    p._skip('made-up property', 'for the test')
    capsys.readouterr()
    p._report_skips()
    out = capsys.readouterr().out
    assert 'made-up property' in out and 'SKIPPED' in out


def test_bands_are_nan_padded_under_an_interior_window(in_example):
    p = _driver(str(in_example / 'bands'))
    p.interior_window(-3.0, 3.0)
    p.bands(ibrav=2, nk=20)
    E_k = p.data_controller.data_arrays['E_k']
    finite = np.isfinite(E_k)
    assert finite.any(), 'no states found in the window on the band path'
    assert np.all(E_k[finite] >= -3.0 - 1e-9) and np.all(E_k[finite] <= 3.0 + 1e-9)


def test_the_two_window_modes_are_mutually_exclusive(in_example):
    p = _driver(str(in_example / 'excl'))
    p.interior_window(-1.0, 1.0)
    with pytest.raises(RuntimeError, match='mutually exclusive'):
        p.energy_window(emin=-12.0, emax=2.2, nev=10)

    q = _driver(str(in_example / 'excl2'))
    q.energy_window(emin=-12.0, emax=2.2, nev=10)
    with pytest.raises(RuntimeError, match='mutually exclusive'):
        q.interior_window(-1.0, 1.0)


def test_interior_window_rejects_an_inverted_range(in_example):
    p = _driver(str(in_example / 'bad'))
    with pytest.raises(ValueError, match='need ehi > elo'):
        p.interior_window(1.0, -1.0)


def test_edge_contamination_is_warned_about(in_example, capsys):
    """A plotted range too close to the window edge must not fail silently.

    States below elo are never computed, so their adaptive-smearing tails are
    missing from the DoS near the edge -- a real error of order 1e-3 relative
    on a coarse mesh.  The margin clamp handles the common case; this warning
    catches the case where the measured widths turn out wider than assumed.
    """
    p = _driver(str(in_example / 'edge'))
    p.interior_window(-3.0, 3.0, smear_margin_eV=0.05)  # deliberately too small
    p.interpolated_hamiltonian(nfft1=MESH, nfft2=MESH, nfft3=MESH)
    capsys.readouterr()
    p.dos(emin=-1.0, emax=1.0, ne=50, do_pdos=False)
    out = capsys.readouterr().out
    assert 'contaminated' in out
    assert 'smear_margin_eV' in out
