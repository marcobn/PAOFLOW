"""The sparse engine uses only well-projected states.

``pao_hamiltonian`` places the non-projectable part of the PAO space at the
shift ``attr['shift']``; those states are not bands.  The engine drops every
state from ``shift - shift_margin`` up at every k-point, and pulls windows and
property ranges that reach past it back below it with a warning.  On
example01 the shift is 7.16 eV above E_F, and nine of the 18 states sit there.

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


def _sparse(outdir, nfft=8, **config):
    from PAOFLOW.PAOFLOW import PAOFLOW

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        p = PAOFLOW(
            savedir='silicon.save',
            outputdir=str(outdir),
            npool=1,
            verbose=False,
            sparse=True,
            sparse_config=dict({'hopping_threshold': 0.0}, **config),
        )
        p.read_atomic_proj_QE()
        p.projectability(pthr=0.95)
        p.pao_hamiltonian()
        p.interpolated_hamiltonian(nfft1=nfft, nfft2=nfft, nfft3=nfft)
    finally:
        os.chdir(cwd)
    return p


def _top(p):
    from PAOFLOW.sparse.config import SHIFT_MARGIN

    return p.data_controller.data_attributes['shift'] - SHIFT_MARGIN


def test_energy_window_is_lowered_to_the_projectable_top(tmp_path, capsys):
    p = _sparse(tmp_path, energy_window={'emax': 20.0})
    p.sparse._ensure_window()
    assert 'energy_window: emax lowered from 20.000' in capsys.readouterr().out
    assert p.sparse._window.emax == pytest.approx(_top(p))
    assert p.sparse._window.ehi == pytest.approx(_top(p))


def test_window_margin_is_cut_silently(tmp_path, capsys):
    """An emax inside the range keeps its value; only the margin past the
    top is cut, which needs no warning."""
    top = 7.16 - 0.2
    p = _sparse(tmp_path, energy_window={'emax': top - 0.5, 'margin': 1.0})
    p.sparse._ensure_window()
    assert 'lowered' not in capsys.readouterr().out
    assert p.sparse._window.emax == pytest.approx(top - 0.5, abs=0.01)
    assert p.sparse._window.ehi == pytest.approx(_top(p))


def test_interior_window_is_lowered_or_refused(tmp_path, capsys):
    p = _sparse(tmp_path / 'a', interior_window={'elo': -1.0, 'ehi': 10.0})
    p.sparse._ensure_window()
    assert 'interior_window: ehi lowered from 10.000' in capsys.readouterr().out
    assert p.sparse._interior == (-1.0, pytest.approx(_top(p)))
    q = _sparse(tmp_path / 'b', interior_window={'elo': 8.0, 'ehi': 10.0})
    with pytest.raises(ValueError, match='above the well-projected'):
        q.sparse._ensure_window()


def test_property_range_is_lowered(tmp_path, capsys):
    p = _sparse(tmp_path)
    p.dos(emin=-12.0, emax=9.0, ne=200)
    assert 'sparse dos: emax lowered from 9.000' in capsys.readouterr().out
    E = np.loadtxt(os.path.join(p.data_controller.data_attributes['opath'], 'dosdk_0.dat'))[:, 0]
    assert E.max() == pytest.approx(_top(p))


def test_placeholder_states_are_dropped_at_every_k(tmp_path):
    """Solving all 18 states keeps only those below the top; the rest are
    the padding an interior window uses."""
    from PAOFLOW.sparse.mesh import PAD_DELTA, PAD_ENERGY_OFFSET

    p = _sparse(tmp_path, energy_window={'emax': 2.2, 'nev': 18})
    p.dos(emin=-12.0, emax=2.0, ne=100)
    arrays = p.data_controller.data_arrays
    E, delta = arrays['E_k'][..., 0], arrays['deltakp'][..., 0]
    top = _top(p)
    padded = E > top + 0.5 * PAD_ENERGY_OFFSET
    assert (E[~padded] < top).all()
    # the nine states at the shift go at every k-point, and some real ones near it
    assert (padded.sum(axis=1) >= 9).all()
    np.testing.assert_array_equal(delta[padded], PAD_DELTA)
    np.testing.assert_array_equal(arrays['velkp'][:, :, :, 0].transpose(0, 2, 1)[padded], 0.0)


def test_dielectric_ignores_the_placeholder_states(tmp_path):
    """Before, solving more states pulled the placeholders into the
    Kubo-Greenwood sum and raised eps' by ~20% on example01.  Now the sum
    sees the same states whatever the window pads to."""
    eps = []
    for nev in (12, 18):
        p = _sparse(tmp_path / str(nev), energy_window={'emax': 6.0, 'nev': nev})
        with p.sparse.fused():
            p.dielectric_tensor(emin=0.0, emax=6.0, ne=61, d_tensor='diag')
        opath = p.data_controller.data_attributes['opath']
        eps.append(np.loadtxt(os.path.join(opath, 'epsr_xx.dat'))[:, 1])
    np.testing.assert_allclose(eps[1], eps[0], rtol=1e-10, atol=1e-10)


def test_shift_margin_must_not_be_negative():
    from PAOFLOW.sparse.config import SparseConfig

    with pytest.raises(ValueError, match='shift_margin'):
        SparseConfig.parse({'shift_margin': -0.1})
