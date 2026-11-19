"""Default of the position dipoles in gradient_and_momenta (15 November 2026).

On for internal projections of a QE run (``arrays['atomic_basis']``, written by ``projections``),
off for projwfc, VASP and models; an automatic default that cannot build the dipoles warns and leaves
``dHksp`` unchanged, an explicit request raises.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _model_run(tmp_path):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import honeycomb_model

    p = PAOFLOW.PAOFLOW(model=honeycomb_model(t=-1.0, t2=0.1, phi=np.pi / 2), outputdir=str(tmp_path), verbose=False)  # fmt: skip
    p.interpolated_hamiltonian(nfft1=6, nfft2=6, nfft3=1)
    p.pao_eigh()
    return p


def _recorder(p, result=True):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return result

    p.position_dipole_correction = fake
    return calls


def test_dipoles_are_off_without_internal_projections(tmp_path):
    p = _model_run(tmp_path)
    calls = _recorder(p)
    p.gradient_and_momenta()
    assert calls == []


def test_dipoles_are_on_by_default_for_internal_projections(tmp_path):
    p = _model_run(tmp_path)
    p.data_controller.data_arrays['atomic_basis'] = [{'atom': 0}]
    p.data_controller.data_attributes['dft'] = 'QE'
    calls = _recorder(p)
    p.gradient_and_momenta()
    assert len(calls) == 1 and calls[0]['strict'] is False and calls[0]['level'] == 'full'


@pytest.mark.parametrize('setting', ['argument', 'attribute', 'vasp'])
def test_the_default_can_be_switched_off(tmp_path, setting):
    p = _model_run(tmp_path)
    arry, attr = p.data_controller.data_dicts()
    arry['atomic_basis'] = [{'atom': 0}]
    attr['dft'] = 'QE'
    calls = _recorder(p)
    if setting == 'argument':
        p.gradient_and_momenta(position_dipoles=False)
    elif setting == 'attribute':
        attr['position_dipoles'] = False
        p.gradient_and_momenta()
    else:
        attr['dft'] = 'VASP'
        p.gradient_and_momenta()
    assert calls == []


def test_explicit_request_is_strict(tmp_path):
    p = _model_run(tmp_path)
    calls = _recorder(p)
    p.gradient_and_momenta(position_dipoles=True)
    assert len(calls) == 1 and calls[0]['strict'] is True


def test_automatic_default_falls_back_when_the_basis_has_no_tables(tmp_path, capsys):
    """A basis record without radial tables: the default warns and keeps dHksp; True raises."""
    reference = _model_run(tmp_path / 'ref')
    reference.gradient_and_momenta()
    p = _model_run(tmp_path / 'auto')
    arry, attr = p.data_controller.data_dicts()
    arry['atomic_basis'] = [{'atom': 0}]
    attr['dft'] = 'QE'
    p.gradient_and_momenta()
    assert 'position dipoles skipped' in capsys.readouterr().out
    assert np.array_equal(arry['dHksp'], reference.data_controller.data_arrays['dHksp'])
    strict = _model_run(tmp_path / 'strict')
    strict.data_controller.data_arrays['atomic_basis'] = [{'atom': 0}]
    with pytest.raises(Exception):
        strict.gradient_and_momenta(position_dipoles=True)
