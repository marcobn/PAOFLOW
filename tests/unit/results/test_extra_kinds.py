"""Result kinds of the run properties added for the example cases (doped transport, Z2, mirror Chern)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from PAOFLOW.results import discover, load

pytestmark = pytest.mark.unit


@pytest.fixture
def folder(tmp_path):
    rows = np.array([[300.0, 0.68, 1.0e4, 1.1e4, 1.2e4, 3.0e-2, -200.0, -210.0, -220.0],
                     [400.0, 0.66, 9.0e3, 9.5e3, 9.9e3, 3.4e-2, -250.0, -260.0, -270.0]])  # fmt: skip
    header = 'T (K), mu (eV), sigma_xx, sigma_yy, sigma_zz, rho (Ohm cm), S_xx, S_yy, S_zz'
    np.savetxt(tmp_path / 'doped_transport_n1e+19.dat', rows, header=header)
    wcc = np.column_stack([np.linspace(0, 1, 5), np.linspace(0.1, 0.9, 5), np.full(5, 0.5)])
    np.savetxt(tmp_path / 'z2_wcc_k3_0.dat', wcc, header='t, centres; k3 = 0, Z2 = 1')
    (tmp_path / 'mirror_chern.json').write_text(json.dumps({'C_M': 2, 'nu': 0}))
    return tmp_path


def test_doped_transport_quantities(folder):
    ref = next(r for r in discover(str(folder)) if r.kind == 'doped_transport')
    assert ref.title == 'Doped semiconductor n1e+19'
    rho = load(ref, str(folder), {})
    assert rho.y_label == 'Resistivity ρ (Ω cm)' and len(rho.series) == 1
    np.testing.assert_allclose(rho.series[0].y, [3.0e-2, 3.4e-2])
    seebeck = load(ref, str(folder), {'quantity': 'Seebeck coefficient S'})
    assert [s.name for s in seebeck.series] == ['S_xx', 'S_yy', 'S_zz']
    assert seebeck.options['quantity']['choices'][-1] == 'Chemical potential μ'


def test_topology_files(folder):
    refs = {r.kind: r for r in discover(str(folder))}
    wcc = load(refs['z2_wcc'], str(folder), {})
    assert refs['z2_wcc'].title == 'Z2: Wannier charge centres, k3 = 0' and len(wcc.series) == 2
    mirror = load(refs['mirror_chern'], str(folder), {})
    assert mirror.family == 'text' and '"C_M": 2' in mirror.text
