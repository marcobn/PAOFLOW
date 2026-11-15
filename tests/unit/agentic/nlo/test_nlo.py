"""Point-group tensor constraints and the non-linear optics adapter on synthetic outputs."""

import numpy as np
import pytest

from PAOFLOW.agentic.nlo import NonlinearModel
from PAOFLOW.agentic.nlo import symmetry as sym


def test_point_groups_close_and_constrain():
    assert sym.point_group_operations('-43m').shape[0] == 24
    assert sym.point_group_operations('Td').shape[0] == 24
    assert sym.point_group_operations('3m_x').shape[0] == 6
    assert sym.point_group_operations('D3h').shape[0] == 12
    assert sym.point_group_operations('mm2_x').shape[0] == 4
    assert sym.point_group_operations('-1').shape[0] == 2
    # inversion kills everything; Td keeps xyz and its permutations only
    assert len(sym.forbidden_components(sym.point_group_operations('-1'))) == 27
    td = sym.point_group_operations('Td')
    forbidden = sym.forbidden_components(td)
    assert set('xyz' + '') - set() and all(
        c not in forbidden for c in ('xyz', 'yzx', 'zxy', 'xzy', 'yxz', 'zyx')
    )
    assert len(forbidden) == 21 and 'xxx' in forbidden and 'xxy' in forbidden
    assert sym.independent_relations(td)['xyz'] == pytest.approx(
        {'xzy': 1.0, 'yxz': 1.0, 'yzx': 1.0, 'zxy': 1.0, 'zyx': 1.0}
    )
    # 3m with the mirror perpendicular to x: yyy = -yxx = -xxy = -xyx, xxx = 0
    c3v = sym.point_group_operations('3m_x')
    rel = sym.independent_relations(c3v)
    assert rel['yyy']['yxx'] == pytest.approx(-1.0) and rel['yyy']['xxy'] == pytest.approx(-1.0)
    assert 'xxx' in sym.forbidden_components(c3v) and 'zzz' not in sym.forbidden_components(c3v)
    # mm2 with the polar axis x: xxx, xyy, xzz, yxy, yyx, zxz, zzx allowed; yyy, xxy forbidden
    mm2 = sym.point_group_operations('mm2_x')
    forb = sym.forbidden_components(mm2)
    assert all(c not in forb for c in ('xxx', 'xyy', 'xzz', 'yxy', 'yyx', 'zxz', 'zzx'))
    assert all(c in forb for c in ('yyy', 'xxy', 'xyx', 'zzz'))


def test_symmetrize_errors_and_kleinman():
    td = sym.point_group_operations('Td')
    t = sym.tensor_from_components(
        {'xyz': 2.0, 'yzx': 2.2, 'zxy': 1.8, 'xzy': 2.0, 'yxz': 2.0, 'zyx': 2.0, 'xxx': 0.1},
        fill=0.0,
    )
    s = sym.symmetrize(t, td)
    assert s[0, 1, 2] == pytest.approx(2.0) and s[0, 0, 0] == pytest.approx(0.0)
    assert np.allclose(s[0, 1, 2], s[1, 2, 0]) and np.allclose(s[0, 1, 2], s[2, 1, 0])
    err = sym.symmetry_error(t, td)
    assert 0.0 < err['relative'] < 0.2
    assert sym.symmetry_error(s, td)['relative'] == pytest.approx(0.0, abs=1e-12)
    k = sym.kleinman_error(
        sym.tensor_from_components({'xxy': 1.0, 'xyx': 1.0, 'yxx': 0.5}, fill=0.0)
    )
    assert k['relative'] > 0.1
    assert sym.kleinman_error(
        sym.tensor_from_components({'xxy': 1.0, 'xyx': 1.0, 'yxx': 1.0}, fill=0.0)
    )['relative'] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(KeyError):
        sym.point_group_operations('nonsense')


def write_outputs(folder, two_d=False):
    folder.mkdir(parents=True, exist_ok=True)
    e = np.linspace(0.0, 4.0, 41)
    chi = 100.0 + 400.0 * np.exp(-0.5 * ((e - 2.6) / 0.3) ** 2)
    for abc, scale in (
        ('xyz', 1.0),
        ('yzx', 1.02),
        ('zxy', 1.0),
        ('xzy', 1.0),
        ('yxz', 1.0),
        ('zyx', 1.0),
        ('xxx', 0.01),
    ):
        np.savetxt(folder / ('shgr_%s.dat' % abc), np.column_stack([e, scale * chi]))
        np.savetxt(
            folder / ('shgi_%s.dat' % abc), np.column_stack([e, scale * 0.5 * chi * (e > 1.4)])
        )
        np.savetxt(
            folder / ('shga_%s.dat' % abc),
            np.column_stack([e, scale * np.abs(chi + 1j * 0.5 * chi * (e > 1.4))]),
        )
        np.savetxt(
            folder / ('shift_%s.dat' % abc),
            np.column_stack([e, -scale * 30.0 * (e > 1.4) * np.exp(-0.5 * ((e - 3.9) / 0.5) ** 2)]),
        )
        np.savetxt(folder / ('shiftm_%s.dat' % abc), np.column_stack([e, np.zeros_like(e)]))
        np.savetxt(
            folder / ('shift_%s_0.dat' % abc), np.column_stack([e, np.zeros_like(e)])
        )  # spin channel: ignored
    return e


def test_nonlinear_model(tmp_path):
    write_outputs(tmp_path)
    model = NonlinearModel(tmp_path, system={'system': 'GaAs', 'nk': '40^3'})
    assert model.kinds == ['shift', 'shiftm', 'shgr', 'shgi', 'shga'] and model.components == [
        'xxx',
        'xyz',
        'xzy',
        'yxz',
        'yzx',
        'zxy',
        'zyx',
    ]
    assert model.energy_ev.size == 41 and model.unit('shga') == 'pm/V'
    assert model.static('shga')['xyz'] == pytest.approx(100.0)
    assert model.value('shga', 'xyz', 2.6) == pytest.approx(500.0 * np.sqrt(1.25), rel=1e-6)
    peak = model.peak('shift', 'xyz')
    assert peak['energy_ev'] == pytest.approx(3.9) and peak['value'] == pytest.approx(-30.0)
    assert model.onset('shift', 'xyz') == pytest.approx(2.6, abs=0.11)  # 2 % of the peak
    assert model.shg('xyz')[0] == pytest.approx(100.0 + 0j)
    t = model.tensor('shg', 0.0, fill=0.0)
    assert t[0, 1, 2] == pytest.approx(100.0) and t[1, 1, 1] == 0.0
    check = model.symmetry_check('Td', 'shg', 0.0)
    assert 'xxx' in check['forbidden'] and check['forbidden_present'] == ['xxx']
    assert check['leakage'] == pytest.approx(0.01 / 1.02, rel=1e-6)
    assert check['relations']['xyz']['yzx'] == pytest.approx(1.0)
    assert check['error'] is None  # not every component was computed
    k = model.kleinman_check()
    assert k['relative'] < 0.05
    assert NonlinearModel.d_coefficient(170.0) == 85.0
    with pytest.raises(ValueError):
        model.effective_3d('shift', 'xyz', 0.6)
    text = model.describe()
    assert '| system | GaAs |' in text and 'xyz 100' in text
    sheet = NonlinearModel(tmp_path, two_d=True, slab_height_nm=1.8)
    assert sheet.unit('shift') == 'nm uA/V^2' and sheet.unit('shga') == 'pm^2/V'
    assert sheet.effective_3d('shift', 'xyz', 0.6)[-1] == pytest.approx(
        model.spectrum('shift', 'xyz')[-1] / 0.6
    )
    assert sheet.effective_3d('shga', 'xyz', 0.6)[0] == pytest.approx(100.0 / 600.0)
    with pytest.raises(FileNotFoundError):
        NonlinearModel(tmp_path / 'empty')
