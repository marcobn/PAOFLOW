"""Readers of the ph.x outputs and of the crystal space group."""

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_io import (
    locate_dfpt_files,
    read_dvscf,
    read_patterns,
    read_space_group,
)

pytestmark = pytest.mark.unit

PATTERNS = """<?xml version="1.0" encoding="UTF-8"?>
<Root>
  <IRREPS_INFO>
    <QPOINT_NUMBER>2</QPOINT_NUMBER>
    <MINUS_Q_SYM>false</MINUS_Q_SYM>
    <NUMBER_IRR_REP>2</NUMBER_IRR_REP>
    <REPRESENTION.1>
      <NUMBER_OF_PERTURBATIONS>1</NUMBER_OF_PERTURBATIONS>
      <PERTURBATION.1>
        <DISPLACEMENT_PATTERN>
 1.0 0.0
 0.0 0.0
 0.0 0.0
        </DISPLACEMENT_PATTERN>
      </PERTURBATION.1>
    </REPRESENTION.1>
    <REPRESENTION.2>
      <NUMBER_OF_PERTURBATIONS>2</NUMBER_OF_PERTURBATIONS>
      <PERTURBATION.1>
        <DISPLACEMENT_PATTERN>
 0.0 0.0
 0.6 0.8
 0.0 0.0
        </DISPLACEMENT_PATTERN>
      </PERTURBATION.1>
      <PERTURBATION.2>
        <DISPLACEMENT_PATTERN>
 0.0 0.0
 0.0 0.0
 0.0 -1.0
        </DISPLACEMENT_PATTERN>
      </PERTURBATION.2>
    </REPRESENTION.2>
  </IRREPS_INFO>
</Root>
"""

SCHEMA = """<?xml version="1.0"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <atomic_structure nat="2" alat="10.0">
      <atomic_positions>
        <atom name="A" index="1">0.0 0.0 0.0</atom>
        <atom name="A" index="2">2.5 2.5 2.5</atom>
      </atomic_positions>
      <cell>
        <a1>-5.0 0.0 5.0</a1>
        <a2>0.0 5.0 5.0</a2>
        <a3>-5.0 5.0 0.0</a3>
      </cell>
    </atomic_structure>
    <symmetries>
      <nsym>2</nsym>
      <symmetry>
        <info name="identity">crystal_symmetry</info>
        <rotation rank="2" dims="3 3">
 1 0 0
 0 1 0
 0 0 1
        </rotation>
        <fractional_translation>0.0 0.0 0.0</fractional_translation>
        <equivalent_atoms size="2" nat="2">1 2</equivalent_atoms>
      </symmetry>
      <symmetry>
        <info name="inversion">crystal_symmetry</info>
        <rotation rank="2" dims="3 3">
 -1 0 0
 0 -1 0
 0 0 -1
        </rotation>
        <fractional_translation>{FT}</fractional_translation>
        <equivalent_atoms size="2" nat="2">{IRT}</equivalent_atoms>
      </symmetry>
      <symmetry>
        <info name="lattice only">lattice_symmetry</info>
        <rotation rank="2" dims="3 3">
 0 1 0
 1 0 0
 0 0 1
        </rotation>
      </symmetry>
    </symmetries>
  </output>
</qes:espresso>
"""


def test_read_patterns_columns_are_modes(tmp_path):
    path = tmp_path / 'patterns.2.xml'
    path.write_text(PATTERNS)
    pat = read_patterns(path)
    assert pat['nirr'] == 2 and pat['npert'] == [1, 2] and pat['nat'] == 1 and not pat['minus_q']
    assert np.allclose(pat['u'][:, 1], [0.0, 0.6 + 0.8j, 0.0])
    assert np.allclose(pat['u'][:, 2], [0.0, 0.0, -1.0j])


def test_read_dvscf_fortran_order_and_size_check(tmp_path):
    nr = (3, 4, 5)
    nmodes = 2
    rng = np.random.default_rng(3)
    data = rng.standard_normal((nmodes,) + nr) + 1j * rng.standard_normal((nmodes,) + nr)
    path = tmp_path / 'x.dvscf_q1'
    # Fortran order: first grid index fastest
    np.concatenate([data[m].ravel(order='F') for m in range(nmodes)]).astype(np.complex128).tofile(
        path
    )
    out = read_dvscf(path, nr, nmodes)
    assert out.shape == (nmodes, 1) + nr
    assert np.allclose(out[:, 0], data)
    with pytest.raises(ValueError):
        read_dvscf(path, (3, 4, 6), nmodes)


def test_locate_dfpt_files_both_layouts(tmp_path):
    pp = tmp_path / 'save'
    (pp / 'x.phsave').mkdir(parents=True)
    for i in (1, 2):
        (pp / ('x.dvscf_q%d' % i)).write_bytes(b'')
        (pp / ('x.dyn_q%d' % i)).write_text('')
        (pp / 'x.phsave' / ('patterns.%d.xml' % i)).write_text('')
    f = locate_dfpt_files(pp)
    assert f.layout == 'pp' and f.prefix == 'x' and f.nq_irr == 2 and f.dyn0 is None
    ph = tmp_path / 'run'
    (ph / '_ph0' / 'x.phsave').mkdir(parents=True)
    (ph / '_ph0' / 'x.q_2').mkdir()
    (ph / '_ph0' / 'x.dvscf1').write_bytes(b'')
    (ph / '_ph0' / 'x.q_2' / 'x.dvscf1').write_bytes(b'')
    for i in (1, 2):
        (ph / '_ph0' / 'x.phsave' / ('patterns.%d.xml' % i)).write_text('')
        (ph / ('x.dyn%d' % i)).write_text('')
    (ph / 'x.dyn0').write_text('')
    f = locate_dfpt_files(ph)
    assert (
        f.layout == 'ph'
        and f.nq_irr == 2
        and f.dvscf[1].endswith('x.q_2/x.dvscf1')
        and f.dyn0 is not None
    )
    with pytest.raises(FileNotFoundError):
        locate_dfpt_files(tmp_path)


def test_read_space_group_checks_the_atom_map(tmp_path):
    save = tmp_path / 'x.save'
    save.mkdir()
    # inversion maps atom 1 (0,0,0) onto itself and atom 2 onto itself only with t = -(2 x_2) + lattice;
    # with t = 0 inversion swaps the sublattices: x_2 = (-1/4, 3/4, -1/4) -> (1/4, -3/4, 1/4) = x_1 + R? no:
    # use the diamond convention: inversion + translation (1/4,1/4,1/4)... here simply irt = (1, 2) with t = 0 is wrong
    (save / 'data-file-schema.xml').write_text(
        SCHEMA.replace('{FT}', '0.0 0.0 0.0').replace('{IRT}', '1 2')
    )
    with pytest.raises(ValueError):
        read_space_group(save)
    # the correct map: -x_2 = x_2 - 2 x_2, and 2 x_2 = (-1/2, 3/2, -1/2) is not a lattice vector,
    # so inversion about the origin exchanges nothing: it maps atom 2 to -x_2 = (1/4,-3/4,1/4) = x_2 + (1/2,-3/2,1/2)
    # which is not an atom; the diamond inversion needs the translation t = -(x_1 + x_2) = (1/4,-3/4,1/4) ...
    ft = np.array([0.25, -0.75, 0.25])
    text = SCHEMA.replace('{FT}', ' '.join('%.6f' % v for v in ft)).replace('{IRT}', '2 1')
    (save / 'data-file-schema.xml').write_text(text)
    group = read_space_group(save)
    assert group.nsym == 2 and group.names[1] == 'inversion'
    assert np.array_equal(group.equivalent_atoms[1], [1, 0])
