"""Unit tests for the EPW coarse-coupling readers (``epbwrite`` files, ``.ukk``)."""

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.elphon.qe_elph_io import read_epw_epb, read_epw_ukk


def _write_record(path, payload, split=None):
    """Write one sequential unformatted record, optionally as gfortran subrecords."""
    data = np.ascontiguousarray(payload).view(np.uint8)
    cuts = [0] + list(split or []) + [data.size]
    with open(path, 'wb') as fh:
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            n = b - a
            more = i < len(cuts) - 2
            np.array([-n if more else n], np.int32).tofile(fh)
            data[a:b].tofile(fh)
            np.array([n], np.int32).tofile(fh)


def _epb_payload(nq, xqc, et, dynq, epmatq, zstar, epsi):
    """Byte payload of one EPW pool record, Fortran (column-major) ordering."""
    parts = [
        np.array([nq], np.int32),
        np.asarray(xqc, np.float64).ravel(),  # xqc(3, nq): q-major rows == C ravel of (nq, 3)
        np.asarray(et, np.float64).ravel(order='F'),
        np.asarray(dynq, np.complex128).ravel(order='F'),
        np.asarray(epmatq, np.complex128).ravel(order='F'),
        np.asarray(zstar, np.float64).ravel(order='F'),
        np.asarray(epsi, np.float64).ravel(order='F'),
    ]
    return np.concatenate([p.view(np.uint8) for p in parts])


def _synthetic_epw(rng, nbnd, nbndep, nk, nat, nq):
    nm = 3 * nat
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)  # noqa: E731
    return {
        'xqc': rng.standard_normal((nq, 3)),
        'et': rng.standard_normal((nbnd, nk)),
        'dynq': cplx(nm, nm, nq),
        'epmatq': cplx(nbndep, nbndep, nk, nm, nq),
        'zstar': rng.standard_normal((3, 3, nat)),
        'epsi': rng.standard_normal((3, 3)),
    }


@pytest.mark.parametrize('npool', [1, 3])
def test_read_epw_epb_concatenates_pools_in_qe_order(tmp_path, npool):
    rng = np.random.default_rng(3)
    nbnd, nbndep, nk, nat, nq = 5, 3, 7, 2, 4  # nk=7 over 3 pools -> 3, 2, 2 k-points
    ref = _synthetic_epw(rng, nbnd, nbndep, nk, nat, nq)
    base, extra = divmod(nk, npool)
    k0 = 0
    for ip in range(npool):
        nks = base + (1 if ip < extra else 0)
        sl = slice(k0, k0 + nks)
        payload = _epb_payload(
            nq,
            ref['xqc'],
            ref['et'][:, sl],
            ref['dynq'],
            ref['epmatq'][:, :, sl],
            ref['zstar'],
            ref['epsi'],
        )
        _write_record(tmp_path / ('pb.epb%d' % (ip + 1)), payload)
        k0 += nks

    out = read_epw_epb(str(tmp_path), 'pb', nbnd, nk, nat, nbndep)

    np.testing.assert_array_equal(out['xq_cart'], ref['xqc'])
    np.testing.assert_array_equal(out['et_ry'], ref['et'].T)
    np.testing.assert_array_equal(out['dynq'], ref['dynq'])
    np.testing.assert_array_equal(out['epmatq'], ref['epmatq'])
    np.testing.assert_array_equal(out['zstar'], np.moveaxis(ref['zstar'], 2, 0))
    np.testing.assert_array_equal(out['epsi'], ref['epsi'])


def test_read_epw_epb_handles_gfortran_subrecords(tmp_path):
    rng = np.random.default_rng(5)
    nbnd, nbndep, nk, nat, nq = 4, 2, 3, 1, 2
    ref = _synthetic_epw(rng, nbnd, nbndep, nk, nat, nq)
    payload = _epb_payload(nq, *(ref[k] for k in ('xqc', 'et', 'dynq', 'epmatq', 'zstar', 'epsi')))
    _write_record(tmp_path / 'pb.epb1', payload, split=[101, 640])

    out = read_epw_epb(str(tmp_path), 'pb', nbnd, nk, nat, nbndep)

    np.testing.assert_array_equal(out['epmatq'], ref['epmatq'])
    np.testing.assert_array_equal(out['et_ry'], ref['et'].T)


def test_read_epw_epb_rejects_wrong_dimensions(tmp_path):
    rng = np.random.default_rng(7)
    ref = _synthetic_epw(rng, 4, 2, 3, 1, 2)
    payload = _epb_payload(2, *(ref[k] for k in ('xqc', 'et', 'dynq', 'epmatq', 'zstar', 'epsi')))
    _write_record(tmp_path / 'pb.epb1', payload)
    with pytest.raises(ValueError, match='expected'):
        read_epw_epb(str(tmp_path), 'pb', 4, 3, 1, 3)  # wrong nbndep


def test_read_epw_epb_reports_atom_count_for_per_species_masses(tmp_path: Path) -> None:
    # Two species, three atoms (MgB2): a per-species mass list gives nat = 2.
    rng = np.random.default_rng(13)
    nbnd, nbndep, nk, nat, nq = 4, 2, 3, 3, 2
    ref = _synthetic_epw(rng, nbnd, nbndep, nk, nat, nq)
    payload = _epb_payload(nq, *(ref[k] for k in ('xqc', 'et', 'dynq', 'epmatq', 'zstar', 'epsi')))
    _write_record(tmp_path / 'mgb2.epb1', payload)
    with pytest.raises(ValueError, match='matches nat=3: masses_amu must give one mass per atom'):
        read_epw_epb(str(tmp_path), 'mgb2', nbnd, nk, 2, nbndep)
    assert read_epw_epb(str(tmp_path), 'mgb2', nbnd, nk, nat, nbndep)['dynq'].shape == (9, 9, nq)


def test_read_epw_epb_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_epw_epb(str(tmp_path), 'pb', 4, 3, 1, 2)


def test_read_epw_ukk_parses_band_bookkeeping(tmp_path):
    rng = np.random.default_rng(11)
    nk, nbnd, nbndskip, nwann = 3, 6, 2, 2
    nbndep = nbnd - nbndskip
    u = rng.standard_normal((nk, nbndep, nwann)) + 1j * rng.standard_normal((nk, nbndep, nwann))
    lwin = rng.random((nk, nbndep)) > 0.3
    lines = ['%12d%12d' % (nbndep, nbndskip)]
    lines += ['%12d' % (b + 1) for b in range(nbndskip, nbnd)]
    lines += [' (%.16E,%.16E)' % (z.real, z.imag) for z in u.ravel()]
    lines += [' T' if f else ' F' for f in lwin.ravel()]
    lines += [' T' if b < nbndskip else ' F' for b in range(nbnd)]
    lines += ['  0.000000000000E+00  0.000000000000E+00  0.000000000000E+00'] * nwann
    path = tmp_path / 'pb.ukk'
    path.write_text('\n'.join(lines) + '\n')

    out = read_epw_ukk(str(path), nk)

    assert (out['nbndep'], out['nbndskip'], out['nbnd'], out['nwann']) == (
        nbndep,
        nbndskip,
        nbnd,
        nwann,
    )
    np.testing.assert_array_equal(out['ibndkept'], np.arange(nbndskip, nbnd))
    np.testing.assert_allclose(out['u'], u, rtol=1e-15)
    np.testing.assert_array_equal(out['lwin'], lwin)
    np.testing.assert_array_equal(out['exband'], np.arange(nbnd) < nbndskip)


def test_vertex_from_epw_uses_kept_bands_and_matches_vertex_pao_R():
    from PAOFLOW.elphon.do_pao_eph import vertex_from_epw
    from PAOFLOW.elphon.elph_bloch import kq_index_map, vertex_pao_R

    rng = np.random.default_rng(13)
    ng = (2, 2, 2)
    nk, nbnd, nawf, ncart = 8, 5, 3, 3
    kept = np.array([2, 3, 4])
    ax = [np.arange(n) / n for n in ng]
    kc = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    A = rng.standard_normal((nbnd, nawf, nk)) + 1j * rng.standard_normal((nbnd, nawf, nk))
    ep = rng.standard_normal((3, 3, nk, ncart)) + 1j * rng.standard_normal((3, 3, nk, ncart))
    q = np.array([0.5, 0.0, 0.5])

    gR = vertex_from_epw(ep, A, kc, q, ng, kept)

    ikq, _ = kq_index_map(kc, q, ng)
    kidx = np.round(kc * np.array(ng)).astype(int) % np.array(ng)
    ref = vertex_pao_R(np.transpose(ep, (2, 0, 1, 3)), A[kept], ikq, kidx, ng)
    np.testing.assert_allclose(gR, ref, rtol=1e-13, atol=1e-13)


def test_vertex_from_epw_rejects_incommensurate_q():
    from PAOFLOW.elphon.do_pao_eph import vertex_from_epw

    kc = np.zeros((8, 3))
    with pytest.raises(ValueError, match='commensurate'):
        vertex_from_epw(
            np.zeros((1, 1, 8, 3), complex),
            np.zeros((1, 1, 8)),
            kc,
            np.array([1 / 3, 0, 0]),
            (2, 2, 2),
            np.array([0]),
        )


def _write_ukk(path, nk, nbnd, nbndskip, lwin=None):
    nbndep = nbnd - nbndskip
    lwin = np.ones((nk, nbndep), bool) if lwin is None else lwin
    lines = ['%12d%12d' % (nbndep, nbndskip)] + ['%12d' % (b + 1) for b in range(nbndskip, nbnd)]
    lines += [' (1.0000000000000000E+00,0.0000000000000000E+00)'] * (nk * nbndep)  # nwann = 1
    lines += [' T' if f else ' F' for f in lwin.ravel()]
    lines += [' T' if b < nbndskip else ' F' for b in range(nbnd)]
    lines += ['  0.0E+00  0.0E+00  0.0E+00']
    path.write_text('\n'.join(lines) + '\n')


def test_load_epw_coupling_infers_prefix_and_converts_q(tmp_path):
    from PAOFLOW.elphon.do_pao_eph import load_epw_coupling

    rng = np.random.default_rng(17)
    nbnd, nbndskip, nk, nat, nq = 5, 2, 4, 1, 3
    bg = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])
    ref = _synthetic_epw(rng, nbnd, nbnd - nbndskip, nk, nat, nq)
    q_cryst = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]])
    ref['xqc'] = q_cryst @ bg
    payload = _epb_payload(nq, *(ref[k] for k in ('xqc', 'et', 'dynq', 'epmatq', 'zstar', 'epsi')))
    _write_record(tmp_path / 'pb.epb1', payload)
    _write_ukk(tmp_path / 'pb.ukk', nk, nbnd, nbndskip)

    out = load_epw_coupling(str(tmp_path), nbnd, nk, nat, bg)

    np.testing.assert_allclose(out['q_cryst'], q_cryst, atol=1e-12)
    np.testing.assert_array_equal(out['ibndkept'], np.arange(nbndskip, nbnd))
    np.testing.assert_array_equal(out['epmatq'], ref['epmatq'])
    assert out['nq'] == nq


def test_load_epw_coupling_rejects_packed_bands(tmp_path):
    from PAOFLOW.elphon.do_pao_eph import load_epw_coupling

    rng = np.random.default_rng(19)
    nbnd, nbndskip, nk, nat, nq = 4, 1, 2, 1, 1
    ref = _synthetic_epw(rng, nbnd, nbnd - nbndskip, nk, nat, nq)
    payload = _epb_payload(nq, *(ref[k] for k in ('xqc', 'et', 'dynq', 'epmatq', 'zstar', 'epsi')))
    _write_record(tmp_path / 'pb.epb1', payload)
    lwin = np.ones((nk, nbnd - nbndskip), bool)
    lwin[1, 2] = False
    _write_ukk(tmp_path / 'pb.ukk', nk, nbnd, nbndskip, lwin)
    with pytest.raises(NotImplementedError, match='window'):
        load_epw_coupling(str(tmp_path), nbnd, nk, nat, np.eye(3))


def test_phonon_modes_from_force_constants():
    from PAOFLOW.elphon.do_pao_eph import phonon_modes_from_force_constants
    from PAOFLOW.elphon.elph_bloch import RY_TO_THZ

    C = np.diag([4.0, 1.0, -0.25]).astype(complex)
    masses = np.array([4.0, 4.0, 4.0])
    freq, z = phonon_modes_from_force_constants(C, masses)
    np.testing.assert_allclose(freq, np.array([-0.25, 0.5, 1.0]) * RY_TO_THZ)
    np.testing.assert_allclose(z @ z.conj().T, np.eye(3), atol=1e-14)


def test_phonon_interp_from_epw_reproduces_spring_model():
    """Nearest-neighbour springs on a cubic lattice: C(q) = 2k sum_i (1 - cos 2 pi q_i) I_3."""
    from PAOFLOW.elphon.do_pao_eph_dense_q import phonon_interp_from_epw
    from PAOFLOW.elphon.elph_bloch import AMU_RY, RY_TO_THZ

    n, k_spring, mass_amu = 4, 0.3, 2.0
    ax = np.arange(n) / n
    q_grid = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    order = np.random.default_rng(23).permutation(len(q_grid))  # EPW star order, not grid order
    q_list = q_grid[order]
    c_of_q = lambda q: 2 * k_spring * np.sum(1 - np.cos(2 * np.pi * q)) * np.eye(3)  # noqa: E731
    dynq = np.stack([c_of_q(q) for q in q_list], axis=-1).astype(complex)

    phonon_at_q = phonon_interp_from_epw(dynq, q_list, [mass_amu], (n, n, n), np.eye(3))

    for q in (np.array([0.25, 0.5, 0.0]), np.array([0.125, 0.3, 0.7])):  # on- and off-grid
        freq, _ = phonon_at_q(q)
        expected = np.sqrt(np.linalg.eigvalsh(c_of_q(q)) / (mass_amu * AMU_RY)) * RY_TO_THZ
        np.testing.assert_allclose(np.sort(freq), np.sort(expected), rtol=1e-10)


def test_phonon_interp_from_epw_requires_full_grid():
    from PAOFLOW.elphon.do_pao_eph_dense_q import phonon_interp_from_epw

    with pytest.raises(ValueError, match='cover'):
        phonon_interp_from_epw(
            np.zeros((3, 3, 1), complex), np.zeros((1, 3)), [1.0], (2, 2, 2), np.eye(3)
        )


def test_read_epw_a2f_three_columns(tmp_path: Path) -> None:
    from PAOFLOW.elphon.qe_elph_io import read_epw_a2f

    path = tmp_path / 'pb.a2f'
    path.write_text(
        ' w[meV] a2f and integrated 2*a2f/w\n'
        '   0.1   0.01   0.0002\n'
        '   0.2   0.04   0.0010\n'
        '   0.3   0.09   0.0020\n'
        ' Integrated el-ph coupling\n'
        '  #     1.158\n'
    )
    omega, a2f, lam = read_epw_a2f(str(path))
    np.testing.assert_allclose(omega, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(a2f, [0.01, 0.04, 0.09])
    np.testing.assert_allclose(lam, [0.0002, 0.0010, 0.0020])


def test_read_epw_a2f_smearing_columns(tmp_path: Path) -> None:
    from PAOFLOW.elphon.qe_elph_io import read_epw_a2f

    omega = np.linspace(0.1, 10.0, 100)
    a2f = np.exp(-((omega - 5.0) ** 2))
    cols = np.column_stack([omega] + [a2f * (1 + 0.01 * i) for i in range(10)])
    footer = (
        ' Integrated el-ph coupling\n  # ' + ' '.join(['1.1'] * 10) + '\n'
        ' Phonon smearing (meV)\n  # ' + ' '.join(['0.1'] * 10) + '\n'
    )
    path = tmp_path / 'pb.a2f'
    with open(path, 'w') as fh:
        fh.write(' w[meV] a2f and integrated 2*a2f/w for 10 smearing values\n')
        np.savetxt(fh, cols)
        fh.write(footer)
    w, a, lam = read_epw_a2f(str(path))
    np.testing.assert_allclose(w, omega)
    np.testing.assert_allclose(a, a2f)
    integrand = 2.0 * a2f / omega
    assert lam[0] == 0.0
    np.testing.assert_allclose(lam[-1], np.trapezoid(integrand, omega), rtol=1e-12)
