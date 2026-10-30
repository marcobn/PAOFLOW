"""Long-range (Fröhlich) coupling on 3C-SiC: coarse-point identity, 1/q divergence, EPW agreement, LO-TO splitting.

Asset ``sic/``: nscf save (3^3 k, 8 bands, 40 Ry), ``ph.x`` 3^3 with ``epsil`` (``save/``), the internal
PAO basis tables (``BASIS_PS/``), and EPW's printed vertex (``lpolar``, wide ``fsthick``) on two k and
seven q of the 12^3 fine grid (``reference/epw_gkk.npz``).
"""

import os

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_vertex import CoarseVertexSetup
from PAOFLOW.elphon.do_pao_eph_dense_q import _phonon_modes_at_q, g_Re_at_q, prepare_dense_vertex
from PAOFLOW.elphon.elph_bloch import (
    AMU_RY,
    RY_TO_EV,
    RY_TO_THZ,
    band_vertex,
    precompute_dense_electrons,
    vertex_phases,
    vertex_ws_coefficients,
)
from PAOFLOW.elphon.long_range import LongRangeModel
from PAOFLOW.elphon.pao_electrons import pao_hamiltonian_from_save

pytestmark = pytest.mark.integration

MASSES = np.array([28.0855, 12.0107])
NK = 12
NVAL = 4


def _read_list(path):
    rows = [ln.split() for ln in open(path).read().splitlines()[1:] if ln.strip()]
    return np.array([[float(x) for x in r[:3]] for r in rows])


def _epw_reference(path):
    data = np.load(path)
    blocks = {}
    i = 0
    while '%d' % i in data:
        kq = data['%d' % i][0]
        key = (tuple(np.round(kq[:3], 6)), tuple(np.round(kq[3:], 6)))
        blocks[key] = {
            name: data['b%d_%s' % (i, name)]
            for name in ('ibnd', 'jbnd', 'nu', 'omega_mev', 're', 'im')
        }
        i += 1
    return blocks


def _mode_groups(omega_mev):
    order = np.argsort(omega_mev)
    groups, start = [], 0
    for i in range(1, order.size + 1):
        if i == order.size or omega_mev[order[i]] - omega_mev[order[i - 1]] > 0.05:
            groups.append(order[start:i])
            start = i
    return groups


@pytest.fixture(scope='module')
def sic(elphon_assets, tmp_path_factory):
    root = os.path.join(os.path.dirname(elphon_assets), 'sic')
    if not os.path.isdir(root):
        pytest.skip('asset has no sic/ directory')
    save, dfpt = os.path.join(root, 'sic.save'), os.path.join(root, 'save')
    work = str(tmp_path_factory.mktemp('sic'))
    el = pao_hamiltonian_from_save(
        save, basispath=os.path.join(root, 'BASIS_PS'), configuration='standard', pthr=0.9,
        expand_wedge=False, workpath=work, outputdir='out',
    )  # fmt: skip
    setup = CoarseVertexSetup(save, dfpt, (3, 3, 3))
    model = LongRangeModel.from_dfpt(setup)
    nscf = el.nscf
    kwargs = dict(
        source='dfpt', masses_amu=MASSES, tau_cryst=nscf['tau_cryst'],
        orbital_positions=nscf['orbital_positions'], save_dir=save,
    )  # fmt: skip
    with_model = prepare_dense_vertex(
        el.projections, nscf['kpts_cryst'], nscf['bg'], nscf['at'], dfpt, (3, 3, 3), None, (3, 3, 3),
        None, long_range=model, **kwargs,
    )  # fmt: skip
    plain = prepare_dense_vertex(
        el.projections, nscf['kpts_cryst'], nscf['bg'], nscf['at'], dfpt, (3, 3, 3), None, (3, 3, 3),
        None, allow_missing_long_range=True, **kwargs,
    )  # fmt: skip
    electrons = precompute_dense_electrons(
        el.HRs, nscf['at'], NK, [0.01], 8, (3, 3, 3), orbital_positions=nscf['orbital_positions']
    )
    return {
        'el': el, 'setup': setup, 'model': model, 'with': with_model, 'plain': plain,
        'electrons': electrons, 'root': root,
    }  # fmt: skip


def _g2_sums(sic, vertex, k, q, use_model):
    """Sum over valence band pairs of |g_nu|^2 (meV^2, with 1/sqrt(2 omega)) and the mode energies."""
    electrons, model = sic['electrons'], sic['model']
    nscf = sic['el'].nscf
    g_ReRp, phonon_at_q = vertex['g_ReRp'], vertex['phonon_at_q']
    Nint_p, W_p, Midx_p = vertex['phonon_ws']
    freq_thz, z = _phonon_modes_at_q(q, phonon_at_q)
    omega_ry = np.abs(freq_thz) / RY_TO_THZ
    zmass = z / np.sqrt(np.repeat(MASSES * AMU_RY, 3))[None, :]
    gn_flat = vertex_ws_coefficients(g_Re_at_q(g_ReRp, q, Nint_p, W_p, Midx_p), electrons)
    V = electrons['V']
    nawf = V.shape[1]
    lr = model.vertex_pao(q, nawf, nscf['orbital_positions']) if use_model else None

    def index(x):
        return int(np.ravel_multi_index(np.rint(np.asarray(x) * NK).astype(int) % NK, (NK,) * 3))

    ik, ikq = index(k), index(np.add(k, q))
    g = band_vertex(
        gn_flat, vertex_phases(electrons, np.array([ik])), V[[ikq]], V[[ik]], zmass, lr
    )[0]
    g = (
        g[:, :NVAL, :NVAL]
        / np.sqrt(2.0 * np.where(omega_ry > 1e-8, omega_ry, np.inf))[:, None, None]
    )
    return (np.abs(g) ** 2).sum(axis=(1, 2)) * (RY_TO_EV * 1e3) ** 2, omega_ry * RY_TO_EV * 1e3


def test_dense_path_with_model_reproduces_the_dfpt_vertex_at_a_coarse_point(sic):
    """At k = Gamma, q = (1/3, 1/3, 0) nothing is interpolated: subtract and restore is an identity."""
    setup = sic['setup']
    q = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    member = next(m for m in setup.members if np.allclose(m.q_cryst, q))
    freq_thz, z = _phonon_modes_at_q(q, sic['with']['phonon_at_q'])
    omega_ry = np.abs(freq_thz) / RY_TO_THZ
    zmass = z / np.sqrt(np.repeat(MASSES * AMU_RY, 3))[None, :]
    d = setup.band_vertex_at_q(member)[0][:NVAL, :NVAL]  # k = Gamma is the first nscf point
    g_dfpt = np.einsum('mnc,vc->vmn', d, zmass) / np.sqrt(2.0 * omega_ry)[:, None, None]
    g2_dfpt = (np.abs(g_dfpt) ** 2).sum(axis=(1, 2)) * (RY_TO_EV * 1e3) ** 2
    g2_pao, omega = _g2_sums(sic, sic['with'], np.zeros(3), q, True)
    for grp in _mode_groups(omega):
        assert abs(g2_pao[grp].sum() / g2_dfpt[grp].sum() - 1.0) < 0.02


def test_lo_coupling_diverges_as_one_over_q_only_with_the_model(sic):
    k = np.zeros(3)
    small, smaller = np.array([2.0, 2.0, 0.0]) / NK, np.array([1.0, 1.0, 0.0]) / NK
    lo = {}
    for name, vertex, use in (('with', sic['with'], True), ('plain', sic['plain'], False)):
        g2a, om_a = _g2_sums(sic, vertex, k, small, use)
        g2b, om_b = _g2_sums(sic, vertex, k, smaller, use)
        lo[name] = (g2a[int(np.argmax(om_a))], g2b[int(np.argmax(om_b))])
    ratio_with = (
        lo['with'][1] / lo['with'][0]
    )  # |q| halves -> |g_LO|^2 grows ~4x (plus the short-range part)
    ratio_plain = lo['plain'][1] / lo['plain'][0]
    assert ratio_with > 2.5 and ratio_plain < 1.6
    assert lo['with'][1] > 3.0 * lo['plain'][1]  # the Frohlich part dominates near Gamma


def test_interpolated_vertex_agrees_with_epw(sic):
    ref = _epw_reference(os.path.join(sic['root'], 'reference', 'epw_gkk.npz'))
    kpts = _read_list(os.path.join(sic['root'], 'reference', 'kpts.txt'))
    qpts = _read_list(os.path.join(sic['root'], 'reference', 'qpts.txt'))
    worst_coarse, worst_fine = 0.0, 0.0
    for k in kpts:
        for q in qpts:
            key = [
                kk
                for kk in ref
                if np.allclose(kk[0], k, atol=2e-6) and np.allclose(kk[1], q, atol=2e-6)
            ]
            if not key:
                continue
            b = ref[key[0]]
            g2, omega = _g2_sums(sic, sic['with'], k, q, True)
            coarse = np.allclose(q * 3, np.rint(q * 3)) and np.allclose(k * 3, np.rint(k * 3))
            for grp in _mode_groups(omega):
                sel = (
                    np.isin(b['nu'], [nu + 1 for nu in grp])
                    & (b['ibnd'] <= NVAL)
                    & (b['jbnd'] <= NVAL)
                )
                epw = (b['re'][sel] ** 2 + b['im'][sel] ** 2).sum()
                if epw < 1e3:
                    continue
                dev = abs(g2[grp].sum() / epw - 1.0)
                if coarse:
                    worst_coarse = max(worst_coarse, dev)
                else:
                    worst_fine = max(worst_fine, dev)
    assert worst_coarse < 0.02  # same DFPT vertex, both interpolations exact there
    assert worst_fine < 0.7  # different localised bases (4 Wannier vs PAO) on a 3^3 coarse grid


def test_phonon_lo_to_splitting_follows_the_dielectric_data(sic):
    model = sic['model']
    phonon_with, phonon_plain = sic['with']['phonon_at_q'], sic['plain']['phonon_at_q']
    q = np.array([1.0, 1.0, 0.0]) / (10 * NK)
    with_model, _ = phonon_with(q)
    plain, _ = phonon_plain(q)
    optical = np.sort(with_model)[-3:]
    split_with = optical[2] - optical[1]
    split_plain = np.sort(plain)[-1] - np.sort(plain)[-2]
    z_avg = 0.5 * (abs(model.born_charges[0, 0, 0]) + abs(model.born_charges[1, 0, 0]))
    masses_ry = MASSES * AMU_RY
    mu = 1.0 / (1.0 / masses_ry[0] + 1.0 / masses_ry[1])
    lst = (
        8.0 * np.pi * z_avg**2 / (model.omega * mu * model.epsilon[0, 0])
    )  # omega_LO^2 - omega_TO^2 (Ry^2)
    w_to = optical[1] / RY_TO_THZ
    w_lo_expected = np.sqrt(w_to**2 + lst) * RY_TO_THZ
    assert abs(optical[2] / w_lo_expected - 1.0) < 0.03
    assert split_with > 5.0 * max(split_plain, 1e-3)
