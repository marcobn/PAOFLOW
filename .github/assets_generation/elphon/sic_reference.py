"""3C-SiC: PAOFLOW's interpolated |g_{mn nu}(k, q)| with the long-range (Frohlich) model vs EPW (prtgkk).

PAOFLOW: native DFPT vertex (dfpt source) on 3^3/3^3, PAO 'minimal' basis, long-range subtract/restore,
dense 12^3 electrons; the k and q lists of epw/kpts.txt, epw/qpts.txt.  Compared quantity (gauge
invariant): sum over degenerate (m, n, nu) subspaces of |g|^2 for the 4 valence bands, in meV^2, with
the zero-point factor 1/sqrt(2 omega) as EPW prints it.  Usage: python parity_sic.py [--no-long-range]
"""

import os
import sys
import time
import numpy as np
from PAOFLOW.elphon.pao_electrons import pao_hamiltonian_from_save
from PAOFLOW.elphon.dfpt_vertex import CoarseVertexSetup
from PAOFLOW.elphon.long_range import LongRangeModel
from PAOFLOW.elphon.do_pao_eph_dense_q import prepare_dense_vertex, _phonon_modes_at_q, g_Re_at_q
from PAOFLOW.elphon.elph_bloch import (
    RY_TO_EV,
    RY_TO_THZ,
    AMU_RY,
    precompute_dense_electrons,
    vertex_ws_coefficients,
    vertex_phases,
    band_vertex,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE, DFPT = os.path.join(HERE, 'paoflow/sic.save'), os.path.join(HERE, 'phonon/save')
_G = int(os.environ.get('SIC_GRID', '3'))
NG, QG, NK = (_G, _G, _G), (_G, _G, _G), 12
MASSES = [28.0855, 12.0107]
use_lr = '--no-long-range' not in sys.argv
BASIS = 'standard' if '--standard' in sys.argv else 'minimal'
EPW_OUT = (
    'epw/epw2_fs100.out'
    if os.path.exists(os.path.join(HERE, 'epw/epw2_fs100.out'))
    else 'epw/epw2.out'
)
EPW_NOLR = 'epw_nolpolar/epw2_fs100.out'
SAVE_REF = '--save-reference' in sys.argv
NVAL = 4


def read_list(path):
    rows = [ln.split() for ln in open(path).read().splitlines()[1:] if ln.strip()]
    return np.array([[float(x) for x in r[:3]] for r in rows])


kpts = read_list(os.path.join(HERE, 'epw/kpts.txt'))
qpts = read_list(os.path.join(HERE, 'epw/qpts.txt'))
el = pao_hamiltonian_from_save(
    SAVE,
    basispath=os.path.join(HERE, 'BASIS_PS') if BASIS == 'standard' else None,
    configuration=BASIS,
    pthr=0.90,
    expand_wedge=False,
    workpath=os.path.join(HERE, 'paoflow'),
    outputdir='out',
    verbose=False,
)
print(
    'PAO basis: %s, %d orbitals, %d bands'
    % (BASIS, el.projections.shape[1], el.projections.shape[0])
)
A, HRs, nscf = el.projections, el.HRs, el.nscf
setup = CoarseVertexSetup(SAVE, DFPT, QG)
model = LongRangeModel.from_dfpt(setup)
print(
    'eps_inf diag',
    np.round(np.diag(model.epsilon), 3),
    ' Z* diag',
    np.round([np.diag(z) for z in model.born_charges], 3),
)
orbital_positions = nscf['orbital_positions']
t0 = time.time()
vertex = prepare_dense_vertex(
    A,
    nscf['kpts_cryst'],
    nscf['bg'],
    nscf['at'],
    DFPT,
    QG,
    None,
    NG,
    None,
    source='dfpt',
    masses_amu=np.array(MASSES),
    tau_cryst=nscf['tau_cryst'],
    orbital_positions=orbital_positions,
    save_dir=SAVE,
    long_range=model if use_lr else None,
    allow_missing_long_range=not use_lr,
)
print('coarse vertex ready (%.0f s); long-range model: %s' % (time.time() - t0, use_lr))
g_ReRp, phonon_at_q = vertex['g_ReRp'], vertex['phonon_at_q']
Nint_p, W_p, Midx_p = vertex['phonon_ws']
mass_flat = np.repeat(np.array(MASSES) * AMU_RY, 3)
electrons = precompute_dense_electrons(
    HRs, nscf['at'], NK, [0.01], 8, NG, orbital_positions=orbital_positions
)
V, E = electrons['V'], electrons['E']  # eigenvectors (nkd, nawf, nawf), energies (nkd, nawf) Ry
nawf = V.shape[1]


def kindex(k):
    lab = np.rint(np.asarray(k) * NK).astype(int) % NK
    return int(np.ravel_multi_index(lab, (NK, NK, NK)))


out = {}
for q in qpts:
    freq_thz, z = _phonon_modes_at_q(q, phonon_at_q)
    omega_ry = np.abs(freq_thz) / RY_TO_THZ
    zmass = z / np.sqrt(mass_flat)[None, :]
    gn_flat = vertex_ws_coefficients(g_Re_at_q(g_ReRp, q, Nint_p, W_p, Midx_p), electrons)
    lr = model.vertex_pao(q, nawf, orbital_positions) if use_lr else None
    for k in kpts:
        ik, ikq = kindex(k), kindex(k + q)
        g = band_vertex(
            gn_flat, vertex_phases(electrons, np.array([ik])), V[[ikq]], V[[ik]], zmass, lr
        )[0]  # (nu, m, n)
        g = (
            g[:, :NVAL, :NVAL]
            / np.sqrt(2.0 * np.where(omega_ry > 1e-8, omega_ry, np.inf))[:, None, None]
        )
        out[(tuple(np.round(k, 6)), tuple(np.round(q, 6)))] = {
            'g2_mev2': np.abs(g) ** 2 * (RY_TO_EV * 1e3) ** 2,
            'omega_mev': omega_ry * RY_TO_EV * 1e3,
            'e_k': E[ik, :NVAL] * RY_TO_EV,
            'e_kq': E[ikq, :NVAL] * RY_TO_EV,
        }
np.save(
    os.path.join(HERE, 'paoflow_g2_%s_%s.npy' % (BASIS, 'lr' if use_lr else 'nolr')),
    out,
    allow_pickle=True,
)
# summary: Frohlich 1/q behaviour of the LO mode at k = Gamma along the (1,1,0) list
print('q (crystal)            |q| (2pi/a)   sum_mn |g_LO|^2 (meV^2)   omega_LO (meV)')
for q in qpts[:5]:
    key = (tuple(np.round(kpts[0], 6)), tuple(np.round(q, 6)))
    r = out[key]
    nu = int(np.argmax(r['omega_mev']))
    print(
        '%-22s %9.4f   %14.4f        %8.3f'
        % (
            np.round(q, 4),
            np.linalg.norm(q @ nscf['bg']),
            r['g2_mev2'][nu].sum(),
            r['omega_mev'][nu],
        )
    )


# ----------------------------------------------------------------------------- EPW comparison
def parse_epw_gkk(path):
    """EPW prtgkk blocks -> {(k_cryst, q_cryst): dict(ibnd, jbnd, nu, e_k, e_kq, omega_mev, re, im)}."""
    blocks, q, k = {}, None, None
    rows = []

    def flush():
        if k is not None and rows:
            arr = np.array(rows)
            blocks[(k, q)] = {
                'ibnd': arr[:, 0].astype(int),
                'jbnd': arr[:, 1].astype(int),
                'nu': arr[:, 2].astype(int),
                'e_k': arr[:, 3],
                'e_kq': arr[:, 4],
                'omega_mev': arr[:, 5],
                'g_abs': arr[:, 7],
                're': arr[:, 8],
                'im': arr[:, 9],
            }

    for ln in open(path):
        s = ln.strip()
        if s.startswith('iq ='):
            flush()
            rows = []
            k = None
            q = tuple(np.round([float(x) for x in s.split('coord.:')[1].split()], 6))
        elif s.startswith('ik ='):
            flush()
            rows = []
            k = tuple(np.round([float(x) for x in s.split('coord.:')[1].split()], 6))
        elif k is not None and s and s[0].isdigit():
            tok = s.split()
            if len(tok) >= 10:
                rows.append([float(t) for t in tok[:10]])
    flush()
    return blocks


epw_path = os.path.join(HERE, EPW_OUT)
if os.path.exists(epw_path):
    epw = parse_epw_gkk(epw_path)
    print(
        '\nEPW blocks: %d; comparing sum_{mn in valence} |g|^2 per degenerate mode group (meV^2)'
        % len(epw)
    )
    print('%-28s %-28s %6s %12s %12s %9s' % ('k', 'q', 'modes', 'PAOFLOW', 'EPW', 'ratio'))
    worst = 0.0
    for (k, q), r in out.items():
        ek = None
        match = [
            key
            for key in epw
            if np.allclose(key[0], k, atol=2e-6) and np.allclose(key[1], q, atol=2e-6)
        ]
        if not match:
            print('no EPW block for k=%s q=%s' % (k, q))
            continue
        b = epw[match[0]]
        omega = r['omega_mev']
        groups, start = [], 0
        order = np.argsort(omega)
        om_sorted = omega[order]
        for i in range(1, 7):
            if i == 6 or om_sorted[i] - om_sorted[i - 1] > 0.05:
                groups.append(order[start:i])
                start = i
        for grp in groups:
            pao = sum(r['g2_mev2'][nu].sum() for nu in grp)
            epw_modes = [nu + 1 for nu in grp]
            sel = np.isin(b['nu'], epw_modes) & (b['ibnd'] <= NVAL) & (b['jbnd'] <= NVAL)
            ref = (b['re'][sel] ** 2 + b['im'][sel] ** 2).sum()
            ratio = pao / ref if ref > 0 else float('nan')
            worst = max(worst, abs(ratio - 1.0)) if ref > 1e-6 else worst
            print(
                '%-28s %-28s %6s %12.4f %12.4f %9.4f'
                % (np.round(k, 4), np.round(q, 4), '+'.join(map(str, epw_modes)), pao, ref, ratio)
            )
    print('worst |ratio - 1| over groups with |g|^2 > 1e-6 meV^2: %.3e' % worst)

# ----------------------------------------------------------------------------- polar contribution only
other = os.path.join(HERE, 'paoflow_g2_%s_%s.npy' % (BASIS, 'nolr' if use_lr else 'lr'))
if (
    os.path.exists(other)
    and os.path.exists(os.path.join(HERE, EPW_NOLR))
    and os.path.exists(epw_path)
):
    pao_other = np.load(other, allow_pickle=True).item()
    epw_lr, epw_nolr = parse_epw_gkk(epw_path), parse_epw_gkk(os.path.join(HERE, EPW_NOLR))
    pao_lr, pao_nolr = (out, pao_other) if use_lr else (pao_other, out)
    print('\nPolar contribution, LO mode (highest): [sum_mn |g|^2 with model] - [without], meV^2')
    print('%-28s %-28s %12s %12s %9s' % ('k', 'q', 'PAOFLOW', 'EPW', 'ratio'))
    worst = 0.0
    for key in pao_lr:
        k, q = key
        m = [
            kk
            for kk in epw_lr
            if np.allclose(kk[0], k, atol=2e-6) and np.allclose(kk[1], q, atol=2e-6)
        ]
        if not m or m[0] not in epw_nolr:
            continue
        b1, b0 = epw_lr[m[0]], epw_nolr[m[0]]
        nu = int(np.argmax(pao_lr[key]['omega_mev'])) + 1
        sel1 = (b1['nu'] == nu) & (b1['ibnd'] <= NVAL) & (b1['jbnd'] <= NVAL)
        sel0 = (b0['nu'] == nu) & (b0['ibnd'] <= NVAL) & (b0['jbnd'] <= NVAL)
        d_epw = (b1['re'][sel1] ** 2 + b1['im'][sel1] ** 2).sum() - (
            b0['re'][sel0] ** 2 + b0['im'][sel0] ** 2
        ).sum()
        d_pao = pao_lr[key]['g2_mev2'][nu - 1].sum() - pao_nolr[key]['g2_mev2'][nu - 1].sum()
        ratio = d_pao / d_epw if abs(d_epw) > 1e-6 else float('nan')
        worst = max(worst, abs(ratio - 1.0)) if abs(d_epw) > 100 else worst
        print(
            '%-28s %-28s %12.1f %12.1f %9.4f'
            % (np.round(k, 4), np.round(q, 4), d_pao, d_epw, ratio)
        )
    print('worst |ratio - 1| (polar LO contribution, |delta| > 100 meV^2): %.3e' % worst)

if SAVE_REF and os.path.exists(os.path.join(HERE, EPW_OUT)):
    ref = parse_epw_gkk(os.path.join(HERE, EPW_OUT))
    np.savez_compressed(
        os.path.join(HERE, 'epw_gkk_reference.npz'),
        **{'%d' % i: np.array([list(k) + list(q)]) for i, (k, q) in enumerate(ref)},
        **{
            'b%d_%s' % (i, name): ref[key][name]
            for i, key in enumerate(ref)
            for name in ('ibnd', 'jbnd', 'nu', 'e_k', 'e_kq', 'omega_mev', 're', 'im')
        },
    )
    print('EPW reference written: %d blocks' % len(ref))
