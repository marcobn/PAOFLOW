#!/usr/bin/env python
"""Example — Route 2 (EPW-like): reconstruct the coupling from QE ``dvscf``.

.. deprecated::
    This internal-reconstruction route is **superseded** by the atomic-orbital
    (Agapito-Bernardi) route in ``example_ao_from_qe_coupling.py``, which reads
    QE's full ``el_ph_mat`` and needs no bare-local / nonlocal / induced
    reconstruction (and handles NLCC / ultrasoft for free).  The reconstruction
    below is kept only as a physics/debugging harness: its coupling magnitude is
    ~2.9x too large for Pb, traced to an irreducible ~0.4% cancellation between
    the bare-local (271x) and induced (228x) terms -- not fixable at the
    reconstruction level.  Prefer the AO route for any production use.

Replace the EPW/Wannier interpolation with PAOFLOW's PAO interpolation:

  1. read the QE coarse-grid induced potential ``dvscf`` and the nscf
     wavefunctions;
  2. reconstruct the *full* DFPT perturbation
     ``dV = dvscf(induced) + dV_loc(bare) + dV_NL(bare)``;
  3. build the Bloch deformation potential ``d_mn(k,q)``;
  4. rotate into the PAO gauge and Fourier-transform to real space ``g(R_e)``;
  5. Wigner-Seitz interpolate electrons (HRs) and the vertex to a dense grid and
     evaluate the Fermi-surface double delta -> ``gamma_{q nu}`` / ``lambda_{q nu}``.

STATUS (see Elphon_module.md): this route produces the correct *structure*
(alpha^2F follows the phonon DOS, correct mode/q dependence) but the coupling
magnitude |g|^2 is currently ~2.5x too large for Pb, traced to a delicate
bare/induced cancellation in the longitudinal/small-G channel. It is provided
as a reference implementation and debugging harness, NOT a production result.

Required QE inputs (coarse pw.x nscf + ph.x DFPT on the SAME k-grid):
  * <save>/                 -- nscf on the FULL k-grid (nosym, noinv), nbnd>nawf,
                               wavefunctions wfc*.dat saved
  * _ph0/... dvscf files    -- induced potential per irreducible q (fildvscf)
  * _ph0/<prefix>.phsave/patterns.*.xml   -- displacement patterns
  * <prefix>.dyn0, <prefix>.dyn{1..nq}    -- dynamical matrices
  * the pseudopotential UPF (for the bare local + nonlocal reconstruction)

Edit the CONFIG block, then::

    conda run -n work python example_epw_like_interpolation.py
"""

import os

import numpy as np

from PAOFLOW import PAOFLOW
from PAOFLOW.elphon.elph_bloch import (
    _DC,
    AMU_RY,
    _load_all_ur,
    deformation_potential_q,
    kq_index_map,
    lambda_q_dense_ws,
    read_nscf,
    vertex_pao_R,
)
from PAOFLOW.elphon.qe_dvloc import bare_dvloc_cart, load_vloc_for_run
from PAOFLOW.elphon.qe_dvnl import (
    average_pp_beta,
    becp_k,
    build_projectors,
    nonlocal_dq,
    read_upf_beta,
)
from PAOFLOW.elphon.qe_dvscf import (
    dvscf_path,
    dvscf_to_cartesian,
    patterns_path,
    read_dvscf,
    read_patterns,
)
from PAOFLOW.elphon.qe_elph_io import read_qe_dyn
from PAOFLOW.projection.do_atwfc_proj import read_QE_wfc

# --------------------------------------------------------------------------- #
# CONFIG -- edit for your system (paths are for the Pb tutorial run).
# --------------------------------------------------------------------------- #
BASE = os.environ.get('ELPH_BASE', './exercise1_epw')  # dir with lead.save, _ph0, *.dyn*
BASIS = os.environ.get('PAOFLOW_BASIS', '../../BASIS')  # PAOFLOW atomic-orbital basis
SAVEDIR = 'lead.save'  # nscf save directory (inside BASE)
PREFIX = 'lead'  # QE prefix (phsave, dvscf naming)
FILDVSCF = 'pbdv'  # ph.x fildvscf tag
DYNPREFIX = 'pb'  # dynamical-matrix file prefix (<DYNPREFIX>.dyn*)
NG = (6, 6, 6)  # coarse coupling k-grid == pw.x SCF k-grid
MASS_AMU = 207.2  # atomic mass (Pb)
NK_DENSE = 18  # dense interpolation grid for the double delta
SIGMAS_RY = np.array([0.02])  # Fermi-surface smearing(s), Ry
NELEC = 14  # valence electrons (for the dense E_F recompute)


def main():
    save = os.path.join(BASE, SAVEDIR)
    ph0 = os.path.join(BASE, '_ph0')

    # --- electronic structure: PAO Hamiltonian HRs + projection matrices A_k --
    pf = PAOFLOW.PAOFLOW(workpath=BASE, outputdir='paoflow_out', savedir=SAVEDIR, verbose=False)
    pf.projections(configuration='standard', basispath=BASIS)
    pf.projectability(pthr=0.90)
    # grab U (= A_k) BEFORE pao_hamiltonian, which deletes it from the arrays.
    A = pf.data_controller.data_arrays['U'][:, :, :, 0].copy()
    pf.pao_hamiltonian()
    HRs = pf.data_controller.data_arrays['HRs']
    print('PAO Hamiltonian grid:', HRs.shape[2:5], ' nawf =', HRs.shape[0])

    # --- geometry, k-list, q-list ------------------------------------------- #
    info = read_nscf(save)
    fft, nk, kcry, bg, at = (
        info['fft'],
        info['nk'],
        info['kpts_cryst'],
        info['bg'],
        info['at'],
    )
    L = open(os.path.join(BASE, '%s.dyn0' % DYNPREFIX)).read().split('\n')
    nq = int(L[1].split()[0])
    q_cart = np.array([[float(x) for x in L[2 + i].split()] for i in range(nq)])
    q_cryst = np.linalg.solve(bg.T, q_cart.T).T

    # --- real-space wavefunctions + nonlocal projector setup ---------------- #
    ur = _load_all_ur(save, nk, fft)
    mass_ry = MASS_AMU * AMU_RY
    vloc_by_type = load_vloc_for_run(info, BASE)
    beta_data = read_upf_beta(os.path.join(BASE, next(iter(info['species'].values()))))
    channels = average_pp_beta(beta_data)
    nat = info['tau_cryst'].shape[0]
    proj = build_projectors(channels, nat)
    tpiba = 2.0 * np.pi / info['alat']
    dc = _DC(save, nspin=1)
    becp_all, dbecp_all = [], []
    for ik in range(nk):
        gk, wf = read_QE_wfc(dc, ik, 0)
        b, db = becp_k(
            gk,
            wf['wfc'],
            channels,
            proj,
            info['tau_cryst'],
            tpiba,
            info['omega'],
            beta_data['r'],
            beta_data['rab'],
            beta_data['kkbeta'],
        )
        becp_all.append(b)
        dbecp_all.append(db)

    kidx = np.round(kcry * np.array(NG)).astype(int) % np.array(NG)

    # --- per-q coupling -> dense lambda ------------------------------------- #
    print('\n q   mode   omega[THz]   lambda_qv   gamma[GHz]')
    for iq in range(nq):
        # 1) full DFPT perturbation in the Cartesian displacement basis
        pat = read_patterns(patterns_path(ph0, PREFIX, iq + 1))
        dv = read_dvscf(dvscf_path(ph0, PREFIX, iq + 1, FILDVSCF), fft, pat['npert'])
        dv_cart = dvscf_to_cartesian(dv, pat['U'])  # induced (QE file)
        dv_cart = dv_cart + bare_dvloc_cart(vloc_by_type, q_cryst[iq], info)  # + bare local

        # 2) Bloch deformation potential d_mn(k,q) (+ bare nonlocal KB term)
        ikq, G0 = kq_index_map(kcry, q_cryst[iq], NG)
        d = deformation_potential_q(ur, ikq, G0, dv_cart, fft)
        for ik in range(nk):
            d[ik] += nonlocal_dq(
                becp_all[ik],
                dbecp_all[ik],
                becp_all[ikq[ik]],
                dbecp_all[ikq[ik]],
                proj,
                nat,
            )

        # 3) PAO gauge + Fourier transform to real space g(R_e)
        gR = vertex_pao_R(d, A, ikq, kidx, NG)

        # 4) phonon eigenvectors (mass-weighted) from the dynamical matrix
        dyn = read_qe_dyn(os.path.join(BASE, '%s.dyn%d' % (DYNPREFIX, iq + 1)))
        zmass = (dyn['eigenvectors'] / np.sqrt(mass_ry)).reshape(dyn['freq_thz'].size, -1)

        # 5) dense Wigner-Seitz interpolation + Fermi double delta
        res = lambda_q_dense_ws(
            gR,
            HRs,
            q_cryst[iq],
            NG,
            at,
            zmass,
            dyn['freq_thz'],
            SIGMAS_RY,
            NK_DENSE,
            nelec=NELEC,
        )
        lam = res['lambda_qnu'][0]  # first (only) sigma
        gam = res['gamma_ghz'][0]
        for v in range(zmass.shape[0]):
            print(
                '%2d   %2d    %8.3f    %9.4f   %9.3f'
                % (iq + 1, v, dyn['freq_thz'][v], lam[v], gam[v])
            )
    print('\nNOTE: |g|^2 is ~2.5x too large for Pb (bare/induced cancellation);')
    print('      structure (mode/q dependence) is correct.  See Elphon_module.md.')


if __name__ == '__main__':
    main()
