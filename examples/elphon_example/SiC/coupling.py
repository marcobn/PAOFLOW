#!/usr/bin/env python3
"""The Fröhlich coupling of 3C-SiC: |g(k, q)| along Gamma-X with and without the long-range model.

    mpirun -np N python coupling.py [--nk 24]

After the scf, ph.x (epsil, fildvscf) and the nscf (K_POINTS automatic, unfolded by symmetry),
the coarse coupling is computed by PAOFLOW from the dvscf files (no EPW), the analytic dipole
vertex of the rigid-ion model (epsilon_inf, Z*) is subtracted on the coarse grid and added back
at every dense q, and the band-resolved |g| at k = Gamma is evaluated for q along Gamma-X on the
dense --nk grid.  The LO mode diverges as 1/|q| (Fröhlich); the plain interpolation of the full
coupling cannot represent that and underestimates the coupling near Gamma.

Outputs (output/coupling): g_path.dat (|q|, mode-group sums of |g|^2 over the valence bands, with
and without the model, and the analytic Fröhlich term alone) and g_frohlich.png.
"""

import argparse
import os

import numpy as np

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
from PAOFLOW.elphon.wavefunction_symmetry import stores_irreducible_wedge

HERE = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(HERE, 'nscf', 'sic.save')
PHONON_DIR = os.path.join(HERE, 'phonon')
BASISDIR = os.path.join(HERE, 'BASIS_PS')
OUTPUTDIR = os.path.join(HERE, 'output', 'coupling')
COARSE_GRID = (6, 6, 6)
QGRID = (6, 6, 6)
MASSES_AMU = np.array([28.0855, 12.0107])  # Si, C
NELEC = 8
NVAL = 4
PTHR = 0.90
QUADRUPOLES = None  # 'quadrupole.fmt' (from quadrupoles.py): adds a dipole + quadrupole model to the comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--nk', type=int, default=24, help='dense grid of the path points (multiples of 1/nk)')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--quadrupoles', default=QUADRUPOLES, metavar='FILE', help='quadrupole.fmt of quadrupoles.py')
    args = parser.parse_args()
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    os.makedirs(OUTPUTDIR, exist_ok=True)
    el = pao_hamiltonian_from_save(
        SAVEDIR, BASISDIR, 'standard', PTHR, expand_wedge=False,
        unfold_kpoints=stores_irreducible_wedge(SAVEDIR), workpath=HERE, outputdir='output/coupling',
    )  # fmt: skip
    nscf = el.nscf
    setup = CoarseVertexSetup(SAVEDIR, PHONON_DIR, QGRID)
    model = LongRangeModel.from_dfpt(setup)
    models = {'model': model}
    if args.quadrupoles:
        models['quadrupole'] = LongRangeModel.from_coupling(
            PHONON_DIR, SAVEDIR, quadrupoles=os.path.join(HERE, args.quadrupoles)
        )
    common = dict(
        source='dfpt', masses_amu=MASSES_AMU, tau_cryst=nscf['tau_cryst'],
        orbital_positions=nscf['orbital_positions'], save_dir=SAVEDIR,
    )  # fmt: skip
    vertex = {}
    variants = [(name, {'long_range': m}) for name, m in models.items()]
    variants.append(('plain', {'allow_missing_long_range': True}))
    for name, extra in variants:
        vertex[name] = prepare_dense_vertex(
            el.projections, nscf['kpts_cryst'], nscf['bg'], nscf['at'], PHONON_DIR, QGRID, None,
            COARSE_GRID, None, **common, **extra,
        )  # fmt: skip
    electrons = precompute_dense_electrons(
        el.HRs, nscf['at'], args.nk, [0.01], NELEC, COARSE_GRID, orbital_positions=nscf['orbital_positions']
    )
    V = electrons['V']
    nawf = V.shape[1]
    mass_flat = np.repeat(MASSES_AMU * AMU_RY, 3)
    x_dir = np.array([1.0, 0.0, 0.0]) @ np.linalg.inv(nscf['bg'])  # Gamma-X: Cartesian (1,0,0) 2 pi/a
    rows = []
    for n in range(1, args.nk // 2 + 1):
        q = np.rint(x_dir * n / (args.nk // 2) * args.nk) / args.nk  # on the dense grid
        q_norm = np.linalg.norm(q @ nscf['bg'])
        line = [q_norm]
        for name in ('model', 'plain') + (('quadrupole',) if 'quadrupole' in vertex else ()):
            g_ReRp, phonon_at_q = vertex[name]['g_ReRp'], vertex[name]['phonon_at_q']
            Nint_p, W_p, Midx_p = vertex[name]['phonon_ws']
            freq_thz, z = _phonon_modes_at_q(q, phonon_at_q)
            omega_ry = np.abs(freq_thz) / RY_TO_THZ
            zmass = z / np.sqrt(mass_flat)[None, :]
            gn_flat = vertex_ws_coefficients(g_Re_at_q(g_ReRp, q, Nint_p, W_p, Midx_p), electrons)
            lr = models[name].vertex_pao(q, nawf, nscf['orbital_positions']) if name in models else None
            ik = 0
            ikq = int(np.ravel_multi_index(np.rint(q * args.nk).astype(int) % args.nk, (args.nk,) * 3))
            g = band_vertex(gn_flat, vertex_phases(electrons, [ik]), V[[ikq]], V[[ik]], zmass, lr)[0]
            g = g[:, :NVAL, :NVAL] / np.sqrt(2.0 * np.where(omega_ry > 1e-8, omega_ry, np.inf))[:, None, None]
            g2 = (np.abs(g) ** 2).sum(axis=(1, 2)) * (RY_TO_EV * 1e3) ** 2  # meV^2 per mode
            order = np.argsort(omega_ry)
            acoustic, to, lo = g2[order[:3]].sum(), g2[order[3:5]].sum(), g2[order[5]]
            line += [lo, to, acoustic]
            if name == 'model':
                omega_lo = omega_ry[order[5]]
                if n == 1:
                    omega_lo_first = omega_lo
        # the analytic dipole vertex alone, LO mode, summed over the valence pairs (|<u_k+q|u_k>|^2 -> 4)
        coeff, _ = model.coefficients(q)
        freq_thz, z = _phonon_modes_at_q(q, vertex['model']['phonon_at_q'])
        zmass = z / np.sqrt(mass_flat)[None, :]
        c_lo = (zmass @ coeff)[np.argmax(np.abs(freq_thz))]
        frohlich = 4.0 * abs(c_lo) ** 2 / (2.0 * omega_lo) * (RY_TO_EV * 1e3) ** 2
        line.append(frohlich)
        rows.append(line)
    rows = np.array(rows)
    if rank == 0:
        header = ('|q| (2pi/a)   LO_model  TO_model  ac_model  LO_plain  TO_plain  ac_plain  LO_dipole_only'
                  + ('  LO_quad  TO_quad  ac_quad' if 'quadrupole' in vertex else '')
                  + '   [sum over the 4 valence bands of |g|^2 / (2 omega), meV^2, k = Gamma, q along Gamma-X]')
        path = os.path.join(OUTPUTDIR, 'g_path.dat')
        np.savetxt(path, rows, fmt='%12.5e', header=header)
        print('Fröhlich coupling of 3C-SiC (k = Gamma, q along Gamma-X, dense %d^3):' % args.nk)
        print('  epsilon_inf %.3f, Z* %.3f / %.3f, omega_LO at the first q %.1f meV'
              % (model.epsilon[0, 0], model.born_charges[0, 0, 0], model.born_charges[1, 0, 0], omega_lo_first * RY_TO_EV * 1e3))
        print('  |q|     |g_LO|^2 model   plain   dipole-only%s  (meV^2)'
              % ('   dipole+quadrupole LO / TO / acoustic (model LO/TO/ac)' if rows.shape[1] > 8 else ''))
        for r in rows[:4]:
            extra = ''
            if rows.shape[1] > 8:
                extra = '   %10.1f / %8.1f / %8.1f  (%8.1f / %8.1f / %8.1f)' % (r[8], r[9], r[10], r[1], r[2], r[3])
            print('  %.4f  %12.1f %12.1f %12.1f%s' % (r[0], r[1], r[4], r[7], extra))
        print('  wrote %s' % path)
        if not args.no_plot:
            plot(rows)


def plot(rows):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    q = rows[:, 0]
    ax.plot(q, rows[:, 7], color='0.55', lw=1.0, ls=':', label='dipole term alone')
    ax.plot(q, rows[:, 4], color='#D55E00', lw=1.4, ls='--', marker='s', ms=4, label='plain interpolation')
    ax.plot(q, rows[:, 1], color='#0072B2', lw=1.6, marker='o', ms=4, label='with the dipole term')
    if rows.shape[1] > 8:
        ax.plot(q, rows[:, 8], color='#009E73', lw=1.6, marker='^', ms=4, label='dipole + quadrupole terms')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('|q| along Γ-X (2π/a)')
    ax.set_ylabel('Σ$_{mn}$ |g$_{LO}$|² (meV²), k = Γ')
    ax.set_title('3C-SiC: Fröhlich 1/|q| coupling of the LO mode', fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    path = os.path.join(OUTPUTDIR, 'g_frohlich.png')
    fig.savefig(path, dpi=200)
    print('  wrote %s' % path)


if __name__ == '__main__':
    main()
