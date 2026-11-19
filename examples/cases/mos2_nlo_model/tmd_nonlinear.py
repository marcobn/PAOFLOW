"""Shift current, injection current and SHG of monolayer MoS2 from the three-band TMD model.

    python tmd_nonlinear.py               # -> output/{shift_yyy.dat, shgr_yyy.dat, shgi_yyy.dat, ...}

Nearest-neighbour model of G.-B. Liu et al., Phys. Rev. B 88, 085433 (2013) (GGA parameters, on-site
spin-orbit coupling lambda L_z S_z): the d_z2, d_xy, d_x2-y2 orbitals of the metal, a_1 along x.  Point
group D3h with the mirror x -> -x, so sigma^{yyy} = -sigma^{yxx} = -sigma^{xxy} and sigma^{xxx} = 0 (shift
current, SHG alike).  Other materials: 'WS2', 'MoSe2', 'WSe2', 'MoTe2', 'WTe2'; 'functional': 'LDA'
(with 'soc': <lambda in eV>, the tabulated lambda being fitted to GGA); 'soc': False for the
spin-degenerate model.  The sheet responses are written in 2D units (twoD=True), the slab height being
the cell height (10 alat by default, or 'thickness' in angstrom).
"""

from PAOFLOW import PAOFLOW


def main():
    model = {'label': 'tmd', 'material': 'MoS2', 'functional': 'GGA', 'soc': True}
    paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./output', verbose=True)
    paoflow.nonlinear_optics(
        nk=(300, 300, 1),
        emin=0.0,
        emax=5.0,
        ne=501,
        eta=0.05,
        components=['yyy', 'yxx', 'xxy', 'xxx'],
        twoD=True,
    )
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
