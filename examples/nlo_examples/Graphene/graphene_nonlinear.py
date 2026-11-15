"""Shift current and second-harmonic generation of gapped graphene (hBN-like) from the tight-binding model.

    python graphene_nonlinear.py          # -> output/{shift_yyy.dat, shgr_yyy.dat, shgi_yyy.dat, ...}

The honeycomb model with a sublattice potential ``delta`` breaks inversion (point group D3h,
a_1 along x): the only independent in-plane components are sigma^{yyy} = -sigma^{yxx} =
-sigma^{xxy} = -sigma^{xyx} (shift current) and chi^{yyy} = -chi^{yxx} = -chi^{xxy} = -chi^{xyx}
(SHG); sigma^{xxx} = 0.  The responses of the sheet are written in 2D units (``twoD=True``:
nm uA/V^2 and pm^2/V, the slab height being the 10 alat vacuum of the model).
"""

from PAOFLOW import PAOFLOW


def main():
    model = {'label': 'graphene', 't': 2.7, 'delta': 1.0}  # eV; delta opens a 1 eV gap at K
    paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='./output', verbose=True)
    paoflow.nonlinear_optics(
        nk=(300, 300, 1),
        emin=0.0,
        emax=6.0,
        ne=601,
        eta=0.05,
        components=['yyy', 'yxx', 'xxy', 'xxx'],
        nelec=2,
        twoD=True,
    )
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
