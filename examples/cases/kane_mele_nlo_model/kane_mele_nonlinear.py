"""Shift current of the Kane-Mele model on both sides of the quantum-spin-Hall / trivial transition.

    python kane_mele_nonlinear.py          # -> output_qsh/, output_trivial/ {shift_yyy.dat, ...}

Kane-Mele model (C. L. Kane and E. J. Mele, Phys. Rev. Lett. 95, 226801 (2005)) with a staggered potential
lambda_v, intrinsic spin-orbit coupling lambda_SO and Rashba coupling lambda_R; without Rashba the gap at K
closes at lambda_v = 3 sqrt(3) lambda_SO (quantum spin Hall below, trivial above).  The staggered potential
breaks inversion (D3h, C3v with Rashba; a_1 along x): sigma^{yyy} = -sigma^{yxx} = -sigma^{xxy},
sigma^{xxx} = 0.  The model is a PAOFLOW.models.models.HoppingModel passed to PAOFLOW(model=...); the dict
label {'label': 'Kane_Mele', 't', 'soc_par', 'r_par', 'v_par' (units of t), 'alat' (angstrom)} builds the
same model.
"""

import numpy as np

from PAOFLOW import PAOFLOW
from PAOFLOW.models.models import kane_mele_model


def main():
    soc, rashba = 0.06, 0.05  # eV, t = 1 eV
    critical = 3 * np.sqrt(3) * soc  # lambda_v at the transition (without Rashba)
    for phase, delta in (('qsh', 0.5 * critical), ('trivial', 1.5 * critical)):
        model = kane_mele_model(t=1.0, soc=soc, rashba=rashba, delta=delta)
        paoflow = PAOFLOW.PAOFLOW(model=model, outputdir=f'./output_{phase}', verbose=True)
        paoflow.nonlinear_optics(
            nk=(300, 300, 1),
            emin=0.0,
            emax=7.0,
            ne=701,
            eta=0.03,
            components=['yyy', 'yxx', 'xxx'],
            injection_current=False,
            second_harmonic=False,
            twoD=True,
        )
        paoflow.finish_execution()


if __name__ == '__main__':
    main()
