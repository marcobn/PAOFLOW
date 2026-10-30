"""Shift current and second-harmonic generation of monolayer GeS (Pmn2_1).

    mpirun -np 8 python main.py          # -> output/{shift_xxx.dat, shift_xyy.dat, shga_xxx.dat, ...}

Monolayer GeS is the polar (mm2) puckered monochalcogenide of Wang et al., Phys. Rev. B 96,
115147 (2017), Fig. 1, and Rangel et al., Phys. Rev. Lett. 119, 067402 (2017): with the polar
axis x (armchair) the allowed in-plane components are sigma^{xxx}, sigma^{xyy} and sigma^{yxy}
(sigma^{yyy} = sigma^{xxy} = 0 by the mirror y -> -y).  The sheet responses are written in 2D
units (twoD=True: nm uA/V^2 and pm^2/V, the slab height being the vacuum cell).  The QE inputs
(relax.in, scf.in, nscf.in) are in this folder (internal projections: no projwfc); the PBE structure was relaxed from a
layer of bulk GeS (see README).
"""

from PAOFLOW import PAOFLOW

NK = 120  # (NK, NK, 1); 300 for converged spectra
ETA = 0.05  # eV
SCISSOR = 0.0  # eV (PBE gap; Wang et al. compare to GW with a scissor)


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='ges.save', smearing=None, npool=1, verbose=True)
    # PAO basis generated from the pseudopotentials (see README): the repository BASIS tables
    # give a projectability of only 0.90 for the Ge 3d bands with these pseudopotentials.
    paoflow.projections(internal=True, basispath='BASIS_PS/', configuration='standard')
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.nonlinear_optics(
        nk=(NK, NK, 1),
        emin=0.0,
        emax=4.0,
        ne=401,
        eta=ETA,
        scissor=SCISSOR,
        components=['xxx', 'xyy', 'yxy', 'yyy', 'xxy'],
        twoD=True,
    )
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
