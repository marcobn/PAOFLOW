"""Shift current and second-harmonic generation of GaAs from the PAO Hamiltonian.

    mpirun -np 8 python main.py          # -> output/{shift_xyz.dat, shgr_xyz.dat, shgi_xyz.dat, shga_xyz.dat, ...}

GaAs (F-43m) has one independent component, xyz (and its permutations).  The PAO
Hamiltonian is interpolated on NK^3 k-points on the fly (no stored k-grid tensors); the
conduction bands are shifted by SCISSOR to the experimental gap (1.42 eV = 0.56 PBE + 0.86),
as in the Wannier-interpolation study of Wang et al., Phys. Rev. B 96, 115147 (2017), whose
Fig. 4 (|chi_xyz| and Im chi_xyz) is the target, with Rashkeev et al. (1998) and Nastos et al.
(2005) for the static value (about 1.5-2.5 x 10^2 pm/V with the scissor) and Ibanez-Azpiroz
et al. (2018) for the shift current.  The QE inputs are those of example10 (same GaAs.save).
"""

from PAOFLOW import PAOFLOW

NK = 40  # 40^3 for a quick look; 100^3 for converged spectra (minutes on 8 ranks)
SCISSOR = 0.86  # eV
ETA = 0.05  # eV


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='GaAs.save', smearing=None, npool=1, verbose=True)
    arry, attr = paoflow.data_controller.data_dicts()
    attr['basispath'] = '../../../BASIS/'
    arry['configuration'] = {
        'Ga': ['3S', '3P', '4S', '4P', '3D', '4D', '5S', '5P', '5D', '6S', '6P'],
        'As': ['3S', '4S', '4P', '3D', '3P', '4D', '5S', '5P', '5D', '6S', '6P'],
    }
    paoflow.projections(
        internal=True, basispath=attr['basispath'], configuration=arry['configuration']
    )
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.nonlinear_optics(
        nk=(NK, NK, NK),
        emin=0.0,
        emax=4.0,
        ne=401,
        eta=ETA,
        scissor=SCISSOR,
        components=['xyz', 'yzx', 'zxy', 'xxx'],  # Kleinman partners and a vanishing one
    )
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
