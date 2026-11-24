import numpy as np
from scipy.constants import e, hbar, m_e

from PAOFLOW import PAOFLOW
from PAOFLOW.boltzmann.TauModel import TauModel


def acoustic_tau(temp, eigs, params):
    """Acoustic deformation-potential relaxation time (s), energies from the band edges."""
    # eigs is |eps_nk| from the VBM: conduction states are measured from the CBM E_c,
    # valence states (holes) stay measured from the VBM
    energy = np.where(eigs >= params['E_c'], eigs - params['E_c'], eigs) * e
    kT = temp * e
    mass = params['ms'] * m_e
    deformation = params['D_ac'] * e
    tau = np.full_like(energy, np.inf)
    moving = energy > 0.0
    numerator = 2 * np.pi * params['rho'] * (hbar**2 * params['v']) ** 2
    tau[moving] = numerator / ((2 * mass) ** 1.5 * deformation**2 * np.sqrt(energy[moving]) * kT)
    return tau


def main():
    # input: silicon.save written by the QE runs of the case (scf.in, nscf.in, proj.in)
    savedir = 'silicon.save'
    outputdir = 'output'

    # PAO Hamiltonian and band velocities on a 36x36x36 mesh
    paoflow = PAOFLOW.PAOFLOW(savedir=savedir, outputdir=outputdir, smearing='gauss')
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    paoflow.interpolated_hamiltonian(nfft1=36, nfft2=36, nfft3=36)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing(min_width=0.01)

    # transport tensors at constant relaxation time, 300 K
    paoflow.transport(tmin=300.0, tmax=300.0, emin=-1.0, emax=1.75, ne=551)

    # chemical potential of n-type Si at 1e19 electrons per cm^3
    paoflow.doping(
        tmin=300.0,
        tmax=800.0,
        nt=11,
        emin=-12.0,
        emax=2.0,
        ne=5601,
        doping_conc=-1.0e19,
    )
    temperature, mu = np.loadtxt('output/doping_n1e+19.dat', unpack=True)

    # resistivity with energy-dependent relaxation times
    tau_parameters = {
        'v': 9.0e3,  # sound velocity (m/s)
        'rho': 2.33e3,  # mass density (kg/m^3)
        'ms': 0.26,  # effective mass (m_e)
        'D_ac': 5.0,  # deformation potential (eV)
        'nI': 1.0e19,  # ionised impurities (cm^-3)
        'Zi': 1,  # impurity charge
        'eps_0': 11.7,
        'eps_inf': 11.7,
    }
    E_CBM = 0.754  # CBM of this run (eV above the VBM)
    custom_acoustic = TauModel(function=acoustic_tau, params={**tau_parameters, 'E_c': E_CBM})
    channel_sets = {
        'built-in acoustic + impurity': ['acoustic', 'impurity'],
        'custom acoustic + built-in impurity': [custom_acoustic, 'impurity'],
    }
    resistivity = {}
    for label, channels in channel_sets.items():
        rho = []
        for t, m in zip(temperature, mu):
            paoflow.transport(
                tmin=t,
                tmax=t,
                emin=m,
                emax=m,
                ne=1,
                scattering_channels=channels,
                tau_dict=tau_parameters,
            )
            row = np.loadtxt('output/sigmagauss_0.dat')
            rho.append(100.0 * 3.0 / row[2:5].sum())
        resistivity[label] = np.array(rho)

    np.savetxt(
        'output/resistivity.dat',
        np.column_stack([temperature, *resistivity.values()]),
        header='T (K), rho built-in (Ohm cm), rho custom acoustic (Ohm cm)',
    )
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
