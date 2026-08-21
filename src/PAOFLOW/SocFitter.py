#first the orbital that will be added SOC
# read the shells information from somewhere, put values (dict with the selected soc orbitals)
# read the FR bands, compare it kpt with ad hoc FR, 
# fit up to the last band with Proj > 95%

import numpy as np
from PAOFLOW import PAOFLOW
from PAOFLOW import GPAO
import matplotlib.pyplot as plt
import scipy.optimize as so
import matplotlib.pyplot as plt
import   numpy as np
import copy

class soc_fitter:
    def __init__(self, 
                 pao_sr,bands_soc_path,orb={},ibrav=0,fermi_level=0,nkpts=100,guess=0.2,N_extra_bands=0):
        arry, attr = pao_sr.data_controller.data_dicts()
        self.pao_sr=pao_sr
        self.pao_fit_sr=pao_sr
        self.nkpts=nkpts
        self.soc_shell_dict,self.which_pdf_dict=soc_mask(self,orb)
        self.ibrav=ibrav
        self.begin_bnd_fit,self.end_bnd_fit = calc_1st_bnd(self),attr['nelec']+N_extra_bands
        self.Ef=fermi_level
        self.guess=guess
        self.FR_bands=read_qe_bands(bands_soc_path,self.Ef)
        self.soc_strengh_dict=fit_soc_strength(self)
        plot_final_bands(self)
    
        #print(self.begin_bnd_fit,self.soc_shell_dict,self.which_pdf_dict)
    def band_error(self, params, dft_bands):

        pao_bands = self.run_paoflow_soc(params)

        nbnd_min = self.begin_bnd_fit
        nbnd_max = self.end_bnd_fit

        pao_fit = pao_bands[:, nbnd_min:nbnd_max]
        dft_fit = dft_bands[:, nbnd_min:nbnd_max]

        if pao_fit.shape != dft_fit.shape:
            raise ValueError(
                f"Band shape mismatch: "
                f"PAOFLOW={pao_fit.shape}, "
                f"DFT={dft_fit.shape}"
            )

        diff = pao_fit - dft_fit
        print(np.mean(diff**2))
        return np.mean(diff**2)
    
    def run_paoflow_soc(self, params):

        paoflow = copy.deepcopy(self.pao_fit_sr)

        soc_dict = build_soc_dict(self, params)

        print('NEW SOC dictionary:', soc_dict)

        paoflow.adhoc_spin_orbit(
            soc_strengh=soc_dict,
            soc_shell_weights=self.soc_shell_dict
        )

        paoflow.bands(
            ibrav=self.ibrav,
            nk=self.nkpts
        )

        bands = np.loadtxt('./output/bands_0.dat')

        paoflow.finish_execution()

        return bands

def fit_soc_strength(self):

    n_soc = sum(
        len(positions)
        for positions in self.which_pdf_dict.values()
    )
    initial = [self.guess] * n_soc

    res = so.minimize(
        self.band_error,
        initial,
        args=(self.FR_bands,),
        method='L-BFGS-B',
        bounds=[(0.0, None)] * len(initial),
        options={'maxiter': 25}
    )
#    res = so.minimize_scalar(
#        lambda x: self.band_error([x], self.FR_bands),
#        bounds=(0.0, 3.0),
#        method='bounded',
#        options={
#            'xatol': 1e-6,
#            'maxiter': 25
#        }
#    )
    print("SOC fitted:", res.x)

    SOC_final_dict = build_soc_dict(self,res.x)
    return SOC_final_dict


def build_soc_dict(self, params):

    soc_dict = {}

    param_index = 0

    for elem, orbitals in self.which_pdf_dict.items():

        values = [0.0, 0.0, 0.0]

        for orb in orbitals:
            if param_index >= len(params):
                raise ValueError(
                    "Number of SOC parameters is smaller than "
                    "the number of requested SOC orbitals."
                )

            values[orb] = params[param_index]
            param_index += 1

        soc_dict[elem] = values

    if param_index != len(params):
        raise ValueError(
            "Number of SOC parameters is larger than "
            "the number of requested SOC orbitals."
        )

    return soc_dict




def read_qe_bands(filename, fermi_level):

    bands = []
    current_band = []

    with open(filename, 'r') as f:

        for line in f:

            line = line.strip()

            # Empty line = end of band block
            if not line:
                if current_band:
                    bands.append(current_band)
                    current_band = []
                continue

            values = line.split()

            if len(values) < 2:
                continue

            k = float(values[0])
            energy = float(values[1]) - fermi_level

            current_band.append((k, energy))

        # Last band
        if current_band:
            bands.append(current_band)

    # Check that every band has the same number of k-points
    nk = len(bands[0])

    for i, band in enumerate(bands):
        if len(band) != nk:
            raise ValueError(
                f"QE band {i} has {len(band)} k-points, "
                f"expected {nk}"
            )

    # Energy matrix: nk × nbands
    dft_bands = np.array([
        [band[k][1] for band in bands]
        for k in range(nk)
    ])

    return dft_bands


def soc_mask(self, orb):

    arry, attr = self.pao_sr.data_controller.data_dicts()

    soc_mask = {}
    pos_pdf_dict = {}

    orbital_position = {
        'P': 0,
        'D': 1,
        'F': 2,
    }

    for elem, shells in arry['configuration'].items():
        mask = [0.0] * len(shells)
        positions = []

        if elem in orb:

            for target_shell in orb[elem]:

                target_shell = target_shell.upper()
                for i, shell in enumerate(shells):
                    if shell.upper() == target_shell:
                        mask[i] = 1.0
                orbital = target_shell[-1]

                if orbital in orbital_position:
                    positions.append(
                        orbital_position[orbital]
                    )
                else:
                    raise ValueError(
                        f"Unsupported orbital '{target_shell}' "
                        f"for element {elem}. "
                        f"S orbital LS=0, there is no SOC"
                    )
        soc_mask[elem] = mask
        pos_pdf_dict[elem] = positions

    return soc_mask, pos_pdf_dict


def calc_1st_bnd(self):
    self.pao_sr.bands(ibrav=self.ibrav, nk=self.nkpts)

    path = './output/bands_0.dat'

    energy_cutoff = -10.0

    with open(path, 'r') as f:
        line = f.readline().split()

    energies = np.array([float(x) for x in line[1:]])

    first_band = np.where(energies > energy_cutoff)[0][0]

    return 2*first_band

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_final_bands(self):

    # ---------------------------------
    # Read final PAOFLOW SOC bands
    # ---------------------------------
    pao_data = np.loadtxt('./output/bands_0.dat')

    pao_kpoints = pao_data[:, 0]
    pao_bands = pao_data[:, 1:]

    # ---------------------------------
    # DFT bands
    # ---------------------------------
    dft_bands = self.FR_bands

    # ---------------------------------
    # Check dimensions
    # ---------------------------------
    #if pao_bands.shape != dft_bands.shape:
    #    raise ValueError(
    #        f'Band shape mismatch: '
    #        f'PAOFLOW={pao_bands.shape}, '
    #        f'DFT={dft_bands.shape}'
    #    )

    # ---------------------------------
    # Band window
    # ---------------------------------
    nbnd_min = self.begin_bnd_fit
    nbnd_max = self.end_bnd_fit

    pao_fit = pao_bands[:, nbnd_min:nbnd_max]
    dft_fit = dft_bands[:, nbnd_min:nbnd_max]

    # ---------------------------------
    # Plot
    # ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(pao_fit.shape[1]):

        ax.plot(
            pao_kpoints,
            pao_fit[:, i],
            linewidth=1.5,color='blue',
            label='PAOFLOW' if i == 0 else None
        )

        ax.plot(
            pao_kpoints,
            dft_fit[:, i],
            linewidth=1.5,color='red',alpha=0.7,
            label='DFT SOC' if i == 0 else None
        )

    ax.set_xlabel(r'$\vec k$')
    ax.set_ylabel(r'$E-E_F$ (eV)')
    ax.set_title(f'{self.soc_strengh_dict}')

    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()

    # ---------------------------------
    # Save
    # ---------------------------------
    output_path = './output/final_soc_fit.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Final SOC band comparison saved to {output_path}')