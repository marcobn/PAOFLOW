import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import copy


class soc_fitter:
    """
    Compare fully relativistic (FR) DFT bands against PAOFLOW bands (SR)
        to fit and extract ad-hoc Spin-Orbit Coupling (SOC) strength parameters for target orbitals.

        Typical workflow
        ----------------
        >>> from PAOFLOW.SocFitter import soc_fitter
        >>> fitter = soc_fitter(
        ...     pao_sr=paoflow_obj,
        ...     bands_soc_path='./Al-fr.gnu',
        ...     orb={"Al": ["3P"]},
        ...     ibrav=1,
        ...     fermi_level=5.6253
        ... )

        Parameters
        ----------
        pao_sr : PAOFLOW
            A initialized scalar-relativistic PAOFLOW instance containing core electronic
            structure and data-controller dictionaries.
        bands_soc_path : str
            Path to the text dat file containing the fully-relativistic (FR) reference
            DFT bands (e.g., Quantum ESPRESSO `.gnu` or `.dat` output). Run write_QE_path=True with paoflow before doing the FR DFT
        orb : dict, optional
            Dictionary mapping element symbols to lists of targeted orbital shell labels
            for SOC fitting (e.g., ``{"Al": ["3P"], "Pt": ["5D"]}``). Default is ``{}``.
        ibrav : int, optional
            Bravais lattice index for PAOFLOW band structure generation. Default is ``0``.
        fermi_level : float, optional
            Fermi energy (in eV) used to shift reference bands relative to $E_F$. Default is ``0``.
        nkpts : int, optional
            Number of $k$-points to evaluate along the band path. Default is ``100``.
        guess : float, optional
            Initial estimate or scaling multiplier for the fitting parameters. Default is ``0.2``.
        N_extra_bands : int, optional
            Number of extra bands beyond the total electron count (``nelec``) to include
            in the fitting energy window. Default is ``0``.

        Attributes
        ----------
        pao_sr : PAOFLOW
            Reference scalar-relativistic PAOFLOW object.
        pao_fit_sr : PAOFLOW
            Working PAOFLOW object used during optimization cycles.
        nkpts : int
            Number of k-points along the high-symmetry path.
        soc_shell_dict : dict
            Binary shell-weight masks per element indicating which orbitals undergo SOC fitting.
        which_pdf_dict : dict
            Mapping of element to orbital angular momentum indices (0=P, 1=D, 2=F) subject to SOC.
        ibrav : int
            Bravais lattice flag.
        begin_bnd_fit : int
            Index of the lowest energy band included in the fitting window.
        end_bnd_fit : int
            Index of the highest band included in the fitting window (``nelec + N_extra_bands``).
        Ef : float
            Fermi energy applied to the system.
        guess : float
            Initial parameter guess provided during setup.
        FR_bands : numpy.ndarray
            Array of shape ``(nkpts, nbands)`` holding aligned reference DFT SOC band energies.
        soc_strengh_dict : dict
            Fitted SOC parameters xi (in eV) per element and orbital type after optimization.
    """

    def __init__(
        self,
        pao_sr,
        bands_soc_path,
        orb={},
        ibrav=0,
        fermi_level=0,
        nkpts=100,
        guess=0.2,
        N_extra_bands=0,
    ):
        arry, attr = pao_sr.data_controller.data_dicts()
        self.pao_sr = pao_sr
        self.pao_fit_sr = pao_sr
        self.nkpts = nkpts
        self.soc_shell_dict, self.which_pdf_dict = soc_mask(self, orb)
        self.ibrav = ibrav
        self.begin_bnd_fit, self.end_bnd_fit = calc_1st_bnd(self), attr['nelec'] + N_extra_bands
        self.Ef = fermi_level
        self.guess = guess
        self.FR_bands = read_qe_bands(bands_soc_path, self.Ef)
        self.soc_strengh_dict = fit_soc_strength(self)
        plot_final_bands(self)

        # print(self.begin_bnd_fit,self.soc_shell_dict,self.which_pdf_dict)

    #    def band_error(self, params, dft_bands):
    #
    #        pao_bands = self.run_paoflow_soc(params)
    #
    #        nbnd_min = self.begin_bnd_fit
    #        nbnd_max = self.end_bnd_fit
    #
    #        pao_fit = pao_bands[:, nbnd_min+1:nbnd_max+1]
    #        dft_fit = dft_bands[:, nbnd_min:nbnd_max]
    #
    #        if pao_fit.shape != dft_fit.shape:
    #            raise ValueError(
    #                f"Band shape mismatch: "
    #                f"PAOFLOW={pao_fit.shape}, "
    #                f"DFT={dft_fit.shape}"
    #            )
    #
    #        diff = pao_fit - dft_fit
    #        print(np.mean(diff**2))
    #        return np.mean(diff**2)
    def band_error(self, params, dft_bands):
        pao_bands = self.run_paoflow_soc(params)

        nbnd_min = self.begin_bnd_fit
        nbnd_max = self.end_bnd_fit

        pao_fit = pao_bands[:, nbnd_min + 1 : nbnd_max + 1]
        dft_fit = dft_bands[:, nbnd_min:nbnd_max]

        if pao_fit.shape != dft_fit.shape:
            raise ValueError(
                f'Band shape mismatch: ' f'PAOFLOW={pao_fit.shape}, ' f'DFT={dft_fit.shape}'
            )

        diff = pao_fit - dft_fit

        error = np.mean(diff**2)

        # ============================================================
        # Plot
        # ============================================================

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))

        nkpts, nbands = pao_fit.shape

        for ibnd in range(nbands):
            plt.plot(
                range(nkpts),
                dft_fit[:, ibnd],
                'k-',
                linewidth=1.5,
                label='DFT' if ibnd == 0 else None,
            )

            plt.plot(
                range(nkpts),
                pao_fit[:, ibnd],
                'r--',
                linewidth=1.0,
                label='PAOFLOW' if ibnd == 0 else None,
            )

        plt.axhline(0, linestyle='--', linewidth=0.8)

        plt.xlabel('k-point')
        plt.ylabel('Energy (eV)')
        plt.legend()
        plt.title(f'MSE = {error:.6e}')

        plt.tight_layout()
        plt.savefig('./output/QExPAO_bnd.png')

        print(error)

        return error

    def run_paoflow_soc(self, params):
        paoflow = copy.deepcopy(self.pao_fit_sr)

        soc_dict = build_soc_dict(self, params)

        print('NEW SOC dictionary:', soc_dict)

        paoflow.adhoc_spin_orbit(soc_strengh=soc_dict, soc_shell_weights=self.soc_shell_dict)

        paoflow.bands(ibrav=self.ibrav, nk=self.nkpts)

        bands = np.loadtxt('./output/bands_0.dat')

        paoflow.finish_execution()

        return bands


def fit_soc_strength(self):
    # single element fit guess
    # "SMART FIT"
    n_soc = sum(len(positions) for positions in self.which_pdf_dict.values())
    element = next(iter(self.soc_shell_dict))
    soc_element = estimate_soc(element)
    initial = [soc_element] * n_soc
    res = so.minimize(
        self.band_error,
        initial,
        args=(self.FR_bands,),
        method='L-BFGS-B',
        bounds=[(soc_element * 0.6, soc_element * 1.4)] * len(initial),
        options={'maxiter': 40},
    )
    print('SOC fitted:', res.x)
    SOC_final_dict = build_soc_dict(self, res.x)

    # UGLIEST FIT
    #    element=next(iter(self.soc_shell_dict))
    #    soc_element=estimate_soc(element)
    #    soc_values=np.linspace(soc_element*0.8,soc_element*3,40)
    #    list_errors=np.array([self.band_error(params=[i],dft_bands=self.FR_bands) for i in soc_values])
    #    output_path = './output/soc_vs_error.png'
    #    fig, ax = plt.subplots(figsize=(8, 6))
    #    ax.scatter(soc_values,list_errors)
    #    ax.set_xlabel(r'$\xi$ (eV)')
    #    ax.set_ylabel(r'(pao_fit - dft)$^2$ (eV$^2$)')
    #    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #    plt.close(fig)
    #    print([np.argmin(list_errors)])
    #    SOC_final_dict=build_soc_dict(self,[soc_values[np.argmin(list_errors)]])
    return SOC_final_dict


def build_soc_dict(self, params):
    soc_dict = {}

    param_index = 0

    for elem, orbitals in self.which_pdf_dict.items():
        values = [0.0, 0.0, 0.0]

        for orb in orbitals:
            if param_index >= len(params):
                raise ValueError(
                    'Number of SOC parameters is smaller than '
                    'the number of requested SOC orbitals.'
                )

            values[orb] = params[param_index]
            param_index += 1

        soc_dict[elem] = values

    if param_index != len(params):
        raise ValueError(
            'Number of SOC parameters is larger than ' 'the number of requested SOC orbitals.'
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
            raise ValueError(f'QE band {i} has {len(band)} k-points, ' f'expected {nk}')

    # Energy matrix: nk × nbands
    dft_bands = np.array([[band[k][1] for band in bands] for k in range(nk)])

    return dft_bands


def soc_mask(self, orb, soc_strenghts=None):
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
                    positions.append(orbital_position[orbital])
                else:
                    raise ValueError(
                        f"Unsupported orbital '{target_shell}' "
                        f'for element {elem}. '
                        f'S orbital LS=0, there is no SOC'
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

    return 2 * first_band


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
    # if pao_bands.shape != dft_bands.shape:
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
            linewidth=1.5,
            color='blue',
            label='PAOFLOW' if i == 0 else None,
        )

        ax.plot(
            pao_kpoints,
            dft_fit[:, i],
            linewidth=1.5,
            color='red',
            alpha=0.7,
            label='DFT SOC' if i == 0 else None,
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


def estimate_soc(El):
    """
    The soc ={} dict contains an approximate value for the soc strenght for desired shell,
    still in testing for optimal values
    """
    soc = {
        # ============================================================
        # p orbitals
        # ============================================================
        # 3p
        'Al': 0.015,
        'Si': 0.030,
        'P': 0.051,
        'S': 0.080,
        'Cl': 0.117,
        # 4p
        'Ga': 0.107,
        'Ge': 0.172,
        'As': 0.251,
        'Se': 0.344,
        'Br': 0.453,
        # 5p
        'In': 0.209,
        'Sn': 0.309,
        'Sb': 0.421,
        'Te': 0.543,
        'I': 0.686,
        # 6p
        'Tl': 0.85,
        'Pb': 0.90,
        'Bi': 1.10,
        'Po': 1.20,
        'At': 1.30,
        # ============================================================
        # d orbitals
        # ============================================================
        # 3d
        'Sc': 0.010,
        'Ti': 0.014,
        'V': 0.018,
        'Cr': 0.022,
        'Mn': 0.027,
        'Fe': 0.032,
        'Co': 0.038,
        'Ni': 0.045,
        'Cu': 0.055,
        'Zn': 0.065,
        # 4d
        'Y': 0.039,
        'Zr': 0.058,
        'Nb': 0.069,
        'Mo': 0.091,
        'Tc': 0.128,
        'Ru': 0.143,
        'Rh': 0.174,
        'Pd': 0.193,
        'Ag': 0.246,
        'Cd': 0.306,
        # 5d
        'Hf': 0.191,
        'Ta': 0.248,
        'W': 0.308,
        'Re': 0.373,
        'Os': 0.441,
        'Ir': 0.514,
        'Pt': 0.556,
        'Au': 0.556,
        'Hg': 0.63,
        # 4F
        'La': 0.2,
        'Ce': 0.2,
        'Pr': 0.2,
        'Nd': 0.2,
        'Pm': 0.2,
        'Sm': 0.2,
        'Eu': 0.2,
        'Gd': 0.2,
        'Tb': 0.2,
        'Dy': 0.2,
        'Ho': 0.2,
        'Er': 0.2,
        'Tm': 0.2,
        'Yb': 0.2,
        'Lu': 0.2,
    }

    if El not in soc:
        raise ValueError(f'No SOC estimate available for element: {El}')

    return soc[El]


def get_orbitals(dict):
    orbital = {
        # ============================================================
        # p orbitals
        # ============================================================
        # 3p
        'Al': '3P',
        'Si': '3P',
        'P': '3P',
        'S': '3P',
        'Cl': '3P',
        # 4p
        'Ga': '4P',
        'Ge': '4P',
        'As': '4P',
        'Se': '4P',
        'Br': '4P',
        # 5p
        'In': '5P',
        'Sn': '5P',
        'Sb': '5P',
        'Te': '5P',
        'I': '5P',
        # 6p
        'Tl': '6P',
        'Pb': '6P',
        'Bi': '6P',
        'Po': '6P',
        'At': '6P',
        # ============================================================
        # d orbitals
        # ============================================================
        # 3d
        'Sc': '3D',
        'Ti': '3D',
        'V': '3D',
        'Cr': '3D',
        'Mn': '3D',
        'Fe': '3D',
        'Co': '3D',
        'Ni': '3D',
        'Cu': '3D',
        'Zn': '3D',
        # 4d
        'Y': '4D',
        'Zr': '4D',
        'Nb': '4D',
        'Mo': '4D',
        'Tc': '4D',
        'Ru': '4D',
        'Rh': '4D',
        'Pd': '4D',
        'Ag': '4D',
        'Cd': '4D',
        # 5d
        'Hf': '5D',
        'Ta': '5D',
        'W': '5D',
        'Re': '5D',
        'Os': '5D',
        'Ir': '5D',
        'Pt': '5D',
        'Au': '5D',
        'Hg': '5D',
        # 4f
        'La': '4F',
        'Ce': '4F',
        'Pr': '4F',
        'Nd': '4F',
        'Pm': '4F',
        'Sm': '4F',
        'Eu': '4F',
        'Gd': '4F',
        'Tb': '4F',
        'Dy': '4F',
        'Ho': '4F',
        'Er': '4F',
        'Tm': '4F',
        'Yb': '4F',
        'Lu': '4F',
    }

    filtered = {}

    for El, orbitals in dict.items():
        if El not in orbital:
            continue

        target = orbital[El]

        filtered[El] = [orb for orb in orbitals if orb == target]

    return filtered


def build_automatic_adhoc_soc(calc):
    arry, attr = calc.data_controller.data_dicts()

    soc_shell_weights = {}
    soc_strengh = {}
    orbital_position = {
        'P': 0,
        'D': 1,
        'F': 2,
    }

    orb = get_orbitals(arry['configuration'])
    print(orb)
    for elem, shells in arry['configuration'].items():
        mask = [0.0] * len(shells)
        positions = []
        pdf_strengh = [0.0, 0.0, 0.0]
        if elem in orb:
            for target_shell in orb[elem]:
                target_shell = target_shell.upper()
                for i, shell in enumerate(shells):
                    if shell.upper() == target_shell:
                        mask[i] = 1.0
                orbital = target_shell[-1]
                pdf_strengh[orbital_position[orbital]] = estimate_soc(elem)

        soc_shell_weights[elem] = mask
        soc_strengh[elem] = pdf_strengh
    print(
        f'YOUR AUTOMATIC PARAMS FOR SOC CALCULATION ARE:\nsoc_strengh={soc_shell_weights}\nsoc_shell_weights={soc_strengh}'
    )
    return soc_strengh, soc_shell_weights
