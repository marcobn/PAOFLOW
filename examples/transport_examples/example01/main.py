from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport


def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='output/qe/al5.save',
        outputdir='output/paoflow',
        smearing='gauss',
        npool=1,
        verbose=True,
        save_overlaps=True,
    )

    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.95)
    paoflow.pao_hamiltonian(
        shift_type=1,
        expand_wedge=False,
    )
    paoflow.projections()

    transport = Transport(paoflow.data_controller)

    transport.build_hamiltonian_blocks(
        datafile_C='./output/qe/al5.save/atomic_proj.xml',
        dimC=20,
        transport_direction=3,
        calculation_type='bulk',
        use_sym=False,
        do_overlap_transformation=False,
        # --- optional debug / advanced flags (defaults shown) ---
        # debug=False,           # write .ham, projectability.txt, kovp.txt during setup
        # surface=False,         # surface-mode lead Green's function
        # ispin=0,               # spin channel for spin-polarized inputs
        # niterx=200, transfer_thr=1.0e-7, nprint=20, nfailx=5,  # lead transfer-matrix convergence
        # shift_L=0.0, shift_C=0.0, shift_R=0.0, shift_corr=0.0,  # rigid on-site energy shifts (eV)
        # do_eigenchannels=False, neigchnx=200000,               # transmission eigenchannels
        # do_eigplot=False, ie_eigplot=0, ik_eigplot=0,          # eigenchannel plotting at one (ie, ik)
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': 'ALL'},
    )

    transport.configure_energy_grid(
        emin=-7.0,
        emax=2.0,
        ne=9001,
        delta=0.0005,
        # --- optional smearing / energy knobs (defaults shown) ---
        # smearing_type='lorentzian',  # or 'gaussian', 'fermi-dirac', 'methfessel-paxton', 'marzari-vanderbilt'
        # delta_ratio=5.0e-3, xmax=25.0,                  # adaptive smearing
        # ne_buffer=1, energy_step=0.001, nx_smear=20000,
    )

    transport.configure_outputs(
        output_dir='./output/paoflow',
        postfix='_bulk',
        # --- optional output flags (defaults shown) ---
        # write_kdata=False, write_green_function=False, write_lead_self_energy=False,
    )

    transport.compute_leads_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transport.compute_transmission()
    transport.compute_dos()


if __name__ == '__main__':
    main()
