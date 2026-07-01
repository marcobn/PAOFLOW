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

    # Define which orbitals form each Hamiltonian block (required, build-time).
    transport.define_blocks(
        H00_C={'rows': 'ALL', 'cols': 'ALL'},
        H_CR={'rows': 'ALL', 'cols': 'ALL'},
    )

    transport.build_hamiltonian_blocks(
        dimC=20,
        transport_direction=3,
        calculation_type='bulk',
        use_sym=False,
        do_overlap_transformation=False,
    )

    transport.configure_onsite_shifts(shift_L=0.0, shift_C=0.0, shift_R=0.0, shift_corr=0.0)
    transport.configure_lead_convergence(
        niterx=200, transfer_thr=1.0e-7, nprint=20, nfailx=5, surface=False
    )
    transport.configure_eigenchannels(
        do_eigenchannels=False, neigchnx=200000, do_eigplot=False, ie_eigplot=0, ik_eigplot=0
    )

    transport.configure_energy_grid(
        emin=-7.0,
        emax=2.0,
        ne=9001,
        delta=0.0005,
    )

    transport.configure_outputs(
        output_dir='./output/paoflow',
        postfix='_bulk',
    )

    transport.compute_leads_self_energy(write=True)
    transport.compute_greens_functions(write=True)
    transport.compute_transmission()
    transport.compute_dos()


if __name__ == '__main__':
    main()
